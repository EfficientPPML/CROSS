"""Read-only packed checkpoints, independent dense oracle and declared domains."""
import hashlib,math,re
import numpy as np
GATE=1e-8
class TrajectoryFailure(ValueError):pass

def metric(actual,expected):
    a=np.asarray(actual);e=np.asarray(expected)
    result=dict(passed=False,finite=False,shape_valid=a.shape==e.shape,max_abs_error=None,count=int(a.size),gate=GATE)
    if a.dtype.kind not in 'fiu' or e.dtype.kind not in 'fiu' or not result['shape_valid'] or not a.size:return result
    if not np.isfinite(a).all() or not np.isfinite(e).all():return result
    with np.errstate(over='ignore',invalid='ignore'):error=np.max(np.abs(a.astype(np.float64)-e.astype(np.float64)))
    if not np.isfinite(error):return result
    result.update(finite=True,max_abs_error=float(error),passed=bool(error<=GATE));return result

def packed_expected(array,kind,sequence):
    a=np.asarray(array,dtype=np.float64)
    if kind=='heads':
        if a.ndim!=4 or a.shape[0]!=1 or a.shape[2]!=sequence or a.shape[3]!=128:raise ValueError('head oracle shape differs')
        rows=a[0].transpose(1,0,2).reshape(sequence,-1)
    elif kind=='final':
        if a.shape!=(1,1024):raise ValueError('final norm oracle shape differs')
        rows=a
    else:
        if a.ndim!=3 or a.shape[0]!=1 or a.shape[1]!=sequence:raise ValueError('row oracle shape differs')
        rows=a[0]
    if rows.shape[1]>4096:raise ValueError('oracle row wider than physical layout')
    result=np.zeros((16,4096));result[:len(rows),:rows.shape[1]]=rows
    return result.ravel()

class Observer:
    def __init__(self,reference,manifest,positions,*,publish=None):
        self.reference=reference;self.manifest=manifest;self.positions=positions;self.sequence=len(positions)
        if self.sequence not in (1,8):raise ValueError('two predeclared trajectory sizes only')
        self.publish=publish;self.operators={};self.blocks={};self.domains={};self.logits={};self.duplicates={};self.last_label=None;self.callbacks=0
    def domain(self,label,value,interval):
        v=np.asarray(value);good=np.isfinite(v)
        row=dict(domain=list(interval),count=int(v.size),minimum=float(v[good].min()) if good.any() else None,
                 maximum=float(v[good].max()) if good.any() else None,below=int(np.sum(v<interval[0])),above=int(np.sum(v>interval[1])),nonfinite=int(np.sum(~good)))
        if label in self.domains:raise TrajectoryFailure('duplicate domain checkpoint: '+label)
        self.domains[label]=row
        if row['below'] or row['above'] or row['nonfinite']:raise TrajectoryFailure('actual packed declared domain failure: '+label)
    def _record(self,target,key,value,expected):
        rawhash=hashlib.sha256(value.tobytes()).hexdigest()
        row=metric(value,expected);row['actual_sha256']=rawhash;row['expected_sha256']=hashlib.sha256(np.asarray(expected).tobytes()).hexdigest()
        if key in target:
            # The final child and parent deliberately expose the same output.
            if key!='final.inverse_sqrt' or target[key]!=row:raise TrajectoryFailure('inconsistent repeated checkpoint: '+str(key))
            self.duplicates[key]=self.duplicates.get(key,0)+1;return
        target[key]=row
        if not row['passed']:raise TrajectoryFailure('ideal all-slot mismatch: '+str(key))
    def __call__(self,label,value,metadata):
        self.last_label=label;self.callbacks+=1;x=value.polynomial
        if type(x)is not np.ndarray or x.dtype!=np.dtype('float64') or x.shape!=(65536,) or not np.isfinite(x).all():raise TrajectoryFailure('nonfinite/wrong physical callback payload: '+label)
        rows=x.reshape(16,4096);m=self.manifest
        norm=re.fullmatch(r'layer(\d+)\.(attention|tail)\.tile0\.(input|q|k|post)_norm\.variance',label)
        final_variance=label=='final.norm.variance'
        if norm or final_variance:
            site='final.inverse_sqrt' if final_variance else norm[1]+'.'+norm[3]+'.inverse_sqrt'
            width=128 if norm and norm[3] in ('q','k') else 1024
            # This callback is before broadcast; every anchor, including dummy
            # inactive anchors, must satisfy the declared polynomial domain.
            self.domain(label,rows[:,::width],m['nonlinear'][site]['domain'])
        soft=re.fullmatch(r'layer(\d+)\.attention\.tile0\.softmax\.K(\d+)\.(.*)',label)
        if soft:
            layer,k,stage=int(soft[1]),int(soft[2]),soft[3];heads=rows[k-1,:512].reshape(16,32);rule=m['softmax'][str(layer)]
            if stage=='normalized_t_branch_mask':
                t=heads[:,:k];self.domain(label+'.centered',t*2.**(rule['squarings']-1),rule['centered_domain'])
                self.domain(label+'.exp_argument',t*.5,m['exp']['domain'])
            elif stage=='initial_mean_anchor_dummy1':self.domain(label,rows[:,::32],m['initial_reciprocal']['domain'])
            elif re.fullmatch(r'round\d+.denominator_anchor_dummy1',stage):self.domain(label,rows[:,::32],m['reciprocal_by_keys'][str(k)]['domain'])
        gate=re.fullmatch(r'layer(\d+)\.tail\.tile0\.gate.linear',label)
        if gate:
            rule=m['nonlinear'][gate[1]+'.silu'];active=rows[:self.sequence,:3072]
            self.domain(label,active,rule['domain'])
            if rule['method']=='rational_tanh':self.domain(label+'.tanh_base_argument',active*2.**-(rule['doublings']+1),rule['base']['domain'])
        denominator=re.fullmatch(r'layer(\d+)\.tail\.tile0\.silu\.double\d+.denominator',label)
        if denominator:self.domain(label,x,m['nonlinear'][denominator[1]+'.silu']['reciprocal']['domain'])
        name=None;kind='rows'
        a=re.fullmatch(r'layer(\d+)\.attention\.tile0\.(input|q|k)_norm\.output',label)
        if a:name=a[1]+'.'+a[2]+'.inverse_sqrt';kind='heads' if a[2]in('q','k')else'rows'
        a=re.fullmatch(r'layer(\d+)\.attention\.tile0\.([qkv]).linear',label)
        if a:name=a[1]+'.'+a[2]+'_linear'
        a=re.fullmatch(r'layer(\d+)\.attention\.tile0\.([qk]).rope.output',label)
        if a:name=a[1]+'.'+a[2]+'_rope';kind='heads'
        a=re.fullmatch(r'layer(\d+)\.attention.output0',label)
        if a:name=a[1]+'.gqa'
        a=re.fullmatch(r'layer(\d+)\.tail\.tile0\.(o|gate|up|down).linear',label)
        if a:name=a[1]+'.'+{'o':'output_linear','gate':'gate_linear','up':'up_linear','down':'gated_mlp'}[a[2]]
        a=re.fullmatch(r'layer(\d+)\.tail\.tile0\.post_norm.output',label)
        if a:name=a[1]+'.post.inverse_sqrt'
        a=re.fullmatch(r'layer(\d+)\.tail\.tile0\.silu.output',label)
        if a:name=a[1]+'.silu'
        if label=='final.norm.output':name='final.inverse_sqrt';kind='final'
        if name:self._record(self.operators,name,x,packed_expected(self.reference.operators[name],kind,self.sequence))
        a=re.fullmatch(r'layer(\d+).block.output0',label)
        if a:
            layer=int(a[1]);self._record(self.blocks,str(layer),x,packed_expected(self.reference.blocks[layer],'rows',self.sequence))
            if self.publish:self.publish(dict(phase='owner_block_completed',layer=layer,callbacks=self.callbacks,operators=len(self.operators),domains=len(self.domains)))
        a=re.fullmatch(r'logits.(\d+)',label)
        if a:
            i=int(a[1]);start=i*4096;stop=min(start+4096,151936);e=np.zeros(65536);e[:stop-start]=self.reference.logits[0,start:stop]
            self._record(self.logits,str(i),x,e)
    def summary(self):
        return dict(callbacks=self.callbacks,last_label=self.last_label,operators=self.operators,blocks=self.blocks,logits=self.logits,domains=self.domains,duplicates=self.duplicates)
    def complete(self):
        if set(self.operators)!=set(self.reference.operators) or len(self.operators)!=421 or set(self.blocks)!={str(i) for i in range(28)} or set(self.logits)!={str(i) for i in range(38)}:raise TrajectoryFailure('complete 421 operators/28 blocks/38 all-slot logits required')
        norms=[name for name in self.domains if name.endswith('.variance')]
        if len(norms)!=113:raise TrajectoryFailure('all113 actual normalization domains required')
        expected_soft=(self.sequence-1)*sum(3+r['squarings'] for r in self.manifest['softmax'].values())
        expected_silu=28+sum(1 for k,r in self.manifest['nonlinear'].items() if k.endswith('.silu') and r['method']=='rational_tanh')+sum(r.get('doublings',0) for k,r in self.manifest['nonlinear'].items() if k.endswith('.silu'))
        if len(self.domains)!=113+expected_soft+expected_silu:raise TrajectoryFailure('complete actual nonlinear domain inventory required')
        return dict(passed=True,operators=421,blocks=28,logit_blocks=38,logits=151936,physical_metric_slots=(421+28+38)*65536,domains=len(self.domains))
