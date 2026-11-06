"""Independent dense public-toy formulas; lazy NumPy supplied by the admitted caller.

No packed graph, layout, BSGS or primitive method is called. Stored reference
records contain one1024-slot period; literal replay compares every64 replica.
The same rounded public coefficients are inputs, not independently calibrated.
"""
def references(np, config, weights, embeddings, rules):
    B,S,H,F,NH,KV,D=2,4,16,32,4,2,8
    if embeddings.shape!=(B,S,H):raise ValueError('fixed toy embedding shape')
    records={};indices={};domains=[];failure=None
    def grid_rows(x):
        result=np.zeros((B,S,S,F),dtype=np.float64)
        result[...,:x.shape[-1]]=x[:,:,None,:]
        return result
    def record(kind,label,grid):
        key=(kind,str(label))
        if key in records:raise ValueError('duplicate dense semantic record')
        nonlocal failure
        if not np.isfinite(grid).all():
            failure={'kind':kind,'label':str(label),'finite_entries':int(np.isfinite(grid).sum()),'total_entries':int(grid.size)}
            raise FloatingPointError('nonfinite independent dense stage')
        records[key]=np.array(grid,dtype=np.float64,copy=True).reshape(1024)
    def number(kind):
        i=indices.get(kind,0);indices[kind]=i+1;return str(i)
    def polynomial(grid,name,weight=1.):
        label=number('polynomial')+':'+name;record('polynomial_before',label,grid)
        rule=rules[name];lo,hi=rule.domain
        domains.append(dict(label=label,minimum=float(grid.min()),maximum=float(grid.max()),domain=[lo,hi],passed=bool(np.all((grid>=lo)&(grid<=hi)))))
        coefficients=np.polynomial.chebyshev.cheb2poly(rule.coefficients)
        if rule.input_power_basis:
            argument=grid;coefficients=coefficients/(hi**np.arange(len(coefficients)))
        else:argument=grid*(2/(hi-lo)) + (-(hi+lo)/(hi-lo))
        powers={1:argument}
        def power(degree):
            if degree not in powers:
                powers[degree]=power(degree//2)*power((degree+1)//2)
            return powers[degree]
        terms=[power(n)*(coefficients[n]*weight) for n in range(1,len(coefficients))if coefficients[n]!=0]
        out=sum(terms)+coefficients[0]*weight if terms else np.zeros_like(argument)+coefficients[0]*weight
        record('polynomial_after',label,out);return out
    def linear(x,name,source_heads=None):
        label=number('linear');record('linear_before',label,grid_rows(x))
        result=x@weights[name+'.weight'].T
        if source_heads is not None:
            result=np.repeat(result.reshape(B,S,source_heads,D),NH//source_heads,axis=2).reshape(B,S,NH*D)
        record('linear_after',label,grid_rows(result));return result
    def norm(x,name):
        label=number('normalize');record('normalize_before',label,grid_rows(x))
        variance=np.sum(x*x,axis=-1,keepdims=True)*(1/H)+config['rms_norm_eps']
        vg=np.broadcast_to(variance[:,:,None,:],(B,S,S,F)).copy()
        inverse=polynomial(vg,'inverse_sqrt')[:,:,0,0:1]
        result=(x*inverse)*weights[name+'.weight']
        record('normalize_after',label,grid_rows(result));return result
    def head_norm(x,name):
        label=number('normalize_heads');record('normalize_heads_before',label,grid_rows(x))
        heads=x.reshape(B,S,NH,D)
        variance=np.sum(heads*heads,axis=-1,keepdims=True)*(1/D)+config['rms_norm_eps']
        vg=np.broadcast_to(variance,(B,S,NH,D)).reshape(B,S,F)
        inverse=polynomial(grid_rows(vg),'inverse_sqrt')[:,:,0,:].reshape(B,S,NH,D)
        result=((heads*inverse)*weights[name+'.weight']).reshape(B,S,F)
        record('normalize_heads_after',label,grid_rows(result));return result
    def rope(x):
        label=number('rope');record('rope_before',label,grid_rows(x))
        heads=x.reshape(B,S,NH,D)
        frequency=config['rope_theta']**(-2*np.arange(D//2,dtype=np.float64)/D)
        angle=np.arange(S,dtype=np.float64)[None,:,None,None]*np.concatenate((frequency,frequency))[None,None,None,:]
        rotated=np.concatenate((-heads[...,D//2:],heads[...,:D//2]),axis=-1)
        out=(heads*np.cos(angle)+rotated*np.sin(angle)).reshape(B,S,F)
        record('rope_after',label,grid_rows(out));return out
    valid=np.tril(np.ones((S,S),dtype=np.float64))[None,:,:,None]
    counts=np.arange(1,S+1,dtype=np.float64)[None,:,None,None]
    def attention(q,k,v):
        label=number('attention');record('attention_before',label,grid_rows(q))
        qh=q.reshape(B,S,NH,D).transpose(0,2,1,3);kh=k.reshape(B,S,NH,D).transpose(0,2,1,3);vh=v.reshape(B,S,NH,D).transpose(0,2,1,3)
        raw=(qh@kh.swapaxes(-1,-2))*(1/(D**.5))
        score=np.repeat(raw.transpose(0,2,3,1),D,axis=-1)*valid
        mean=np.sum(score,axis=2,keepdims=True)*(1/counts)
        centered=score-mean
        exp=polynomial(centered,'exp',valid)
        denominator=np.broadcast_to(np.sum(exp,axis=2,keepdims=True)*(1/counts),(B,S,S,F)).copy()
        inverse=polynomial(denominator,'reciprocal',1/counts)
        probability=exp*inverse
        probabilities=probability.reshape(B,S,S,NH,D)[...,0].transpose(0,3,1,2)
        out=(probabilities@vh).transpose(0,2,1,3).reshape(B,S,F)
        record('attention_after',label,grid_rows(out));return out
    logits=None;complete=False;error_type=None
    try:
        x=np.array(embeddings,dtype=np.float64,copy=True);record('input','original_embeddings',grid_rows(x))
        for layer in range(3):
            p=f'model.layers.{layer}.';a=p+'self_attn.'
            z=norm(x,p+'input_layernorm')
            v=linear(z,a+'v_proj',KV);q=linear(z,a+'q_proj',NH);k=linear(z,a+'k_proj',KV)
            q=rope(head_norm(q,a+'q_norm'));k=rope(head_norm(k,a+'k_norm'))
            attended=attention(q,k,v)
            x=x+linear(attended,a+'o_proj');record('phase',f'packed Qwen layer {layer}: attention ready',grid_rows(x))
            z=norm(x,p+'post_attention_layernorm')
            raw_gate=linear(z,p+'mlp.gate_proj');active=np.zeros(F);active[:24]=1.
            gate=polynomial(grid_rows(raw_gate),'silu',active)[:,:,0,:24]
            up=linear(z,p+'mlp.up_proj')
            x=x+linear(gate*up,p+'mlp.down_proj');record('phase',f'packed Qwen layer {layer}: MLP ready',grid_rows(x))
        final_norm=norm(x,'model.norm');logits=linear(final_norm,'model.embed_tokens')
        record('output','logits',grid_rows(logits))
        complete=True
    except (FloatingPointError,OverflowError) as error:
        error_type=type(error).__name__
    return {'records':records,'domains':domains,'logits':None if logits is None else logits[:,-1,:], 'counts':indices,'complete':complete,'error_type':error_type,'first_nonfinite':failure}
