"""Capture actual native Qwen outputs for strict approximate-circuit comparisons.

This observer preserves native arithmetic and never supplies model outputs.
It supports eager, CPU float32, unpadded full-prefill contexts (including full
prefix recomputation for decode). Captures are transient float64 copies. The
historical *.inverse_sqrt labels denote complete weighted RMSNorm outputs,
not the scalar reciprocal-square-root factor.
"""
import numpy as np


class NativeCapture:
    def __init__(self,model):
        self.model=model;self.handles=[];self.active=False;self.entered=False
        self.operators={};self.blocks={};self.module=None;self.original=None;self.wrapper=None
        self.layers=tuple(model.model.layers)
        if not self.layers:raise ValueError('native capture requires decoder layers')
        self.required={f'{i}.{label}' for i in range(len(self.layers)) for label in (
            'input.inverse_sqrt','post.inverse_sqrt','q.inverse_sqrt','k.inverse_sqrt',
            'q_linear','k_linear','v_linear','q_rope','k_rope','gqa','output_linear',
            'gate_linear','up_linear','silu','gated_mlp')}|{'final.inverse_sqrt'}

    @staticmethod
    def _copy(value):
        import torch
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu' or value.dtype!=torch.float32:
            raise ValueError('native captures require actual CPU float32 tensors')
        result=value.detach().numpy().astype(np.float64,copy=True)
        if not np.isfinite(result).all():raise ValueError('native captured output is nonfinite')
        return result

    def _store(self,label,value):
        if label in self.operators:raise ValueError('duplicate native operator output: '+label)
        self.operators[label]=self._copy(value)

    def _hook(self,label,transform=None):
        def capture(module,args,result):
            if self.active:self._store(label,transform(result) if transform is not None else result)
            # Returning None leaves the actual native output unchanged.
        return capture

    def _restore(self,primary=None):
        self.active=False;self.operators={};self.blocks={};errors=[]
        if self.module is not None and self.original is not None:
            try:self.module.eager_attention_forward=self.original
            except BaseException as error:errors.append(error)
        handles,self.handles=self.handles,[]
        for handle in handles:
            try:handle.remove()
            except BaseException as error:errors.append(error)
        self.entered=False
        if errors:
            if primary is not None:
                for error in errors:primary.add_note('Native capture cleanup failed: '+repr(error))
            else:
                first=next((error for error in errors if isinstance(error,(KeyboardInterrupt,SystemExit))),errors[0])
                for error in errors:
                    if error is not first:first.add_note('Additional native capture cleanup failure: '+repr(error))
                raise first

    def __enter__(self):
        if self.entered:raise ValueError('native capture is already installed')
        from transformers.models.qwen3 import modeling_qwen3 as m
        import torch
        self.module=m;self.original=m.eager_attention_forward
        if getattr(self.original,'_strict_native_capture',False):raise ValueError('nested native capture is unsupported')
        self.entered=True
        try:
            targets={id(layer.self_attn):i for i,layer in enumerate(self.layers)}
            for i,layer in enumerate(self.layers):
                if layer.self_attn.layer_idx!=i:raise ValueError('native layer indices differ from model order')
                for label,module in (('input',layer.input_layernorm),('post',layer.post_attention_layernorm)):
                    self.handles.append(module.register_forward_hook(self._hook(f'{i}.{label}.inverse_sqrt')))
                for label,module in (('q',layer.self_attn.q_norm),('k',layer.self_attn.k_norm)):
                    def head_norm(value):
                        if value.ndim!=4:raise ValueError('native head norm must be B,S,H,D')
                        return value.transpose(1,2)
                    self.handles.append(module.register_forward_hook(self._hook(f'{i}.{label}.inverse_sqrt',head_norm)))
                for label,module in (('q_linear',layer.self_attn.q_proj),('k_linear',layer.self_attn.k_proj),
                                     ('v_linear',layer.self_attn.v_proj),('output_linear',layer.self_attn.o_proj),
                                     ('gate_linear',layer.mlp.gate_proj),('up_linear',layer.mlp.up_proj),
                                     ('silu',layer.mlp.act_fn),('gated_mlp',layer.mlp.down_proj)):
                    self.handles.append(module.register_forward_hook(self._hook(f'{i}.{label}')))
                def block(module,args,result,index=i):
                    if not self.active:return
                    if index in self.blocks:raise ValueError('duplicate native block output')
                    if not isinstance(result,tuple) or not result:raise ValueError('native decoder output must be a tuple')
                    self.blocks[index]=self._copy(result[0])
                self.handles.append(layer.register_forward_hook(block))
            def last(value):
                if value.ndim!=3:raise ValueError('native final norm must be B,S,H')
                return value[:,-1]
            self.handles.append(self.model.model.norm.register_forward_hook(self._hook('final.inverse_sqrt',last)))
            original=self.original
            def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
                if self.active:
                    if id(module) not in targets:raise ValueError('captured attention belongs to another model')
                    if module.training or dropout!=0 or query.ndim!=4 or key.ndim!=4 or value.ndim!=4:
                        raise ValueError('native capture requires eager eval attention')
                    sequence=query.shape[-2]
                    if sequence<1 or key.shape[-2]!=sequence or value.shape[-2]!=sequence:
                        raise ValueError('native capture requires full-prefix recomputation, not a KV cache')
                    if sequence>1 and attention_mask is None:raise ValueError('non-singleton native context requires a causal mask')
                    if attention_mask is not None:
                        if attention_mask.device.type!='cpu' or attention_mask.dtype!=torch.float32:
                            raise ValueError('native attention mask must be CPU float32')
                        mask=attention_mask[:,:,:,:sequence].detach().cpu().numpy()
                        target=(query.shape[0],query.shape[1],sequence,sequence)
                        expected=np.tril(np.ones((sequence,sequence),dtype=bool))
                        expanded=np.broadcast_to(mask,target);allowed=np.broadcast_to(expected,target)
                        if (not np.array_equal(expanded==0,allowed)
                                or np.any(~allowed & ~((expanded==np.finfo(np.float32).min)|np.isneginf(expanded)))):
                            raise ValueError('native attention mask differs from unpadded causal contract')
                    index=targets[id(module)]
                    self._store(f'{index}.q_rope',query);self._store(f'{index}.k_rope',key)
                result=original(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
                if self.active:
                    attended=result[0]
                    if attended.ndim!=4 or attended.shape[:2]!=(query.shape[0],query.shape[-2]):
                        raise ValueError('native attended output must be B,S,H,D')
                    self._store(f'{index}.gqa',attended.reshape(attended.shape[0],attended.shape[1],-1))
                return result
            attention._strict_native_capture=True
            self.wrapper=attention;m.eager_attention_forward=attention
            return self
        except BaseException as error:
            self._restore(error);raise

    def begin(self):
        if not self.entered or self.active:raise ValueError('native capture must be installed and inactive')
        if self.model.training:raise ValueError('native capture requires model.eval()')
        self.operators={};self.blocks={};self.active=True

    def end(self):
        if not self.active:raise ValueError('native capture is not active')
        self.active=False
        if set(self.operators)!=self.required or set(self.blocks)!=set(range(len(self.layers))):
            missing=sorted(self.required-set(self.operators));extra=sorted(set(self.operators)-self.required)
            raise ValueError(f'incomplete native operator/block capture: missing={missing}, extra={extra}, blocks={sorted(self.blocks)}')
        blocks=[self.blocks[i] for i in range(len(self.layers))];operators=dict(self.operators)
        self.blocks={};self.operators={}
        return blocks,operators

    def __exit__(self,kind,error,traceback):
        self._restore(error)
        return False
