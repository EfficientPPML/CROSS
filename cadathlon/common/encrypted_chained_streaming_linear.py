"""Optional chained rotations for static, keyless distinct-row CKKS linears.

The original StreamingLinear and packed-transformer defaults are unchanged.
This class borrows their validation/ownership and queue/memory helpers. Its
rotation-key inventory follows actual sparse gaps. Chaining changes key-switch
noise history; encrypted error requires workload-specific validation.
"""
from dataclasses import dataclass
from contextlib import closing
from types import SimpleNamespace
import math
import numpy as np
from encrypted_streaming_linear import StreamingLinear,RowTileLayout,_finite_real_array

def validate_rotation_strategy(strategy):
    if type(strategy) is not str or strategy not in ('direct','chained'):
        raise ValueError('rotation_strategy must be exactly direct or chained')
    return strategy


@dataclass(frozen=True)
class LinearRotationPlan:
    strategy: str
    baby_indices: tuple
    baby_steps: tuple
    giant_indices: tuple
    giant_steps: tuple
    final_giant: int
    required_keys: tuple
    nonzero_rotation_calls: int


def linear_rotation_plan(source, *, strategy='direct'):
    """Public actual gaps; sparse matrices need not share the dense3-key plan."""
    validate_rotation_strategy(strategy)
    babies=tuple(sorted({d % source.n1 for d in source.active}-{0}))
    giants=tuple(sorted({d // source.n1 * source.n1 for d in source.active},reverse=strategy=='chained'))
    if strategy=='direct':
        baby_steps=babies;giant_steps=giants;final=0
    else:
        baby_steps=tuple(index-previous for previous,index in zip((0,*babies),babies))
        giant_steps=tuple(previous-current for previous,current in zip(giants,giants[1:]))
        final=min(giants) if giants else 0
    shifts=(*baby_steps,*giant_steps,final)
    keys=tuple(sorted({shift % source.dimension for shift in shifts if shift % source.dimension}))
    calls=sum(bool(shift % source.dimension) for shift in shifts)
    return LinearRotationPlan(strategy,babies,baby_steps,giants,giant_steps,final,keys,calls)


def require_chained_rotation_keys(context, required):
    """Check initialized public key/map membership without generating/reading keys."""
    from collections.abc import Mapping
    cache=getattr(context,'_param_cache',None)
    if cache is None or context.secret_key is not None or cache.secret_key is not None or context.parameters.get('secret_key') is not None:
        raise ValueError('chained linear requires an initialized keyless context')
    if not required:return
    if getattr(cache,'per_level_rotation_keys',False):
        raise ValueError('chained linear requires supplied reusable public rotation keys')
    keys=getattr(cache,'raw_rotation_keys',None);maps=getattr(cache,'coef_maps',None)
    if not isinstance(keys,Mapping) or not isinstance(maps,Mapping):
        raise ValueError('chained linear requires public rotation-key/map inventories')
    for key in required:
        entry=keys.get(key)
        if (not isinstance(entry,Mapping) or entry.get('a') is None or entry.get('b') is None
                or maps.get(key) is None):
            raise ValueError(f'chained linear lacks supplied public key/map for rotation{key}')



def _validate_input_metadata(value):
    if type(getattr(value,'_ckks_nsd',None)) is not int or value._ckks_nsd!=1:
        raise ValueError('chained linear requires explicit integer noise-scale degree one')
    scale=getattr(value,'_ckks_scale',None)
    if isinstance(scale,(bool,np.bool_)) or not isinstance(scale,(int,float,np.integer,np.floating)):
        raise ValueError('chained linear requires a finite positive actual input scale')
    try:
        binary64_scale=float(scale)
        valid=math.isfinite(binary64_scale) and binary64_scale>0
    except (OverflowError,TypeError,ValueError):valid=False
    if not valid:raise ValueError('chained linear requires a finite positive actual input scale')


def _chained_linear(graph,value,weight,bias=None,*,input_mask):
    from encrypted_packed_qwen import MatrixDiagonals
    from encrypted_packed_transformer import (giant_chunks,pending_limit,output_payload_bytes,
        DEFAULT_EXTRA_PAYLOAD_BYTES,DEFAULT_MAX_PENDING)
    source=MatrixDiagonals(weight,graph.layout.slots,graph.features)
    plan=linear_rotation_plan(source,strategy='chained')
    bias_slots=0. if bias is None else graph.layout.feature(bias)
    if getattr(graph.arithmetic,'refresher',None) is not None:
        raise ValueError('chained linear requires static capacity without implicit refresh')
    require_chained_rotation_keys(graph.ctx,plan.required_keys)
    if not source.active:
        return graph.arithmetic.weighted_sum([value],[0.],bias=bias_slots)
    (value,)=graph.arithmetic._ensure_capacity((value,),1)
    count_bytes=output_payload_bytes(graph,value)
    babies={0:value};previous=value
    for index,step in zip(plan.baby_indices,plan.baby_steps):
        babies[index]=graph.arithmetic.rotate(previous,step);previous=babies[index]
    del previous
    groups={}
    for diagonal in source.active:
        giant=diagonal//source.n1*source.n1;groups.setdefault(giant,[]).append(diagonal)
    limit=pending_limit(count_bytes,len(groups),
        extra_payload_bytes=getattr(graph,'linear_extra_payload_bytes',DEFAULT_EXTRA_PAYLOAD_BYTES),
        max_pending=getattr(graph,'linear_max_pending',DEFAULT_MAX_PENDING))
    def make_inner(giant,indices):
        diagonals=source.materialize_diagonals(indices);values,weights=[],[]
        for index in indices:
            # Both the public mask and matrix diagonal use source coordinates.
            diagonal=diagonals[index]*np.roll(input_mask,-index)
            values.append(babies[index%source.n1]);weights.append(np.roll(diagonal,giant))
        return graph.arithmetic.weighted_sum(values,weights)
    result=None;previous_giant=None
    ordered=((giant,groups[giant]) for giant in plan.giant_indices)
    with closing(giant_chunks(ordered,make_inner,limit)) as chunks:
        for giant,inner in chunks:
            if result is None:result=inner
            else:
                rotated=graph.arithmetic.rotate(result,previous_giant-giant)
                result=graph.arithmetic.add(rotated,inner);del rotated
            previous_giant=giant;del inner
    result=graph.arithmetic.rotate(result,plan.final_giant)
    if bias is not None:
        level=graph.arithmetic._level(result);graph.arithmetic._prepare(level)
        constant=graph.arithmetic._encode_public(bias_slots,level,result._ckks_scale)
        result=graph.ctx.he_add[level].add_plain(result,constant)
    return graph.arithmetic._finish(result)


class ChainedStreamingLinear(StreamingLinear):
    """Explicit static-capacity chained schedule; borrowed context ownership.

    Construction, canonical ciphertext/security validation, stream length checks
    and clear() are inherited unchanged. Positive actual input scales need not
    equal the canonical scale: the existing weighted sum aligns plaintext scales.
    Partial row and feature padding, including bias, retain their original rules.
    Semantic bounds are not propagated. Outputs lose all _public_* tags; a later
    bound-aware consumer must declare a justified new bound.
    """
    @staticmethod
    def required_rotation_indices(weight,layout):
        if not isinstance(layout,RowTileLayout):raise TypeError('expected RowTileLayout')
        from encrypted_packed_qwen import MatrixDiagonals
        weight=_finite_real_array(weight,'public weight')
        return linear_rotation_plan(MatrixDiagonals(weight,layout.slots,layout.features),strategy='chained').required_keys

    def linear_tile(self,value,weight,bias=None,*,valid_rows,input_mask=None):
        # Reuse the inherited public-layout validators and ciphertext/security
        # gate. The base has no kernel dispatch hook, so only this small public
        # preparation/dispatch method is overridden; context ownership is shared.
        self.layout._valid_rows(valid_rows)
        weight=_finite_real_array(weight,'public weight')
        if weight.ndim!=2:raise ValueError('public weight must be a matrix')
        self.layout._width(weight.shape[0]);self.layout._width(weight.shape[1])
        if bias is not None:
            bias=_finite_real_array(bias,'public bias')
            if bias.shape!=(weight.shape[0],):raise ValueError('public bias differs from the output width')
        mask=self.layout.input_mask(weight.shape[1],valid_rows=valid_rows)
        if input_mask is not None:
            input_mask=_finite_real_array(input_mask,'public source mask')
            if input_mask.shape!=(self.layout.slots,):raise ValueError('public source mask must contain every physical slot')
            mask*=input_mask
        self._validate_ciphertext(value);_validate_input_metadata(value)
        active_layout=SimpleNamespace(slots=self.layout.slots,
            feature=lambda values:self.layout.feature(values,valid_rows=valid_rows))
        graph=SimpleNamespace(ctx=self.ctx,layout=active_layout,features=self.layout.features,arithmetic=self.arithmetic)
        result=_chained_linear(graph,value,weight,bias,input_mask=mask)
        for name in tuple(vars(result)):
            if name.startswith('_public_'):delattr(result,name)
        return result
