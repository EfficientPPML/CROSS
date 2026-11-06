"""Qwen masked PV geometry extending the frozen head-routing owner.

Probabilities are ciphertext inputs from the caller's separately implemented
softmax. This module does not inspect, normalize, decrypt or refresh them.
"""
from dataclasses import dataclass
import numpy as np
from encrypted_qwen_head_routing import QwenHeadRouting,SLOTS

@dataclass(frozen=True)
class QwenAttendedTile:
    output: object
    query_positions: tuple
    key_positions: tuple
    counts: dict
    level: int
    scale: float

class QwenAttentionRouting(QwenHeadRouting):
    """Borrow the predecessor's exact context/security/continuation guards."""
    def required_rotation_indices(self):
        return tuple(sorted(set(super().required_rotation_indices())|
            {s%SLOTS for s in [*range(1,len(self.layout.positions)),*[-96*h for h in range(1,16)],*[-(1<<bit) for bit in range(7)]]}))

    def attend(self,probabilities,expanded_value_tiles,*,query_tile):
        """Causal P@V into16×128 query-head features per distinct token row.

        Probabilities occupy32*h+k. Each public causal lane is routed to its
        head128 anchor and broadcast; expanded V is row-broadcast separately.
        Masking at source extraction rejects hostile future/padding values.
        The final real weighted mask consumes one level and clears inactive
        rows/features in the ideal circuit. Inputs need depth3 plus reserve2.
        """
        self._guard();positions=self.layout.tile_positions(query_tile)
        if type(expanded_value_tiles)is not tuple or len(expanded_value_tiles)!=self.layout.tile_count:
            raise ValueError('exact ordered expanded-value tile tuple required')
        def body(rotate,weighted,multiply,add,keep,guard):
            total=None
            for key,keypos in enumerate(self.layout.positions):
                shifted=rotate(probabilities,key);values=[];weights=[]
                for head in range(16):
                    mask=np.zeros((16,4096))
                    for row,pos in enumerate(positions):
                        if keypos<=pos:mask[row,head*128]=1
                    values.append(rotate(shifted,-96*head));weights.append(mask.ravel())
                repeated=weighted(values,weights,f'key{key}_probability_anchors')
                for bit in range(7):repeated=add(repeated,rotate(repeated,-(1<<bit)))
                keep('current_probability',repeated)
                source=expanded_value_tiles[key//16];values=[];weights=[]
                for row in range(len(positions)):
                    mask=np.zeros((16,4096));mask[row,:2048]=1
                    values.append(rotate(source,((key%16)-row)*4096));weights.append(mask.ravel())
                broadcast=weighted(values,weights,f'key{key}_value_broadcast')
                product=multiply(repeated,broadcast,f'key{key}_weighted_value')
                total=product if total is None else add(total,product)
                keep('aggregate',total);guard()
            mask=np.zeros((16,4096));mask[:len(positions),:2048]=1
            return weighted([total],[mask.ravel()],'attended_output_mask')
        output=self._execute((probabilities,*expanded_value_tiles),3,body)
        return QwenAttendedTile(output,positions,self.layout.positions,dict(self.last_run['counts']),self.last_run['output_level'],self.last_run['output_scale'])
