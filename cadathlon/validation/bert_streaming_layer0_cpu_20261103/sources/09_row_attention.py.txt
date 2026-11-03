"""Plaintext slot-layout simulator, not encrypted arithmetic or a BERT backend.

Every vector models one 65,536-slot ciphertext. Exact exp/division are opaque
oracle operators; their CKKS implementation, domains and refresh errors remain
separate work. Masks depend only on public positions. Keys are masked; actual
padded query rows are preserved.
"""
from collections import Counter
import math
import numpy as np


class RowPackedAttention:
    def __init__(self, rows=16, features=4096, heads=12, head_dim=64, diagnostic_domain=None):
        if rows<1 or features<1 or rows*features!=65536 or head_dim&(head_dim-1) or heads*head_dim>features:
            raise ValueError('unsupported row/head layout')
        self.rows,self.features,self.heads,self.head_dim=rows,features,heads,head_dim
        self.slots=rows*features;self.hidden=heads*head_dim
        self.anchors=np.arange(heads)*head_dim
        self.anchor_mask=np.zeros((rows,features));self.anchor_mask[:,self.anchors]=1
        self.anchor_mask=self.anchor_mask.reshape(-1)
        self.counts=Counter();self.rotations=set();self.ranges={};self.diagnostic_domain=diagnostic_domain

    def pack_rows(self, values, tile_index):
        values=np.asarray(values,dtype=np.float64)
        if values.ndim!=2 or values.shape[1]!=self.hidden or not np.isfinite(values).all():
            raise ValueError('expected finite sequence×hidden values')
        tile=np.zeros((self.rows,self.features))
        start=tile_index*self.rows;part=values[start:start+self.rows]
        tile[:len(part),:self.hidden]=part
        return tile.reshape(-1)

    def rotate(self, value, shift):
        shift=int(shift)%self.slots
        if shift==0:return value
        self.counts['rotations']+=1;self.rotations.add(shift)
        return np.roll(value,-shift)

    def rotate_rows(self, value, offset):
        return self.rotate(value,offset*self.features)

    def head_reduce(self, value, factor=1., mask=None):
        result=value
        for bit in range((self.head_dim-1).bit_length()):
            result=result+self.rotate(result,1<<bit)
            self.counts['ciphertext_additions']+=1
        self.counts['public_multiplications']+=1
        return result*self.anchor_mask*factor*(1 if mask is None else mask)

    def head_broadcast(self, value):
        # The input must be anchor-only. Otherwise inactive features can wrap
        # into a neighboring head or token row during the negative rotations.
        if np.any(np.asarray(value)*(1-self.anchor_mask)):
            raise ValueError('head broadcast requires anchor-only input')
        result=value
        for bit in range((self.head_dim-1).bit_length()):
            result=result+self.rotate(result,-(1<<bit))
            self.counts['ciphertext_additions']+=1
        return result

    def key_mask(self, mask, tile_index, offset):
        start=tile_index*self.rows
        local=np.zeros(self.rows)
        part=np.asarray(mask,dtype=bool)[start:start+self.rows]
        local[:len(part)]=part
        aligned=np.roll(local,-offset)
        return (aligned[:,None]*self.anchor_mask.reshape(self.rows,self.features)).reshape(-1)

    def record(self,name,values):
        values=np.asarray(values)
        if not values.size:return
        row=self.ranges.setdefault(name,{'minimum':float('inf'),'maximum':float('-inf'),'count':0,'nonfinite':0})
        finite=values[np.isfinite(values)]
        if finite.size:
            row['minimum']=min(row['minimum'],float(finite.min()));row['maximum']=max(row['maximum'],float(finite.max()))
        row['count']+=int(values.size);row['nonfinite']+=int(np.sum(~np.isfinite(values)))
        if self.diagnostic_domain is not None and name.startswith('centered_'):
            low,high=self.diagnostic_domain
            row['comparison_domain']=list(self.diagnostic_domain)
            row['below']=row.get('below',0)+int(np.sum(values<low))
            row['above']=row.get('above',0)+int(np.sum(values>high))

    def attention(self,q,k,v,key_mask):
        q,k,v=map(lambda value:np.asarray(value,dtype=np.float64),(q,k,v))
        mask=np.asarray(key_mask,dtype=bool)
        if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=2 or q.shape[1]!=self.hidden or mask.shape!=(len(q),) or not mask.any():
            raise ValueError('expected equal sequence×hidden Q/K/V and a nonempty public key mask')
        if not all(np.isfinite(value).all() for value in (q,k,v)):
            raise ValueError('finite Q/K/V required')
        tiles=(len(q)+self.rows-1)//self.rows
        output=np.zeros_like(q);valid_count=int(mask.sum())
        for query_tile in range(tiles):
            query=self.pack_rows(q,query_tile);mean=np.zeros(self.slots)
            for key_tile in range(tiles):
                key=self.pack_rows(k,key_tile)
                for offset in range(self.rows):
                    valid=self.key_mask(mask,key_tile,offset)
                    product=query*self.rotate_rows(key,offset);self.counts['ciphertext_products']+=1
                    score=self.head_reduce(product,1/(math.sqrt(self.head_dim)*valid_count),valid)
                    mean+=score;self.counts['ciphertext_additions']+=1
            numerator=np.zeros(self.slots);denominator=np.zeros(self.slots)
            for key_tile in range(tiles):
                key=self.pack_rows(k,key_tile);value=self.pack_rows(v,key_tile)
                for offset in range(self.rows):
                    valid=self.key_mask(mask,key_tile,offset)
                    product=query*self.rotate_rows(key,offset);self.counts['ciphertext_products']+=1
                    score=self.head_reduce(product,1/math.sqrt(self.head_dim))
                    centered=(score-mean)*valid
                    self.counts['ciphertext_additions']+=1;self.counts['public_multiplications']+=1
                    rows=min(self.rows,len(q)-query_tile*self.rows)
                    centers=centered.reshape(self.rows,self.features)[:rows,self.anchors]
                    active=valid.reshape(self.rows,self.features)[:rows,self.anchors].astype(bool)
                    query_valid=mask[query_tile*self.rows:query_tile*self.rows+rows,None]
                    self.record('centered_valid_queries',centers[active&query_valid])
                    self.record('centered_padded_queries',centers[active&~query_valid])
                    # Invalid key slots and inactive feature columns see zero,
                    # never masked extreme scores, before the nonlinear call.
                    exponential=np.exp(centered)*valid
                    self.counts['opaque_exponentials']+=1;self.counts['public_multiplications']+=1
                    self.record('unmasked_exp',exponential[valid.astype(bool)])
                    denominator+=exponential;self.counts['ciphertext_additions']+=1
                    broadcast=self.head_broadcast(exponential)
                    numerator+=broadcast*self.rotate_rows(value,offset)
                    self.counts['ciphertext_products']+=1;self.counts['ciphertext_additions']+=1
            rows=min(self.rows,len(q)-query_tile*self.rows)
            self.record('denominator_mean_exp',denominator.reshape(self.rows,self.features)[:rows,self.anchors]/valid_count)
            # Neutral inactive denominators are public ones; no comparison or
            # secret-dependent reciprocal branch is part of this layout graph.
            inverse=1/(denominator+(1-self.anchor_mask))*self.anchor_mask
            self.counts['opaque_reciprocals']+=1;self.counts['public_multiplications']+=1
            final=numerator*self.head_broadcast(inverse);self.counts['ciphertext_products']+=1
            rows=min(self.rows,len(q)-query_tile*self.rows)
            output[query_tile*self.rows:query_tile*self.rows+rows]=final.reshape(self.rows,self.features)[:rows,:self.hidden]
        return output

    def plan(self,sequence):
        tiles=(sequence+self.rows-1)//self.rows
        pairs=tiles*tiles*self.rows
        return dict(rows=self.rows,features=self.features,slots=self.slots,sequence=sequence,tiles=tiles,
            ciphertext_slot_period=self.slots,attention_rotation_indices=sorted(self.rotations),
            actual_simulated_counts=dict(self.counts),
            expected_counts={'ciphertext_products':3*pairs+tiles,'opaque_exponentials':pairs,'opaque_reciprocals':tiles},
            critical_depth_from_block_input='7 + E + R (including one Q/K/V projection level; E/R are opaque exp/reciprocal depths)',
            output_projection_extra_levels=1,
            notes=['Exact exp/division establish only layout semantics, not realizable CKKS nonlinear operators.',
                   'The unnormalized exponential route differs from the previously composed normalized repeated-squaring circuit.',
                   'Batches may stream one example at a time; token rows are never replicas of every key/feature pair.',
                   'Only public key masks apply; all actual padded query rows are retained.'])
