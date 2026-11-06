"""Synthetic phase specification for a future diagnostic; no ciphertext I/O."""
import math

class PhaseLedger:
    """Opaque checkpoint order and post-server client admission, not an HE guard."""
    def __init__(self, checkpoints):
        self.expected=tuple((row['index'],row['kind'],row['label'],row['level']) for row in checkpoints)
        if not self.expected or any(type(x[0]) is not int or x[0]!=i for i,x in enumerate(self.expected)):
            raise ValueError('contiguous exact checkpoint indices required')
        self.position=0;self.state='server';self.receipts=[]
    def saved(self,index,*,kind,label,level,scale,nsd,actual_expected_scale,sha256):
        if self.state!='server' or type(index)is not int or self.position>=len(self.expected):raise ValueError('server phase/order')
        if type(level)is not int or type(nsd)is not int or nsd!=1:raise ValueError('exact metadata types')
        if any(type(v)is not float or not math.isfinite(v) or v<=0 for v in (scale,actual_expected_scale)):raise ValueError('finite actual scale required')
        if (index,kind,label,level)!=self.expected[self.position] or scale!=actual_expected_scale:raise ValueError('checkpoint contract mismatch')
        if type(sha256)is not str or len(sha256)!=64 or any(c not in '0123456789abcdef' for c in sha256):raise ValueError('opaque file digest required')
        self.receipts.append((index,sha256));self.position+=1
    def seal(self,*,source_verified,keyless_verified,server_group_absent):
        if self.state!='server' or self.position!=len(self.expected) or any(x is not True for x in (source_verified,keyless_verified,server_group_absent)):
            raise ValueError('complete server outputs and public terminal evidence required')
        self.state='sealed'
    def decode(self,index):
        if self.state not in ('sealed','client') or type(index)is not int or not 0<=index<len(self.expected):raise ValueError('private decode before server seal')
        self.state='client';return self.receipts[index]
    def fail(self):self.state='failed'
