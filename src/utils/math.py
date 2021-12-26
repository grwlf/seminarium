from math import exp
from typing import (Optional,List,Dict,Any,Tuple)

Bit=int

def i2bits(i:int, nbits:Optional[int]=None)->List[Bit]:
  """ Convert integer `i` into a list of bits. Add some bits to the left to
  match the `nbits` requirement. """
  assert i>=0
  assert nbits>=i.bit_length() if nbits else True
  blen=i.bit_length()
  nbits=nbits if nbits is not None else blen
  acc:list=[]
  while i>0:
    acc.append(i&1)
    i=i>>1
  acc.extend([0 for _ in range(nbits-blen)])
  return list(reversed(acc))

def bits2i(bs:List[Bit])->int:
  """ Convert list of bits `bs` back into an integer """
  assert all([b==0 or b==1 for b in bs])
  acc:int=0
  for b in bs:
    acc<<=1
    acc|=b
  return acc

def sigmoid(x):
  return 1 / (1 + exp(-x))
