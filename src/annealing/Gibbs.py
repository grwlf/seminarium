from pylightnix import (Registry, Config, Build, DRef, RRef, realize1,
                        instantiate, mklens, mkdrv, mkconfig, selfref,
                        match_only, build_wrapper, writejson, readjson,
                        writestr, filehash, shell, pyobjhash, realize,
                        match_latest, autostage, autostage_, fsinit, store_gc,
                        rmref, current_registry, mkregistry, realizeU)

import numpy as np
import numpy.random
from numpy.testing import assert_allclose
from numpy.random import seed as np_seed, choice as np_choice
from math import sin, exp, ceil, log
from typing import (Any, Optional, Union, Callable, List, Tuple, TypeVar,
                    Generic, Dict, Iterator)
from functools import partial
from json import dump as json_dump, load as json_load, loads as json_loads
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain
from os.path import join, dirname

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist, subplots
from dataclasses_json import dataclass_json

from reports.lib import kshow
from reports.lib import *

from utils.math import bits2i, sigmoid, i2bits

plt.style.use('dark_background')
fsinit(join(dirname(__file__),'..','..','_pylightnix'),use_as_default=True)

@dataclass_json
@dataclass
class Task:
  weights:np.ndarray
  """ ^ A rectangular matrix of weights, only upper half matters """
def tsave(t:Task, file:str)->None:
  np.save(file,t.weights)
def tload(file:str)->Task:
  return Task(np.load(file))


def tisize(t:Task)->int:
  assert len(t.weights.shape)==2, \
    f"A flat matrix required, not {t.weights.shape}"
  assert t.weights.shape[0]==t.weights.shape[1], \
    f"A rectangular matrix required, not {t.weights.shape}"
  return t.weights.shape[0]

VStamp=int

def vstamp(v:np.ndarray)->VStamp:
  """ Calculate a uniqe stamp of `v` """
  return bits2i([(1 if b==1 else 0) for b in v.tolist()])

def tenergy(t:Task, v:np.ndarray)->float:
  sz=tisize(t)
  assert v.shape[0]==sz, f"Task-vector size mismatch"
  s=0
  for i in range(sz):
    for j in range(i+1,sz):
      s+=t.weights[i,j]*v[i]*v[j]
  return -s

def trandom(sz:int=10)->Task:
  w=np.random.uniform(-1.0,1.0,size=(sz,sz))
  for i in range(sz):
    for j in range(i+1,sz):
      w[j,i]=w[i,j]
  assert_allclose(w,w.T)
  return Task(w)

# @dataclass_json
# @dataclass
# class GibbsResults:
#   t:Task
#   Xs:List[np.ndarray]
#   T:float


# def gibbs(t:Task, T:float=1.0, maxsteps:Optional[int]=100)->GibbsResults:
#   sz=tisize(t)
#   v=np.zeros(shape=(sz,),dtype=int)
#   step=0
#   Xs=[]
#   while True:
#     if maxsteps is not None and step>=maxsteps:
#       break
#     for j in range(sz):
#       s=0
#       for i in range(sz):
#         if i!=j:
#           s+=t.weights[i,j]*v[i]
#       P1=sigmoid((1/T)*s)
#       v[j]=np_choice([1,0],p=[P1,1.0-P1])
#     Xs.append(v.copy())
#     step+=1
#   return GibbsResults(t,Xs,T)


@dataclass_json
@dataclass
class PDist:
  """ 1-D Probability distribution """
  pdf:np.ndarray


def mkpdist(ps:np.ndarray)->PDist:
  assert_allclose(sum(ps),1.0)
  return PDist(ps)

def gibbsPD_ideal(t:Task,T:float=1)->PDist:
  """ Gibbs distribution for task `T`, calculated by definition (ineffective).
  """
  sz=tisize(t)
  assert sz<=10, f"Are you crazy?"
  Z=0.0
  ps=np.zeros(2**sz)
  for i in range(2**sz):
    v=np.array([(1 if b>0 else -1) for b in i2bits(i,nbits=sz)])
    p=exp(-tenergy(t,v)/T)
    Z+=p
    ps[i]=p
  return mkpdist(ps/Z)

@dataclass_json
@dataclass
class Dataset:
  ds:np.ndarray

def mkds(ds:np.ndarray)->Dataset:
  assert len(ds.shape)==2
  assert ds.shape[0]>0
  for i in range(ds.shape[0]):
    for j in range(ds.shape[1]):
      assert ds[i,j]==1 or ds[i,j]==-1, f"ds[{i},{j}]=={ds[i,j]}"
  assert all(ds.shape)==2
  return Dataset(ds)

def dssize(d:Dataset)->int:
  return d.ds.shape[0]

def dsitem(d:Dataset,i:int)->np.ndarray:
  return d.ds[i,:]


def gibbsP(t:Task,
           T:float=1.0,
           maxsteps:Optional[int]=100,
           d:Optional[Dataset]=None
           )->Iterator[np.ndarray]:
  """ Gibbs-sample data points according to Gibbs distribution for task `T`.
  Probabilities are calculated approximately. During the sampling, some of
  neurons may be "clamped" to the points specified by the optional Dataset `ds`.
  """
  sz=tisize(t)
  ds=dssize(d) if d is not None else 0
  assert ds<=sz, f"Dataset size ({ds}) should be <= than task size ({sz})."
  if maxsteps is not None:
    assert ds<=maxsteps, \
      f"Gibbs sampler can't visit {ds} dataset items in {maxsteps} iterations"
  v=np.zeros(shape=(sz,),dtype=int)
  step=0
  dsiter=0
  while True:
    if maxsteps is not None and step>=maxsteps:
      break
    # Clamp some neurons
    a=dsitem(d,dsiter) if d is not None else np.array([])
    for j in range(ds):
      v[j]=a[j]
    dsiter=dsiter+1 if (dsiter+1)<ds else 0
    # Sample other neurons
    for j in range(ds,sz):
      s=0
      for i in range(sz):
        if i!=j:
          s+=t.weights[i,j]*v[i]
      P1=sigmoid((2/T)*s)
      v[j]=np_choice([1,-1],p=[P1,1.0-P1])
    yield v
    step+=1


def stat_PD(t:Task, gen:Iterator[np.ndarray])->Iterator[PDist]:
  sz=tisize(t)
  ps=np.zeros(2**sz)
  step=0
  for v in gen:
    ps[vstamp(v)]+=1
    if step>0 and step%100==0:
      yield mkpdist(ps/(step+1))
    step+=1


def KL(a, b):
  a=np.asarray(a,dtype=float)
  b=np.asarray(b,dtype=float)
  return np.sum(np.where(a!=0,a*np.log(a/b),0))


@autostage(name='gibbstask',sz=5,out=[selfref,'out.npy'])
def stage_gibbstask(build:Build,name,sz,out):
  t=trandom(sz)
  tsave(t,out)
  t2=tload(out)
  assert_allclose(t.weights,t2.weights)


@autostage(name='plotKL',T=1.0,out=[selfref,'out.png'],
           sourcedeps=[gibbsP,stat_PD,gibbsPD_ideal])
def stage_plotKL(build:Build,name,reft,out,T=1.0):
  t=tload(reft.out)
  pd1=gibbsPD_ideal(t,T)
  acc=[]
  for pd2 in stat_PD(t,gibbsP(t,T,maxsteps=100*1024)):
    kl=KL(pd1.pdf,pd2.pdf)
    acc.append(kl)
  assert kl<0.001
  plt.close()
  plt.style.use('default')
  plt.plot(acc,label='KL-divirgence')
  plt.grid()
  plt.legend()
  plt.gca().set_xlabel('#samples*100')
  plt.savefig(out)


def run():
  with current_registry(mkregistry()) as r:
    reft=stage_gibbstask()
    refp=stage_plotKL(reft=reft)
    return realizeU(instantiate(refp))


