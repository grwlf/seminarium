from pylightnix import (Registry, Config, Build, DRef, RRef, realize1,
                        instantiate, mklens, mkdrv, mkconfig, selfref,
                        match_only, build_wrapper, writejson, readjson,
                        writestr, filehash, shell, pyobjhash, realize,
                        match_latest, autostage, autostage_, fsinit, store_gc,
                        rmref, current_registry, mkregistry)

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

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist, subplots
from dataclasses_json import dataclass_json

from reports.lib import kshow
from reports.lib import *

from utils.math import bits2i, sigmoid, i2bits

plt.style.use('dark_background')
fsinit('_pylightnix',use_as_default=True)


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
  return bits2i(v.tolist())

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

@dataclass_json
@dataclass
class GibbsResults:
  t:Task
  Xs:List[np.ndarray]
  T:float


def gibbs(t:Task, T:float=1.0, maxsteps:Optional[int]=100)->GibbsResults:
  sz=tisize(t)
  v=np.zeros(shape=(sz,),dtype=int)
  step=0
  Xs=[]
  while True:
    if maxsteps is not None and step>=maxsteps:
      break
    for j in range(sz):
      s=0
      for i in range(sz):
        if i!=j:
          s+=t.weights[i,j]*v[i]
      P1=sigmoid((1/T)*s)
      v[j]=np_choice([1,0],p=[P1,1.0-P1])
    Xs.append(v.copy())
    step+=1
  return GibbsResults(t,Xs,T)


@dataclass_json
@dataclass
class PDist:
  pdf:np.ndarray


def mkpdist(ps:np.ndarray)->PDist:
  assert_allclose(sum(ps),1.0)
  return PDist(ps)

def tPideal(t:Task,T:float=1)->PDist:
  sz=tisize(t)
  assert sz<=10, f"Are you crazy?"
  Z=0.0
  ps=np.zeros(2**sz)
  for i in range(2**sz):
    v=np.array(i2bits(i,nbits=sz))
    p=exp(-tenergy(t,v)/T)
    Z+=p
    ps[i]=p
  return mkpdist(ps/Z)


def gibbsPI(t:Task, T:float=1.0, maxsteps:Optional[int]=100)->Iterator[PDist]:
  sz=tisize(t)
  v=np.zeros(shape=(sz,),dtype=int)
  step=0
  ps=np.zeros(2**sz)
  while True:
    if maxsteps is not None and step>=maxsteps:
      break
    for j in range(sz):
      s=0
      for i in range(sz):
        if i!=j:
          s+=t.weights[i,j]*v[i]
      P1=sigmoid((1/T)*s)
      v[j]=np_choice([1,0],p=[P1,1.0-P1])
    ps[vstamp(v)]+=1
    step+=1
    if step%100==0:
      yield mkpdist(ps/step)


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


@autostage(name='plotKL',T=1.0,out=[selfref,'out.png'])
def stage_plotKL(build:Build,name,reft,out,T=1.0):
  t=tload(reft.out)
  pd1=tPideal(t,T)
  acc=[]
  for pd2 in gibbsPI(t,T,maxsteps=100*1024):
    kl=KL(pd1.pdf,pd2.pdf)
    acc.append(kl)
  plt.close()
  plt.style.use('default')
  plt.plot(acc,label='KL-dvg')
  plt.grid()
  plt.savefig(out)
  # kittyshow()

def run():
  with current_registry(mkregistry()) as r:
    reft=stage_gibbstask()
    refp=stage_plotKL(reft=reft)
    return realize1(instantiate(refp))


