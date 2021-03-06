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
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist, subplots
from dataclasses_json import dataclass_json

from reports.lib import kshow
from reports.lib import *

from utils.math import bits2i, sigmoid, i2bits
from datasets.mnist import stage_bwmnist

plt.style.use('dark_background')
# fsinit(join(dirname(__file__),'..','..','_pylightnix'),use_as_default=True)

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

def tenergy2(t:Task, v:np.ndarray)->float:
  sz=tisize(t)
  assert v.shape[0]==sz, f"Task-vector size mismatch"
  v2=v.reshape((v.shape[0],1))
  vv=v2@v2.transpose()
  return -np.sum((t.weights*vv)/2)


def test_tenergy():
  sz=10
  t=trandom(sz)
  v=np.random.choice([-1.0,1.0],size=(sz,))
  e=tenergy(t,v)
  e2=tenergy2(t,v)
  assert_allclose(e,e2)

def trandom(sz:int=10)->Task:
  w=np.random.uniform(-1.0,1.0,size=(sz,sz))
  for i in range(sz):
    for j in range(i,sz):
      w[j,i]=w[i,j] if i!=j else 0.0
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
  X:np.ndarray
  Y:np.ndarray

def mkds(x:np.ndarray, y:np.ndarray)->Dataset:
  assert len(x.shape)==2
  assert len(y.shape)==1
  assert x.shape[0]==y.shape[0]
  assert x.shape[0]>0, f"{x.shape}"
  ex=x[(x!=1)&(x!=-1)]
  assert all(d==0 for d in ex.shape), f"{ex.shape}"
  ey=y[(y<0)|(y>9)]
  assert all(d==0 for d in ey.shape), f"{ey.shape}"
  return Dataset(x,y)


def dsmnist(path:str, limit:Optional[int]=None)->Dataset:
  d=np.load(path,allow_pickle=True)
  x=d['x_train']
  y=d['y_train']
  if limit is not None:
    x=x[:limit]
    y=y[:limit]
  x=x.reshape((x.shape[0],-1)).astype(int)
  x[x==0.0]=-1
  x[x==1.0]=1
  return mkds(x,y)

def dssize(d:Dataset)->int:
  """ Number of images in the dataset """
  return d.X.shape[0]

def dsitemsz(d:Dataset)->int:
  """ Size of images in pixels """
  return d.X.shape[1]

def dsitem(d:Dataset,i:int)->np.ndarray:
  return d.X[i,:]

def dsslice(d:Dataset,f,t)->Dataset:
  assert f<t
  df=t-f
  f2=f%(d.X.shape[0]-df)
  return mkds(d.X[f2:f2+df,:],d.Y[f2:f2+df])

def dslogprob(d:Dataset, t:Task)->float:
  """ Calculate the log-probability of sampling the dataset `d` out of
  Gibbs-distribution with weights `t`. """
  s=0.0
  for i in range(dssize(d)):
    s+=tenergy2(t,dsitem(d,i))
    # if i%100==0:
    #   print(i)
  return s


def dslogprob2(d:Dataset, t:Task)->float:
  """ Calculate the log-probability of sampling the dataset `d` out of
  Gibbs-distribution with weights `t`. Cant swallow large datasets. """
  dX=d.X # [:1000,:]
  # assert dX.shape[2]==3
  assert len(dX.shape)==2, f"{dX.shape}"
  X=dX.reshape((dX.shape[0],dX.shape[1],1))
  return -np.sum((X@X.transpose([0,2,1]))*t.weights)/2


def test_logprob():
  sz=10
  t=trandom(sz)
  d=mkds(np.random.choice([-1,1],size=(3,sz)),
         np.random.choice(range(10),size=(3,)))
  lp1=dslogprob(d,t)
  lp2=dslogprob2(d,t)
  assert_allclose(lp1,lp2)



def gibbsP(t:Task,
           T:float=1.0,
           X0:Optional[np.ndarray]=None,
           maxsteps:Optional[int]=100,
           Xclamp:Optional[np.ndarray]=None
           )->Iterator[np.ndarray]:
  """ Gibbs-sample data points according to a Gibbs distribution for task `T`.
  Probabilities are calculated approximately. Some of the neurons may be
  "clamped" to the values specified by the optional Dataset `ds`.  """
  sz=tisize(t)
  ts=Xclamp.shape[0] if Xclamp is not None else 0
  assert ts<=sz, f"Xclamp length ({ts}) should be <= task size ({sz})."
  v=X0 if X0 is not None else np.zeros(shape=(sz,),dtype=int)
  assert len(v.shape)==1
  assert v.shape[0]==sz
  step=0
  # Clamp dirst `ts` neurons
  if Xclamp is not None:
    v[:ts]=Xclamp
  while True:
    if maxsteps is not None and step>=maxsteps:
      break
    # Sample j-th node given all other nodes
    for j in range(ts,sz):
      s=0
      for i in range(sz):
        if i!=j:
          s+=t.weights[i,j]*v[i]
      P1=sigmoid((2/T)*s)
      # print(P1)
      v[j]=np_choice([1,-1],p=[P1,1.0-P1])
    # Yield the node state vector
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


# def stat_Boltzmann(t:Task, d:Dataset, epsilon=0.01)->Iterator[Task]:
#   sz=tisize(t)
#   ds=dssize(d)
#   ts=dsitemsz(d)
#   W=deepcopy(t.weights)
#   step=0
#   T=1.0
#   while True:
#     dLdW=np.zeros(shape=t.weights.shape,dtype=float)
#     G=gibbsP(t,T,maxsteps=ds*100,d=d)
#     for nv1,v in enumerate(G):
#       for j in range(sz):
#         for i in range(sz):
#           if i!=j:
#             dLdW[i,j]+=v[i]*v[j]
#     G=gibbsP(t,T,maxsteps=ds*100,d=None)
#     for nv2,v in enumerate(G):
#       for j in range(ts,sz):
#         for i in range(sz):
#           if i!=j:
#             dLdW[i,j]-=v[i]*v[j]
#     dLdW/=2*((nv1+1)+(nv2+1))
#     W+=epsilon*dLdW/T
#     yield Task(W)
#     step+=1

def boltzstep(t:Task, d:Dataset,
              initstep:int=0,
              X0:Optional[np.ndarray]=None,
              maxsteps=100,
              T=1.0)->Iterator[Tuple[int,Task,np.ndarray]]:
  sz=tisize(t)
  ts=dsitemsz(d)
  W=deepcopy(t.weights)
  epsilon=0.01
  warmup=100
  batch=1
  step=initstep
  dbatch=dsslice(d,0,batch)
  bs=dssize(dbatch)
  X=X0 if X0 is not None else np.random.choice([1,-1],size=sz)
  Xbatch=np.random.choice([-1,1],size=(bs,sz,1))
  dLdW=np.zeros(shape=t.weights.shape,dtype=float)
  tW=t
  while True:
    if step>=maxsteps:
      break
    for i in range(bs):
      for X in gibbsP(tW,T,maxsteps=warmup,X0=X,Xclamp=dsitem(dbatch,i)):
        pass
      Xbatch[i,:,0]=X
    s1=np.sum((Xbatch@Xbatch.transpose([0,2,1]))*tW.weights,axis=0)/2
    for i,X in enumerate(gibbsP(tW,T,maxsteps=warmup+bs,X0=X,Xclamp=None)):
      if i>=warmup:
        Xbatch[i-warmup,:,0]=X
    s2=np.sum((Xbatch@Xbatch.transpose([0,2,1]))*tW.weights,axis=0)/2
    dLdW=s1-s2
    W+=epsilon*dLdW/T
    tW=Task(W)
    yield step,tW,X
    step+=1

def boltztrain(t:Task, d:Dataset):
  acc=[]
  P=dslogprob2(d,t)
  print(P)
  acc.append(P)
  for step,tn,v in boltzstep(t,d,0):
    P=dslogprob2(d,tn)
    print(step,P)
    acc.append(P)
  return acc


def KL(a, b):
  """ Calculate the KL-divergence providing that both arrays encode probability
  distributions of the same discrete random variable. """
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


@autostage(name='boltztrain',
           size=28*28+100,
           # size=28*28,
           mnist=stage_bwmnist,
           out_W=[selfref,'W.npy'],
           out_steps=[selfref,'steps.npy'],
           out_X=[selfref,'X.npy'],
           mnist_limit=1000,
           maxsteps=10000,
           T=0.01)
def stage_boltztrain(build:Build,
                     name, size, mnist,
                     out_W, out_steps, out_X, mnist_limit, maxsteps, T,
                     refcontinue:Optional[RRef]=None):
  v=5
  d=dsmnist(mnist.out,limit=mnist_limit)
  if refcontinue is not None:
    assert size==mklens(refcontinue).size.val
    t=Task(np.load(mklens(refcontinue).out_W.syspath))
    X0=np.load(mklens(refcontinue).out_X.syspath)
    step=np.load(mklens(refcontinue).out_steps.syspath)+1
    print(f'continue from {step}')
  else:
    t=trandom(dsitemsz(d))
    step=0
    X0=None
    # P=dslogprob2(d,t)
    # print('initial',P)
    print('initial')
  try:
    for s,tn,X in boltzstep(t,d,initstep=step,X0=X0,maxsteps=maxsteps,T=T):
      # P=dslogprob2(d,tn)
      # print(s,P)
      print(s)
      np.save(out_W,tn.weights)
      np.save(out_steps,s)
      np.save(out_X,X)
  except KeyboardInterrupt:
    pass


def boltzrun():
  cont=realizeU(instantiate(stage_boltztrain))

def boltzcont():
  _,clo=instantiate(stage_boltztrain)
  cont=realizeU(clo,assert_realized=clo.targets)
  return realizeU(instantiate(stage_boltztrain,refcontinue=cont._rref),
                  force_rebuild=clo.targets)


def boltzsample(T=1):
  n=6
  # T=1000000000
  # T=10000
  a=realizeU(instantiate(stage_boltztrain))
  step=np.load(a.out_steps)
  print(f"Loading Boltzmann weights after {step} steps of training")
  t=Task(np.load(a.out_W))
  print(f"Done, shape {t.weights.shape}")
  d=dsmnist(a.mnist.out,limit=a.mnist_limit)
  # print(t)
  warmup=5
  acc=[]
  G=gibbsP(t,T,maxsteps=warmup+n,Xclamp=None)
  for i,v in enumerate(G):
    if i>=warmup:
      acc.append(deepcopy(v))
  plt.figure()
  for i,v in enumerate(acc):
    plt.subplot(ceil(len(acc)/3),3,i+1)
    plt.imshow(v[:28*28].reshape((28,28)),cmap=plt.get_cmap('gray'),
               interpolation='none',
               resample=False)
  plt.savefig('_out.png')
  kittydraw('_out.png')


