import sympy

from pylightnix import (Registry, Config, Build, DRef, RRef, realize1,
                        instantiate, mklens, mkdrv, mkconfig, selfref,
                        match_only, build_wrapper, writejson, readjson,
                        writestr, filehash, shell, pyobjhash, realize,
                        match_latest, autostage, fsinit)

import numpy as np
from numpy.random import seed as np_seed, choice as np_choice
from math import sin, exp
from typing import (Any, Optional, Union, Callable, List, Tuple, TypeVar,
                    Generic)
from functools import partial
from json import dump as json_dump, load as json_load, loads as json_loads
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, hist, subplots
from dataclasses_json import dataclass_json
from math import ceil

from reports.lib import kshow
from reports.lib import *

plt.style.use('dark_background')
fsinit('_pylightnix',use_as_default=True)

@dataclass_json
@dataclass
class Task:
  size:int
  ks:List[float]

def tload(ref:Union[RRef,Build])->Task:
  return Task(mklens(ref).task_size.val,np.load(mklens(ref).out_ks.syspath))

def teval(x:int,t:Task)->float:
  """ A random function to find the minimum of """
  acc:float=0.0
  for p,k in enumerate(t.ks):
    acc+=sin(k*float(x))
  return acc

def tsuggestX(X:int,t:Task)->int:
  X2=X+np_choice([1,-1])
  if X2<0:
    X2=float(t.size-1)
  if X2>=t.size:
    X2=0
  return int(X2)


def tsize(t:Task,dim:int)->int:
  return ceil(t.size**(1.0/dim))

def tflatten(t:Task, X:List[int])->int:
  S=tsize(t,len(X))
  k=1
  acc=0
  for x in reversed(X):
    acc+=x*k
    k*=S
  return acc

def tsuggestX_dim(X:List[int],t:Task,dim:int)->List[int]:
  S=tsize(t,dim)
  while True:
    X2=np.array(X)+np_choice([1,-1],size=(dim,))
    x2=tflatten(t,X2)
    if x2>=0 and x2<t.size:
      break
  return X2.tolist()

def teval_dim(xs:List[int],t:Task)->float:
  """ A random function to find the minimum of """
  x=tflatten(t,xs)
  acc:float=0.0
  for p,k in enumerate(t.ks):
    acc+=sin(k*float(x))
  return acc

V=TypeVar('V')

@dataclass_json
@dataclass
class MetropolisResults(Generic[V]):
  Xs:List[V]
  Ys:List[float]
  Ps:List[float]
  Ac:List[int]



def metropolis(F:Callable[[V],float],
               G:Callable[[V],V],
               T:float,
               X_0:V,
               maxsteps:Optional[int]=100,
               maxaccepts:Optional[int]=None)->MetropolisResults[V]:
  """ Find the minimum of `F` using the MCMC algorithm. Evolve the F's argument
  according to the `MC` step function """
  step=0
  naccepts=0
  X=X_0
  Y=F(X)
  Xs=[]; Ys=[]; Ps=[]; Ac=[]
  while True:
    if maxsteps is not None and step>=maxsteps:
      break
    if maxaccepts is not None and naccepts>=maxaccepts:
      break
    X2=G(X)
    Y2=F(X2)
    if Y2<Y:
      Ps.append(1.0)
    else:
      Ps.append(exp((Y-Y2)/T))
    assert Ps[-1]>=0.0
    assert Ps[-1]<=1.0
    accept=np_choice([True,False],p=[Ps[-1],1.0-Ps[-1]])
    if accept:
      X=X2
      Y=Y2
    Xs.append(X)
    Ys.append(Y)
    Ac.append(1 if accept else 0)
    naccepts+=1 if accept else 0
    step+=1
  return MetropolisResults(Xs,Ys,Ps,Ac)


#  ____  _
# / ___|| |_ __ _  __ _  ___  ___
# \___ \| __/ _` |/ _` |/ _ \/ __|
#  ___) | || (_| | (_| |  __/\__ \
# |____/ \__\__,_|\__, |\___||___/
#                |___/


TASK_SIZE=100
@autostage(name='koefs',
           index=0,
           N=100,
           scale=1,
           task_size=TASK_SIZE,
           out_ks=[selfref,'ks.npy'],
           sourcedeps=[metropolis,teval,tsuggestX])
def stage_koef(build:Build,name,index,N,scale,out_ks,task_size):
  print('Generating coeffs')
  r=np.random.rand(N)
  r-=0.5
  r*=2*scale
  np.save(out_ks,r)


@autostage(T0min=0.0001,
           T0max=1e6,
           factor=2,
           waitsteps=10,
           Paccept_tgt=0.99,
           out_T0=[selfref,'T0.npy'],
           dim=4,
           sourcedeps=[metropolis,teval,tsuggestX])
def stage_findT0(build:Build,ref_koef,T0min,T0max,factor,
                 waitsteps,Paccept_tgt,out_T0,dim):
  t=tload(ref_koef._rref)
  F=partial(teval_dim,t=t)
  G=partial(tsuggestX_dim,t=t,dim=dim)
  T0i=float(T0min)
  while T0i<T0max:
    r=metropolis(F=F,G=G,T=T0i,X_0=[0]*dim,maxsteps=waitsteps)
    Paccept=np.mean(r.Ac)
    print(f"T0i {T0i} Paccept {Paccept}")
    if Paccept>=Paccept_tgt:
      break
    T0i*=factor
  np.save(out_T0,T0i)


def same(l,rtol)->bool:
  return all(np.isclose(x,l[0],rtol=rtol) for x in l)

@autostage(decay=0.85,rtol=0.001,patience=10,
           out_results=[selfref,'results.json'],
           sourcedeps=[metropolis,teval,tsuggestX],
           name='annealing')
def stage_annealing(build:Build,ref_T0,decay,rtol,patience,out_results,name):
  T=np.load(ref_T0.out_T0)
  t=tload(ref_T0.ref_koef._rref)
  dim=ref_T0.dim
  F=partial(teval_dim,t=t)
  G=partial(tsuggestX_dim,t=t,dim=dim)
  rs:List[MetropolisResults[List[int]]]=[]
  while True:
    print(f"T is {T}")
    rs.append(metropolis(F=F,G=G,T=T,X_0=rs[-1].Xs[-1] if rs else [0]*dim,
                         maxaccepts=t.size,
                         maxsteps=10*t.size))
    solutions=[r.Ys[-1] for r in rs[-patience:]]
    if len(solutions)==patience and same(solutions,rtol):
      break
    with open(out_results,'w') as f:
      json_dump([r.to_dict() for r in rs],f,indent=4) # type: ignore
    T*=decay

def stage_experiment2(r:Registry):
  ref_koef=stage_koef(r)
  ref_T0=stage_findT0(ref_koef=ref_koef,r=r)
  ref_ann=stage_annealing(ref_T0=ref_T0,r=r)
  return (ref_koef,ref_T0,ref_ann)

#  ____
# |  _ \ _   _ _ __  _ __   ___ _ __ ___
# | |_) | | | | '_ \| '_ \ / _ \ '__/ __|
# |  _ <| |_| | | | | | | |  __/ |  \__ \
# |_| \_\\__,_|_| |_|_| |_|\___|_|  |___/
#

def runA(force=False)->RRef:
  (_,_,rAnn),clo,ctx=realize(instantiate(stage_experiment2),force_rebuild=force)
  assert isinstance(rAnn,DRef)
  return ctx[rAnn][0]

ref_ann=runA()


def plotsA(rref:RRef=ref_ann):
  t:Task=tload(mklens(rref).ref_T0.ref_koef.rref)
  with open(mklens(rref).out_results.syspath) as f:
    rs=[MetropolisResults[List[int]].from_dict(x) for x in json_load(f)] # type: ignore
  fXs=np.arange(0,t.size-1)
  fYs=list(map(lambda x:teval(x,t),fXs))
  fig,(axF,axXs,ax1,ax2)=subplots(4)
  axF.plot(fXs,fYs,label='F')
  allXs=[tflatten(t,x) for x in chain(*[r.Xs for r in rs])]
  allYs=list(chain(*[r.Ys for r in rs]))
  axXs.plot(allXs,'.',color='red',label='X(time)')
  ax1.hist(allXs,label='Visited X')
  ax2.hist(allYs,color='orange',label='Visited Y')
  fig.legend()
  kshow()
  plt.close()
  print(f"Result: X={allXs[-1]} Y={allYs[-1]}")


