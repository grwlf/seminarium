import sympy

from pylightnix import (Registry, Config, Build, DRef, RRef, realize1,
                        instantiate, mklens, mkdrv, mkconfig, selfref,
                        match_only, build_wrapper, writejson, readjson,
                        writestr, filehash, shell, pyobjhash, realize,
                        match_latest)

import numpy as np
from numpy.random import seed as np_seed, choice as np_choice
from math import sin, exp
from typing import (Any, Optional, Union, Callable, List, Tuple)
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

plt.style.use('dark_background')


@dataclass_json
@dataclass
class Task:
  size:int
  ks:List[float]

def tload(ref:Union[RRef,Build])->Task:
  return Task(
    mklens(ref).task_size.val,
    np.load(mklens(ref).ref_koefs.out_ks.syspath))

def teval(x:float,t:Task)->float:
  """ A random function to find the minimum of """
  acc:float=0.0
  for p,k in enumerate(t.ks):
    acc+=sin(k*x)
  return acc

def tsuggestX(X:float,t:Task)->float:
  X2=X+np_choice([1,-1])
  if X2<0:
    X2=float(t.size-1)
  if X2>=t.size:
    X2=0
  return float(X2)


@dataclass_json
@dataclass
class MetropolisResults:
  Xs:List[float]
  Ys:List[float]
  Ps:List[float]
  Ac:List[int]


def metropolis(F:Callable[[float],float],
               G:Callable[[float],float],
               T:float,
               X_0:float,
               maxsteps:Optional[int]=100,
               maxaccepts:Optional[int]=None)->MetropolisResults:
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


def stage_koefs(index:int, r:Registry)->DRef:
  def config():
    nonlocal index
    name='koefs'
    N=100
    scale=1
    out_ks = [selfref, 'ks.npy']
    depends = pyobjhash([metropolis,teval,tsuggestX,stage_koefs])
    return mkconfig(locals())
  def make(b:Build):
    # np_seed(17)
    print('Generating coeffs')
    r=np.random.rand(mklens(b).N.val)
    r-=0.5
    r*=2*mklens(b).scale.val
    np.save(mklens(b).out_ks.syspath,r)
  return mkdrv(config(), match_only(), build_wrapper(make), r)

def stage_metropolis(ref_koefs:DRef, r:Registry):
  def config():
    nonlocal ref_koefs
    task_size=100
    steps=100*task_size
    T=4.0
    out_results=[selfref, "results.json"]
    depends = pyobjhash([metropolis,teval,tsuggestX,stage_metropolis])
    return mkconfig(locals())
  def make(b:Build):
    print('Running metropolis')
    t=tload(b)
    F=partial(teval,t=t)
    G=partial(tsuggestX,t=t)
    mr=metropolis(F=F,G=G,
                    T=mklens(b).T.val,
                    X_0=0,
                    maxsteps=mklens(b).steps.val)
    with open(mklens(b).out_results.syspath,'w') as f:
      json_dump(mr.to_dict(),f,indent=4) # type: ignore
  return mkdrv(config(), match_latest(), build_wrapper(make), r=r)


def stage_findT0(ref_metr:DRef, r:Registry):
  def config():
    nonlocal ref_metr
    T0min=0.0001
    T0max=1e6
    factor=2
    waitsteps=10
    Paccept_tgt=0.99
    out_T0 = [selfref, 'T0.npy']
    depends = pyobjhash([metropolis,teval,tsuggestX,stage_findT0])
    return mkconfig(locals())
  def make(b:Build):
    t=tload(mklens(b).ref_metr.rref)
    F=partial(teval,t=t)
    G=partial(tsuggestX,t=t)
    T0i=float(mklens(b).T0min.val)
    while T0i<mklens(b).T0max.val:
      r=metropolis(F=F,G=G,T=T0i,X_0=0,maxsteps=mklens(b).waitsteps.val)
      Paccept=np.mean(r.Ac)
      print(f"T0i {T0i} Paccept {Paccept}")
      if Paccept>=mklens(b).Paccept_tgt.val:
        break
      T0i*=mklens(b).factor.val
    np.save(mklens(b).out_T0.syspath,T0i)
  return mkdrv(config(), match_latest(), build_wrapper(make), r=r)


def same(l,rtol)->bool:
  return all(np.isclose(x,l[0],rtol=rtol) for x in l)

def stage_annealing(ref_T0:DRef, r:Registry):
  def config():
    nonlocal ref_T0
    decay = 0.85
    rtol = 0.001
    patience = 10
    out_results = [selfref, 'results.json']
    depends = pyobjhash([metropolis,teval,tsuggestX,stage_annealing])
    v=3
    return mkconfig(locals())
  def make(b:Build):
    T=np.load(mklens(b).ref_T0.out_T0.syspath)
    t=tload(mklens(b).ref_T0.ref_metr.rref)
    decay=mklens(b).decay.val
    patience=mklens(b).patience.val
    rtol=mklens(b).rtol.val
    F=partial(teval,t=t)
    G=partial(tsuggestX,t=t)
    rs:List[MetropolisResults]=[]
    while True:
      print(f"T is {T}")
      rs.append(metropolis(F=F,G=G,T=T,X_0=rs[-1].Xs[-1] if rs else 0,
                           maxaccepts=t.size,
                           maxsteps=10*t.size))
      solutions=[r.Ys[-1] for r in rs[-patience:]]
      if len(solutions)==patience and same(solutions,rtol):
        break
      with open(mklens(b).out_results.syspath,'w') as f:
        json_dump([r.to_dict() for r in rs],f,indent=4) # type: ignore
      T*=decay
  return mkdrv(config(), match_latest(), build_wrapper(make), r=r)


def stage_experiment1(r:Registry):
  ref_koefs=stage_koefs(0,r)
  ref_metr=stage_metropolis(ref_koefs,r)
  ref_T0=stage_findT0(ref_metr,r)
  return (ref_metr,ref_T0)

def stage_experiment2(r:Registry):
  ref_koefs=stage_koefs(0,r)
  ref_metr=stage_metropolis(ref_koefs,r)
  ref_T0=stage_findT0(ref_metr,r)
  ref_ann=stage_annealing(ref_T0,r)
  return (ref_metr,ref_T0,ref_ann)

#  ____
# |  _ \ _   _ _ __  _ __   ___ _ __ ___
# | |_) | | | | '_ \| '_ \ / _ \ '__/ __|
# |  _ <| |_| | | | | | | |  __/ |  \__ \
# |_| \_\\__,_|_| |_|_| |_|\___|_|  |___/
#

def run()->Tuple[RRef,RRef,MetropolisResults]:
  (r,rT0),clo,ctx=realize(instantiate(stage_experiment1))
  assert isinstance(r,DRef)
  assert isinstance(rT0,DRef)
  with open(mklens(ctx[r][0]).out_results.syspath) as f:
    mr=MetropolisResults.from_dict(json_load(f)) # type: ignore
  return (ctx[r][0],ctx[rT0][0],mr)

curref,curT0,curmr=run()
print('T0 =',np.load(mklens(curT0).out_T0.syspath))

def runA(force=False)->RRef:
  (_,_,rAnn),clo,ctx=realize(instantiate(stage_experiment2),force_rebuild=force)
  assert isinstance(rAnn,DRef)
  return ctx[rAnn][0]

# currefT0=realize1(instantiate(stage_findT0

def plots(rref=curref,mr=curmr):
  t:Task=tload(rref)
  allXs=np.arange(0,t.size-1)
  Ys=list(map(lambda x:teval(x,t),allXs))
  fig,(axF,axXs,ax1,ax2)=subplots(4)
  axF.plot(allXs,Ys,label='F')
  axXs.plot(mr.Xs,color='red',label='X(time)')
  ax1.hist(mr.Xs,label='Visited X')
  ax2.hist(mr.Ys,color='orange',label='Visited Y')
  fig.legend()
  kshow()
  plt.close()


def plotsA(rref:RRef):
  t:Task=tload(mklens(rref).ref_T0.ref_metr.rref)
  with open(mklens(rref).out_results.syspath) as f:
    rs=[MetropolisResults.from_dict(x) for x in json_load(f)] # type: ignore
  fXs=np.arange(0,t.size-1)
  fYs=list(map(lambda x:teval(x,t),fXs))
  fig,(axF,axXs,ax1,ax2)=subplots(4)
  axF.plot(fXs,fYs,label='F')
  allXs=list(chain(*[r.Xs for r in rs]))
  allYs=list(chain(*[r.Ys for r in rs]))
  axXs.plot(allXs,color='red',label='X(time)')
  ax1.hist(allXs,label='Visited X')
  ax2.hist(allYs,color='orange',label='Visited Y')
  fig.legend()
  kshow()
  plt.close()
  print(f"Result: X={allXs[-1]} Y={allYs[-1]}")


