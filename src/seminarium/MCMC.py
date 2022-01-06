import sympy

from pylightnix import (Registry, Config, Build, DRef, RRef, realize1,
                        instantiate, mklens, mkdrv, mkconfig, selfref,
                        match_only, build_wrapper, writejson, readjson,
                        writestr, filehash, shell, pyobjhash, realize,
                        match_latest, autostage, autostage_, fsinit, store_gc,
                        rmref)

import numpy as np
import numpy.random
from numpy.random import seed as np_seed, choice as np_choice
from math import sin, exp, ceil, log
from typing import (Any, Optional, Union, Callable, List, Tuple, TypeVar,
                    Generic)
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

from multiprocessing import pool,Pool
def _f(a,f):
  f(**a)
def pool_starstarmap(p:pool.Pool, f:Callable[...,Any], lkwargs:List[dict]):
  p.map(partial(_f,f=f),lkwargs)


plt.style.use('dark_background')
# fsinit(join(dirname(__file__),'..','..','_pylightnix'),use_as_default=True)

@dataclass_json
@dataclass
class Task:
  """ Task describes a particular task of finding the minimum of a
  function specified in `teval`. `ks` define the coefficients required to do the
  calculation """
  size:int
  ks:List[float]

def tload(ref:Union[RRef,Build])->Task:
  return Task(mklens(ref).task_size.val,np.load(mklens(ref).out_ks.syspath))

def teval(x:int,t:Task)->float:
  """ A random function to find the minimum of """
  acc:float=0.0
  xf=float(x)/t.size
  for p,k in enumerate(t.ks):
    acc+=sin(500*k*xf)
  acc+=(1000/8)*(xf-0.5)*(xf-0.5)
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
  scale=float(t.size)/float(S**len(X))
  k=1
  acc=0
  for x in reversed(X):
    acc+=x*k
    k*=S
  return int(acc*scale)

# def tsuggestX_dim(X:List[int],t:Task,maxattempts:int=100)->List[int]:
#   dim=len(X)
#   S=tsize(t,dim)
#   x1=tflatten(t,X)
#   if x1>=t.size:
#     X=[0]*dim
#   while True:
#     X2=np.array(X)+np_choice([1,0,-1],size=(dim,))
#     for j in range(len(X2)):
#       if X2[j]<0:
#         X2[j]=S-1
#     x2=tflatten(t,X2)
#     if x2!=x1 and x2>=0 and x2<t.size:
#       return X2.tolist()

def tsuggestX_dim(X:List[int],t:Task,maxattempts:int=100)->List[int]:
  dim=len(X)
  S=tsize(t,dim)
  x1=tflatten(t,X)
  if x1>=t.size:
    X=[0]*dim
  while True:
    X2=np.array(X)+np_choice([1,0,-1],size=(dim,))
    for j in range(len(X2)):
      if X2[j]<0:
        X2[j]=S-1
      if X2[j]>=S:
        X2[j]=0
    x2=tflatten(t,X2)
    if x2!=x1 and x2>=0 and x2<t.size:
      return X2.tolist()

def teval_dim(xs:List[int],t:Task)->float:
  """ A random function to find the minimum of """
  return teval(tflatten(t,xs),t)

V=TypeVar('V')

@dataclass_json
@dataclass
class MetropolisResults(Generic[V]):
  Xs:List[V]
  Ys:List[float]
  Ps:List[float]
  Ac:List[int]
  T:float



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
  return MetropolisResults(Xs,Ys,Ps,Ac,T)


#  ____  _
# / ___|| |_ __ _  __ _  ___  ___
# \___ \| __/ _` |/ _` |/ _ \/ __|
#  ___) | || (_| | (_| |  __/\__ \
# |____/ \__\__,_|\__, |\___||___/
#                |___/


TASK_SIZE=10000
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


@autostage(T0min=0.001,
           Tstart=1,
           T0max=1e6,
           factor=2,
           waitsteps=100,
           Paccept_tgt=0.80,
           Paccept_sol=0.01,
           out_T0=[selfref,'T0.npy'],
           out_T1=[selfref,'T1.npy'],
           out_stepsused=[selfref,'stepsused.npy'],
           dim=4,
           sourcedeps=[metropolis,teval,tsuggestX])
def stage_findT0(build:Build,ref_koef,Tstart,T0min,T0max,factor,
                 waitsteps,Paccept_tgt,out_T0,dim,Paccept_sol,out_T1,
                 out_stepsused):
  t=tload(ref_koef._rref)
  F=partial(teval_dim,t=t)
  G=partial(tsuggestX_dim,t=t)
  T0i=Tstart
  nsteps=0
  while T0i<T0max:
    X0=np_choice(range(int(t.size**(1/dim))),size=(dim,)).tolist()
    r=metropolis(F=F,G=G,T=T0i,X_0=X0,maxsteps=waitsteps)
    Paccept=sum(ac for ac in r.Ac if ac>0)/len(r.Ac)
    print(f"T0i {T0i} Paccept {Paccept} steps {nsteps+waitsteps}")
    if Paccept>=Paccept_tgt:
      break
    T0i*=factor
    nsteps+=waitsteps
  T0e=Tstart
  while T0e>T0min:
    X0=np_choice(range(int(t.size**(1/dim))),size=(dim,)).tolist()
    r=metropolis(F=F,G=G,T=T0e,X_0=X0,maxsteps=waitsteps)
    Paccept=sum(ac for ac in r.Ac if ac>0)/len(r.Ac)
    print(f"T0e {T0e} Paccept {Paccept} steps {nsteps+waitsteps}")
    if Paccept<=Paccept_sol:
      break
    T0e/=factor
    nsteps+=waitsteps
  np.save(out_T1,T0e)
  np.save(out_T0,T0i)
  np.save(out_stepsused,nsteps)


def same(l,rtol)->bool:
  return all(np.isclose(x,l[-1],rtol=rtol) for x in l)


def _worker(ref_T0,decay,rtol,patience,out_results,name,
         maxsteps,maxaccepts,rindex,out_rindex)->None:
  print(ref_T0)
  numpy.random.seed(13*rindex)
  T=float(np.load(ref_T0.out_T0))
  assert T>0.0, f"T0={T}<=0.0 ??"
  T1=np.load(ref_T0.out_T1)
  assert T1>0.0, f"T1={T1}<=0.0 ??"
  t=tload(ref_T0.ref_koef._rref)
  maxaccepts2=t.size if maxaccepts is None else maxaccepts
  task_size=ref_T0.ref_koef.task_size
  budget=task_size-np.load(ref_T0.out_stepsused)
  assert budget>0, f"{budget}<=0 ??"
  dim=ref_T0.dim
  X0=np_choice(range(int(task_size**(1/dim))),size=(dim,)).tolist()
  F=partial(teval_dim,t=t)
  G=partial(tsuggestX_dim,t=t)
  rs:List[MetropolisResults[List[int]]]=[]
  while True:
    maxloops=log(T1/T,decay)
    if budget<=0 or maxloops<=0:
      break
    maxsteps=min(1000,task_size/20,int(budget/maxloops)) # Maxsteps per T
    assert maxsteps>0, f"{maxsteps}<=0 ??"
    rs.append(metropolis(F=F,G=G,T=T,X_0=rs[-1].Xs[-1] if rs else X0,
                         maxaccepts=maxaccepts2,
                         maxsteps=maxsteps))
    nsteps=sum(len(r.Xs) for r in rs)
    solutions=[r.Ys[-1] for r in rs[-patience:]]
    print(f"T {T:0.3f} ns {nsteps} bu {budget} ms {maxsteps} : {solutions}")
    if len(solutions)==patience and same(solutions,rtol):
      break
    with open(out_results,'w') as f:
      acc=[]
      for r in rs:
        dd=r.to_dict() # type: ignore
        acc.append(dd)
      json_dump(acc,f,indent=4)
    T*=decay
    budget-=len(rs[-1].Xs)
    np.save(out_rindex,rindex)

@autostage_(decay=0.85,
            rtol=0.01,
            patience=3,
            out_results=[selfref,'results.json'],
            out_rindex=[selfref,'rindex.npy'],
            sourcedeps=[_worker,metropolis,teval,tsuggestX_dim],
            maxaccepts=100,
            maxsteps=None,
            name='annealing',
            nouts=10)
def stage_annealing(build:Build,arglist):
  with Pool() as p:
    pool_starstarmap(p, _worker, [a.__dict__ for a in arglist])

@dataclass_json
@dataclass
class AnnealingResult:
  Xs:Tuple[float,float]
  Ys:Tuple[float,float]
  mean_budget:float

@autostage(name='results', out_results=[selfref,'results.npy'])
def stage_results(build:Build,ref_ann,name,out_results,always_multiref=True):
  t:Task=tload(mklens(ref_ann[0]._rref).ref_T0.ref_koef.rref)
  Xs=[]; Ys=[]; budgets=[]
  for a in ref_ann:
    with open(a.out_results) as f:
      rs=[MetropolisResults[List[int]].from_dict(x) for x in json_load(f)] # type: ignore
    Xs.append(rs[-1].Xs[-1])
    Ys.append(rs[-1].Ys[-1])
    budgets.append(sum([len(r.Xs) for r in rs])+np.load(a.ref_T0.out_stepsused))
  writejson(out_results,AnnealingResult((float(np.mean(Xs)),float(np.std(Xs))),
                                        (float(np.mean(Ys)),float(np.std(Ys))),
                                        float(np.mean(budgets))).to_dict())


@autostage(name='bruteforce',
           out_results=[selfref,'results.npy'],
           out_rindex=[selfref,'rindex.npy'],
           budget=100,nouts=10)
def stage_bruteforce(build:Build,ref_koef,name,out_results,out_rindex,
                     budget,rindex):
  assert budget>0
  t:Task=tload(ref_koef._rref)
  budget=min(budget,t.size)
  X=np_choice(list(range(t.size)),size=(1,))
  Ymin=teval(X,t); Xmin=X
  print(f'Bruteforceing with budget {budget}')
  while budget>0:
    Y=teval(X,t)
    if Y<Ymin:
      Ymin=Y
      Xmin=X
    X=X+1 if X+1<t.size else 0
    budget-=1
  np.save(out_results,Ymin)
  np.save(out_rindex,rindex)

#  ____
# |  _ \ _   _ _ __  _ __   ___ _ __ ___
# | |_) | | | | '_ \| '_ \ / _ \ '__/ __|
# |  _ <| |_| | | | | | | |  __/ |  \__ \
# |_| \_\\__,_|_| |_|_| |_|\___|_|  |___/
#

def runA(force=False,dim=10,task_size=TASK_SIZE,view=0,pngfile=None)->None:
  r=Registry()
  ref_koef=stage_koef(r,task_size=task_size)
  ref_T0=stage_findT0(ref_koef=ref_koef,r=r,dim=dim,
                      waitsteps=task_size/100 if task_size<=10000 else 1000)
  ref_ann=stage_annealing(ref_T0=ref_T0,r=r)
  # ref_res=stage_results(ref_ann=ref_ann,r=r)
  rAnn,clo,ctx=realize(instantiate(ref_ann,r=r),force_rebuild=force)
  assert isinstance(rAnn,DRef)
  rref=ctx[rAnn][view]
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
  if pngfile is None:
    kshow()
  else:
    plt.savefig(pngfile)
  plt.close()
  print(f"len(Xs)={len(allXs)}")
  print(f"Result: X={allXs[-1]} Y={allYs[-1]}")

def run2(force=False,task_size=TASK_SIZE,pngfile=None)->None:
  acc:list=[]
  bf_mean:list=[]; bf_std:list=[]; ann_budgets:list=[]
  dims=list(range(10))[1:]
  for dim in dims:
    print(f"dim {dim}")
    r=Registry()
    ref_koef=stage_koef(r,task_size=task_size)
    ref_T0=stage_findT0(ref_koef=ref_koef,r=r,dim=dim,
                        waitsteps=task_size/100 if task_size<=10000 else 1000)
    ref_ann=stage_annealing(ref_T0=ref_T0,r=r)
    ref_res=stage_results(ref_ann=ref_ann,r=r)
    rRes,clo,ctx=realize(instantiate(ref_res,r=r),force_rebuild=force)
    assert isinstance(rRes,DRef)
    with open(mklens(ctx[rRes][0]).out_results.syspath) as f:
      acc.append(AnnealingResult.from_dict(json_load(f)))
    ann_budgets.append(acc[-1].mean_budget)
    ref_bf=stage_bruteforce(ref_koef=ref_koef,budget=ann_budgets[-1],r=r)
    rBf,clo,ctx=realize(instantiate(ref_bf,r=r),force_rebuild=force)
    assert isinstance(rBf,DRef)
    Ybf=[]
    for rref in ctx[rBf]:
      Ybf.append(np.load(mklens(rref).out_results.syspath))
    bf_mean.append(np.mean(Ybf))
    bf_std.append(np.std(Ybf))

  t:Task=tload(mklens(ctx[rBf][0]).ref_koef.rref)
  fig,(Af,Aperf,Abudget)=subplots(3)
  means=np.array([r.Ys[0] for r in acc])
  stds=np.array([r.Ys[1] for r in acc])
  bf_mean_a=np.array(bf_mean); bf_std_a=np.array(bf_std)
  fXs=np.arange(0,t.size-1)
  fYs=list(map(lambda x:teval(x,t),fXs))
  Af.plot(fXs,fYs,label='F')
  Aperf.plot(dims,means,color='red',label='ann')
  Aperf.fill_between(dims,means-stds,means+stds,alpha=0.5)
  Aperf.plot(dims,bf_mean,color='blue',label='bf')
  Aperf.fill_between(dims,bf_mean_a-bf_std_a,bf_mean_a+bf_std_a,alpha=0.5)
  Abudget.plot(dims,ann_budgets)
  if pngfile is None:
    kshow()
  else:
    plt.savefig(pngfile)
  plt.close()

def runGC()->None:
  acc:list=[]
  for task_size in [TASK_SIZE,10*TASK_SIZE]:
    for dim in list(range(15))[1:]:
      print(f"dim {dim}")
      r=Registry()
      ref_koef=stage_koef(r,task_size=task_size)
      ref_T0=stage_findT0(ref_koef=ref_koef,r=r,dim=dim,
                          waitsteps=task_size/100 if task_size<=10000 else 1000)
      ref_ann=stage_annealing(ref_T0=ref_T0,r=r)
      ref_res=stage_results(ref_ann=ref_ann,r=r)
      acc.extend([ref_res])
  rmdrefs,rmrrefs=store_gc(keep_drefs=acc, keep_rrefs=[])
  for rref in rmrrefs:
    rmref(rref)
  for dref in rmdrefs:
    rmref(dref)

