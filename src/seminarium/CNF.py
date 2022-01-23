import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from dataclasses_json import dataclass_json

from utils.math import bits2i,log

from numpy.random import choice as npchoice
from numpy.testing import assert_allclose
from numpy.linalg import eig
from pylightnix import *

from seminarium.MCMC import (anneal,findT,metropolis,MetropolisResults,transmat)

DIM=6
B0=[False for _ in range(DIM)]


def maxcnf_eval(x:np.ndarray)->int:
  """ `x` is the array of `0` and `1` representing formula variables.  Return
  `1` if the formula evaluates to `True` or 0 otherwise. """
  assert x.shape==(DIM,), f"{x.shape}"
  ans=0
  ans+=(x[0]==1 or x[1]==1)
  ans+=(x[0]==0 or x[2]==0)
  ans+=(x[2]==1 or x[1]==1)
  ans+=(x[0]==0 or x[5]==1)
  ans+=(x[3]==1 or x[2]==0)
  ans+=(x[5]==0 or x[1]==1)
  return -ans

def maxcnf_suggest(x:np.ndarray)->np.ndarray:
  assert x.shape==(DIM,)
  v=npchoice([1,0],size=(DIM,))
  x2=np.array(x+v)
  x2[x2==2]=0
  return x2

def maxcnf_init()->np.ndarray:
  return npchoice([0,1],size=(DIM,))

def maxcf_serialize(x:np.ndarray)->int:
  base=1
  acc=0
  for i in range(x.shape[0]):
    acc+=base*x[i]
    base*=2
  return acc


def test_findT():
  res=findT(F=maxcnf_eval,G=maxcnf_suggest,fX0=maxcnf_init)
  print(res)

def test_anneal():
  F=maxcnf_eval
  G=maxcnf_suggest
  T0i,T0e,steps=findT(F,G,fX0=maxcnf_init)
  budget=10000-steps
  decay=0.85
  def _m(T,X0,budget):
    maxloops=log(T0e/T,decay)
    maxsteps=min(1000,int(budget/maxloops)) # Maxsteps per T
    return metropolis(F=F,G=G,T=T,X_0=X0,maxsteps=maxsteps)
  for rs in anneal(_m,T0i,T0e,maxcnf_init(),budget=budget,decay=decay):
    last=rs
  print(last)


def test_transmat():
  F=maxcnf_eval
  G=maxcnf_suggest
  X0=maxcnf_init()
  T=10
  maxsteps=10000
  mr=metropolis(F=F,G=G,T=T,X_0=X0,maxsteps=maxsteps)
  print(mr)
  tm=transmat(2**DIM,maxcf_serialize,mr)
  print(tm)
  assert_allclose(np.ones(shape=(2**DIM,)),np.sum(tm,axis=1))
  print(eig(tm)[0])

def flatten(x:np.ndarray)->int:
  """ Convert `x` in to an integer, treating it as binary representation """
  return bits2i(x.astype(int).tolist())

@autostage(T=1, B0=B0, maxsteps=1000,
           out_track=[selfref,'track.json'],
           out_png=[selfref,'out.png'])
def stage_anneal_cnf(build:Build, T, B0, out_track, out_png, maxsteps):
  x0=np.array([1 if b else 0 for b in B0],dtype=float)
  mr=metropolis(F=maxcnf_eval,G=maxcnf_suggest,T=T,X_0=x0,maxsteps=maxsteps)
  # np.save(out_track,
  writejson(out_track,mr.to_dict())

  fig,(axXs,ax1,ax2)=plt.subplots(3)
  allXs=[flatten(x) for x in mr.Xs]
  allYs=list(mr.Ys)
  axXs.plot(allXs,'.',color='red',label='X(time)')
  ax1.hist(allXs,label='Visited X')
  ax2.hist(allYs,color='orange',label='Visited Y')
  fig.legend()
  plt.savefig(out_png)


