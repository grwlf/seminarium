from seminarium.MCMC import metropolis, MetropolisResults
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from utils.math import bits2i

from numpy.random import choice
from pylightnix import *

import matplotlib.pyplot as plt
import numpy as np


DIM=6
B0=[False for _ in range(DIM)]


def cnf_eval(x:np.ndarray)->int:
  """ `x` is the array of `0` and `1` representing formula variables.  Return
  `1` if the formula evaluates to `True` or 0 otherwise. """
  assert x.shape==(DIM,)
  b=[i==1 for i in x.tolist()]
  return 1 if \
    (b[0] or b[1]) and (b[0] or not b[2]) and (not b[2] or b[1]) and \
    (b[0] or b[5]) and (b[3] or not b[2]) and (not b[5] or b[1]) else 0


def cnf_suggest(x:np.ndarray)->np.ndarray:
  """ FIXME: change ALL the variables, not just one """
  assert x.shape==(DIM,)
  v=choice(range(x.shape[0]),size=(1,))
  x2=x.copy()
  x2[v]=1-x2[v]
  return x2

def flatten(x:np.ndarray)->int:
  """ Convert `x` in to an integer, treating it as binary representation """
  return bits2i(x.astype(int).tolist())

@autostage(T=1, B0=B0, maxsteps=1000,
           out_track=[selfref,'track.json'],
           out_png=[selfref,'out.png'])
def stage_anneal_cnf(build:Build, T, B0, out_track, out_png, maxsteps):
  x0=np.array([1 if b else 0 for b in B0],dtype=float)
  mr=metropolis(F=cnf_eval,G=cnf_suggest,T=T,X_0=x0,maxsteps=maxsteps)
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


