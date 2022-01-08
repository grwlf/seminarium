from typing import Optional
from os.path import join, dirname
from os import makedirs
from math import ceil
from pdb import set_trace

from pylightnix import (Registry, DRef, fsinit, realizeU, instantiate,
                        fetchurl2, selfref, autostage)
from reports.lib import *

from seminarium.Gibbs import Dataset as Gibbs_Dataset, mkds

import matplotlib.pyplot as plt
import numpy as np

# fsinit(join(dirname(__file__),'..','..','_pylightnix'),use_as_default=True)


def stage_mnist(r:Optional[Registry]=None)->DRef:
  """ Fetch the MNIST dataset from the Internet """
  return fetchurl2(
    name='mnist',
    mode='as-is',
    url='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
    sha256='731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1',
    r=r)


@autostage(name='bwmnist', mnist=stage_mnist, out=[selfref,'bwmnist.npz'])
def stage_bwmnist(build, name, mnist, out):
  f=np.load(mnist.out, allow_pickle=True)
  xtrain,ytrain=f['x_train'],f['y_train']
  xtest,ytest=f['x_test'],f['y_test']
  xtrain=np.round(xtrain/256)
  xtest=np.round(xtest/256)
  np.savez(out,**{'x_train':xtrain,
                  'y_train':ytrain,
                  'x_test':xtest,
                  'y_test':ytest})


@autostage(name='pmnist', samples=[42,3,17,423,42,68],
           mnist=stage_mnist, out=[selfref,'out.png'])
def stage_mnist_plot(name, build, samples, mnist, out):
  f=np.load(mnist.out, allow_pickle=True)
  xtrain,ytrain=f['x_train'],f['y_train']
  assert len(samples)<9
  plt.figure()
  for i,p in enumerate(samples):
    plt.subplot(ceil(len(samples)/3),3,i+1)
    plt.imshow(xtrain[p],cmap=plt.get_cmap('gray'),
               interpolation='none',
               resample=False)
  plt.savefig(out)


def stage_bwmnist_plot(r=None)->DRef:
  return stage_mnist_plot(name='pbwmnist',
                          mnist=stage_bwmnist(r=r),r=r)
