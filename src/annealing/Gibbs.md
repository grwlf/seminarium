-   [Gibbs sampling and Botlzmann
    machines](#gibbs-sampling-and-botlzmann-machines)
    -   [Resources](#resources)
        -   [Gibbs sampler](#gibbs-sampler)
        -   [Boltzmann machines](#boltzmann-machines)
        -   [Others](#others)
    -   [Comparing ideal and Gibbs-sampled
        distributions](#comparing-ideal-and-gibbs-sampled-distributions)

# Gibbs sampling and Botlzmann machines

## Resources

### Gibbs sampler

-   http://probability.ca/jeff/ftpdir/varnew.pdf
    -   Rates of Convergence for Gibbs Sampling for Variance Component
        Models
    -   1993
    -   Rosenthal
-   https://jwmi.github.io/BMS/chapter6-gibbs-sampling.pdf
    -   Chapter 6 of unknown book
-   https://ocw.mit.edu/courses/economics/14-384-time-series-analysis-fall-2013/lecture-notes/MIT14_384F13_lec26.pdf
    -   Lecture 26. MCMC: Gibbs Sampling
    -   Mikusheva
    -   2007

### Boltzmann machines

-   https://arxiv.org/pdf/1806.07066.pdf
    -   Restricted Boltzmann Machines: Introduction and Review
    -   2018
    -   Montufar
-   https://christian-igel.github.io/paper/TRBMAI.pdf
    -   Training Restricted Boltzmann Machines: An Introduction
    -   2014
    -   Fischer, Igel
-   https://www.csrc.ac.cn/upload/file/20170703/1499052743888438.pdf
    -   A Practical Guide to Training Restricted Boltzmann Machines,
        version 1
    -   2010.09.03
    -   Hinton
-   https://www.cs.toronto.edu/\~hinton/csc321/readings/boltz321.pdf
    -   Boltzmann Machines
    -   2007.03.25
    -   Hinton

### Others

-   https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    -   KL-divirgence

## Comparing ideal and Gibbs-sampled distributions

Calculating the Gibbs distribution in a brute-force manner

``` python
def tPideal(t:Task,T:float=1)->PDist:
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
```

Callculating an approximation to the same Gibbs distribution using Gibbs
sampler.

``` python
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
      P1=sigmoid((2/T)*s)
      v[j]=np_choice([1,-1],p=[P1,1.0-P1])
    ps[vstamp(v)]+=1
    step+=1
    if step%100==0:
      yield mkpdist(ps/step)
```

Comparing the results using KL-divergence

``` python
@autostage(name='plotKL',T=1.0,out=[selfref,'out.png'],
           sourcedeps=[gibbsPI, tPideal])
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
```

![](img/2893561013825138459.png)
