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
    -   Chapter 6 of unknown book with lots of examples
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
    -   A shallow description of Boltzmann machine’s principles

### Others

-   https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
    -   KL-divirgence

## Comparing ideal and Gibbs-sampled distributions

Calculating the Gibbs distribution in a brute-force manner

``` sourceError
SOURCE ERROR in "Gibbs.md.in" near line 57:
The pattern given by "include" option "start_regex" was not found
```

Callculating an approximation to the same Gibbs distribution using Gibbs
sampler.

``` sourceError
SOURCE ERROR in "Gibbs.md.in" near line 66:
The pattern given by "include" option "start_regex" was not found
```

Comparing the results using KL-divergence

``` python
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
```

![](img/7032249505444918607.png)
