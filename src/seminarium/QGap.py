# -*- coding: utf-8 -*-
"""Копия блокнота "2-SAT Quantum Spectral Gaps"

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jxI-mO5HxOFVUUEITET-k6AT-S05t-os
"""

import itertools
import functools
import numpy as np
import math
from pprint import pprint
import scipy.integrate
import matplotlib.pyplot as plt

cache = lambda f: functools.lru_cache(maxsize=None)(f)

@cache
def unit1():
  return np.identity(n=2)

@cache
def unit(n):
  return np.eye(N=2**n)

@cache
def pauli_z():
  return np.array([[1, 0], [0, -1]])

@cache
def pauli_x():
  return np.array([[0, 1], [1, 0]])

@cache
def pauli_y():
  return np.array([[0, -1j], [1j, 0]])

def pauli_index(i, n, pauli_matrix):
  left = [unit1()] * (i - 1)
  right = [unit1()] * (n - i)
  parts = left + [pauli_matrix] + right
  return functools.reduce(np.kron, parts)  

@cache
def pauli_z_index(i, n):
  return pauli_index(i, n, pauli_z())

@cache
def pauli_x_index(i, n):
  return pauli_index(i, n, pauli_x())

@cache
def pauli_y_index(i, n):
  return pauli_index(i, n, pauli_y())

@cache
def variable_to_operator(i, n):
  assert i != 0
  sign = 1 if i > 0 else -1
  i = i * sign
  return 0.5*(unit(n) - sign * pauli_z_index(i, n))

@cache
def clause_to_operator(x, y, n):
  return variable_to_operator(x, n) * variable_to_operator(y, n)

def formula_to_operator(clauses, n):
  mapper = lambda clause: clause_to_operator(clause[0], clause[1], n)
  ops = list(map(mapper, clauses))
  return sum(ops)

def spectrac_gap(op):
  eps = 1e-16
  e, _ = np.linalg.eig(op)
  n = op.shape[0]
  i = 1
  while i < n and e[i] - e[i-1] < eps:
    i += 1
  return np.abs(e[i] - e[i-1]) if i < n  else 0

# (x1 or x2) (x1 or not x2) (not x1 or x2)
# \00> - True True
formula_to_operator([(1, 2), (1, -2), (-1, 2), (-1, -2)], 2)

spectrac_gap(formula_to_operator(((1, 2), (1, -2)), 2))

def generate_all_clauses(n):
  variables = list(range(1, n+1))
  pairs = itertools.combinations(variables, 2)
  signs = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
  def mapper(pair_sign):
    pair, sign = pair_sign
    return (sign[0]*pair[0], sign[1]*pair[1])
  clauses = map(mapper, itertools.product(pairs, signs))
  return list(clauses)

def generate_all_formulas(n):
  clauses = generate_all_clauses(n)
  return itertools.chain.from_iterable(
      itertools.combinations(clauses, r) for r in range(1, len(clauses) + 1))

def spectral_gaps(n):
  return [spectrac_gap(formula_to_operator(f, n)) for f in generate_all_formulas(n)]

gaps2 = spectral_gaps(2)

gaps3 = spectral_gaps(3)

_ = plt.hist(gaps2)

_ = plt.hist(gaps3, log=True)



"""Solve Schroedinger equation, plot spectrum and demonstrate convergency."""

n = 2
dim = 2 ** n

# rng = np.random.default_rng(7)
# H0 = rng.random((dim, dim))
# H0 = H0 + H0.transpose()

H0 = pauli_z_index(1, n) + pauli_x_index(2, n)

# True True /00>
# H0 = formula_to_operator([(1, 2), (-1, -2), (-1, 2)], n)

# False False /11>
H1 = formula_to_operator([(1, 2), (1, -2), (-1, 2)], n)

_, v = np.linalg.eigh(H0)
psi0 = v[:, 0]

T = 1000.0

def Ht(t):
  return (1 - t/T) * H0 + t/T * H1

def schroedenger_rhs(t, psi_):
  return -1j * Ht(t) @ psi_

sol = scipy.integrate.solve_ivp(schroedenger_rhs, (0, T), psi0 + 0j)

numpoints = len(sol['t'])

spectrum = np.zeros((numpoints, dim))

for i in range(numpoints):
    spectrum[i, :], _ = np.linalg.eigh(Ht(sol['t'][i]))
  
psiT = sol['y'][:, -1]

print(np.abs(psiT))

plt.figure()
plt.title('Spectrum')
plt.plot(spectrum)
plt.show()

plt.figure()
plt.title(f'Psi(t) for t = T (={T})')
plt.bar(range(dim), np.abs(psiT))
plt.show()

pauli_x() @ pauli_y() - pauli_y() @ pauli_x()

H0