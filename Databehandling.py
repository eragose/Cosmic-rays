
import numpy as np
import pylab as pl
import scipy.optimize
from pip._internal.utils import virtualenv
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from scipy.stats import poisson
import math

import os
import csv

import re


def conv(s):
  try:
    s = int(s)
  except:
    pass
  return s

# Denne her kode aflæser og fordeler dataet i ders baretemte label.

with open('./CSMHUNT_12301_2022-9-7_16-3-44.cvs', 'r') as file:
  csvreader = csv.reader(file, delimiter=';')
  var = []
  for i in csvreader:
    t = [conv(s) for s in i]
    var.append(t)
  data = list(zip(*var))
  #print(data)
  # Nu plotter jeg dataet

x= np.array(sorted(data[9][2:]))
x_mu = np.mean(x)

ys = poisson.pmf(x,x_mu)

y = [t/sum(x) for t in x ]


fig, ax = plt.subplots()

ax.scatter(x,ys)


#%%
plt.hist(x,50 ,density=True, edgecolor='black')
plt.show()



