
import numpy as np
import pylab as pl
import scipy.optimize
from pip._internal.utils import virtualenv
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
from scipy.stats import poisson

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

with open('./CSMHUNT_12301_2022-9-7_14-23-3.cvs', 'r') as file:
  csvreader = csv.reader(file, delimiter=';')
  var = []
  for i in csvreader:
    t = [conv(s) for s in i]
      var.append(t)
  data = list(zip(*var))
  #print(data)
  # Nu plotter jeg dataet


#x= np.array(data[9][2:])
#y= x/sum(x)
#print(y)
fig, ax = plt.subplots()


ax.stem(x,y )

ax.set(xlim=(0, 8), xticks=np.arange(-1, len(x)+2),
       ylim=(0, 1000), yticks=np.linspace(0, 1100, 10))

plt.show()

  print(x)

#  data = [int(x) for x in data]
 # print(data)

 # [conv(s) for s in data]
  #plt.plot(data[1,1],data[9,1])

  #%%

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

y = [t/sum(x) for t in x ]
print(y)



fig, ax = plt.subplots()


ax.scatter(x,y)

#ax.set(xlim=(0, len(x)), xticks=np.arange(0,len(x),1200),
       #ylim=(0, 1), yticks=np.linspace(0, 1, 10))

plt.show()

#%%
plt.hist(x,density=True, edgecolor='black')
plt.
plt.show()


