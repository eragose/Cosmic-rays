from scipy.optimize import curve_fit
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
import scipy.stats as ss

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
print("x", type(x[0]))
ys = poisson.pmf(x,x_mu)

y = [t/sum(x) for t in x ]

plt.rc("font", family=["Helvetica", "Arial"]) # skifter skrifttype
plt.rc("axes", labelsize=16)   # skriftstørrelse af `xlabel` og `ylabel`
plt.rc("xtick", labelsize=14, top=True, direction="in")  # skriftstørrelse af ticks og viser ticks øverst
plt.rc("ytick", labelsize=14, right=True, direction="in")
plt.rc("axes", titlesize=16)

fig1, ax1 = plt.subplots()
fig1.set_size_inches(6,5,forward=True)
fig, ax = plt.subplots()

ax.plot(x,ys, label="poisson(" + str(x_mu) +")")


#%%
ax.hist(x,25 ,density=True, edgecolor='black')
ax.legend()
ax.set_title("ax")
fig.show()

def funlin(x, a):
  return poisson.pmf(x, a)



yler = np.array((y))*0.1

#plt.errorbar(x, y, yler, fmt='o', ms=6, capsize=3)

pinit1 = 931
xhelp1 = np.linspace(800,1050,100)
#yhelp1 = funlin(xhelp1, pinit1)
#plt.plot(xhelp1, yhelp1, 'r.')
#plt.show()

#%%
popt, pcov = curve_fit(funlin, x, y, p0=pinit1, sigma=yler, absolute_sigma=True)
print('a (hældning):',popt[0])
perr = np.sqrt(np.diag(pcov))
print('usikkerheder:',perr)
chmin = np.sum(((y-funlin(x, *popt))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))

ax1.errorbar(x, y, yler, fmt="o", ms=6, capsize= 3, label = "data")
ax1.plot(xhelp1, funlin(xhelp1, *popt), 'k-.', label = "fit")
ax1.legend()
ax1.set_ylabel("Counts")
ax1.set_xlabel("Angle (degree)")

ax1.set_title("ax1")
fig1.show()









