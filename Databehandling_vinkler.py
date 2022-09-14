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

with open('./Data_vinkel.cvs', 'r') as file:
  csvreader = csv.reader(file, delimiter=';')
  var = []
  for i in csvreader:
    t = [conv(s) for s in i]
    var.append(t)
  data = list(zip(*var))
  #print(data)
  # Nu plotter jeg dataet

print(data[9][2:])
y = np.array(data[9][2:])
print(y)
x = np.linspace(0,90,7)

coefficients = np.polyfit(x, y, 3)
fx = np.linspace(-1, 100, 100)
fy = np.polyval(coefficients, fx)
plt.plot(fx, fy, '-')
plt.plot(x, y, 'ro')
plt.show()



x_mu = np.mean(x)

#y = np.array()

fig, ax = plt.subplots()

ax.plot(x,y)

#plt.show()


#%%

#Tager fit fra 2. semester


def funlin(x, a, b,c):
  return np.cos(a * (x)/90*np.pi/2)**2 * b + c



yler = np.array((y)**0.5)

plt.errorbar(x, y, yler, fmt='o', ms=6, capsize=3)

pinit1 = np.array([1, 120.,0])
xhelp1 = np.linspace(0.,100.,90)
yhelp1 = funlin(xhelp1, pinit1[0], pinit1[1], pinit1[2])
plt.plot(xhelp1, yhelp1, 'r.')
plt.show()

#%%
popt, pcov = curve_fit(funlin, x, y, p0=pinit1, sigma=yler, absolute_sigma=True)
print('a (hældning):',popt[0],'    b (intercept):',popt[1])
perr = np.sqrt(np.diag(pcov))
print('usikkerheder:',perr)
chmin = np.sum(((y-funlin(x, *popt))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))

plt.errorbar(x, y, yler, fmt="o", ms=6, capsize= 3)
plt.plot(xhelp1, funlin(xhelp1, *popt), 'k-.')
plt.show()

