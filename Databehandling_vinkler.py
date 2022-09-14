
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
y = data[9][2:]
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

def funlin(x, a, b):
  return np.cos((x/90*np.pi/2)*a)**2


xlin = np.array([0.8, 2.5, 4.4, 5.2, 7.4, 9.0])
ylin = np.array([1.20, 2.41, 3.54, 4.44, 4.30, 6.10])
yler = np.array([0.4,0.4,0.4,0.4,0.4,0.4,0.4])
plt.errorbar(x, y, yler, fmt='o', ms=6, capsize=3)


pinit1 = np.array([1., 60.])
xhelp1 = np.linspace(0.,90.,50)
yhelp1 = funlin(xhelp1, pinit1[0], pinit1[1])
plt.plot(xhelp1, yhelp1, 'r.')
plt.show()
