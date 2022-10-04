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

plt.rc("font", family=["Helvetica", "Arial"]) # skifter skrifttype
plt.rc("axes", labelsize=16)   # skriftstørrelse af `xlabel` og `ylabel`
plt.rc("xtick", labelsize=14, top=True, direction="in")  # skriftstørrelse af ticks og viser ticks øverst
plt.rc("ytick", labelsize=14, right=True, direction="in")
plt.rc("axes", titlesize=16)

fig, ax = plt.subplots()
fig.set_size_inches(6,5,forward=True)


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
#plt.plot(fx, fy, '-')
#plt.plot(x, y, 'ro')
#plt.show()



x_mu = np.mean(x)

#y = np.array()


#ax.plot(x,y)

#plt.show()


#%%

#Tager fit fra 2. semester


def funlin(x, a, b,c):
  return np.cos(a * (x)/90*np.pi/2)**2 * b + c



yler = np.array((y)**0.5)

#plt.errorbar(x, y, yler, fmt='o', ms=6, capsize=3)

pinit1 = np.array([1, 120.,0])
xhelp1 = np.linspace(0.,90.,90)
yhelp1 = funlin(xhelp1, pinit1[0], pinit1[1], pinit1[2])
#plt.plot(xhelp1, yhelp1, 'r.')
#plt.show()

#%%
popt, pcov = curve_fit(funlin, x, y, p0=pinit1, sigma=yler, absolute_sigma=True)
print('a (hældning):',popt[0],'    b (intercept):',popt[1])
perr = np.sqrt(np.diag(pcov))
print('usikkerheder:',perr)
chmin = np.sum(((y-funlin(x, *popt))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))

ax.errorbar(x, y, yler, fmt="o", ms=6, capsize= 3, label = "data")
ax.plot(xhelp1, funlin(xhelp1, *popt), 'k-.', label = "fit")
ax.legend()
ax.set_ylabel("Counts")
ax.set_xlabel("Angle (degree)")
ax.set_xticks(ticks = np.linspace(0, 90, 7))
ax.set_title("Counts through building")
fig.savefig("angles")
plt.show()


with open('./Data_vinkel2.cvs', 'r') as file:
  csvreader = csv.reader(file, delimiter=';')
  var = []
  for i in csvreader:
    t = [conv(s) for s in i]
    var.append(t)
  data2 = list(zip(*var))

yc = np.concatenate((y, data2[9][2:]))
xc = np.concatenate((x,x))

fig1, ax1 = plt.subplots()
fig1.set_size_inches(6,5,forward=True)

#%%
ylerc = np.array((yc)**0.5)

popt, pcov = curve_fit(funlin, xc, yc, p0=pinit1, sigma=ylerc, absolute_sigma=True)
print('kombinerede værdier a (hældning):',popt[0],'    b (intercept):',popt[1], '    c (offset):',popt[2])
perr = np.sqrt(np.diag(pcov))
print('kombinerede usikkerheder:',perr)
chmin = np.sum(((y-funlin(x, *popt))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))

ax1.errorbar(xc, yc, ylerc, fmt="o", ms=6, capsize= 6, label = "data towards parkinglot",color = 'r')
ax1.errorbar(x, y, yler, fmt="o", ms=6, capsize= 6, label = "data towards main building")
ax1.plot(xhelp1, funlin(xhelp1, *popt), 'k-.', label = "fit")
ax1.legend()
ax1.set_ylabel("Counts")
ax1.set_xlabel("Angle (degree)")
ax1.set_xticks(ticks = np.linspace(0, 90, 7))
ax1.set_title("All angle counts")
ax1.text(0,20,'cos('+str(round(popt[0],2))+' x/90*pi/2)^2 * '+ str(round(popt[1],2))+ ' +'+str(round(popt[2],2)))
fig1.savefig("angles combined")
fig1.show()

print("1 45 degree", xc[3], yc[3], "60 degree", xc[5], yc[5])
print("2 45 degree", xc[10], yc[10], "60 degree", xc[12], yc[12])