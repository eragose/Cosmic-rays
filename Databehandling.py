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

x_counts, bin = np.histogram(x, bins=15)


print(len(x))
#plt.hist(x, bins=15)
#plt.show()

x_mu = np.mean(x)
#print("x", type(x[0]))
ys = poisson.pmf(x,x_mu)

x_frequency = x_counts/len(x)
#print(x_frequency, len(x_frequency), bin)

bin_points = []
for i in range(len(bin)-1):
  bin_points.append((bin[i]+bin[i+1])/2)


y = x_frequency
x = bin_points




plt.rc("font", family=["Helvetica", "Arial"]) # skifter skrifttype
plt.rc("axes", labelsize=16)   # skriftstørrelse af `xlabel` og `ylabel`
plt.rc("xtick", labelsize=14, top=True, direction="in")  # skriftstørrelse af ticks og viser ticks øverst
plt.rc("ytick", labelsize=14, right=True, direction="in")
plt.rc("axes", titlesize=16)


fig1, ax1 = plt.subplots()
fig1.set_size_inches(6,5,forward=True)

#ax.plot(x,y, label="poisson(" + str(x_mu) +")")

def funlin(x, a):
  return poisson.pmf(x, a)


yler = np.array((y))*0.1
pinit1 = 931
xhelp1 = np.linspace(int(x[0]),int(x[-1]),int(x[-1])-int(x[0])+1)
print(xhelp1)
popt, pcov = curve_fit(funlin, x, y, p0=pinit1, sigma=yler, absolute_sigma=True)
print('a (hældning):',popt[0])
perr = np.sqrt(np.diag(pcov))
print('usikkerheder:',perr)
chmin = np.sum(((y-funlin(x, *popt))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))


mu = 931
variance = 100
sigma = math.sqrt(variance)


def normfit(x, mu, variance):
  sigma = math.sqrt(variance)
  return ss.norm.pdf(x, mu, sigma)

pinit =[mu, variance]

popt1, pcov1 = curve_fit(normfit, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
print('mu :',popt1[0])
print('varians :',popt1[1])
perr = np.sqrt(np.diag(pcov1))
print('usikkerheder:',perr)
chmin = np.sum(((y-normfit(x, *popt1))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))

ax1.errorbar(x, y,yerr=yler, color = "r", label = "data", fmt = 'o', capsize = 10)
ax1.plot(xhelp1, funlin(xhelp1, *popt), 'k-.', label = "fitpoisson")
ax1.plot(xhelp1, normfit(xhelp1, *popt1), 'b-.', label = "fitnorm")
ax1.legend()
ax1.set_ylabel("Frequency")
ax1.set_xlabel("Counts")

ax1.set_title("Count distribution")
fig1.savefig("Count distribution ")
plt.show()

print(len(x))



""" fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
fig.set_size_inches(6,5,forward=True)

ax.plot(x,ys, label="poisson(" + str(x_mu) +")")

x = set_of_x
#%%
#ax.hist(x,25 ,density=True, edgecolor='black')
ax.scatter(x,y, color = "r")
ax.legend()
ax.set_title("ax")
#plt.show()

def funlin(x, a):
  return poisson.pmf(x, a)



yler = np.array((y))*0.1

#plt.errorbar(x, y, yler, fmt='o', ms=6, capsize=3)

pinit1 = 931
xhelp1 = np.linspace(x[0],x[-1],x[-1]-x[0]+1)
#yhelp1 = funlin(xhelp1, pinit1)
#plt.plot(xhelp1, yhelp1, 'r.')
#plt.show()

#print(funlin(xhelp1, 940))

#%%
popt, pcov = curve_fit(funlin, x, y, p0=pinit1, sigma=yler, absolute_sigma=True)
print('a (hældning):',popt[0])
perr = np.sqrt(np.diag(pcov))
print('usikkerheder:',perr)
chmin = np.sum(((y-funlin(x, *popt))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))

mu = 931
variance = 100
sigma = math.sqrt(variance)
#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)


def normfit(x, mu, variance):
  sigma = math.sqrt(variance)
  return ss.norm.pdf(x, mu, sigma)

pinit =[mu, variance]

popt1, pcov1 = curve_fit(normfit, x, y, p0=pinit, sigma=yler, absolute_sigma=True)
print('mu :',popt1[0])
print('varians :',popt1[1])
perr = np.sqrt(np.diag(pcov1))
print('usikkerheder:',perr)
chmin = np.sum(((y-normfit(x, *popt1))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))

ax1.scatter(x, y, color = "r", label = "data")
ax1.plot(xhelp1, funlin(xhelp1, *popt), 'k-.', label = "fitpoisson")
ax1.plot(xhelp1, normfit(xhelp1, *popt1), 'b-.', label = "fitnorm")
ax1.legend()
ax1.set_ylabel("Frequency")
ax1.set_xlabel("Counts")

ax1.set_title("Count distribution")
fig1.savefig("Count distribution ")
plt.show()

print(len(x)) """








