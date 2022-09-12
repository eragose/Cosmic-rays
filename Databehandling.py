
import numpy as np
import scipy.optimize
from pip._internal.utils import virtualenv
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks

import os
import csv
import re


def conv(s):
  try:
    s = float(s)
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
  data =  zip(var)
  data = [_ for _ in zip(data[0],data[1:])]
  #print(data)
  print(data,'y')
  #plt.plot(data[0,1],data[9,1])


#  data = [int(x) for x in data]
 # print(data)

 # [conv(s) for s in data]
  #plt.plot(data[1,1],data[9,1])

