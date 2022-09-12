
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from scipy.misc import electrocardiogram
from scipy.signal import find_peaks
import plotly.graph_objects as go
import os

# assign directory
directory = 'CSMHUNT_12301_2022-9-7_14-23-3.cvs'

# iterate over files in that directory

s = open('CSMHUNT_12301_2022-9-7_14-23-3.cvs').read()
for i in s:
    print(i)
print(s)


