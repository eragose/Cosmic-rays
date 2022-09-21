# Necesery pakages
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as ss

#Collection of data from folder
folder = 'Data_vinkler'                                 #Specify folder name
col_names = []                                          #As str
seperator = ';'                                         #Seperator in data file
file_type = '*.cvs'                                     #Data type, keep "*"
col_num_for_x = 0
col_num_for_y = None
xlabel = 'Angle'
ylabel = 'Counts'
plot_title = 'Muon Count as a function of angle'
ticks = np.linspace(0, 90, 7)

def funlin(x, *params):
  return np.cos(params[0] * (x)/90*np.pi/2)**2 * params[1] + params[2]          #Fitting function

pinit1 = np.array([1, 120.,0])                          #Help variables
xhelp1 = np.linspace(0.,90.,90)                         #Adjust to data
yhelp1 = funlin(xhelp1, pinit1[0], pinit1[1], pinit1[2])



######################################################################################
read_files = glob.glob(os.path.join(folder, file_type))  #Collects all data files

#Creates a list of all file names
np_array_values = []
for files in read_files:
    pdfile = pd.read_csv(files, sep=seperator)           #Specify seperator
    np_array_values.append(pdfile)

#Converts data to readable array format
if len(np_array_values) > 1:
  merge_values = np.vstack(np_array_values)
else:
  merge_values = np_array_values
data = pd.DataFrame(merge_values)
data.columns = col_names
print(data)

numpy_data = data.to_numpy()
print(numpy_data)
x_data = numpy_data[:,col_num_for_x]
y_data = numpy_data[:,col_num_for_y]                     #Can be changed to function on x data

yler = y_data**0.5

#Fitting section
popt, pcov = curve_fit(funlin, x_data, y_data, p0=pinit1, sigma=yler, absolute_sigma=True)
print('Coefficients:',*popt)
perr = np.sqrt(np.diag(pcov))
print('Uncertaity:',perr)
chmin = np.sum(((y_data-funlin(x_data, *popt))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))


#Beautification of plots
plt.rc("font", family=["Helvetica", "Arial"]) # skifter skrifttype
plt.rc("axes", labelsize=16)   # skriftstørrelse af `xlabel` og `ylabel`
plt.rc("xtick", labelsize=14, top=True, direction="in")  # skriftstørrelse af ticks og viser ticks øverst
plt.rc("ytick", labelsize=14, right=True, direction="in")
plt.rc("axes", titlesize=16)


fig, ax = plt.subplots()
fig.set_size_inches(6,5,forward=True)

ax.errorbar(x_data, y_data, yler, fmt="o", ms=6, capsize= 3, label = "data")
ax.plot(xhelp1, funlin(xhelp1, *popt), 'k-.', label = "fit")
ax.legend()
ax.set_ylabel(ylabel)
ax.set_xlabel(xlabel)
ax.set_xticks(ticks)
ax.set_title(plot_title)
plt.show()