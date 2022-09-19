import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as ss

plt.rc("font", family=["Helvetica", "Arial"]) # skifter skrifttype
plt.rc("axes", labelsize=16)   # skriftstørrelse af `xlabel` og `ylabel`
plt.rc("xtick", labelsize=14, top=True, direction="in")  # skriftstørrelse af ticks og viser ticks øverst
plt.rc("ytick", labelsize=14, right=True, direction="in")
plt.rc("axes", titlesize=16)

fig, ax = plt.subplots()
fig.set_size_inches(6,5,forward=True)

'''
Henter path og læser filerne
'''
directory = os.getcwd()
folder = './Data atage'

read_files = glob.glob(os.path.join(folder,'*.cvs'))

np_array_values = []
for files in read_files:
    etager = pd.read_csv(files, sep=';')
    np_array_values.append(etager)

'''
Samler dataen så den kan op deles
'''  
merge_values = np.vstack(np_array_values)
data = pd.DataFrame(merge_values)
data.columns = ['num' , 'coinc', 'date' , 'time' , 'sec' , 'RecTime' , 'A' , 'B' , 'C' , 'COINC' , 'Pressure' , 'Temp' , 'Humidity' , 'Altitude' ]

etager = data['COINC'][1:][::2] # skal kun bruge hver anden data punkt
print("etager", etager)



xa = np.array([0,398.+393.3])
xb = np.array([4,6,8])*394.8
x = list(xa) + list(xb)

def funlin(x, a, b,c):
  return a*np.exp(x*b)+c

y = list(etager)

yler = np.array((etager)**0.5)

#plt.errorbar(x, y, yler, fmt='o', ms=6, capsize=3)

pinit1 = np.array([1, 120.,0])
xhelp1 = np.linspace(0.,90.,90)
yhelp1 = funlin(xhelp1, pinit1[0], pinit1[1], pinit1[2])
#plt.plot(xhelp1, yhelp1, 'r.')
#plt.show()
print("x", x, "y", y)
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
ax.set_title("Mc shizzle")
fig.show()

plt.scatter(xs, etager)
plt.xlabel('Altitude [cm]')
plt.ylabel('Coincidents')
plt.show()

print(data["time"])

#%%



