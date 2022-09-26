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
#print("etager", etager)



xa = np.array([0,398.0+393.3])
xb = np.array([4,6,8])*394.8
x = np.concatenate((xa, xb))

def funlin(x, a, b,c):
  return a*np.exp(x*b)+c


y = etager.to_numpy()
print("type", type(y))
print("Atype", type(y[0]))

yler = y**0.5
yler = yler.astype(np.float64)
print("yler", yler)
#plt.errorbar(x, y, yler, fmt='o', ms=6, capsize=3)

pinit1 = np.array([4, 0.001,90])
xhelp1 = np.linspace(0.,x[-1],90)
yhelp1 = funlin(xhelp1, pinit1[0], pinit1[1], pinit1[2])
#plt.plot(xhelp1, yhelp1, 'r.')
#plt.show()
print("x", x, "y", y)
#%%
print("sigma type",type(yler))
print("sigma arg type", type(yler[4]))

popt, pcov = curve_fit(funlin, x, y, p0=pinit1, sigma=yler, absolute_sigma=True)
print('a (hældning):',popt[0],'    b (intercept):',popt[1], '    c', popt[2])
perr = np.sqrt(np.diag(pcov))
print('usikkerheder:',perr)
chmin = np.sum(((y-funlin(x, *popt))/yler)**2)
print('chi2:',chmin,' ---> p:', ss.chi2.cdf(chmin,4))

ux,uy = x[1],195
uyl = uy ** 0.5

ax.errorbar(ux,uy,uyl, fmt="o", ms=6, capsize= 3, label = "data outside",color= 'r')
ax.errorbar(x, y, yler, fmt="o", ms=6, capsize= 3, label = "data")
ax.plot(xhelp1, funlin(xhelp1, *popt), 'k-.', label = "fit")
ax.legend()
ax.set_ylabel("Counts")
ax.set_xlabel("Altitude (cm)")
ax.set_xticks(ticks = np.linspace(0, 3200, 5))
ax.set_title("Counts as a function of altitude")
ax.grid()

plt.savefig('etager')
plt.show()





