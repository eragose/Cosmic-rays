import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
print(etager)



xs = np.array([0,3.98+393.3])
x = np.array([4,6,8])*394.8
xs = list(xs) + list(x)

plt.scatter(xs, etager)
plt.xlabel('Altitude [cm]')
plt.ylabel('Coincidents')
plt.show()

print(data["time"])

#%%


