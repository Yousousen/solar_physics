# -*- coding: utf-8 -*-

"""
Created on Thu Jan  9 15:19:30 2020

@author: Rasjied Sloot
"""

#This short code can be used to plot stellar atmosphere flux models from Coelho (2014)

#more information: http://specmodels.iag.usp.br/

#download data: http://specmodels.iag.usp.br/fits_search/compress/s_coelho14_highres.tgz

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np

# load a model
data=fits.getdata('t06000_g+5.0_m05p00_hrplc.fits')
data2=fits.getdata('t06000_g+1.0_m10p00_hr.fits')
data3=fits.getdata('t05750_g+3.5_m01p04_hrplc.fits')
# Resolution and Range of spectrum
labda_start = 2500
resolution = 0.02

# list of wavelengths
wavelength = np.arange(len(data))*resolution+labda_start

#display specrum within a certain range (xlim)

plt.figure(figsize=(20,10))
plt.xlim(3900,4000)
plt.plot(wavelength, data, markersize=0.1)
plt.plot(wavelength, data2, markersize=0.1)
plt.plot(wavelength, data3, markersize=0.1)
plt.show()

# combine to array and crop

spectrum = []

[wavelength[1], data[1]]

for i in range( len(wavelength)):
    if wavelength[i] < 6570 and  wavelength[i] > 6560:
        spectrum.append([np.round(wavelength[i],2),data[i]])
    

