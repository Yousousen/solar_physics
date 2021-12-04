# -*- coding: utf-8 -*-
"""
Created on Wed May 27 20:08:31 2020

@author: Rasjied Sloot
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# Load an image and show the array

# data = fits.getdata('C:/OBSERVATIONS/Masters/M101/M101_L.fit')
data = fits.getdata('C:/OBSERVATIONS/Masters/NGC 1491/NGC1491 Ha_master.fit')

print(data)
print('shape of array:', data.shape)

# Some statistics

mean = np.mean(data)
print ('mean value:', mean)

stdev = np.std(data)
print ('standard deviation:', stdev)

median = np.median(data)
print ('median value:',median)

minimum = np.min(data)
print('minimum value:', minimum)

maximum = np.max(data)
print ('max value:',maximum)

# plot the image

plt.imshow(data)
plt.show()

# or a bit better

from matplotlib.colors import LogNorm

def plot_image(image):
        

    vmin, vmax = np.percentile(image, [5, 95])
    fig, ax1 = plt.subplots(1,1, figsize=(20,30))
    plt.imshow(image, cmap='gray' , norm=LogNorm(vmin=0.9*vmin,vmax=1.5*vmax))
    plt.show()
    
plot_image(data)

# make a selection/crop

crop2 = data[:,2000:3500]
plot_image(crop2)

crop1 = data[1700:2100]
plot_image(crop1)

intensity_profile=np.sum(crop1, axis=0)
plt.plot(intensity_profile)

# Show data header

data_header = fits.getheader('C:/OBSERVATIONS/Masters/NGC 1491/NGC1491 Ha_master.fit')

print(data_header)

# display one specific element:

header_info = data_header['FILTER']
print('FILTER USED:', header_info)


header_info2 = data_header['IMAGETYP']
print('IMAGE TYPE:', header_info2)