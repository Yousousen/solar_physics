import os
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from lmfit import models
from pathlib import Path
from astropy.io import fits
import scipy.constants as sc
import astropy.constants as ac

def pixels_to_wavelenghts(pixel, slope, intercept):
    '''
    This function converts pixel values to wavelengths using a linear model to be determined by a fit.
    '''
    return slope * pixel + intercept

def import_files(path, filename):
    '''
    Import files in the syntax that I used in the notebook. That is, choose the path that contains
    all the fit files, then for the filename choose something like bias* to obtain all bias files
    as a dictionary that can be accessed by filename 
    '''
    files =  [ p for p in Path(path).glob(filename) ]
    file_dict = dict()
    for file in files:
        file_dict[os.path.basename(file)] = fits.getdata(file) # key of the dict is the filename.

    headers_dict = dict()
    for x in file_dict.keys():
        headers_dict[x] = fits.getheader(path + x)

    return file_dict, headers_dict 

def get_peak_wavelength(wavelengths, irradiances):
    index = list(irradiances).index(max(irradiances))
    return  wavelengths[index]

def get_effective_temperature(wavelength_peak):
    return ( 2.897771955 * 10**-3 ) / (wavelength_peak * 10**-9)

def stack(values):
    return  np.median( [ x for x in values ], axis=0 )

def planck_scaled(temperature, wavelength):
    return (S := ( ( ( ac.R_sun / ac.au ) ** 2 )  *  ( 2 * sc.Planck * sc.speed_of_light ** 2 ) /
        ( (wavelength**5) * ( np.exp( ( sc.Planck * sc.speed_of_light) / ( wavelength * sc.k * temperature ) ) - 1 ) ) ) )

df_fl = pd.read_csv('./data/fraunhofer_lines.csv')
df_fl.keys()

#plt.xlabel('pixel')
#plt.ylabel('wavelength (nm)')
#plt.plot(df_fl['pixel'], df_fl['wavelength'], '.')
#plt.figure()

model = models.LinearModel()
fit_pixel_wavelength = model.fit(x=df_fl['pixel'], data=df_fl['wavelength'])
#print(fit_pixel_wavelength.params['slope'])
#print(fit_pixel_wavelength.params['intercept'])
#fit_pixel_wavelength.plot(xlabel='pixel', ylabel='Wavelength (nm)')

flat_dict, flat_headers = import_files(
    "C:/Users/bvptr/academia/physics/year2/natuurkunde_en_sterrenkunde_practicum2/solar_physics_non_git/solar_physics_data/NP2 zonnefysica/20211112 - LISA daglicht spectra/" ,
    "flat_alt*"
)

light_dict, light_headers = import_files(
    "C:/Users/bvptr/academia/physics/year2/natuurkunde_en_sterrenkunde_practicum2/solar_physics_non_git/solar_physics_data/20201109 sky zenith/" ,
    "skyspectrum*")

dark_dict, dark_headers = import_files(
    "C:/Users/bvptr/academia/physics/year2/natuurkunde_en_sterrenkunde_practicum2/solar_physics_non_git/solar_physics_data/20201109 sky zenith/" ,
    "Dark*"
)

bias_dict, bias_headers = import_files(
    "C:/Users/bvptr/academia/physics/year2/natuurkunde_en_sterrenkunde_practicum2/solar_physics_non_git/solar_physics_data/NP2 zonnefysica/SunsetLISA/" ,
    "bias*"
)

#print(
#    light_headers['skyspectrum zenit-0001.fit']['IMAGETYP'],
#    flat_headers['flat_alt-0001.fit']['IMAGETYP'],
#    dark_headers['Dark-0001.fit']['IMAGETYP'],
#    bias_headers['bias-001.fit']['IMAGETYP'],
#)

light_stack = stack( light_dict.values() )
flat_stack = stack( flat_dict.values() )
dark_stack = stack( dark_dict.values() )
bias_stack = stack( bias_dict.values() )

# NOTE: Dark is currently broken (it has a neon frame through it) so we ignore it.
# TODO: mean or sum or median? I currently don't konw what difference it makes.
#dark_master = np.median(dark_stack)
bias_master = np.median(bias_stack)
flat_master = np.median([flat - bias_master for flat in flat_stack ])
calibrated = [ (light - bias_master) / flat_master for light in light_stack ]

# For some reason when we use this spectrum we only get what we expect if we take only part of it.
# Currently the subtraction of bias and flat does nothing.
spectrum = np.median(calibrated[300:800], axis=0)
#spectrum = np.median(light_stack[300:800], axis=0)

# We now convert the pixel values of the spectrum to wavelengths
wavelengths = pixels_to_wavelenghts(range(len(spectrum)), fit_pixel_wavelength.values['slope'], fit_pixel_wavelength.values['intercept'])

# crude method to determine effective temperature.
T_accepted = 5772
print(wavelength_peak := get_peak_wavelength(wavelengths, spectrum)) # 481.6998282285298
print(T_eff := get_effective_temperature(wavelength_peak)) # 6015.72137913495
print("%f%% error from the accepted value." % float( ( T_eff / T_accepted ) * 100 - 100) ) #4.22% error

#plt.xlabel('wavelength (nm)')
#plt.ylabel('intensity (counts)')
#plt.plot(wavelengths, spectrum)

# fitting
print(type(list(spectrum)))
print(type(wavelengths))
f = lambda T, wvlen :  ( ( ac.R_sun / ac.au ) ** 2 )  *  ( 2 * sc.Planck * sc.speed_of_light ** 2 ) / \
    ( (wvlen**5) * ( np.exp( ( sc.Planck * sc.speed_of_light) / ( wvlen * sc.k * T ) ) - 1 ) )  
model_planck_scaled = models.Model(f, name="Planck_scaled")
fit_planck_scaled = model_planck_scaled.fit(list(spectrum), wvlen=list(wavelengths), temperature=T_accepted)
fit_planck_scaled.plot(xlabel="$\lambda$ (nm)", ylabel="Spectral irradiance")
plt.show()