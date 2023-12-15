"""Analysis script
"""

import argparse

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from loguru import logger
from astropy.io import fits
from scipy.stats import norm
from scipy.optimize import curve_fit

#from pressure_analysis.models import gaussian, exponential

__description__ = \
"""Fits and plots the track sampled from a GPD detector, in particular takes
reconstructed data files and plots the ADC counts of the tracks, does a 
Gaussian fit and prints the optimal parameters. 
"""

# Parser object.
SECULAR_PRESSURE_ANALYSIS_ARGPARSER = argparse.ArgumentParser(description=__description__)
SECULAR_PRESSURE_ANALYSIS_ARGPARSER.add_argument('path_to_data_directory', type=str,
            help='path to the directory containing data')
SECULAR_PRESSURE_ANALYSIS_ARGPARSER.add_argument('attribute_name', type=str, help= 'The\
                                                attribute to be analyzed. It can be chosen\
                                                from the list of attributes in a recon\
                                                .fits file.')


def file_opening(file_path : str) -> list:
    """Opens a fits file and returns the events inside.
    Arguments
    ---------
    file_path : str
        file path to the .fits file to be opened
    Return 
    ------
    data : list 
        list containing events 
    """
    with fits.open(file_path) as hdu_list:
        data = hdu_list['EVENTS'].data
        return data

if __name__ == '__main__':
    #Defining the data id number
    data_id = '020_0001869'
    filepath_to_data_dir = SECULAR_PRESSURE_ANALYSIS_ARGPARSER.parse_args().path_to_data_directory
    attribute = SECULAR_PRESSURE_ANALYSIS_ARGPARSER.parse_args().attribute_name
    file_path = f"{filepath_to_data_dir}/{data_id}/{data_id}_data_recon.fits"
    events = file_opening(file_path)
    #Defining the cuts 
    #(events['TRK_SIZE'] > 0) & (events['TRK_SIZE'] < 300)
    borders_mask = (events['ABSX'] < 7.5) & (events['ABSY'] < 7.5)
    trk_size_mask = (events['TRK_SIZE'] > 0) & (events['TRK_SIZE'] < 300)
    mask = np.logical_and(borders_mask,trk_size_mask)
    data = events[attribute]
    masked_data = data[mask]
    print(f'-------- Analyzing attribute {attribute} (with cuts): mean = {np.mean(masked_data)}, std = {np.std(masked_data)} -----------')
    
    plt.figure()
    plt.hist(data, bins = 50)
    plt.hist2d(pha_cutted, trksize_cutted, bins=[50,50], range=[[0,20000],[0,300]])

    plt.show()
