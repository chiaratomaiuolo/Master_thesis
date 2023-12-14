"""Analysis script
"""
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from astropy.io import fits
from scipy.stats import norm
from scipy.optimize import curve_fit

from pressure_analysis.models import Gauss

"""This script is done for fitting and plotting the track sampled from a
GPD detector, in particular takes a data ascquisition in input and plots
the ADC counts of the tracks, does a Gaussian fit and prints the optimal
parameters. 
"""

def secular_pressure_analysis(dataset_id : np.array) -> None:
    """Opens files and performs the analysis.

    Arguments
    ---------
    dataset_id : np.array 
        An array-like containing the IDs of the datasets to be analyzed.
    
    """
    return




if __name__ == '__main__':
    #Collecting .fits data from file 
    f=fits.getdata("/Users/chiara/Desktop/Thesis_material/data/drive-download-20230317T130511Z-001/020_0001777/020_0001777_data_recon.fits")
    #Selecting the data of the primary peak (the cutting value in chosen after plotting the whole set preliminarly)
    cutting_value = 5000
    primary_peak = f['PHA'][f['PHA']> cutting_value]

    #Starting value for Gaussian model (chosen after plotting the whole set preliminarly)
    p01 = np.array([1.,7000,150])

    #Fitting  and plotting the primary peak histogram

    plt.figure(1)
    n, edges, patches = plt.hist(primary_peak, bins=100)
    bincenters = (edges[:-1] + edges[1:]) / 2
    plt.xlabel('ADC counts [ADC]')
    plt.ylabel('Occurrencies')

    popt, pcov = curve_fit(Gauss, bincenters, n, p0=p01)
    norm_fit, mu_fit, sigma_fit = popt
    print(f'Optimal parameters: norm = {norm_fit} +/- {np.sqrt(pcov[0][0])}, mu = {mu_fit} +/- {np.sqrt(pcov[1][1])}, sigma = {sigma_fit} +/- {np.sqrt(pcov[2][2])}')

    label = "N = {:.1f} +/- {:.1f}\n$\mu$ = {:.1f} +/- {:.1f}\n$\sigma$ = {:.1f} +/- {:.1f}".format(norm_fit,np.sqrt(pcov[0][0]), mu_fit, np.sqrt(pcov[1][1]), sigma_fit, np.sqrt(pcov[2][2]))
    z = np.linspace(min(bincenters), max(bincenters), 1000)
    plt.plot(z, Gauss(z, *popt), label=label)

    plt.legend(loc='best')
    plt.show()