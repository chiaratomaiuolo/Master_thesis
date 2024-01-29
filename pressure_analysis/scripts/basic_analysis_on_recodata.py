#This code is basically a cut-and-paste from ../sandbox/lbaldini/hv_scan.py re adapted
#for the analysis of a generic quantity from a GPD dataset (for example, the trk_size). 

#it needs the right PYTHONPATH to work. It is necessary to source setup.sh in /gpdsuite repository.

import argparse

from astropy.io import fits
import numpy as np
from loguru import logger
from uncertainties import ufloat

from gpdsuite.fitting import fit_histogram, fit_gaussian_iterative
from gpdsuite.hist import Histogram1d
from gpdsuite.matplotlib_ import plt
from gpdsuite.modeling import Gaussian, ArFe55Spectrum

__description__ = \
""" 
    Analysis of a physical quantity from a reconstructed GPD dataset. 
    This script performs some quality cuts on data and plots the pdf
    of the interested quantity. 
    
"""

def analyze(file_path : str, **kwargs) -> [float, float, float, float]: #returning quantity mean, error, energy res, error on energy res
    """
    """
    #Saving parser arguments needed for the analysis
    #file_path = kwargs['infile']
    cut_radius = kwargs['cutradius']
    target_name = kwargs['target']
    udm_target = kwargs['udm_target']

    #Opening .fits file
    with fits.open(file_path) as hdu_list:
        data = hdu_list['EVENTS'].data
        print(f'Filepath {file_path} opened')
    #Saving in variables the quantities needed for defining the mask
    pha = data['PHA']
    trk_size = data['TRK_SIZE']
    x = data['BARX']
    y = data['BARY']
    absx = data['ABSX']
    absy = data['ABSY']
    #Defining the mask for quality selection (borders, track size and positive signal)
    geometrical_mask = (trk_size > 0) & (trk_size < 300) & (np.abs(absx)<7.) & (np.abs(absy)<7.) & (pha > 0)
    
    #Applying mask on PHA in order to fit a Gaussian model for this quantity (needed for defining the 2nd mask)
    pha = pha[geometrical_mask]
    binning_pha = np.linspace(kwargs['phamin'], kwargs['phamax'], kwargs['bins'])
    h_pha = Histogram1d(binning_pha, xlabel='PHA value [adc counts]').fill(pha)
    h_pha.plot()
    model = fit_gaussian_iterative(h_pha)
    peak = ufloat(model.parameter_value('Peak'), model.parameter_error('Peak'))
    sigma = ufloat(model.parameter_value('Sigma'), model.parameter_error('Sigma'))
    res = sigma/peak
    plt.figure(1)
    h_pha.plot()
    model.plot()
    model.stat_box()

    target = data[target_name]
    if target_name == 'PHA': #if the target was PHA, analysis is over, parameters are Gaussian ones
        print(f'-------- Analyzing attribute {target_name} (with cuts): peak value = {peak}, std on peak = {np.std(target)/np.sqrt(len(target))} and energy resolution {res} -----------')
        if kwargs['show'] == 'True':
            plt.show()
        return model.parameter_value('Peak'), model.parameter_error('Peak'), res.nominal_value, res.std_dev
    
    #Computing mean and std of target variable around PHA peak (1.5 sigmas borders)
    number_of_sigmas = kwargs['n_sigmas']
    mask2 = np.logical_and(pha > model.parameter_value('Peak') - number_of_sigmas*model.parameter_value('Sigma'),
                           pha < model.parameter_value('Peak') + number_of_sigmas*model.parameter_value('Sigma'))
    target = data[target_name][geometrical_mask]
    target = target[mask2]
    print(f'++++++++++++++++++++ {len(target)} ++++++++++++++++++=')
    plt.figure(2)
    binning = np.linspace(min(target), max(target), kwargs['bins'])
    h = Histogram1d(binning, xlabel=f'{target_name} {udm_target}').fill(target)
    h.plot()
    plt.axvline(x=np.mean(target), color='r', label=fr'$\mu_{{target}}$')
    plt.legend()
    
    if args.target == 'trk_size':
        print(f'-------- Analyzing attribute {target_name} (with cuts): mean = {np.mean(target)}, std = {np.std(target)/np.sqrt(len(target))} and energy resolution {res} -----------')
        plt.figure(3)
        plt.title('track size vs PHA')
        plt.hist2d(pha[mask2], target, bins=[50,50], range=[[0,20000],[0,300]])
        plt.xlabel('PHA [ADC counts]')
        plt.ylabel('Track size [number of pixels]')
        plt.ylim(0,300)
        plt.colorbar()
        if kwargs['show'] == 'True':
            plt.show()
        return np.mean(target), np.std(target)/np.sqrt(len(target)), res.nominal_value, res.std_dev

    print(f'-------- Analyzing attribute {target_name} (with cuts): mean = {np.mean(target)}, std = {np.std(target)/np.sqrt(len(target))} and energy resolution {res} -----------')
    
    if kwargs['show'] == 'True':
        plt.show()
        return np.mean(target), np.std(target)/np.sqrt(len(target)), res.nominal_value, res.std_dev 
    
    return np.mean(target), np.std(target)/np.sqrt(len(target)), res.nominal_value, res.std_dev
        



if __name__ == '__main__':
    #Parsing arguments
    parser = argparse.ArgumentParser(description=__description__)
    parser.add_argument('path_to_data_dir', type=str, help='Path to file of dataset to be analyzed')
    #parser.add_argument('data_ids', type=list[str], help='list containing data ids of all datasets to analyze')
    parser.add_argument('target', type=str, help='Target quantity to be analyzed')
    parser.add_argument('pressure', type=int, help='Pressure in mbar')
    parser.add_argument('show', type=str, default='False', help='if True, plots the distribution of the target attribute')
    parser.add_argument('--n_sigmas', type=float, default=1.5, help='Number of sigmas on PHA fit to be applied on cutting mask')
    parser.add_argument('--udm_target', type=str, default='[number of pixels]', help='unit measure of target (for plotting it on xaxis)')
    parser.add_argument('--cutradius', type=float, default=5., help='Radius of the area to be kept')
    parser.add_argument('--phamin', type=float, default=0., help='Minimum vaule of PHA to be kept')
    parser.add_argument('--phamax', type=float, default=25000., help='Maximum value of PHA to be kept')
    parser.add_argument('--bins', type=int, default=100, help='Number of bins to be used for the pdf')
    args = parser.parse_args()
    #Defining lists for saving interesting quatitites
    energy_res = []
    d_energy_res = []
    target_list = []
    dtarget_list = []
    # Defining data ids with a dictionary
    # key = pressure in mbar
    # items = data ids of hv scan for that pressure
    data_ids_dict = {600: ['020_0001892', '020_0001893', '020_0001898', '020_0001895', '020_0001896', '020_0001897'],
                700: ['020_0001884', '020_0001885', '020_0001890', '020_0001887', '020_0001888', '020_0001889'],
                800: ['020_0001877', '020_0001878', '020_0001876', '020_0001880', '020_0001881', '020_0001882'],
                900: ['020_0001869', '020_0001870', '020_0001871', '020_0001867', '020_0001873', '020_0001874']}
    
    data_ids = data_ids_dict[args.pressure]

    for data_id in data_ids:
        filepath_to_data_dir = args.path_to_data_dir
        file_path = f"{filepath_to_data_dir}/{data_id}/{data_id}_data_recon.fits"
        logger.info(f'Analyzing file 020_000{data_id}_data_recon.fits ...')

        mean, sigma, res, sigma_res = analyze(file_path, **args.__dict__)
        energy_res.append(res)
        d_energy_res.append(sigma_res)
        target_list.append(mean)
        dtarget_list.append(sigma)

    target_list = ['%e' % elem for elem in target_list]
    target_list = [float(elem) for elem in target_list]
    dtarget_list = ['%.2e' % elem for elem in dtarget_list]
    dtarget_list = [float(elem) for elem in dtarget_list]

    energy_res = ['%e' % elem for elem in energy_res]
    energy_res = [float(elem) for elem in energy_res]
    d_energy_res = ['%.1e' % elem for elem in d_energy_res]
    d_energy_res = [float(elem) for elem in d_energy_res]



    print(f'{args.target} in {args.pressure} mbar HV scan lists means and errors:')
    print(target_list)
    print(dtarget_list)
    print(f'Energy resolution in {args.pressure} mbar HV scan lists means and errors:')
    print(energy_res)
    print(d_energy_res)