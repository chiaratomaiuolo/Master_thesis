import argparse

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *


__description__ = \
"""This script fits and plots the peak of the ADC curve
    of GPD done in HV scans at different pressures. 
"""

HV_SCAN_GAIN_ARGPARSER = argparse.ArgumentParser(description=__description__)
HV_SCAN_GAIN_ARGPARSER.add_argument('gas_type', type=str,
            help="gas type hv scan to be analyzed. It can be chosen between 'Ar'\
                and 'DME'")

"""Defining the dictionaries filled with data: 
    HV value [V]
    Principal peak mean [adc counts]
    Error on principal peak mean [adc counts]
    Resoultion [adc counts]
    Error on resolution [adc counts]
    Track size mean [# pixel]
    Error on Track size mean [# pixel]
    Track size std [# pixel]
    Marker (for plot)
    Color (for plot)
"""

#Argon filled HV scan
data_Ar = {1050: (np.array([470,475,480,485,490,495]), 
               np.array([7.62542e+03, 8.87178e+03, 1.02663e+04, 1.18671e+04, 1.37032e+04, 1.57368e+04]), 
               np.array([7.82002e+00, 8.53888e+00, 9.61986e+00, 1.04984e+01, 1.18495e+01, 1.31743e+01]), 
               np.array([2.41983e-01, 2.24806e-01, 2.18380e-01, 2.07768e-01, 2.01876e-01, 1.95274e-01]), 
               np.array([2.32420e-03, 2.18750e-03, 2.16568e-03, 1.99791e-03, 1.98421e-03, 1.93137e-03]),
               np.array([98,105,112,119,126,132]),
               np.array([23,25,26,28,30,31]),
               '.',
               'tab:orange'),
            950: (np.array([455,460,465,470,475,480]),
              np.array([7.17503e+03, 8.38174e+03, 9.72369e+03, 1.13145e+04, 1.30451e+04, 1.50717e+04]),
              np.array([8.49603e+00, 9.14891e+00, 9.81264e+00, 1.09992e+01, 1.23929e+01, 1.33978e+01]), 
              np.array([2.64847e-01, 2.43231e-01, 2.26907e-01, 2.16511e-01, 2.10544e-01, 1.97267e-01]), 
              np.array([2.71184e-03, 2.50197e-03, 2.31902e-03, 2.24659e-03, 2.17428e-03, 2.03899e-03]),
              np.array([103,111,120,128,136,144]),
              np.array([25,27,29,31,33,36]),
              '.',
              'tab:blue'),
            850: (np.array([440,445,450,455,460,465]),
                  np.array([6.53579e+03,7.74855e+03,9.07075e+03,1.05516e+04,1.22048e+04,1.40925e+04]),
                  np.array([9.75008e+00,1.04135e+01,1.05858e+01,1.20356e+01,1.29825e+01,1.39050e+01]),
                  np.array([3.19173e-01,2.86031e-01,2.54087e-01,2.38682e-01,2.24541e-01,2.08841e-01]),
                  np.array([3.37182e-03,3.12297e-03,2.69829e-03,2.66632e-03,2.39975e-03,2.27133e-03]),
                  np.array([106,117,128,138,147,157]),
                  np.array([27,29,32,34,37,40]),
                  '.',
                  'tab:green'),
            750: (np.array([430,435,440,445,450,455]), 
                  np.array([6.94290e+03,8.20903e+03,9.63402e+03,1.13148e+04,1.31130e+04,1.51764e+04]),
                  np.array([1.33684e+01,1.33652e+01,1.37944e+01,1.46493e+01,1.55749e+01,1.57513e+01]),
                  np.array([3.84587e-01,3.28564e-01,2.85835e-01,2.57298e-01,2.34863e-01,2.19281e-01]),
                  np.array([4.38797e-03,3.62416e-03,3.29435e-03,2.97945e-03,2.78117e-03,2.38081e-03]),
                  np.array([119,133,147,160,173,185]),
                  np.array([33,35,38,42,46,49]),
                  '.',
                  'tab:red')}
#DME filled HV scan
data_DME = {900: (np.array([555, 550, 545, 540, 535, 530]),
                np.array([1.50486e+04, 1.31859e+04, 1.14255e+04, 1.00099e+04, 8.68993e+03, 7.55785e+03]),
                np.array([2.98336e+01, 2.42018e+01, 9.94992e+00, 2.06765e+01, 1.73132e+01, 1.57285e+01]),
                np.array([2.42859e-01, 2.48126e-01, 2.47019e-01, 2.49135e-01, 2.41524e-01, 2.51318e-01]),
                np.array([4.56488e-03, 4.27562e-03, 1.98905e-03, 4.68997e-03, 4.54468e-03, 4.78248e-03]),
                np.array([67.27, 64.27, 62.25, 59.82, 57.73, 55.54]),
                np.array([0.23, 0.19, 0.09, 0.19, 0.19, 0.17]),
                np.array([16,16,15,14,14,13]),
                #np.array([62,59,58,56,53,51]),
                #np.array([20,19,18,17,17,16]),
                '^',
                'tab:brown'),
            800: (np.array([525,520,515,510,505,500]),
                np.array([1.54167e+04,1.33590e+04,1.16904e+04,1.01651e+04,8.75024e+03,7.65104e+03]),
                np.array([3.05281e+01,2.65245e+01,2.36091e+01,1.92836e+01,1.87316e+01,1.59432e+01]),
                np.array([2.22818e-01,2.26844e-01,2.30069e-01,2.29987e-01,2.38199e-01,2.36385e-01]),
                np.array([4.50918e-03,4.49710e-03,4.60714e-03,4.31215e-03,5.06318e-03,4.71253e-03]),
                np.array([76.90, 73.63, 70.65, 68.02, 65.31, 62.41]),
                np.array([0.27, 0.25, 0.25, 0.22, 0.22, 0.41]),
                np.array([18, 18, 17, 16, 15, 15]),
                #np.array([70,68,65,63,60,58]),
                #np.array([23,22,21,20,19,18]),
                '^',
                'tab:pink'),
            700: (np.array([495,490,485,480,475,470]), 
                np.array([1.53335e+04,1.33529e+04,1.15509e+04,9.96740e+03,8.68138e+03,7.51250e+03]),
                np.array([2.97322e+01,2.53484e+01,2.21335e+01,1.89092e+01,1.74724e+01,1.56408e+01]),
                np.array([2.16828e-01,2.14935e-01,2.17729e-01,2.09969e-01,2.26136e-01,2.34229e-01]),
                np.array([4.48426e-03,4.35292e-03,4.45490e-03,4.34961e-03,4.60379e-03,4.79673e-03]),
                np.array([89.3, 85.82, 82.38, 78.31, 75.78, 71.52]),
                np.array([0.3, 0.29, 0.27, 0.27, 0.25, 0.23, 0.09]),
                np.array([21, 20, 19, 18, 17, 16]),
                #np.array([82.,79.,75.,72.,69.,66.]),
                #np.array([27.,26.,24.,24.,23.,21.]),
                '^',
                'tab:gray'),
            600: (np.array([465,460,455,450,445,440]),
                np.array([1.50914e+04, 1.31539e+04, 1.13804e+04, 9.84107e+03, 8.48239e+03, 7.27940e+03]),
                np.array([2.92899e+01, 2.32623e+01, 2.19038e+01, 1.95515e+01, 1.66600e+01, 1.50333e+01]),
                np.array([2.06922e-01, 2.04551e-01, 2.09664e-01, 2.14846e-01, 2.17263e-01, 2.32728e-01]),
                np.array([4.61977e-03, 4.06534e-03, 4.42278e-03, 4.51371e-03, 4.52059e-03, 4.71713e-03]),
                np.array([108.0, 103.71, 98.82, 94.41, 89.04, 83.84]),
                np.array([0.3, 0.34, 0.33, 0.32, 0.29, 0.27, 0.04]),
                np.array([25, 24, 22, 21, 20, 18]),
                #np.array([98, 94, 90, 86, 81, 77]),
                #np.array([34, 32, 30, 28, 26, 25]),
                '^',
                'tab:olive')}

def line(x, m, q):
    return m*x + q

def expo(x,norm, scale):
    return norm*np.exp((x-400)/scale)

def plot_HV_scan(volt, adc_counts, d_adc_counts, resolution, d_resolution,
                 track_size_mean, d_track_size_mean, track_size_std, pressure, gas_type, reference_voltage):
    #Fitting the gain with an exponential curve
    popt, pcov = curve_fit(expo, volt, adc_counts, p0=(adc_counts.min(), 35), sigma=d_adc_counts) 
    norm, scale = popt
    norm_u = ufloat(norm, np.sqrt(pcov[0][0]))
    scale_u = ufloat(scale, np.sqrt(pcov[1][1]))
    print(f'Optimal parameters for {gas_type} at P={pressure} mbar : norm = {norm} +/- {np.sqrt(pcov[0][0])}, scale = {scale} +/- {np.sqrt(pcov[1][1])}')
    #Printing curve value at reference gain
    gain =  norm_u*exp((reference_voltage-400)/scale_u)
    print(f'----Gain at reference voltage DeltaV = {reference_voltage}: gain = {gain}-----')
    #Fitting the mean track size with a line
    popt_trk, pcov_trk = curve_fit(line, volt, track_size_mean, sigma=d_track_size_mean)
    m, q = popt_trk
    m_u = ufloat(m, np.sqrt(pcov_trk[0][0]))
    q_u = ufloat(q, np.sqrt(pcov_trk[1][1]))
    mean_trk_size_u = m_u*reference_voltage + q_u
    print(f'-------Mean track size at reference voltage DeltaV = {reference_voltage}: track_size = {mean_trk_size_u}-----')


    z = np.linspace(np.min(volt),np.max(volt),1000)

    plt.figure(1)
    plt.subplot(211)
    plt.errorbar(volt,adc_counts,yerr=d_adc_counts, fmt=marker, color=color, label=f'{gas_type}, {pressure} [mbar]')
    plt.axvline(x=reference_voltage, color='g', linestyle='dashed', label='reference voltage')
    plt.yscale('log')
    #plt.ylim(min(adc_counts)-100,max(adc_counts)+1000)
    plt.xlabel('HV [V]')
    plt.ylabel('peak value [ADC]')
    plt.plot(z, expo(z,*popt))
    plt.legend()
    plt.grid(True)
    
    res = (adc_counts - expo(volt,*popt))
    plt.subplot(212)
    plt.xlabel('HV [V]')
    plt.ylabel('peak value [ADC]')
    plt.errorbar(volt, res, d_adc_counts, fmt=marker, color=color, label=f'{gas_type}, {pressure} [mbar]')
    plt.axvline(x=reference_voltage, color='r', linestyle='dashed', label='reference voltage')
    plt.legend()
    plt.grid(True)

    plt.figure(2)
    plt.xlabel('HV [V]')
    plt.ylabel('energy resolution')
    plt.errorbar(volt, resolution, d_resolution, fmt=marker, color=color, label=f'{gas_type}, {pressure} [mbar]')
    plt.axvline(x=reference_voltage, color='r', linestyle='dashed', label='reference voltage')
    plt.legend()
    plt.grid(True)

    plt.figure(3)
    plt.xlabel('mean track size [number of pixels]')
    plt.ylabel('energy resolution')
    plt.errorbar(track_size_mean, resolution, d_resolution, fmt=marker, color=color, label=f'{gas_type}, {pressure} [mbar]')
    plt.legend()
    plt.grid(True)

    plt.figure(4)
    plt.title(fr'Mean track size as a function of $\Delta$V')
    plt.errorbar(volt, track_size_mean, fmt=marker, color=color, label=f'{gas_type}, {pressure} [mbar]')
    plt.legend()
    plt.plot(z, line(z,*popt_trk))
    plt.axvline(x=reference_voltage, color='r', linestyle='dashed', label='reference voltage')
    plt.xlabel('HV [V]')
    plt.ylabel('mean track size [number of pixels]')
    plt.grid(True)

    plt.figure(5)
    plt.xlabel('pressure [mbar]')
    plt.ylabel('$\lambda$ [V]')
    plt.errorbar(pressure, scale, np.sqrt(pcov[1][1]), 10, fmt=marker, color=color)
    plt.grid(True)



    return scale, np.sqrt(pcov[1][1])


if __name__ == '__main__':
    #Defining the reference voltage 
    ref_voltage = 490 #V
    #Defining lists to be filled during the scan
    scale_array_DME = []
    d_scale_array_DME = []
    pressure_array_DME = []
    if HV_SCAN_GAIN_ARGPARSER.parse_args().gas_type == 'DME':
        for pressure in data_DME.keys():
            (HV, peak, dpeak, res, dres, trk_size_mean, trk_size_std, d_trk_size_mean, marker, color) = data_DME[pressure]

            scale, d_scale = plot_HV_scan(HV, peak, dpeak, res, dres, trk_size_mean, d_trk_size_mean, trk_size_std, pressure, 'DME', ref_voltage)
            pressure_array_DME.append(pressure)
            scale_array_DME.append(scale)
            d_scale_array_DME.append(d_scale)

        print(scale_array_DME)
        scale_array_DME=np.array(scale_array_DME)
        d_scale_array_DME=np.array(d_scale_array_DME)
        pressure_array_DME=np.array(pressure_array_DME)
        popt_scale, pcov_scale = curve_fit(line, pressure_array_DME, scale_array_DME, sigma = d_scale_array_DME)
        chi2_DME = (((scale_array_DME- line(pressure_array_DME, *popt_scale))/d_scale_array_DME)**2).sum()
        print(f'chi^2 = {chi2_DME}')
        w = np.linspace(600, 900, 100)
        plt.plot(w, line(w, *popt_scale), color ='tab:cyan', label=f'm = {popt_scale[0]:.6f} +/- {np.sqrt(pcov_scale[0][0]):.6f} \n q = {popt_scale[1]:.2f} +/- {np.sqrt(pcov_scale[1][1]):.2f}')
        print(f'Parameters for line fitting of scale factors: m = {popt_scale[0]} +/- {np.sqrt(pcov_scale[0][0])} , q = {popt_scale[1]} +/- {np.sqrt(pcov_scale[1][1]) }')
        plt.legend(loc='best')
        plt.show()
    if HV_SCAN_GAIN_ARGPARSER.parse_args().gas_type == 'Ar':
        scale_array_Ar = []
        d_scale_array_Ar = []
        pressure_array_Ar = []

        for pressure in data_Ar.keys():
            (HV, peak, dpeak, res, dres, trk_size_mean, d_trk_size_mean, trk_size_std, marker, color) = data_Ar[pressure]

            scale, d_scale = plot_HV_scan(HV, peak, dpeak, res, dres, trk_size_mean, d_trk_size_mean, trk_size_std, pressure, 'Ar', ref_voltage)
            pressure_array_Ar.append(pressure)
            scale_array_Ar.append(scale)
            d_scale_array_Ar.append(d_scale)



        popt_scale, pcov_scale = curve_fit(line, pressure_array_Ar, scale_array_Ar, sigma = d_scale_array_Ar)
        w = np.linspace(700, 1050, 100)
        plt.plot(w, line(w, *popt_scale), color ='tab:cyan', label=f'm = {popt_scale[0]:.6f} +/- {np.sqrt(pcov_scale[0][0]):.6f} \n q = {popt_scale[1]:.2f} +/- {np.sqrt(pcov_scale[1][1]):.2f}')
        print(f'Parameters for line fitting of scale factors: m = {popt_scale[0]} +/- {np.sqrt(pcov_scale[0][0])} , q = {popt_scale[1]} +/- {np.sqrt(pcov_scale[1][1]) }')
        plt.legend(loc='best')

        plt.show()
