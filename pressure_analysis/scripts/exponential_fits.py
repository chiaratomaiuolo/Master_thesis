import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.fft import fft, fftfreq, fftshift
from scipy.optimize import curve_fit

from labviewdataprocessing.labviewdatareading import LabViewdata_reading

__description__ = \
"This script is used for performing the data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS."

def expo(x, A0, tau, c):
    return A0*(np.exp(-x/tau)) + c

def double_exp(x, A1, tau1, A2, tau2, c):
    return A1*(np.exp(-x/tau1)) + A2*(np.exp(-x/tau2)) + c

def fixed_exp(x, A2, tau2, c):
    return expo(x,  11.685, 1.145, 0.) + expo(x, A2, tau2, c)

if __name__ == "__main__":
    #Datafiles are briefly descripted above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times

    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']

    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
    start_times = [['2024-02-26 15:51:00.000']]
    stop_times = [[None]]

    #Obtaining arrays of data
    data_list = LabViewdata_reading(paths_to_data, start_times, stop_times)

    #Obtaining interesting data
    data_list = LabViewdata_reading(paths_to_data, start_times, stop_times)
    timestamps = data_list[0]
    T5 = data_list[6]
    T6 = data_list[7]
    P4 = data_list[15]
    dP4 = data_list[17]
    P3 = data_list[13]
    TJ = data_list[9]
    t_diffs = data_list[18] #s 
    t_hours = t_diffs/3600 #hours

    #We want to exploit the optimal parameter variation as we increase the number
    #of data. 
    i=5
    mask = t_hours < i
    popt, pcov = curve_fit(double_exp, t_hours, P4, p0=[19.45, 1.4, 58., 25., 1111.])
    #Printing optimal parameters
    print(f'Optimal parameters: A1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
          A2 = {popt[2]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau2 = {popt[3]} +/- {np.sqrt(pcov[1][1])} [hours],\
          c = {popt[4]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    chisq = (((P4 - double_exp(t_hours, *popt))/(dP4))**2).sum()
    ndof = len(P4) - len(popt)
    #Plotting pts and fit
    plt.figure()
    plt.title(r'$P_4$ inside AC with epoxy samples inside as a function of time from DME filling - data and fits')
    z = np.linspace(0, max(t_hours), 2000)
    plt.errorbar(t_hours, P4, marker='.', alpha=0.5, markeredgewidth=0, linestyle='', color='darkorange', label='Data')
    plt.plot(z, double_exp(z, *popt), label=f'Double exponential with $t < {i} [hours]$')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    print(mask[-1])
    print(mask)
    print(type(mask[-1]))
    print(mask[-1] == False)
    while mask[-1] == False:
        print('entering loop')
        i+=10
        mask = t_hours < 5+i
        popt, pcov = curve_fit(double_exp, t_hours, P4, p0=[19.45, 1.4, 58., 25., 1111.])
        print(f'Optimal parameters for t_hours< {i}: A1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
              tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
              A2 = {popt[2]} +/- {np.sqrt(pcov[0][0])} [mbar],\
              tau2 = {popt[3]} +/- {np.sqrt(pcov[1][1])} [hours],\
              c = {popt[4]} +/- {np.sqrt(pcov[2][2])} [mbar]')
        chisq = (((P4 - double_exp(t_hours, *popt))/(dP4))**2).sum()
        ndof = len(P4) - len(popt)
        plt.plot(z, double_exp(z, *popt), label=f'Double exponential with $t < {i} [hours]$')


    #We want to exploit the optimal parameter variation as we increase the number
    #of data. 
    i=5
    mask = t_hours < i
    popt, pcov = curve_fit(fixed_exp, t_hours, P4, p0=[58., 25., 1111.])
    #Printing optimal parameters
    print(f'Optimal parameters: A1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
          c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    #Plotting pts and fit
    plt.figure()
    plt.title(r'$P_4$ inside AC with epoxy samples inside as a function of time from DME filling - data and fits')
    z = np.linspace(0, max(t_hours), 2000)
    plt.errorbar(t_hours, P4, marker='.', alpha=0.5, markeredgewidth=0, linestyle='', color='darkorange', label='Data')
    plt.plot(z, fixed_exp(z, *popt), label=f'Double exponential with $t < {i} [hours]$')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    print(mask[-1])
    print(mask)
    print(type(mask[-1]))
    print(mask[-1] == False)
    while mask[-1] == False:
        print('entering loop')
        i+=10
        mask = t_hours < 5+i
        popt, pcov = curve_fit(fixed_exp, t_hours, P4, p0=[58., 25., 1111.])
        print(f'Optimal parameters: A1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
                tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
                c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar]')
        chisq = (((P4 - fixed_exp(t_hours, *popt))/(dP4))**2).sum()
        ndof = len(P4) - len(popt)
        plt.plot(z, fixed_exp(z, *popt), label=f'Double exponential with $t < {i} [hours]$')

    
    
    
    
    
    
    
    plt.legend()
    plt.show()