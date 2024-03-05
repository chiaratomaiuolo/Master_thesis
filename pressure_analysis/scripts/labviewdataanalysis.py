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

if __name__ == "__main__":
    #Datafiles are briefly descripted above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times
    '''
    #Datafile from 12/2/2024 to 20/2/2024 - AC N2 filled.
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_N2_measurements.txt']
    #Datafile from 12/2/2024 to 20/2/2024 - AC N2 filled.
    start_times = [None]
    stop_times = [None]
    '''
    '''
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled.
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']
    '''

    '''
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled, selected times intervals where T_ambient is stable.
    
    start_times = [['2024-02-12 16:00:00.000', '2024-02-13 16:30:00.000', '2024-02-14 16:00:00.000',\
                   '2024-02-15 16:00:00.000', '2024-02-16 16:00:00.000','2024-02-17 16:00:00.000',\
                    '2024-02-18 16:00:00.000', '2024-02-21 16:00:00.000', '2024-02-22 16:00:00.000']]
    stop_times = [['2024-02-12 20:00:00.000', '2024-02-13 15:00:00.000', '2024-02-14 15:00:00.000',\
                   '2024-02-15 20:00:00.000','2024-02-16 16:00:00.001', '2024-02-17 20:00:00.000', \
                   '2024-02-18 20:00:00.000', '2024-02-21 20:00:00.000', '2024-02-22 18:00:00.000']]
    '''
    
    
    '''
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled, full dataset selection
    start_times = [['2024-02-16 18:30:00.000']]
    stop_times = [['2024-02-19 11:00:00.000']]
    '''
    
    
    '''
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled, selection of data having T_Julabo = 40 C (RISING)
    start_times = [['2024-02-12 11:00:00.000']]
    stop_times = [['2024-02-19 16:20:00.000']]
    '''
    '''
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled, selection of data having T_Julabo from 40 to 22 C (DECREASING)
    start_times = [['2024-02-19 16:53:00.000']]
    stop_times = [['2024-02-19 22:00:00.000']]
    '''

    '''
    #Datafile from 20/02/2024 - AC DME filled, selection of data having T_Julabo from 22 to 10 C (DECREASING)
    start_times = [['2024-02-20 12:37:00.000']]
    stop_times = [['2024-02-20 18:07:00.000']]
    '''
    '''
    #Datafile from 20/02/2024 - AC DME filled, selection of data having T_Julabo from 10 to 22 C (RISING)
    start_times = [['2024-02-20 18:07:00.000']]
    stop_times = [['2024-02-22 18:06:00.000']]
    '''
    
    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
    start_times = [['2024-02-26 15:51:00.000']]
    stop_times = [[None]]
    '''
    start_times = [['2024-02-26 15:51:00.000','2024-02-27 8:00:00.000', 
                    '2024-02-28 8:00:00.000','2024-02-29 8:00:00.000',
                    '2024-03-01 8:00:00.000','2024-03-02 8:00:00.000',
                    '2024-03-03 8:00:00.000']]
    stop_times = [['2024-02-26 22:00:00.000','2024-02-27 22:00:00.000',
                   '2024-02-28 22:00:00.000','2024-02-29 22:00:00.000',
                   '2024-03-01 22:00:00.000','2024-03-02 22:00:00.000',
                   '2024-03-03 22:00:00.000']]
    '''

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
    #Computing time in hours
    t_hours = t_diffs/3600 #hours
    
    #Looking at overall data - P4, T5, T_room
    fig, axs = plt.subplots(3)
    fig.suptitle(fr'Absolute pressure inside AC and corresponding temperature - Gas DME,$T_{{Julabo}}$ = {np.mean(TJ):.2f}°C')
    axs[0].plot(timestamps, P4, color='firebrick')
    axs[0].set(xlabel=r'Timestamp', ylabel=r'$P_4$ [mbar]')
    axs[0].grid(True)
    axs[1].plot(timestamps, T5)
    axs[1].set(xlabel=r'Timestamp', ylabel=r'$T_5$ [°C]')
    axs[1].grid(True)
    axs[2].plot(timestamps, T6, color='red')
    axs[2].set(xlabel=r'Timestamp', ylabel=r'$T_{\text{ambient}}$ [°C]')
    axs[2].grid(True)

    #Fitting absolute pressure with a single exponential
    popt, pcov = curve_fit(expo, t_hours, P4, p0=[3.75, 10., 1140.])
    print(f'Optimal parameters: \n\
          A0 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
          tau = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\n\
          c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    chisq = (((P4 - expo(t_hours, *popt)))**2).sum()
    ndof = len(P4) - len(popt)
    print(f'chisq/ndof = {chisq}/{ndof}')
    z = np.linspace(0, max(t_hours), 2000)
    plt.figure()
    plt.title(r'$P_4$ inside AC with epoxy samples inside as a function of time from DME filling - data and fits')
    plt.errorbar(t_hours, P4, marker='.', alpha=0.5, markeredgewidth=0, linestyle='', color='darkorange', label='Data')
    plt.plot(z, expo(z, *popt), color='forestgreen',label='Single exponential')
    #plt.errorbar(t_hours, P4, marker='.', linestyle='', color='firebrick')
    plt.xlabel('Time from filling [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    plt.grid()
    plt.legend()

    #Fitting absolute pressure with a double exponential
    popt, pcov = curve_fit(double_exp, t_hours, P4, p0=[19.45, 1.4, 58., 25., 1111.])
    print(f'Optimal parameters: A1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
          A2 = {popt[2]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau2 = {popt[3]} +/- {np.sqrt(pcov[1][1])} [hours],\
          c = {popt[4]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    chisq = (((P4 - double_exp(t_hours, *popt)))**2).sum()
    ndof = len(P4) - len(popt)
    print(f'chisq/ndof = {chisq}/{ndof}')
    plt.plot(z, double_exp(z, *popt), color='red', label='Double exponential')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')

    
    plt.show()
