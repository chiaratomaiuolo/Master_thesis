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
    
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled.
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']
    
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled, selected times intervals where T_ambient is stable.
    '''
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
    

    #Plotting pressure variations and temperature variations
    fig, axs = plt.subplots(2)
    fig.suptitle(fr'Consecutive pressure variations inside AC and corresponding temperature variations - Gas DME, $T_{{Julabo}}$ = {np.mean(TJ):.2f}°C')
    axs[0].plot(timestamps[1:], np.diff(P4), color='red')
    axs[0].set(xlabel=r'Timestamp', ylabel=r'$\Delta P_4$ [mbar]')
    axs[0].grid(True)
    axs[1].plot(timestamps[1:], np.diff(T5))
    axs[1].set(xlabel=r'Timestamp', ylabel=r'$\Delta T_5$ [°C]')
    axs[1].grid(True)


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

    #Fitting absolute pressure with a double exponential
    t_hours = t_diffs/3600 #hours

    popt, pcov = curve_fit(double_exp, t_hours, P4, p0=[19.45, 1.4, 58., 25., 1111.])
    print(f'Optimal parameters: A1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
          A2 = {popt[2]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau2 = {popt[3]} +/- {np.sqrt(pcov[1][1])} [hours],\
          c = {popt[4]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    chisq = (((P4 - double_exp(t_hours, *popt))/(dP4))**2).sum()
    ndof = len(P4) - len(popt)
    print(f'chisq/ndof = {chisq}/{ndof}')
    print(f'Estimation of temperature after 48 hours: {double_exp(48, *popt)}')
    plt.figure()
    plt.title(r'$P_4$ inside AC with epoxy samples inside as a function of time from DME filling - data and fits')
    z = np.linspace(0, max(t_hours), 2000)
    plt.errorbar(t_hours, P4, marker='.', alpha=0.5, markeredgewidth=0, linestyle='', color='darkorange', label='Data')
    plt.plot(z, double_exp(z, *popt), color='red', label='Double exponential')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')

    #Fitting absolute pressure with a single exponential
    #t = np.array([acq_time*i for i in range(len(timestamps))])
    popt, pcov = curve_fit(expo, t_hours, P4, p0=[3.75, 10., 1140.])
    print(f'Optimal parameters: P0 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
          c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    t_year = 5*popt[1] #5 times characteristic time
    chisq = (((P4 - expo(t_hours, *popt))/(dP4))**2).sum()
    ndof = len(P4) - len(popt)
    print(f'chisq/ndof = {chisq}/{ndof}')
    print(f'Estimation of the asymptotic value: {expo(t_year,*popt)}')
    #plt.figure()
    #plt.title(r'$P_4$ as a function of time from DME filling')
    #z = np.linspace(0, max(t_hours), 2000)
    plt.plot(z, expo(z, *popt), color='forestgreen',label='Single exponential')
    #plt.errorbar(t_hours, P4, marker='.', linestyle='', color='firebrick')
    plt.xlabel('Time from filling [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    plt.grid()
    plt.legend()

    plt.figure('P/T')
    plt.errorbar(P4/(expo(t_hours, *popt)), T6)
    plt.xlabel(r'$\frac{P_4}{P_{4_{fitted}}}$')
    plt.ylabel(r'$T_{ambient}$ [°C]')
    plt.grid()

    T_eff = T5+0.16*(T6-T5)/1.16
    T_eff = T_eff + 273.15 #Kelvin
    fig, ax1 = plt.subplots()
    #Cutting on time interval - both times and all the other interesting quantities
    mask = t_hours>0
    P4 = P4[mask]
    T6 = T6[mask]
    T5 = T5[mask]
    t_hours = t_hours[mask]
    #Computing index for performing the time shift of 216 seconds
    log_time = 5000e-3 #s
    #delay_Troom_Tgas = 216 #s #computed 'by hand'
    delay_Troom_Tgas = 1200 #s #computed 'by hand'
    delay_idx_start = int(np.floor(delay_Troom_Tgas/log_time)) #index for translating P4 forward of n s
    delay_idx_stop = len(P4) - (delay_idx_start) #index for cutting the last 216 seconds of measures (for arrays compatibility)
    
    ax1.set_xlabel('Time from new $T_{Julabo}$ settings [hours]')
    ax1.set_ylabel(fr'$P_4$ [mbar]')
    ax1.plot(t_hours, (P4-expo(t_hours, *popt))/(max(P4)-min(P4)), color='red', label=r'$P_4$ [mbar]')
    #ax1.plot(t_hours[:delay_idx_stop], P4[delay_idx_start:], color='green', label=r'$P_4$ [mbar]')


    ax2 = ax1.twinx()  # instantiate a second pair of axes that shares the same x-axis
    ax2.set_ylabel(r'$T_{ambient} - T_5$ [°C]', color='steelblue')
    T_eff = T_eff + 273.15 #Kelvin
    ax2.plot(t_hours, (T_eff[mask])/(max(T_eff)-min(T_eff)), linestyle = 'dotted', color='steelblue', label=r'$T_{ambient} - T_5$')
    #ax2.plot(t_hours, (T6-T5)/(max(T6-T5)-min(T6-T5)), linestyle = 'dotted', color='steelblue', label=r'$T_{ambient} - T_5$')
    ax2.tick_params(axis='y', labelcolor='steelblue')

    plt.figure()
    plt.errorbar(P4[delay_idx_start:]/(expo(t_hours[delay_idx_start:], *popt)), T_eff[:delay_idx_stop])
    plt.xlabel(r'$\frac{P_4}{P_{4_{fitted}}}$')
    plt.ylabel(r'$T_{gas}$ [°C]')
    plt.grid()
    '''
    plt.figure()
    print(fftfreq(len(T_eff), (log_time/3600)))
    print(fft(T_eff))
    freq = fftfreq(len(T_eff), (log_time/3600))
    print((log_time)*len(T_eff))
    print(len(fft((T_eff)/(max(T_eff)-min(T_eff)))))
    print(len(fftfreq(len(T_eff), (log_time/3600))))
    sp = fft((T_eff)/(max(T_eff)-min(T_eff)))
    plt.plot(freq, sp.real)
    plt.xlim(-0.2,0.2)
    #plt.ylim(0,1000)
    '''


    fig.tight_layout()
    fig.legend()
    
    
    plt.show()
