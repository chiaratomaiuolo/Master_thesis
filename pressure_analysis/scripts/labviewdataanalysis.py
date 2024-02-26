import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

from labviewdataprocessing.labviewdatareading import LabViewdata_reading

def expo(x, A0, tau, c):
    return A0*(np.exp(-x/tau)) + c

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
    
    start_times = [['2024-02-12 16:00:00.000', '2024-02-13 16:30:00.000', '2024-02-14 16:00:00.000',\
                   '2024-02-15 16:00:00.000', '2024-02-16 16:00:00.000','2024-02-17 16:00:00.000',\
                    '2024-02-18 16:00:00.000', '2024-02-21 16:00:00.000', '2024-02-22 16:00:00.000']]
    stop_times = [['2024-02-12 20:00:00.000', '2024-02-13 15:00:00.000', '2024-02-14 15:00:00.000',\
                   '2024-02-15 20:00:00.000','2024-02-16 16:00:00.001', '2024-02-17 20:00:00.000', \
                   '2024-02-18 20:00:00.000', '2024-02-21 20:00:00.000', '2024-02-22 18:00:00.000']]
    
    
    '''
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled, full dataset selection
    start_times = [['2024-02-12 18:00:00.001']]
    stop_times = [['2024-02-20 12:30:00.000']]
    '''
    '''
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled, selection of data having T_Julabo = 40 C (RISING)
    start_times = [['2024-02-19 11:00:00.000']]
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

    #Fitting absolute pressure with a single exponential
    t_hours = t_diffs/3600 #hours
    #t = np.array([acq_time*i for i in range(len(timestamps))])
    popt, pcov = curve_fit(expo, t_hours, P4, p0=[5.75, 24., 1193.], sigma = dP4)
    print(f'Optimal parameters: P0 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
          c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    t_year = 5*popt[1] #5 times characteristic time
    chisq = (((P4 - expo(t_hours, *popt))/(dP4))**2).sum()
    ndof = len(P4) - len(popt)
    print(f'chisq/ndof = {chisq}/{ndof}')
    print(f'Estimation of the asymptotic value: {expo(t_year,*popt)}')
    plt.figure()
    plt.title(r'$P_4$ as a function of time from DME filling')
    z = np.linspace(0, max(t_hours), 2000)
    plt.plot(z, expo(z, *popt), color='steelblue')
    plt.errorbar(t_hours, P4, yerr= dP4, marker='.', linestyle='', color='firebrick')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    plt.grid()
    '''
    fig, ax1 = plt.subplots()
    #Cutting on time interval - both times and all the other interesting quantities
    mask = (t_hours>2.5) & (t_hours< 4.2)
    P4 = P4[mask]
    T6 = T6[mask]
    T5 = T5[mask]
    t_hours = t_hours[mask]
    #Computing index for performing the time shift of 216 seconds
    log_time = 5000e-3 #s
    delay_Troom_Tgas = 216 #s #computed 'by hand'
    delay_idx_start = int(np.floor(delay_Troom_Tgas/log_time)) #index for translating P4 forward of 216 s
    delay_idx_stop = len(P4) - (delay_idx_start) #index for cutting the last 216 seconds of measures (for arrays compatibility)

    ax1.set_xlabel('Time from new $T_{Julabo}$ settings [hours]')
    ax1.set_ylabel(fr'$P_4$ [mbar]')
    ax1.plot(t_hours, P4, color='red', label=r'$P_4$ [mbar]')
    ax1.plot(t_hours[:delay_idx_stop], P4[delay_idx_start:], color='green', label=r'$P_4$ [mbar]')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(r'$T_{ambient} - T_5$ [°C]', color='steelblue')
    ax2.plot(t_hours, (T6-T5), linestyle = 'dotted', color='steelblue', label=r'$T_{ambient} - T_5$')
    ax2.tick_params(axis='y', labelcolor='steelblue')

    fig.tight_layout()
    fig.legend()
    '''
    
    plt.show()
