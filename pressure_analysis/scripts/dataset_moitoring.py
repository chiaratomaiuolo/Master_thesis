import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.fft import fft, fftfreq, fftshift
from scipy.optimize import curve_fit

from pressure_analysis.labviewdatareading import LabViewdata_reading, plot_with_residuals
from pressure_analysis.models import expo, double_expo, triple_expo, alpha_expo_scale

__description__ = \
"This script is used for performing a full-dataset monitoring of measurements\
inside AC for studying the secular pressure trends"


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
    start_times = [['2024-02-12 18:30:00.000']]
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
    '''
    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
    start_times = [['2024-02-26 15:51:00.000']]
    stop_times = [[None]]
    '''
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

    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples_40degrees.txt"]
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from2602.txt"]
    start_times = [['2024-02-26 15:51:00.000']]
    stop_times = [[None]]
    T_Julabo = 22 #°C


    #Obtaining interesting data
    data_list = LabViewdata_reading(paths_to_data, start_times, stop_times)
    timestamps = data_list[0]
    T5 = data_list[6]
    T6 = data_list[7]
    P4 = data_list[15]
    P3 = data_list[13]
    TJ = data_list[9]
    t_diffs = data_list[17] #s

    #Computing time in hours and effective temperature
    t_hours = t_diffs/3600 #hours
    T_eff = T5+0.16*(T6-T5)/1.16 #°C

    #Filtering the effective temperature in order to compensate time lag and heat capacity of AC
    #T_eff_filtered = temperature_butterworth_filtering(T_eff, log_time)
    T_eff_filtered = T_eff

    #Fitting P4/T_eff_filtered with an exponential with a power law dependance
    P_eq = (((P4*100)/(T_eff_filtered+273.15))*(T_Julabo+273.15))/100 #mbar

    #Masking data just to see if the second trend can be described by some of the known models
    #mask = t_hours>44
    #P_eq = P_eq[mask]
    #t_hours = t_hours[mask]
    #T_eff_filtered = T_eff_filtered[mask]
    
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

    plt.figure('Pressure variations with epoxy samples inside')
    plt.title('Pressure variations with epoxy samples inside')
    plt.errorbar(t_hours[t_hours<180], P_eq[t_hours<180], marker='.', label='First set of epoxy samples')

    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from0804.txt"]
    start_times = [['2024-04-08 11:35:35.000']]
    stop_times = [[None]]
    T_Julabo = 22 #°C


    #Obtaining interesting data
    data_list = LabViewdata_reading(paths_to_data, start_times, stop_times)
    timestamps = data_list[0]
    T5 = data_list[6]
    T6 = data_list[7]
    P4 = data_list[15]
    P3 = data_list[13]
    TJ = data_list[9]
    t_diffs = data_list[17] #s

    #Computing time in hours and effective temperature
    t_hours = t_diffs/3600 #hours
    T_eff = T5+0.16*(T6-T5)/1.16 #°C

    #Filtering the effective temperature in order to compensate time lag and heat capacity of AC
    #T_eff_filtered = temperature_butterworth_filtering(T_eff, log_time)
    T_eff_filtered = T_eff

    #Fitting P4/T_eff_filtered with an exponential with a power law dependance
    P_eq = (((P4*100)/(T_eff_filtered+273.15))*(T_Julabo+273.15))/100 #mbar


    plt.errorbar(t_hours, P_eq, marker='.', label='Second set of epoxy samples')
    plt.grid()
    plt.legend()

    def trial(t, P0, Delta, tau):
        return alpha_expo_scale(t, P0, Delta, 0.5, tau)
    
    popt, pcov = curve_fit(trial, t_hours, P_eq, p0=[1204., 120., 48.])
    print(f'Popt for a model with alpha=0.5')
    print(popt)

    fig, axs=plot_with_residuals(t_hours, P_eq, trial, popt)
    fig.suptitle('Dataset from 08/04/2024 - Second set of epoxy samples fitted with alpha expo scale with alpha=0.5')
    
    print(f'Asymptotic values for alpha expo scale with alpha=0.5 fixed:')
    print(f'asymptotic value = {trial(4*popt[-1],*popt)}')
    print(f'4 charasteric times are {(4*popt[-1])/24} days')


    plt.show()
    