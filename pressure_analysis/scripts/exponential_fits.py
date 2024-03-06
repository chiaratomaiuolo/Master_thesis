import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

from pressure_analysis.filtering import temperature_butterworth_filtering
from pressure_analysis.labviewdatareading import LabViewdata_reading, plot_with_residuals
from pressure_analysis.models import expo, expo_P0_frozen, double_expo, triple_expo, double_exp_P0_frozen

__description__ = \
"This script is used for performing the data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS."

def iterative_double_exponential_fit(x: np.array, y: np.array, mask: np.array, popt: np.array, hours_start:float=3, hours_lag: float=10):
    """Performs an iterative fit on an increasing-lenght dataset until its end.
    At the end of the fit prints the parameters and plots the current fitting 
    curve on the latest figure of the script.

    Arguments
    ---------
    - x : np.array
        Array containing x data
    - y_data : np.array
        Array containing x data
    - mask: np.array
        Array mask used for cutting on data
    - popt : np.array
        Array containing the starting initial parameters of the double exp.
    - hours_lag : float
        Float that states the part of the dataset (in terms of hours) that
        is added to the fitted data at every iteration.
    """
    i = hours_start #states the start of the dataset in terms of hours from first data object
    while mask[-1] == False:
        mask = x < i
        popt, pcov = curve_fit(double_exp_P0_frozen, x[mask], y[mask], p0=[*popt])
        print('Fit with double exponential - 4 free parameters')
        print(f'Optimal parameters of double exp:\n\
                Delta1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
                tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\n\
                Delta2 = {popt[2]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
                tau2 = {popt[3]} +/- {np.sqrt(pcov[1][1])} [hours],')
        plt.plot(x, double_exp_P0_frozen(x, *popt), label=f'Double exponential with $t < {i} [hours]$')
        i += hours_lag
    return

if __name__ == "__main__":
    #Datafiles are briefly descripted above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times

    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']

    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
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
    start_times = [['2024-02-26 15:51:00.000']]
    stop_times = [[None]]
    log_time = 5000e-3 #s (from logbook)
    T_Julabo = 22 #°C

    #Obtaining data
    timestamps, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange,\
    P4, P5, t_diffs = LabViewdata_reading(paths_to_data, start_times, stop_times)

    #Computing time in hours and effective temperature
    t_hours = t_diffs/3600 #hours
    T_eff = T5+0.16*(T6-T5)/1.16 #°C

    #Filtering the effective temperature in order to compensate time lag and heat capacity of AC
    T_eff_filtered = temperature_butterworth_filtering(T_eff, log_time)

    #Computing the equivalent pressure
    P_eq = (((P4*100)/(T_eff_filtered+273.15))*(T_Julabo+273.15))/100 #mbar

    #We want to exploit the optimal parameter variation as we increase the number
    #of data. 
    #Before doing that, let's fit the first hours of data with a single
    #and then double exp.
    #With fixed P0.
    #We want to fix the parameters of the 'fast' exponential in order to 
    #analyze only parameters of the long term exponential.
    i=2
    mask = t_hours < i
    popt_init, pcov_init = curve_fit(expo_P0_frozen, t_hours[mask], P_eq[mask], p0=[50., 10.])
    print(f'Optimal parameters of single exp for the first 3 hours of data:\n\
          Delta = {popt_init[0]} +/- {np.sqrt(pcov_init[0][0])} [mbar],\n\
          tau = {popt_init[1]} +/- {np.sqrt(pcov_init[1][1])} [hours],')
    #Plotting pts, fit and residuals for single exponential
    fig, axs = plot_with_residuals(t_hours[mask], P_eq[mask], expo_P0_frozen(t_hours[mask], *popt_init))
    axs[0].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ [mbar]')
    axs[0].grid(True)
    axs[1].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ normalized residuals')
    axs[1].grid(True)
    popt_init, pcov_init = curve_fit(double_exp_P0_frozen, t_hours[mask], P_eq[mask], p0=[50., 10., 100., 100.])
    print(f'Optimal parameters of double exp for the first 3 hours of data:\n\
          Delta1 = {popt_init[0]} +/- {np.sqrt(pcov_init[0][0])} [mbar],\n\
          tau1 = {popt_init[1]} +/- {np.sqrt(pcov_init[1][1])} [hours],\n\
          Delta2 = {popt_init[2]} +/- {np.sqrt(pcov_init[0][0])} [mbar],\n\
          tau2 = {popt_init[3]} +/- {np.sqrt(pcov_init[1][1])} [hours],')
    #Plotting pts, fit and residuals
    fig, axs = plot_with_residuals(t_hours[mask], P_eq[mask], double_exp_P0_frozen(t_hours[mask], *popt_init))
    axs[0].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ [mbar]')
    axs[0].grid(True)
    axs[1].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ normalized residuals')
    axs[1].grid(True)

    #Plotting pts and fit
    plt.figure()
    iterative_double_exponential_fit(t_hours, P_eq, mask, popt_init, hours_start=i, hours_lag=15)
    plt.title(rf'$P_4$ inside AC with epoxy samples, DME filled - all parameters free')
    plt.errorbar(t_hours, P4, marker='.', alpha=0.5, markeredgewidth=0, linestyle='', color='darkorange', label='Data')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    plt.grid()
    plt.legend()

    plt.show()