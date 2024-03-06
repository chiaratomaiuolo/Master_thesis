import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

from pressure_analysis.filtering import temperature_butterworth_filtering
from pressure_analysis.labviewdatareading import LabViewdata_reading, plot_with_residuals
from pressure_analysis.models import expo, expo_P0_frozen, double_expo, double_exp_P0_frozen, triple_expo, triple_expo_P0_frozen

__description__ = \
"This script is used for performing the data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS."

def iterative_exponential_fit(x: np.array, y: np.array, model, popt: np.array, hours_start:float=3, hours_lag: float=10):
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
    popts = []
    hours = [hours_start]
    mask = x < i
    popt_, pcov_ = curve_fit(model, t_hours[mask], P_eq[mask], p0=popt)

    print(f'Optimal parameters of {model.__name__} for the first 3 hours of data:')
    popts.append(popt_)
    print(popt_)
    #Plotting pts, fit and residuals for fit of the first 3 hours
    fig, axs = plot_with_residuals(t_hours[mask], P_eq[mask], model(t_hours[mask], *popt_))
    axs[0].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ [mbar]')
    axs[0].grid(True)
    axs[1].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ normalized residuals')
    axs[1].grid(True)
    
    plt.figure()
    plt.errorbar(x, y, marker='.', linestyle='', label=f'Data')
    plt.title(f'Iterative fit with {model.__name__}')
    while mask[-1] == False:
        mask = x < i
        hours.append(i)
        popt_, pcov_ = curve_fit(model, x[mask], y[mask], p0= popts[-1])
        popts.append(popt_)
        print(f'Optimal parameters of {model.__name__}:\n')
        print(popt_)
        plt.plot(x, model(x, *popt_), label=f'{model.__name__} with $t < {i} [hours]$')
        i += hours_lag
    plt.grid()
    plt.legend()
    return hours, popts

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
    hours, popts = iterative_exponential_fit(t_hours, P_eq, triple_expo_P0_frozen,\
                      popt=[4.69301877, 0.27440838, 11.9865074, 2.43070955, 123.03394475,\
                      55.79114849], hours_start=3, hours_lag=10)
    
    #After having computed parameters, let's frozen the first exp
    def frozen_triple_exp(x, Delta2, tau2, Delta3, tau3):
        return triple_expo_P0_frozen(x, popts[0][0], popts[0][1], Delta2, tau2, Delta3, tau3)
    
    hours, popts = iterative_exponential_fit(t_hours, P_eq, frozen_triple_exp,\
                    popt=[33.86507696, 18.45662877, 186.66209992, 216.86914815],\
                    hours_start=3, hours_lag=10)

    plt.show()