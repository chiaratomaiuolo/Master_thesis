import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt

from pressure_analysis.filtering import temperature_butterworth_filtering
from pressure_analysis.labviewdatareading import LabViewdata_reading, plot_with_residuals
from pressure_analysis.models import expo, double_expo, triple_expo

__description__ = \
"This script is used for performing the data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS."

def double_exp_P0_frozen(x, Delta1, tau1, Delta2, tau2):
    return double_expo(x, 1198.8, Delta1, tau1, Delta2, tau2)

if __name__ == "__main__":
    #Datafiles are briefly descripted above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times

    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
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

    #Filtering the effective temperature in order to compensate time lag and heat capacity of AC
    T_eff_filtered = temperature_butterworth_filtering(T_eff, log_time)

    #Fitting P4/T_eff_filtered with a triple exponential
    P_corrected = (((P4*100)/(T_eff_filtered+273.15))*(22+273.15))/100 #mbar
    popt, pcov = curve_fit(double_exp_P0_frozen, t_hours, P_corrected, p0=[50., 10., 100., 100.])
    print(f' Fixed parameters: A0 = 1.00 [Pa/K], tau0 = 113.4 [hours], c0 = 0 [Pa/K]\n\
          Optimal parameters of remaining double exp:\n\
          Delta1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
          tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\n\
          Delta2 = {popt[2]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
          tau2 = {popt[3]} +/- {np.sqrt(pcov[1][1])} [hours],')

    fig, axs = plot_with_residuals(t_hours, P_corrected, double_exp_P0_frozen(t_hours, *popt))
    axs[0].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ [mbar]')
    axs[0].grid(True)
    axs[1].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ normalized residuals')
    axs[1].grid(True)


    plt.show()
