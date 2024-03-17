import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

from pressure_analysis.filtering import temperature_butterworth_filtering
from pressure_analysis.labviewdatareading import LabViewdata_reading, plot_with_residuals
from pressure_analysis.models import expo, expo_P0_frozen, double_expo, triple_expo, double_exp_P0_frozen, alpha_expo_scale

__description__ = \
"This script is used for performing the data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS."

if __name__ == "__main__":
    #Datafiles are briefly descripted above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times

    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled.
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']
    #Datafile from 12/2/2024 to 20/2/2024 - AC DME filled, full dataset selection
    start_times = [['2024-02-12 18:30:00.000', '2024-02-16 20:00:00.000']]
    stop_times = [['2024-02-16 14:00:00.000', '2024-02-19 11:00:00.000']]
    log_time = 5000e-3 #s (from logbook)
    T_Julabo = 22 #°C
    '''

    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
    start_times = [['2024-02-26 15:51:00.000']]
    stop_times = [[None]]
    log_time = 5000e-3 #s (from logbook)
    T_Julabo = 22 #°C
    '''

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
    #T_eff_filtered = temperature_butterworth_filtering(T_eff, log_time)
    T_eff_filtered = T_eff

    #Fitting P4/T_eff_filtered with an exponential with a power law dependance
    P_eq = (((P4*100)/(T_eff_filtered+273.15))*(T_Julabo+273.15))/100 #mbar
    #P_eq = P4

    def model(t, Delta, alpha, tau):
      return alpha_expo_scale(t, 1198., Delta, alpha, tau)
    
    def expo_(x, Delta, tau):
       return expo(x, 1198., Delta, tau)

    popt, pcov = curve_fit(expo_, t_hours, P_eq, p0=[[6., 100.]])
    print(popt)
    print(np.sqrt(np.diag(pcov)))
    print(pcov)
    '''
    print(f'Optimal parameters of {alpha_expo_scale.__name__}:\n\
          P0 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
          Delta = {popt[1]} +/- {np.sqrt(pcov[1][1])} [mbar],\n\
          alpha = {popt[2]} +/- {np.sqrt(pcov[2][2])},\n\
          tau = {popt[3]} +/- {np.sqrt(pcov[3][3])} [hours],\n')
    '''

    fig, axs = plot_with_residuals(t_hours, P_eq,  expo_(t_hours, *popt))
    #fig.suptitle(r'$P_{eq}$ as a function of time fitted with $P_{eq}(t) = P_0 -\Delta\cdot (e^{-(\frac{t}{\tau})^{\alpha}})$')
    axs[0].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ [mbar]')
    axs[0].grid(True)
    axs[1].set(xlabel=r'Time [hours]', ylabel=r'$\frac{P_{eq}(t) - P_{eq,data}}{P_{eq,data}}$')
    axs[1].grid(True)


    plt.show()
