import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt

from pressure_analysis.labviewdatareading import LabViewdata_reading

__description__ = \
"This script is used for performing a full-dataset monitoring of measurements\
inside AC for studying the secular pressure trends"

if __name__ == "__main__":
    #Datafiles are briefly described above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times
    
    #Datafile from 19/4/2024 to - , filling GPD in BFS with DME.
    #paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from1704.txt"]
    #start_times = [['2024-04-19 12:26:30.000']]
    #stop_times = [[None]]
    log_time = 5000e-3 #s

    #Datafiles from 12/02/2024 to 22/02/2024 - AC DME filled without epoxy samples
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']

    start_times = [['2024-02-12 15:31:00.000']]
    stop_times = [['2024-02-19 11:30:00.000']]

    #Obtaining interesting data
    data_list = LabViewdata_reading(paths_to_data, start_times, stop_times)
    timestamps = data_list[0]
    T2 = data_list[3] #GPD temperature
    T6 = data_list[7] #room temperature
    P5 = data_list[16] #GPD pressure
    t_diffs = data_list[17] #s

    #Mask selecting only 12-21 time band everyday
    #mask = [np.logical_and(t.hour >= 12, t.hour <= 21) for t in timestamps]
    #No selection mask
    mask = [1>0]

    #Computing time in hours and effective temperature
    t_hours = t_diffs/3600 #hours
    mask2 = t_hours > 0
    mask_tot = np.logical_and(mask, mask2)

    #Computing the equivalent pressure
    P_eq = (((P5*100)/(T2+273.15))*(np.mean(T2)+273.15))/100 #mbar

    #Looking at overall data - P5, T2, T_room
    fig, axs = plt.subplots(3)
    fig.suptitle(fr'Absolute pressure inside GPD and corresponding temperature - Gas DME,$T$ = {np.mean(T2):.2f}°C')
    axs[0].plot(timestamps, P5, color='firebrick')
    axs[0].set(xlabel=r'Timestamp', ylabel=r'$P_{GPD}$ [mbar]')
    axs[0].grid(True)
    axs[1].plot(timestamps, T2)
    axs[1].set(xlabel=r'Timestamp', ylabel=r'$T_{GPD}$ [°C]')
    axs[1].grid(True)
    axs[2].plot(timestamps, T6, color='red')
    axs[2].set(xlabel=r'Timestamp', ylabel=r'$T_{\text{ambient}}$ [°C]')
    axs[2].grid(True)

    #Showing the trends
    plt.show()
    