import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt

from pressure_analysis.labviewdatareading import LabViewdata_reading

__description__ = \
"This script is used for performing the data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS."

def expo(x, A0, tau, c):
    return A0*(np.exp(-x/tau)) + c

def double_exp(x, DeltaP1, tau1, DeltaP2, tau2):
    return 1201. - DeltaP1*(1-np.exp(-x/tau1)) - DeltaP2*(1-np.exp(-x/tau2))

def triple_exp(x, A1, tau1, A2, tau2, c):
    return expo(x, 1.00, 113.4, 0.) + double_exp(x, A1, tau1, A2, tau2, c)

def temperature_filtering(T: np.array, sampling_time: float):
    """The following function applies a double Butterworth filter to data.
    It is used for compensating for the time lag of the measurement 
    and the heat capacity of the AC.
    Filter parameters are fixed.

    Arguments
    ---------
    - T : np.array
        Array containing temperature measurements.
    - sampling_time : float
        sampling time in seconds. 
    
    Return
    ------
    - T_filtered : np.array
        Array containing the filtered T.

    """
    #Constructing a lowpass Butterworth filter with defined cutoff frequency
    f_cutoff = 1/(1.5*3600) #Hz
    sos = butter(4, f_cutoff, 'low', fs=1/sampling_time, analog=False, output='sos') #creating a lowpass filter
    all_sos = [sos]
    sos2 = butter(2, [0.0002,0.0003], 'bandstop', fs=1/sampling_time, analog=False, output='sos') #creating a bandstop filter
    all_sos.append(sos2)
    sos = np.vstack(all_sos)
    T_butterworth = sosfilt(sos, T-T[0]) #It is needed to shift data in order to let them start from 0
    T_butterworth += T[0] #Re-shifting the overall array
    return T_butterworth

if __name__ == "__main__":
    #Datafiles are briefly descripted above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times

    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
    start_times = [['2024-02-26 15:51:00.000']]
    stop_times = [[None]]
    log_time = 5000e-3 #s (from logbook)
    T_Julabo = 22 #°C

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
    T_eff_filtered = temperature_filtering(T_eff, log_time)

    #Fitting P4/T_eff_filtered with a triple exponential
    P_corrected = (((P4*100)/(T_eff_filtered+273.15))*(22+273.15))/100 #mbar
    popt, pcov = curve_fit(double_exp, t_hours, P_corrected, p0=[50., 10., 100., 100.])
    print(f' Fixed parameters: A0 = 1.00 [Pa/K], tau0 = 113.4 [hours], c0 = 0 [Pa/K]\n\
          Optimal parameters of remaining double exp:\n\
          A1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [Pa/K],\n\
          tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\n\
          A2 = {popt[2]} +/- {np.sqrt(pcov[0][0])} [Pa/K],\n\
          tau2 = {popt[3]} +/- {np.sqrt(pcov[1][1])} [hours],')
    #chisq = (((P4 - triple_exp(t_hours, *popt)))**2).sum()
    #ndof = len(P4) - len(popt)
    #print(f'chisq/ndof = {chisq}/{ndof}')

    fig, axs = plt.subplots(2)
    axs[0].plot(t_hours, double_exp(t_hours, *popt), label='Double exponential')
    axs[0].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ [mbar]')
    axs[0].grid(True)
    res = (P_corrected - double_exp(t_hours, *popt))/P_corrected
    axs[1].plot(t_hours, res, label='Residuals')
    axs[1].set(xlabel=r'Time [hours]', ylabel=r'Residuals')
    axs[1].grid(True)


    plt.show()
