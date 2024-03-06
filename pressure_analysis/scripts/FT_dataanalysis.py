from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt

from pressure_analysis.labviewdatareading import LabViewdata_reading

__description__ = \
"This script is used for performing the data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS.\
In particular, the goal is the analysis of background absorption of DME on the\
aluminium walls of the chamber.\
Having no direct measurement of the temperature of the gas inside the chamber but\
only consequently outside, a filtering of the temperature data is done in order to\
compensate time lags of pressure wrt temperature due to different point of measurements."

def expo(x, A0, tau, c):
    return A0*(np.exp(-x/tau)) + c

def double_exp(x, A1, tau1, A2, tau2, c):
    return A1*(np.exp(-x/tau1)) + A2*(np.exp(-x/tau2)) + c


if __name__ == "__main__":
    #Datafiles are briefly descripted above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times

    #Datafile containing measurements with DME, no epoxy samples inside - FOR BACKGROUND EFFECTS STUDIES
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt"]
    start_times = [['2024-02-16 18:30:00.000']]
    stop_times = [['2024-02-19 11:00:00.000']]
    log_time = 5000e-3 #s (from logbook)

    '''
    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    start_times = [['2024-02-26 15:51:00.000']]
    stop_times = [[None]]
    log_time = 5000e-3 #s (from logbook)
    '''

    #Obtaining arrays of data from .txt file(s)
    data_list = LabViewdata_reading(paths_to_data, start_times, stop_times)

    #Obtaining interesting data
    data_list = LabViewdata_reading(paths_to_data, start_times, stop_times)
    timestamps = data_list[0]
    T5 = data_list[6] #T of thermoregulator liquid consequently outside chamber
    T6 = data_list[7] #Room temperature
    P4 = data_list[15] #Pressure inside chamber
    TJ = data_list[9] #Temperature set for Julabo termoregulator
    t_diffs = data_list[18] #s from starting point (set as t=0 s)

    #Computing interesting derivate quantities
    t_hours = t_diffs/3600 #hours
    #Defining an effective temperature that should represent
    #gas temperature (weighted mean of T5 and room temperature
    #where the weight of the room temperature has been computed
    #in a time interval where T5 could be considered as constant.
    T_eff = T5+0.16*(T6-T5)/1.16 #°C
    #T_eff = T_eff + 273.15 #Kelvin

    #Looking at overall data - P4, T5, T_room
    fig, axs = plt.subplots(4)
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
    axs[3].plot(timestamps, T_eff, color='green')
    axs[3].set(xlabel=r'Timestamp', ylabel=r'$T_{\text{eff}}$ [°C]')
    axs[3].grid(True)
    
    #Performing FT of the effective temperature and of pressure inside chamber
    #in order to understand how to construct a Butterworth filter. 
    # Number of sample points - note that FFT changes profundly if the number of sample pts is wrong!
    N = len(t_diffs)
    # sample spacing
    T = log_time #that is, 1/sample_freq
    #It is good practice to generate the linspace AFTER thee definition of N and T
    t = np.linspace(0.0, N*T, N, endpoint=False)
    #Defining Nyquist frequency
    f_nyquist = 1/(2*T)
    #Computing frequencies associated with sampling
    xf = fftfreq(N, T)[:N//2]
    #Performing DFTs OF MINMAX NORMALIZED QUANTITIES and generating the right frequency array
    Tft = fft((T_eff-np.mean(T_eff))/(np.max(T_eff)-np.min(T_eff))) #DFT of effective temperature
    T6ft = fft((T6-np.mean(T6))/(np.max(T6)-np.min(T6))) #DFT of room temperature
    Pft = fft((P4-np.mean(P4))/(np.max(P4)-np.min(P4))) #DFT of effective pressure

    #Constructing a lowpass Butterworth filter with defined cutoff frequency
    f_cutoff = 1/(1.5*3600) #Hz
    sos = butter(4, f_cutoff, 'low', fs=1/T, analog=False, output='sos')
    all_sos = [sos]
    sos2 = butter(2, [0.0002,0.0003], 'bandstop', fs=1/T, analog=False, output='sos')
    all_sos.append(sos2)
    sos = np.vstack(all_sos)
    T_butterworth = sosfilt(sos, T_eff-T_eff[0]) #It is needed to shift data in order to let them start from 0
    T_butter_ft = fft((T_butterworth-np.mean(T_butterworth))/(np.max(T_butterworth)-np.min(T_butterworth))) #DFT of T_butterworth

    #Plotting the absolute value of interesting spectra
    plt.figure('DFTs')
    #plt.plot(xf, 2.0/N * np.abs(T6ft[0:N//2]), label=r'$T_{6} - \mu_{T_{6}}$ DFT') #Room temperature FT
    plt.plot(xf, 2.0/N * np.abs(Tft[0:N//2]), label=r'$T_{eff} - \mu_{T_{eff}}$ DFT') #The factor 2.0/N is for normalization,(remember the Nyquist frequency expression)
    plt.plot(xf, 2.0/N * np.abs(Pft[0:N//2]), label=r'$P_{4} - \mu_{P_{4}}$ DFT')
    plt.plot(xf, 2.0/N * np.abs(T_butter_ft[0:N//2]), label=r'$T_{eff,filtered} - \mu_{T_{eff,filtered}}$ DFT')
    plt.xlim(0, 0.0003)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('DFT coefficient')
    plt.legend()
    plt.grid()

    #Some vlines for reference
    #plt.axvline(1.16e-05, color='r', linestyle='dashed', label='1/24h line')
    #plt.axvline(2.31e-05, color='pink', linestyle='dashed', label='1/12h line')
    #plt.axvline(1/3600, color='brown', label='1/1h line')

    #Plotting hysteresis curves: P as a function of the effective temperature
    #Fitting data with a single exponential in order to compare pts with expected values
    popt, pcov = curve_fit(expo, t_hours, P4, p0=[3.75, 10., 1140.])
    
    #Plotting pressure and Butterworth filtered effective temperature in time domain
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Time from new $T_{Julabo}$ settings [hours]')
    ax1.set_ylabel(r'$P_4 - P_{4,fitted}$ [mbar]', color='red')
    ax1.plot(t_hours, (P4-expo(t_hours,*popt))/(np.max(P4)-np.min(P4)),\
            color='red', label=r'$P_4 - P_{4,fitted}$ [mbar]')
    ax1.tick_params(axis='y', labelcolor='red')
    ax2 = ax1.twinx()  # instantiate a second pair of axes that shares the same x-axis
    ax2.set_ylabel(r'$T_{eff,filtered}$ [°C]', color='steelblue')
    ax2.plot(t_hours, (T_eff-T_eff[0])/((max(T_eff)-min(T_eff))), alpha=0.5, label=r'$T_{eff}$ [°C]')
    ax2.plot(t_hours, T_butterworth/((max(T_butterworth)-min(T_butterworth))),\
            label=fr'T_{{eff}} Butterworth filtered [°C], $f_{{cutoff}} = {f_cutoff:.2e}$ [Hz]')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    fig.legend()

    #Plotting hysteresis curves in ordert to understand if time lag has been compensated
    plt.figure('P-T curves')
    #Plotting pressures in Pa and temperatures in Kelvin
    plt.errorbar((P4-expo(t_hours,*popt))*100, (T_eff-T_eff[0])+273.15, label=r'$T_{eff}$')
    plt.errorbar((P4-expo(t_hours,*popt))*100, T_butterworth+273.15, label=r'$T_{eff,filtered}$')
    plt.xlabel(r'$P_4 - P_{4,fitted}$ [Pa]')
    plt.ylabel(r'$T_{eff,filtered}$ [K]')
    plt.legend()
    plt.grid()

    #When hysteresis curve is reasonably closed, it's time to fit P/T with an
    #exponential.
    T_butterworth_K = T_butterworth + 273.15 #Kelvin
    P4_Pa = P4*100
    p_over_tfiltered = P4_Pa/T_butterworth_K
    t=t_hours
    popt, pcov = curve_fit(expo, t, p_over_tfiltered, p0=[*popt])
    print(f'Optimal parameters: \n\
          A0 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [Pa/K],\n\
          tau = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\n\
          c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [Pa/K]')
    plt.figure('P4/T_eff as a function of time')
    plt.errorbar(t, P4_Pa/(T_eff-T_eff[0]+273.15), alpha=0.5, label=r'$\frac{P_4}{T_{eff}}$')
    plt.errorbar(t, p_over_tfiltered, label=r'$\frac{P_4}{T_{eff,filtered}}$')
    plt.plot(t, expo(t,*popt), label='Exponential fit')
    plt.ylabel(r'$\frac{P_4}{T_{eff,filtered}}$ [$\frac{Pa}{K}$]')
    plt.xlabel(r'time [hours]')
    plt.legend()
    plt.grid()
    
    plt.show()