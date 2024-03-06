import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.fft import fft, fftfreq, fftshift
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt

from pressure_analysis.labviewdatareading import LabViewdata_reading

__description__ = \
"This script is used for performing the data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS."

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

def expo(x, DeltaP, tau):
    return 1201. - DeltaP*(1-np.exp(-x/tau))

def double_exp(x, DeltaP1, tau1, DeltaP2, tau2):
    return 1201. - DeltaP1*(1-np.exp(-x/tau1)) - DeltaP2*(1-np.exp(-x/tau2))
'''
def fixed_exp(x, DeltaP2, tau2):
    return 1201. - DeltaP(1-np.exp(-x/tau)) - DeltaP2(1-np.exp(-x/tau2))
'''

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

    #Computing time in hours and effective temperature
    t_hours = t_diffs/3600 #hours
    T_eff = T5+0.16*(T6-T5)/1.16 #°C

    #Filtering the effective temperature in order to compensate time lag and heat capacity of AC
    T_eff_filtered = temperature_filtering(T_eff, log_time)

    #Computing the equivalent pressure
    P_eq = (((P4*100)/(T_eff_filtered+273.15))+(T_Julabo)+273.15)/100 #mbar

    #We want to exploit the optimal parameter variation as we increase the number
    #of data. 
    #Before doing that, let's fit the first hours of data with a single exp.
    #We want to fix the parameters of the 'fast' exponential in order to 
    #analyze only parameters of the long term exponential.
    i=3
    mask = t_hours < i
    popt_init, pcov_init = curve_fit(double_exp, t_hours[mask], P_eq[mask], p0=[50., 10., 100., 100.])
    print(f'Optimal parameters for single exp fit of the first {i} hours of data:\n\
          A = {popt_init[0]} +/- {np.sqrt(pcov_init[0][0])}\n,\
          tau = {popt_init[1]} + {np.sqrt(pcov_init[1][1])}\n\
          c = {popt_init[2]} + {np.sqrt(pcov_init[2][2])}')
    #chisq = ((P4 - expo(t_hours, *popt_init))**2).sum()
    #ndof = len(P4) - len(popt_init)
    #print(f'MSE/ndof = {chisq/ndof}')
    
    #Plotting pts and fit
    plt.figure()
    plt.title(rf'$P_4$ inside AC with epoxy samples, DME filled - all parameters free')
    plt.errorbar(t_hours, P4, marker='.', alpha=0.5, markeredgewidth=0, linestyle='', color='darkorange', label='Data')
    plt.plot(t_hours, double_exp(t_hours, *popt_init), linestyle='dotted', label=f'Single exponential with $t < {i} [hours]$')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    while mask[-1] == False:
        mask = t_hours < i
        popt, pcov = curve_fit(double_exp, t_hours[mask], P4[mask], p0=[50., 1., 100., 100.])
        print('Fit with double exponential - 5 free parameters')
        print(f'Optimal parameters for t_hours<{i}:\n\
            A1 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
            tau1 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\n\
            A2 = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar],\n\
            tau2 = {popt[3]} +/- {np.sqrt(pcov[3][3])} [hours],')

        #chisq = ((P4 - double_exp(t_hours, *popt))**2).sum()
        #ndof = len(P4) - len(popt)
        #print(f'MSE/ndof = {chisq/ndof}')
        plt.plot(t_hours, double_exp(t_hours, *popt), label=f'Double exponential with $t < {i} [hours]$')
        i+=10
    plt.grid()
    plt.legend()

    '''
    #Fixing parameters of the first exp considering the fit of the first 5 hours. 
    i=3
    mask = t_hours < i
    popt, pcov = curve_fit(fixed_exp, t_hours[mask], P4[mask], p0=[58., 25., 1111.])
    #Printing optimal parameters
    print(f'Fit with double exponential \n\
          First exp parameters fixed to: \
          A1 = 4.029 [mbar],\n\
          tau1 = 0.28 [hours],\n\
          c = 0 [mbar]')
    print(f'Optimal parameters of second exp  for t_hours<{i}:\n\
        A2 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
        tau2 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\n\
        c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    #Plotting pts and fit
    plt.figure()
    plt.title(rf'$P_4$ inside AC with epoxy samples, DME filled, first exp parameter fixed')
    z = np.linspace(0, max(t_hours), 2000)
    plt.errorbar(t_hours, P4, marker='.', alpha=0.5, markeredgewidth=0, linestyle='', color='darkorange', label='Data')
    plt.plot(z, fixed_exp(z, *popt), label=f'Double exponential with $t < {i} [hours]$')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    while mask[-1] == False:
        i+=10
        mask = t_hours < 5+i
        popt, pcov = curve_fit(fixed_exp, t_hours[mask], P4[mask], p0=[134., 94., 954.])
        print(f'Optimal parameters of second exp  for t_hours<{i}:\n\
            A2 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\n\
            tau2 = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\n\
            c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar]')
        chisq = (((P4 - fixed_exp(t_hours, *popt))/(dP4))**2).sum()
        ndof = len(P4) - len(popt)
        plt.plot(z, fixed_exp(z, *popt), label=f'Double exponential with $t < {i} [hours]$')

    plt.grid()
    plt.legend()
    '''
    plt.show()