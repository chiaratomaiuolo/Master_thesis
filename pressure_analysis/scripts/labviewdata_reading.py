import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

__description__ = \
"""This script contains the functions for reading and analyzing data written
   using LabView.
   The actual features of datasets (and their order) is shown below:

   #Timestamp	T0 - GPD IFplate (�C)	T1 - GPD drift (�C)	T2 - GPD mech (�C)\
    T3 - Tank TC (�C)	T4 - GTC TF Outlet (�C)	T5 - AC TF Outlet (�C)	T6 - Ambient (�C)\
    T7 - Tank PT100 (�C)	TJulaboInternal (�C)	P0 - Setra AC (mbar)\
    P1 - Setra Filling (mbar)	P2 - Omega AC (mbar)	P3 - Omega GTC (mbar)	PressFullrange (mbar)\
    P4 - MKS AC (mbar)	P5 - MKS GPD (mbar)
"""
def uncertainty_computation(val):
    """This function associates an uncertainty to every LabView data point, taking 
    as uncertainty half of the least significant digit of the measure. 
    (Hopefully, this function is necessary only until rms is inserted into the logged dataset)
    """
    if isinstance(val, float) and val.is_integer():
        return 0.5
    else:
        return (10 ** -np.floor(np.log10(abs(val))))*50

def LabViewdata_reading(paths_to_datafile: list[str], start_times: list[str], stop_times: list[str]):
    """This function takes as input a list of paths to .txt data files and two lists
        containing times of start and stop for delimiting the data objects that needs to be merged
        in a single file. 

        Arguments
        ---------
        - paths_to_datafile : list[str]
            list of strings containing the interested paths to the datafiles.
            Datafiles must be in cronological order because are scanned in 
            incremental order;
        - start_times : list[str]
            list of timestamps in format "%Y-%m-%d %H:%M:%S.%f" that delimit
            the interested starting point of every datafile. 
            It must be of same lenght of paths_to_datafile. If there is no
            interest in a starting point (starting from first row), this 
            quantity is None;
        - stop_times : list[str]
            list of timestamps in format "%Y-%m-%d %H:%M:%S.%f" that delimit
            the interested stopping point of every datafile. 
            It must be of same lenght of paths_to_datafile. If there is no
            interest in a stopping point (going until the end of dataset), 
            this quantity is None;
    """
    #Defining empty lists for every quantity in datafile
    timestamp = np.array([])
    T0 = np.array([])
    T1 = np.array([])
    T2 = np.array([])
    T3 = np.array([])
    T4 = np.array([])
    T5 = np.array([])
    T6 = np.array([])
    T7 = np.array([])
    TJ = np.array([])
    P0 = np.array([])
    P1 = np.array([])
    P2 = np.array([])
    P3 = np.array([])
    PressFullrange = np.array([])
    P4 = np.array([])
    P5 = np.array([])
    for idx, datafile in enumerate(paths_to_datafile):
        #Opening data file and unpacking columns
        timestamp_day_tmp, timestamp_hour_tmp, T0_tmp, T1_tmp, T2_tmp, T3_tmp, T4_tmp,\
        T5_tmp, T6_tmp, T7_tmp, TJ_tmp, P0_tmp, P1_tmp, P2_tmp, P3_tmp,\
        PressFullrange_tmp, P4_tmp, P5_tmp = np.loadtxt(datafile, dtype=str, unpack=True, skiprows=1)
        #Converting timestamps to datatime in order to compare data
        timestamp_tmp = np.array([str(date) + ' ' + str(hour) for date, hour in zip(timestamp_day_tmp, timestamp_hour_tmp)])
        timestamp_tmp = np.array([datetime.strptime(timestp, "%Y-%m-%d %H:%M:%S.%f") for timestp in timestamp_tmp])
        if start_times[idx] is not None:
            #start_times[idx] should be a list, in order, if necessary, to select multiple 
            #intervals in a same datafile
            #turning timestamp into datetime in order to compare it with timestamps
            mask = [False]*len(timestamp_tmp)
            for t_idx, t_start in enumerate(start_times[idx]):
                start_time = datetime.strptime(t_start, "%Y-%m-%d %H:%M:%S.%f")
                if stop_times[idx][t_idx] is not None:
                    #For every element in the list, the mask is constructed
                    #turning timestamp into datetime in order to compare it with timestamps
                    stop_time = datetime.strptime(stop_times[idx][t_idx], "%Y-%m-%d %H:%M:%S.%f")
                    #need to cut on start and stop
                    mask = np.logical_or(mask, ((timestamp_tmp >= start_time) & (timestamp_tmp <= stop_time)))
                else:
                    #need to cut only on start
                    mask = np.logical_or(mask, (timestamp_tmp >= start_time))
        else:
            mask = [False]*len(timestamp_tmp)
            if stop_times[idx] is not None:
                #turning timestamp into datetime in order to compare it with timestamps
                stop_time = datetime.strptime(stop_times[idx], "%Y-%m-%d %H:%M:%S.%f")
                #Need to cut only on stop
                mask = np.logical_or(mask, (timestamp_tmp <= stop_time))
            else:
                #no cuts, taking all data of the dataset. Creating mask with all true
                mask = [True]*len(timestamp_tmp)
        #Filling lists containing interesting data 
        timestamp = np.append(timestamp, timestamp_tmp[mask])
        T0 = np.append(T0, T0_tmp[mask].astype(float))
        T1 = np.append(T1, T1_tmp[mask].astype(float))
        T2 = np.append(T2, T2_tmp[mask].astype(float))
        T3 = np.append(T3, T3_tmp[mask].astype(float))
        T4 = np.append(T4, T4_tmp[mask].astype(float))
        T5 = np.append(T5, T5_tmp[mask].astype(float))
        T6 = np.append(T6, T6_tmp[mask].astype(float))
        T7 = np.append(T7, T7_tmp[mask].astype(float))
        TJ = np.append(TJ, TJ_tmp[mask].astype(float))
        P0 = np.append(P0, P0_tmp[mask].astype(float))
        P1 = np.append(P1, P1_tmp[mask].astype(float))
        P2 = np.append(P2, P2_tmp[mask].astype(float))
        P3 = np.append(P3, P3_tmp[mask].astype(float))
        PressFullrange = np.append(PressFullrange, PressFullrange_tmp[mask].astype(float))
        P4 = np.append(P4, P4_tmp[mask].astype(float))
        P5 = np.append(P5, P5_tmp[mask].astype(float))

    dP4 = np.array([uncertainty_computation(p) for p in P4])
    timestamps_diff = np.array([(tmstp.timestamp() - timestamp[0].timestamp()) for tmstp in timestamp]).astype(float) #difference from the starting pt in s


    
    return [timestamp, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange, P4, P5, dP4, timestamps_diff]

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
                    '2024-02-18 16:00:00.000', '2024-02-19 16:00:00.000']]
    stop_times = [['2024-02-12 20:00:00.000', '2024-02-13 15:00:00.000', '2024-02-14 15:00:00.000',\
                   '2024-02-15 20:00:00.000','2024-02-16 16:00:00.001', '2024-02-17 20:00:00.000', \
                   '2024-02-18 20:00:00.000', '2024-02-19 20:00:00.000']]
    
    '''
    start_times = [['2024-02-12 18:00:00.001']]
    stop_times = [['2024-02-20 12:30:00.000']]
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
    acq_time = 100e-3 #s
    #t = np.array([acq_time*i for i in range(len(timestamps))])
    popt, pcov = curve_fit(expo, t_diffs/3600, P4, p0=[5.75, 20., 1194.], sigma = dP4)
    print(f'Optimal parameters: P0 = {popt[0]} +/- {np.sqrt(pcov[0][0])} [mbar],\
          tau = {popt[1]} +/- {np.sqrt(pcov[1][1])} [hours],\
          c = {popt[2]} +/- {np.sqrt(pcov[2][2])} [mbar]')
    t_year = 0.5*365*24*60*60 #1year time in seconds
    print(f'Estimation of the asymptotic value: {expo(t_year,*popt)}')
    plt.figure()
    plt.title(r'$P_4$ as a function of time from DME filling')
    z = np.linspace(0, 200, 100000)
    plt.plot(z, expo(z, *popt), color='steelblue')
    plt.errorbar(t_diffs/3600, P4, yerr= dP4, marker='.', linestyle='', color='firebrick')
    plt.xlabel('Time [hours]')
    plt.ylabel(r'$P_4$ [mbar]')
    plt.grid()

    '''
    plt.figure()
    plt.plot(timestamps, P3)
    plt.xlabel(r'Timestamp')
    plt.ylabel(r'$P_3$ [mbar]')
    plt.grid()
    '''

    plt.show()
