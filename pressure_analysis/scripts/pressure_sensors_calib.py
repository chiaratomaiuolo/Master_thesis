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
#For reference, our .txt data contain, by now, the following features:
#Timestamp	T0 - GPD IFplate (�C)	T1 - GPD drift (�C)	T2 - GPD mech (�C)
#T3 - Tank TC (�C)	T4 - GTC TF Outlet (�C)	T5 - AC TF Outlet (�C)	T6 - Ambient (�C)
#T7 - Tank PT100 (�C)	TJulaboInternal (�C)	P0 - Setra AC (mbar)	
#P1 - Setra Filling (mbar)	P2 - Omega AC (mbar)	P3 - Omega GTC (mbar)	PressFullrange (mbar)
#P4 - MKS AC (mbar)	P5 - MKS GPD (mbar)

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
            #turning timestamp into datetime in order to compare it with timestamps
            start_time = datetime.strptime(start_times[idx], "%Y-%m-%d %H:%M:%S.%f")
            if stop_times[idx] is not None:
                #turning timestamp into datetime in order to compare it with timestamps
                stop_time = datetime.strptime(stop_times[idx], "%Y-%m-%d %H:%M:%S.%f")
                #need to cut on start and stop
                mask = ((timestamp_tmp >= start_time) & (timestamp_tmp <= stop_time))
            else:
                #need to cut only on start
                mask = (timestamp_tmp >= start_time)
        else:
            if stop_times[idx] is not None:
                #turning timestamp into datetime in order to compare it with timestamps
                stop_time = datetime.strptime(stop_times[idx], "%Y-%m-%d %H:%M:%S.%f")
                #Need to cut only on stop
                mask = (timestamp_tmp <= stop_time)
            else:
                #no cuts, taking all data of the dataset. Creating mask with all true
                mask = 1>0
        #Filling lists containing interesting data 
        timestamp = np.append(timestamp, timestamp_tmp[mask])
        T0 = np.append(T0, T0_tmp[mask])
        T1 = np.append(T1, T1_tmp[mask])
        T2 = np.append(T2, T2_tmp[mask])
        T3 = np.append(T3, T3_tmp[mask])
        T4 = np.append(T4, T4_tmp[mask])
        T5 = np.append(T5, T5_tmp[mask])
        T6 = np.append(T6, T6_tmp[mask])
        T7 = np.append(T7, T7_tmp[mask])
        TJ = np.append(TJ, TJ_tmp[mask])
        P0 = np.append(P0, P0_tmp[mask])
        P1 = np.append(P1, P1_tmp[mask])
        P2 = np.append(P2, P2_tmp[mask])
        P3 = np.append(P3, P3_tmp[mask])
        PressFullrange = np.append(PressFullrange, PressFullrange_tmp[mask])
        P4 = np.append(P4, P4_tmp[mask])
        P5 = np.append(P5, P5_tmp[mask])

    
    return [timestamp, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange, P4, P5]

if __name__ == "__main__":
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-07_1222.txt',\
                    '/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/bfslog_2024-02-07_1246.txt']
    start_times = ['2024-02-07 12:44:33.707', '2024-02-07 16:31:25.989']
    stop_times = [None, None]
    data_list = LabViewdata_reading(paths_to_data, start_times, stop_times)
    timestamps = data_list[0]
    T5 = data_list[6]
    P4 = data_list[15]

    #Plotting pressure variations and temperature variations
    fig, axs = plt.subplots(2)
    fig.suptitle('Pressure variations inside AC and corresponding temperature variations')
    axs[0].plot(timestamps[1:], np.diff(P4.astype(float)), color='red')
    axs[0].set(xlabel=r'Timestamp', ylabel=r'$\Delta P_4$ [mbar]')
    axs[1].plot(timestamps[1:], np.diff(T5.astype(float)))
    axs[1].set(xlabel=r'Timestamp', ylabel=r'$\Delta T_5$ [°C]')

    fig, axs = plt.subplots(2)
    fig.suptitle('Pressure variations inside AC and corresponding temperature variations')
    axs[0].plot(timestamps, P4.astype(float), color='red')
    axs[0].set(xlabel=r'Timestamp', ylabel=r'$P_4$ [mbar]')
    axs[1].plot(timestamps, T5.astype(float))
    axs[1].set(xlabel=r'Timestamp', ylabel=r'$T_5$ [°C]')

    plt.show()
