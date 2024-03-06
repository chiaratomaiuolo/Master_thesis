from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

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

    timestamps_diff = np.array([(tmstp.timestamp() - timestamp[0].timestamp()) for tmstp in timestamp]).astype(float) #difference from the starting pt in s
    
    return [timestamp, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange, P4, P5, timestamps_diff]


def plot_with_residuals(x: np.array, y_data: np.array, y_fitted: np.array):
    """Function for creating a figure with two subplots. 
    The first figure contains data errorbars and data fit, the second figure
    plots the residuals normalized with data values. 

    Arguments
    ---------
    - x : np.array
        Array containing x data
    - y_data : np.array
        Array containing x data
    - y_fitted : np.array
        Array containing fit on y_data
    
    Return
    ------
    - fig, axs of the figure (in order to customize the figure externally)
    """
    fig, axs = plt.subplots(2)
    axs[0].errorbar(x, y_data, marker='.', linestyle='', label='Data')
    axs[0].plot(x, y_fitted, label='Fit to data')
    axs[0].legend()
    res_normalized = (y_data - y_fitted)/y_data
    axs[1].plot(x, res_normalized)
    return fig, axs
