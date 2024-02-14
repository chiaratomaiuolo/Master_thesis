import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

#Timestamp	T0 - GPD IFplate (�C)	T1 - GPD drift (�C)	T2 - GPD mech (�C)
#T3 - Tank TC (�C)	T4 - GTC TF Outlet (�C)	T5 - AC TF Outlet (�C)	T6 - Ambient (�C)
#T7 - Tank PT100 (�C)	TJulaboInternal (�C)	P0 - Setra AC (mbar)	
#P1 - Setra Filling (mbar)	P2 - Omega AC (mbar)	P3 - Omega GTC (mbar)	PressFullrange (mbar)
#P4 - MKS AC (mbar)	P5 - MKS GPD (mbar)

__description__ = \
""" 
    
"""

# Parser object.
PRESSURE_MONITORING_ARGPARSER = argparse.ArgumentParser(description=__description__)
PRESSURE_MONITORING_ARGPARSER.add_argument('path_to_datafile', type=str, help='Absolute\
                                           path to data file')
PRESSURE_MONITORING_ARGPARSER.add_argument('--start_time', type=str, default=None, help='String\
                                           containing the starting time from which pressure and\
                                           temperature are plot')
PRESSURE_MONITORING_ARGPARSER.add_argument('--stop_time', type=str, default=None, help='String\
                                           containing the stopping time to which pressure and\
                                           temperature are plot')
PRESSURE_MONITORING_ARGPARSER.add_argument('--logging_time', type=float, default=5000, help='logging\
                                           time in ms')
PRESSURE_MONITORING_ARGPARSER.add_argument('--acquisition_time', type=float, default=10, help='acquisition\
                                           time in ms')

def LabViewDataPlots(path_to_datafile: str, logging_time: float, acquisition_time: float, start_time: str, stop_time: str):
    """Performs a plot of the data from BFS taken with LabView software. 
    Goal of the analysis is, by now, the study of the stability of pressure
    inside the Absorption Chamber (AC) for inspecting absorption phenomenon. 
    Because of that, the plots shown are the ones useful for this scope.
    FUNCTION TO BE GENERALIZED IF NEEDED. 

    Arguments
    ----------
    - path_to_file : str
        string containing the absolute or relative path of the data file 
        (and the data file name);
    - logging_time : float
        logging time (LoopTimeLogging) set on LabView, that is, the time between
        two data in the .txt file
    - acquisition_time : float
        acquisition time (LoopAcquisitionTime) set on LabView, that is, the time 
        between two acquisitions from the electronics (ADC).
    - start_time : str
        String having format 'hh:mm:ss.sss' that identifies, looking at the 
        timestamp of the data object, the starting time that we want to plot.
    - stop_time : str
        String having format 'hh:mm:ss.sss' that identifies, looking at the 
        timestamp of the data object, the stopping time that we want to plot.

    """
    #Opening data file and unpacking columns
    timestamp_day, timestamp_hour, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange, P4, P5 = np.loadtxt(f'{path_to_datafile}', dtype=str, unpack=True)
    #Converting timestamps into datetime type in order to compare times between each data object
    timestamp_day = np.array([datetime.strptime(date, "%Y-%m-%d") for date in timestamp_day])
    timestamp_hour = np.array([datetime.strptime(hour, "%H:%M:%S.%f") for hour in timestamp_hour])
    #If there are time limits, use them for constructing time mask
    if start_time and stop_time is not None:
        #Creating mask based on time, comparing start and stop time to timestamp
        start_time = datetime.strptime(start_time, "%H:%M:%S.%f")
        mask = (timestamp_hour > start_time) & (timestamp_hour < stop_time)
        

    #mask = timestamp_hour
    #print(timestamp_day)
    print(timestamp_hour)

    return



if __name__ == "__main__":
    args = PRESSURE_MONITORING_ARGPARSER.parse_args()
    LabViewDataPlots(**vars(PRESSURE_MONITORING_ARGPARSER.parse_args()))