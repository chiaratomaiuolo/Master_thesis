from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

#Timestamp	T0 - GPD IFplate (�C)	T1 - GPD drift (�C)	T2 - GPD mech (�C)
#T3 - Tank TC (�C)	T4 - GTC TF Outlet (�C)	T5 - AC TF Outlet (�C)	T6 - Ambient (�C)
#T7 - Tank PT100 (�C)	TJulaboInternal (�C)	P0 - Setra AC (mbar)	
#P1 - Setra Filling (mbar)	P2 - Omega AC (mbar)	P3 - Omega GTC (mbar)	PressFullrange (mbar)
#P4 - MKS AC (mbar)	P5 - MKS GPD (mbar)


def LabViewDataPlots(path_to_file: str, logging_time: float, acquisition_time: float, start_time: str, stop_time: str):
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
    timestamp, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange, P4, P5 = np.loadtxt(f'{path_to_file}', unpack=True)
    #Creating mask based on time, comparing start and stop time to timestamp
    timestamp = datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S")
    print(timestamp)

    return



