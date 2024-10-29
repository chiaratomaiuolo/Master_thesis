import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt

"""Signal filtering functions
"""

def butterworth_filtering(signal: np.array, N, Wn, btype='low', analog=False, fs=None) -> np.array:
    """The following function wraps from scipy.signal.butter: creates a
    Butterworth filter with output='sos' and applies it to the signal,
    given as output.
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html

    Arguments
    ---------
    -signal : np.array
        Array containing the signal to be filtered
    For all the other parameters see link above. 
    """
    butterworth_filter = butter(N, Wn, btype, analog, fs)
    filtered_signal = sosfilt(signal, butterworth_filter)
    return filtered_signal

def temperature_butterworth_filtering(T: np.array, sampling_time: float):
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