import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt


def expo(x, P0, Delta, tau):
    """Decreasing single exponential
    """
    return P0 - Delta*(1-np.exp(-x/tau))

def empty_AC_exp(t):
    """Decreasing single exponential with parameters fixed by empty chamber dataset fit.
    It should correspond to surface asbsorption of DME gas by steel walls of AC.
    """
    return -6.230*(1-np.exp(-t/80.57))

def alpha_expo_scale(t, P0, Delta, alpha, tau):
    """Decreasing exponential with exponent on time dependance. 
    It should describe processes of subdiffusion (alpha<1) or superdiffusion
    (alpha>1).
    """
    return P0 - (Delta*(1-np.exp(-((t/tau)**alpha))))

def double_expo(x, P0, Delta1, tau1, Delta2, tau2):
    """Decreasing double exponential
    """
    return P0 - (Delta1*(1-np.exp(-x/tau1)) + Delta2*(1-np.exp(-x/tau2)))

def triple_expo(x, P0, Delta1, tau1, Delta2, tau2, Delta3, tau3):
    """Decreasing triple exponential
    """
    return P0 - (Delta1*(1-np.exp(-x/tau1)) + Delta2*(1-np.exp(-x/tau2)) + Delta3*(1-np.exp(-x/tau3)))