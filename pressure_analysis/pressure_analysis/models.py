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

def expo_P0_frozen(x, Delta, tau):
    return expo(x, 1198.8, Delta, tau)

def double_expo(x, P0, Delta1, tau1, Delta2, tau2):
    """Decreasing double exponential
    """
    return P0 - (Delta1*(1-np.exp(-x/tau1)) + Delta2*(1-np.exp(-x/tau2)))

def double_exp_P0_frozen(x, Delta1, tau1, Delta2, tau2):
    return double_expo(x, 1198.8, Delta1, tau1, Delta2, tau2)


def triple_expo(x, P0, Delta1, tau1, Delta2, tau2, Delta3, tau3):
    """Decreasing triple exponential
    """
    return P0 - (Delta1*(1-np.exp(-x/tau1)) + Delta2*(1-np.exp(-x/tau2)) + Delta3*(1-np.exp(-x/tau3)))

def triple_expo_P0_frozen(x, Delta1, tau1, Delta2, tau2, Delta3, tau3):
    """Decreasing triple exponential
    """
    return triple_expo(x, 1198.8, Delta1, tau1, Delta2, tau2, Delta3, tau3)