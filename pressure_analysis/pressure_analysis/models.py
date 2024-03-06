import argparse
from datetime import datetime

import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from scipy.signal import butter, sosfilt




def expo(x, P0, Delta, tau):
    """Decreasing single exponential
    """
    return P0 - Delta*(np.exp(-x/tau))

def double_expo(x, P0, Delta1, tau1, Delta2, tau2):
    """Decreasing double exponential
    """
    return P0 - (Delta1*(np.exp(-x/tau1)) - Delta2*(np.exp(-x/tau2)))


def triple_exp_P0_fixed(x, P0, Delta1, tau1, Delta2, tau2, Delta3, tau3):
    """Decreasing triple exponential
    """
    return P0 - (Delta1*(np.exp(-x/tau1)) - Delta2*(np.exp(-x/tau2)) - Delta3*(np.exp(-x/tau3)))