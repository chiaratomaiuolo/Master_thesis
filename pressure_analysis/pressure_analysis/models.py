"""Fit models.
"""

from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

def gaussian(x : np.array, norm : float, mu : float, sigma : float) -> np.array:
    """Gaussian curve definition
    """
    return norm*np.exp(-0.5*((x-mu)/sigma)**2)
    

def exponential(x : np.array, norm : float, scale : float) -> np.array:
    """Exponential curve definition
    """
    return norm*np.exp((x-400)/scale)
