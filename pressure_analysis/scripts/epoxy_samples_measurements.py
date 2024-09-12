""" This script is used for propagating the uncertainties on volumes and areas
of the epoxy samples for the absorption measurements.
"""
import numpy as np
from uncertainties import ufloat

def density(m, V):
    return m/V

def V(t, d_e, d_i):
    ''' Volume of the holed cylinder
    '''
    return (d_e**2 - d_i**2)*t*(np.pi/4)

def S(t, d_e, d_i):
    ''' Area of the holed cylinder
    '''
    return np.pi*(0.5*(d_e**2 - d_i**2) + t*(d_e + d_i))

if __name__ == "__main__":
    # Defining the measurements of the first set of samples
    #m = np.array([ufloat(5.456, 0.001), ufloat(3.628, 0.001), ufloat(2.850, 0.001)]) #g
    #t = np.array([ufloat(2.26, 0.21), ufloat(1.50, 0.01), ufloat(1.50, 0.01)]) #mm
    #de = np.array([ufloat(55.05, 0.05), ufloat(55.03, 0.05), ufloat(50.05, 0.05)]) #mm
    #di = np.array([ufloat(28.93, 0.05), ufloat(28.7, 0.1), ufloat(27.91, 0.05)]) #mm

    # Defining the measurements of the second set of samples
    #m = np.array([ufloat(1.348, 0.001), ufloat(1.166, 0.001), ufloat(1.278, 0.001)]) #g
    #t = np.array([ufloat(0.80, 0.05), ufloat(0.65, 0.05), ufloat(0.60, 0.05)]) #mm
    #de = np.array([ufloat(50.2, 0.1), ufloat(55.1, 0.1), ufloat(55.1, 0.1)]) #mm
    #di = np.array([ufloat(27.9, 0.1), ufloat(28.0, 0.1), ufloat(28.0, 0.1)]) #mm

    # Defining the measurements of the third set of samples
    m = np.array([ufloat(12.596, 0.001)]) #g
    t = np.array([ufloat(5.25, 0.05)]) #mm
    de = np.array([ufloat(55.1, 0.1)]) #mm
    di = np.array([ufloat(28.2, 0.1)]) #mm

    for thick, dext, dint, mass in zip(t, de, di, m):
        print(thick, dext, dint)
        print(f'Volume: {V(thick, dext, dint)}, Area:{S(thick, dext, dint)}')
        print(f'V/S: {V(thick, dext, dint)/S(thick, dext, dint)}')
        print(f'density: {density(mass, V(thick, dext, dint))}')
    
    print(np.sum(m))
    print(np.sum(V(t, de, di)))
    print(np.sum(S(t, de, di)))
    print(np.sum(V(t, de, di)/S(t, de, di)))