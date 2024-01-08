import argparse

import numpy as np 
import matplotlib.pyplot as plt 
from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit 


"""This script computes the conversion factors between the GPD setup assembled
   in INFN PI and the one assembled in INFN TO. 
   It contains functions for plotting interesting quantities, divided by GPD_ID
   and setup used for the measures. 
"""

"""Defining a dictionary of dictionaries filled with data: 
    Key(1): GPD ID
        Key(2): LABEL str describing the setup
            HV value of DeltaV [V]
            Rate [Hz]
            Uncertainty on rate [Hz]
            Mean of PHA peak [adc counts]
            Uncertainty on PHA peak mean [adc counts]
            Resolution on PHA [adc counts]
            Uncertainty on resolution on PHA [adc counts]
            Track size mean [mm]
            Track size std [mm]

    Note that for GPD 33 and GPD 35 separate dictionaries were created because 
    there weren't PI reference measures. 
"""

data = {'GPD 30' :
         {'PI source, PI setup' : {'run_id' : [6386, 6384, 6385] ,\
                              'HV' : [435, 440, 445],\
                              'rate' : [7.963, 7.737, 7.927],
                              'd_rate' : [0.137, 0.069, 0.138],
                              'PHA mean' : [15662, 18279, 21201],\
                              'd PHA mean' : [40, 28, 53],\
                              'PHA sigma' : [1671, 1971, 2167],\
                              'd PHA sigma' : [50, 36, 69],\
                              'mean trk size' : [0.6812, 0.6814, 0.6816],\
                              'd mean trk size' : [0.0036, 0.0021, 0.0036]
                              },
          'TO source, PI setup' : {'run_id' : [399, 397, 398],\
                              'HV' : [435, 440, 445],\
                              'rate' : [32.136, 32.092, 32.058],
                              'd_rate' : [0.525, 0.286, 0.564],
                              'PHA mean' : [15663, 18248, 21059],\
                              'd PHA mean' : [40, 26, 52],\
                              'PHA sigma' : [1742, 1901, 2086],\
                              'd PHA sigma' : [50, 33, 67],\
                              'mean trk size' : [0.6766, 0.6852, 0.6850],\
                              'd mean trk size' : [0.0034, 0.0021, 0.0036]
                              },
          'TO source and HV, PI back-end' : {'run_id' : [404, 402, 403],\
                              'HV' : [435, 440, 445],\
                              'rate' : [31.799, 31.433, 32.645],
                              'd_rate' : [0.549, 0.283, 0.496],
                              'PHA mean' : [15314, 17762, 20559],\
                              'd PHA mean' : [39, 25, 44],\
                              'PHA sigma' : [1598, 1838, 1989],\
                              'd PHA sigma' : [50, 32, 56],\
                              'mean trk size' : [0.6790, 0.6852, 0.6880],\
                              'd mean trk size' : [0.0036, 0.0021, 0.0033]
                              },
          'TO source, TO setup' : {'run_id' : [412, 410, 411],\
                              'HV' : [435, 440, 445],\
                              'rate' : [32.186, 32.001, 33.667],
                              'd_rate' : [0.558, 0.284, 0.574],
                              'PHA mean' : [15115, 17582, 20498],\
                              'd PHA mean' : [37, 25, 51],\
                              'sigma' : [1548, 1773, 2121],\
                              'd sigma' : [48, 32, 65],\
                              'mean trk size' : [0.6801, 0.6797, 0.6906],\
                              'd mean trk size' : [0.0036, 0.0021, 0.0038]
                              }
            },
        'GPD_37' : 
            {'PI source, PI setup' : {'run_id' : [6393, 6390, 6391],\
                              'HV' : [425, 430, 435],\
                              'rate' : [7.851, 7.695, 7.601],
                              'd_rate' : [0.138, 0.069, 0.136],
                              'PHA mean' : [15254, 17872, 20692],\
                              'd PHA mean' : [50, 34, 66],\
                              'sigma' : [2150, 2454, 2731],\
                              'd sigma' : [62, 43, 84],\
                              'mean trk size' : [0.6876, 0.6926, 0.6951],\
                              'd mean trk size' : [0.0036, 0.0021, 0.0036]
                              },
             'TO source, TO setup' : {'run_id' : [418, 416, 417],\
                              'HV' : [425, 430, 435],\
                              'rate' : [30.863, 30.637, 33.437],
                              'd_rate' : [0.543, 0.270, 0.586],
                              'PHA mean' : [14926, 17330, 20240],\
                              'd PHA mean' : [47, 32, 63],\
                              'sigma' : [2026, 2280, 2634],\
                              'd sigma' : [59, 41, 79],\
                              'mean trk size' : [0.6904, 0.6977, 0.7019],\
                              'd mean trk size' : [0.0037, 0.0021, 0.0037]
                              }
            },
        'GPD_38' :
            {'PI source, PI setup' : {'run_id' : [6399, 6397, 6398],\
                              'HV' : [430, 435, 440],\
                              'rate' : [7.957, 7.754, 8.078],
                              'd_rate' : [0.141, 0.068, 0.138],
                              'PHA mean' : [15056, 17553, 20411],\
                              'd PHA mean' : [42, 28, 59],\
                              'sigma' : [1750, 2700, 2344],\
                              'd sigma' : [54, 36, 77],\
                              'mean trk size' : [0.6657, 0.6768, 0.6830],\
                              'd mean trk size' : [0.0034, 0.0021, 0.0036]
                              },
             'TO source, TO setup' : {'run_id' : [426, 424, 425],\
                              'HV' : [430, 435, 440],\
                              'rate' : [32.455, 31.809, 33.064],
                              'd_rate' : [0.561, 0.282, 0.556],
                              'PHA mean' : [14726, 17084, 19889],\
                              'd PHA mean' : [38, 26, 48],\
                              'sigma' : [1549, 1834, 1947],\
                              'd sigma' : [50, 33, 59],\
                              'mean trk size' : [0.6684, 0.6779, 0.6818],\
                              'd mean trk size' : [0.0035, 0.0021, 0.0037]
                              }
            
            },
}
data33 = {'GPD_33' :
            {'TO source, TO setup' : {'run_id' : [432, 430, 431],\
                              'HV' : [440, 445, 450],\
                              'rate' : [30.146, 30.587, 30.589],
                              'd_rate' : [0.526, 0.269, 0.504],
                              'PHA mean' : [17397, 19262, 22943],\
                              'd PHA mean' : [41, 26, 49],\
                              'sigma' : [1666, 1854, 2042],\
                              'd sigma' : [53, 33, 64],\
                              'mean trk size' : [0.7284, 0.7276, 0.7380],\
                              'd mean trk size' : [0.0039, 0.0023, 0.0038]
                              }


            }
}

data35 = {'GPD_35' :
            {'TO source, TO setup' : {'run_id' : [436, 434, 435],\
                              'HV' : [445, 450, 455],\
                              'rate' : [29.685, 29.541, 30.513],
                              'd_rate' : [0.534, 0.272, 0.535],
                              'PHA mean' : [20435, 22618, 26733],\
                              'd PHA mean' : [47, 33, 66],\
                              'sigma' : [1907, 20307, 2421],\
                              'd sigma' : [59, 43, 89],\
                              'mean trk size' : [0.7464, 0.7439, 0.7556],\
                              'd mean trk size' : [0.0041, 0.0024, 0.0043]
                              }


            }
}
          

def GPD_control_plots(gpd_id, gpd_label, run_id, HV, rate, d_rate, pha_mean, d_pha_mean,
                      pha_sigma, dpha_sigma, mean_trk_size, d_mean_trk_size):
    """
        This function creates the following summary plots for GPD tracks:
            - PHA mean as a function of Delta V;
            - PHA sigma as a function of Delta V;
            - mean rate as a function of Delta V;
            - mean track size a function of Delta V. 
    """
    plt.figure(f'PHA peak value, ID : {gpd_id}')
    plt.title(f'PHA peak value, ID : {gpd_id}')
    plt.errorbar(HV, pha_mean, d_pha_mean, marker = '.', label = f'{gpd_label}')
    plt.xlabel(r'$\Delta$V [V]')
    plt.ylabel('PHA [a.u.]')

    plt.legend()
    plt.grid(True)
    
    plt.figure(f'PHA sigma value, ID : {gpd_id}')
    plt.title(f'PHA sigma value, ID : {gpd_id}')
    plt.errorbar(HV, pha_sigma, dpha_sigma, marker = '.', label = f'{gpd_label}')
    plt.xlabel(r'$\Delta$V [V]')
    plt.ylabel('PHA [a.u.]')

    plt.legend()
    plt.grid(True)
    

    plt.figure(f'Mean rate, ID : {gpd_id}')
    plt.title(f'Mean rate, ID : {gpd_id}')
    plt.errorbar(HV, rate, d_rate, marker = '.', label = f'{gpd_label}')
    plt.xlabel(r'$\Delta$V [V]')
    plt.ylabel('Rate [Hz]')

    plt.legend()
    plt.grid(True)
    
    plt.figure(f'Mean track size, ID : {gpd_id}')
    plt.title(f'Mean track size, ID : {gpd_id}')
    plt.errorbar(HV, mean_trk_size, d_mean_trk_size, marker = '.', label = f'{gpd_label}')
    plt.xlabel(r'$\Delta$V [V]')
    plt.ylabel('Track size [mm]')

    plt.legend()
    plt.grid(True)
    
    return 

def fit_with_a_const(x,y,dy,label):
    """
        This function performs a fit with a constant model and plots the result
        in a figure. It returns the optimal constant value and its uncertainty. 
    """
    popt, pcov = curve_fit(constant, x, y, sigma=dy)
    opt_c = ufloat(popt, np.sqrt(pcov))
    plt.errorbar(x, y, dy, marker='o', label=label)
    plt.axhline(popt, color=plt.gca().get_lines()[-1].get_color())
    plt.legend()
    plt.grid(True)
    return popt, pcov

def constant(x, c):
    """
        Definition of a constant model for further usage. 
    """
    return c

def ratio_uncertainty(x,y,dx,dy):
    """
        Defining the uncertainty on the ratio of two quatities. 
    """
    return np.sqrt((dx/y)**2 + ((x*dy)/(y**2))**2)

if __name__ == "__main__":
    for gpd_id in data.keys():
        data_id = data[gpd_id]
        for label, d in data_id.items():
            if label == 'PI source, PI setup':
                pi_data = d
            elif label == 'TO source, TO setup':
                to_data = d
            #GPD_control_plots(gpd_id, label, *d.values()) #For plotting all quantities for every GPD_ID
        r_PHA = np.array(to_data['PHA mean'])/np.array(pi_data['PHA mean'])
        dr_PHA = np.sqrt((np.array(to_data['d PHA mean'])/np.array(pi_data['PHA mean']))**2
                     + ((np.array(to_data['PHA mean'])*np.array(pi_data['d PHA mean']))/(np.array(pi_data['PHA mean'])**2))**2)
        r_rate = np.array(pi_data['rate'])/np.array(to_data['rate'])
        dr_rate = np.sqrt((np.array(pi_data['d_rate'])/np.array(to_data['rate']))**2
                     + ((np.array(pi_data['rate'])*np.array(to_data['d_rate']))/(np.array(to_data['rate'])**2))**2)
        
        plt.figure(fr'PHA mean ratio $\frac{{TO}}{{PI}}$')
        plt.title(fr'Ratio $\frac{{TO}}{{PI}}$ on PHA mean value')
        popt_pha, pcov_pha = fit_with_a_const(d['HV'], r_PHA, dr_PHA, f'{gpd_id}')
        plt.xlabel(r'$\Delta$V [V]')
        plt.ylabel(r'PHA ratio $\frac{TO}{PI}$')
        opt_c = ufloat(popt_pha, np.sqrt(pcov_pha))
        
        plt.figure(fr'rate ratio $\frac{{PI}}{{TO}}$')
        plt.title(fr'Ratio $\frac{{PI}}{{TO}}$ on event rate')
        popt_rate, pcov_rate = fit_with_a_const(d['HV'], r_rate, dr_rate, f'{gpd_id}')
        plt.xlabel(r'$\Delta$V [V]')
        plt.ylabel(r'rate ratio $\frac{TO}{PI}$')
        opt_c_rate = ufloat(popt_rate, np.sqrt(pcov_rate))

    #Saving some variables in order to print them on Python terminal. 
    pi_pha30 = ufloat(18279, 28)
    to_pha30 = ufloat(17582, 25)
    ratio_pha30 = to_pha30/pi_pha30
    pi_rate30 = ufloat(7.737, 0.069)
    to_rate30 = ufloat(32.001, 0.284)
    ratio_rate30 = pi_rate30/to_rate30

    pi_pha37 = ufloat(17872, 34)
    to_pha37 = ufloat(17330, 32)
    ratio_pha37 = to_pha37/pi_pha37
    pi_rate37 = ufloat(7.695, 0.069)
    to_rate37 = ufloat(30.637, 0.270)
    ratio_rate37 = pi_rate37/to_rate37

    pi_pha38 = ufloat(17553, 28)
    to_pha38 = ufloat(17084, 26)
    ratio_pha38 = to_pha38/pi_pha38
    pi_rate38 = ufloat(7.754, 0.068)
    to_rate38 = ufloat(31.809, 0.282)
    ratio_rate38 = pi_rate38/to_rate38

    #Printing some interesting values on Python terminal. 

    print(f'Ratio between full PI setup and full TO setup for GPD_30 at DeltaV = 440 V :'
        f'mean PHA ratio (TO/PI) = {ratio_pha30}, mean rate ratio (PI/TO) = {ratio_rate30}')
    print(f'Ratio between full PI setup and full TO setup for GPD_37 at DeltaV = 430 V :'
        f'mean PHA ratio (TO/PI) = {ratio_pha37}, mean rate ratio (PI/TO) = {ratio_rate37}')
    print(f'Ratio between full PI setup and full TO setup for GPD_38 at DeltaV = 435 V :'
        f'mean PHA ratio (TO/PI) = {ratio_pha38}, mean rate ratio (PI/TO) = {ratio_rate38}')
    

plt.show()
    
