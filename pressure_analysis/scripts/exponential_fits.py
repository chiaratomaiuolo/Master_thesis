import argparse
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

from pressure_analysis.models import expo, alpha_expo_scale, double_expo, empty_AC_exp
from pressure_analysis.filtering import temperature_butterworth_filtering
from pressure_analysis.labviewdatareading import LabViewdata_reading, plot_with_residuals

__description__ = \
"This script is used for performing data analysis of some LabView datasets\
for the study of the behaviour of gases inside Absorption Chamber (AC) of the BFS.\
In particular, in the following, an iterative fit is performed, in order to study the\
trend of the optimal parameters over time."

funcs = {'expo' :expo, 'alpha_expo_scale' :alpha_expo_scale, 'double_expo' :double_expo}

EXPONENTIAL_FITS_ARGPARSER = argparse.ArgumentParser(description=__description__)
EXPONENTIAL_FITS_ARGPARSER.add_argument('-exponential_model','--func', dest='func', choices=['expo', 'alpha_expo_scale', 'double_expo'],
            help="Exponential model to be used for fitting data.")
EXPONENTIAL_FITS_ARGPARSER.add_argument('-p0','--list', nargs='+', type=float, required=True, 
                                        default=None, help="List\
                                        containing the initial parameters. Lenght\
                                        must be the same as the number of parameters\
                                        of exponential_model function.")

def iterative_exponential_fit(x: np.array, y_data: np.array, exp_model, p0: np.array, hours_start:float=24, hours_lag: float=24):
    """Performs an iterative fit on an increasing-lenght dataset until its end.
    At the end of the fit prints the parameters and plots the last fitting 
    curve (containing all data of the dataset).

    Arguments
    ---------
    - x : np.array
        Array containing x data
    - y_data : np.array
        Array containing y data
    - exp_model : function
        Exponential model for fitting parameters
    - mask: np.array
        Array mask used for cutting on data
    - p0 : np.array
        Array containing the starting initial parameters of the double exp.
    - hours_lag : float
        Float that states the part of the dataset (in terms of hours) that
        is added to the fitted data at every iteration.
    """
    i = hours_start #states the start of the dataset in terms of hours from first data object
    popts = []
    hours = [hours_start]
    mask = x < i
    popt_, pcov_ = curve_fit(exp_model, x[mask], y_data[mask], p0=p0)

    print(f'Optimal parameters of {exp_model.__name__} for the first {hours_start} hours of data:')
    popts.append(popt_)
    print(popt_)
    #Plotting pts, fit and residuals for fit of the first 3 hours
    '''
    fig, axs = plot_with_residuals(t_hours[mask], P_eq[mask], exp_model(t_hours[mask], *popt_))
    fig.suptitle(f'Fit of the first {hours_start} hours of data with a {exp_model.__name__}')
    
    axs[0].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ [mbar]')
    axs[0].grid(True)
    axs[1].set(xlabel=r'Time [hours]', ylabel=r'$P_{eq}$ normalized residuals')
    axs[1].grid(True)
    '''
    while mask[-1] == False:
        mask = x < i
        hours.append(i)
        popt_, pcov_ = curve_fit(exp_model, x[mask], y_data[mask], p0=popts[-1])
        popts.append(popt_)
        i += hours_lag
    #Fit considering entire dataset after having iterated on time intervals
    popt_, pcov_ = curve_fit(exp_model, x, y_data, p0=popts[-1])
    popts.append(popt_)
    hours.append(t_hours[-1])
    print(f'Optimal parameters for {exp_model.__name__} fit of the entire dataset:')
    print(f'{popt_} +/- {np.sqrt(np.diag(pcov_))}')
    print(f'Covariance matrix: {pcov_}')
    fig, axs = plot_with_residuals(x, y_data, exp_model, popt_)
    axs[0].set(ylabel=r'$P_{eq}$ [mbar]', xlabel='Time [hours]')
    axs[1].axhline(y=0., color='r', linestyle='-')
    fig.suptitle(f'Fit of {hours[-1]} [hours] with {exp_model.__name__}')
    return hours, popts

def parameters_plots(x, popts):
    #Plotting trend of parameters of the fit model
    n_of_params = len(popts[0])
    fig, axs = plt.subplots(n_of_params)
    fig.suptitle(fr'Optimal parameters trends')
    for i in range(n_of_params):
        popt_list = [p[i] for p in popts]
        axs[i].errorbar(x, popt_list, marker='.', linestyle='')
        axs[i].set(xlabel=r'Time [hours]', ylabel=f'popt[{i}]')
        axs[i].grid(True)
    
    return fig, axs


if __name__ == "__main__":
    #Datafiles are briefly descripted above their pathfile line. 
    #Select the interested one and comment the other paths_to_data, start_times, stop_times

    #paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']

    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples_40degrees.txt"]
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from2602.txt']
    '''
    start_times = [['2024-02-26 15:51:00.000','2024-02-27 8:00:00.000', 
                    '2024-02-28 8:00:00.000','2024-02-29 8:00:00.000',
                    '2024-03-01 8:00:00.000','2024-03-02 8:00:00.000',
                    '2024-03-03 8:00:00.000']]
    stop_times = [['2024-02-26 22:00:00.000','2024-02-27 22:00:00.000',
                   '2024-02-28 22:00:00.000','2024-02-29 22:00:00.000',
                   '2024-03-01 22:00:00.000','2024-03-02 22:00:00.000',
                   '2024-03-03 22:00:00.000']]
    '''
    #start_times = [['2024-02-26 15:50:35.000']]
    #stop_times = [['2024-02-26 15:58:35.000']]
    #stop_times = [['2024-03-15 9:00:00.000']]
    #stop_times = [['2024-02-28 9:51:00.000']]
    #start_times = [['2024-03-18 00:00:00.000']]
    #stop_times=[[None]]
    start_times = [['2024-02-26 15:50:35.000', '2024-03-25 13:00:00.000']]
    stop_times = [['2024-03-15 9:00:00.000', None]]

    #Data sampling parameters (from logbook)
    log_time = 5000e-3 #s
    T_Julabo = 40 #°C

    #Obtaining data
    timestamps, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange,\
    P4, P5, t_diffs = LabViewdata_reading(paths_to_data, start_times, stop_times)

    #Computing time in hours and effective temperature
    t_hours = t_diffs/3600 #hours
    T_eff = T5+0.16*(T6-T5)/1.16 #°C

    #Filtering the effective temperature in order to compensate time lag and heat capacity of AC
    T_eff_filtered = temperature_butterworth_filtering(T_eff, log_time)
    #T_eff_filtered = T_eff

    #Computing the equivalent pressure
    P_eq = (((P4*100)/(T_eff_filtered+273.15))*(T_Julabo+273.15))/100 #mbar

    #Obtaining argument parser objects
    args = EXPONENTIAL_FITS_ARGPARSER.parse_args()
    model = funcs[args.func]
    hours, popts = iterative_exponential_fit(t_hours[t_hours<500], P_eq[t_hours<500], model,\
                      p0=args.list, hours_start=38, hours_lag=24)
    
    fig, axs=plot_with_residuals(t_hours, P_eq, model, popts[-1])
    fig.suptitle('Full dataset from 26/02/2024')
    
    
    print(f'asymptotic value = {model(4*popts[-1][-1],*popts[-1])}')
    print(f'4 charasteric times are {(4*popts[-1][-1])/24} days')
    
    '''
    fig, axs = parameters_plots(hours, popts)
    axs[0].set(ylabel=r'$P_0$ [mbar]')
    axs[1].set(ylabel=r'$\Delta_1$ [mbar]')
    axs[2].set(ylabel=r'$\alpha$')
    axs[3].set(ylabel=r'$\tau$ [hours]')
    '''
    '''

    #Defining a custom function and redoing iterative fit
    def exp_with_alpha_fixed(t, P0, Delta, tau):
        return alpha_expo_scale(t, P0, Delta, 0.5, tau)
    
    def double_alpha(t, P0, delta1, tau1, delta2, alpha2, tau2):
        return expo(t, P0, delta1, tau1) + alpha_expo_scale(t+0.13, 0., delta2, alpha2, tau2)
    
    
    hours, popts = iterative_exponential_fit(t_hours, P_eq, double_alpha,\
                      p0=[1202., 2.2, 0.397, 564., 0.525, 1569.5], hours_start=38, hours_lag=24)
    
    print(f'asymptotic value = {double_alpha(4*popts[-1][-1],*popts[-1])}')
    print(f'4 charasteric times are {(4*popts[-1][-1])/24} days')
    

    fig, axs = parameters_plots(hours, popts)
    axs[0].set(ylabel=r'$P_0$ [mbar]')
    axs[1].set(ylabel=r'$\Delta_1$ [mbar]')
    axs[2].set(ylabel=r'$\tau_1$ [hours]')
    axs[3].set(ylabel=r'$\Delta_2$ [hours]')
    axs[4].set(ylabel=r'$\alpha$')
    axs[1].set(ylabel=r'$\Delta_2$ [mbar]')
    
    '''
    '''
    hours, popts = iterative_exponential_fit(t_hours, P_eq, exp_with_alpha_fixed,\
                      p0=[[1.19855940e+03, 2.74738884e+02, 2.96778217e+02]], hours_start=24, hours_lag=24)

    
    popt, pcov = curve_fit(exp_with_alpha_fixed, t_hours, P_eq, p0=[[1201., 671., 500.]])
    #plot_with_residuals(t_hours, P_eq, alpha_expo_scale(t_hours, *popt))
    #print(popt)
    print(f'asymptotic value = {exp_with_alpha_fixed(4*popts[-1][-1],*popts[-1])}')
    print(f'4 charasteric times are {(4*popt[-1])/24} days')
    plt.figure()
    z = np.linspace(0, 10000, 5000)
    plt.plot(z, exp_with_alpha_fixed(z, *popt))

    fig, axs = parameters_plots(hours, popts)
    '''
    
    

    

    

    plt.show()