import argparse
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from uncertainties import unumpy

from pressure_analysis.models import expo, alpha_expo_scale, double_expo
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

def iterative_exponential_fit(x: np.array, y_data: np.array, exp_model, p0: np.array=None, yerr: np.array=None, start:float=24, hours_lag: float=24):
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
        Array containing the starting initial parameters of the exponential model.
    - hours_lag : float
        Float that states the part of the dataset (in terms of hours) that
        is added to the fitted data at every iteration.

    Return
    -------
    - hours : list 
        List containing the times from filling in hours of every fit.
    - popts : list
        List containing the optimal parameters of every fit performed.
    - pcovs : list
        List containing the covariance matrix of the parameters
        of every fit performed.

    """
    #Stating the start of the dataset in terms of hours from first data object
    i = start 
    # Creating the lists that store the trend of the optimal parameters
    popts = []
    pcovs = []
    hours = [start]
    mask = x < i #Using the mask (if any)
    if yerr is None: # Fitting without errors if they are not provided
        try:
            popt_, pcov_ = curve_fit(exp_model, x[mask], y_data[mask], p0=p0)
            popts.append(popt_)
            while mask[-1] == False:
                mask = x < i
                hours.append(i)
                popt_, pcov_ = curve_fit(exp_model, x[mask], y_data[mask], yerr= np.full(len(y_data[mask]), 0.2), p0=popts[-1])
                popts.append(popt_)
                i += hours_lag
            #Fit using the entire dataset
            popt_, pcov_ = curve_fit(exp_model, x, y_data, p0=popts[-1])
            popts.append(popt_)
            pcovs.append(pcov_)
            hours.append(x[-1])
            print(f'Optimal parameters for {exp_model.__name__} fit of the entire dataset:')
            print(f'{popt_} +/- {np.sqrt(np.diag(pcov_))}')
            print(f'Covariance matrix:\n {pcov_}')
            fig, axs = plot_with_residuals(x, y_data, exp_model, popt_)
            axs[0].set(ylabel=r'$P_{eq}$ [mbar]', xlabel='Time [hours]')
            axs[1].axhline(y=0., color='r', linestyle='-')
            fig.suptitle(f'Fit of {hours[-1]} [hours] with {exp_model.__name__}')
            return hours, popts, pcovs
        except RuntimeError as e:
            print(e)
            plt.figure()
            plt.errorbar(x[mask], y_data[mask], marker='.', label='Dataset')
            plt.plot(x[mask], exp_model(x[mask], *p0), label='Curve with initial parameters')
            plt.grid()
            plt.legend()
            return None, None, None
    else: #yerr are provided, fitting using the errors on y
        try:
            popt_, pcov_ = curve_fit(exp_model, x[mask], y_data[mask], p0=p0, sigma=yerr[mask], absolute_sigma=True)
            popts.append(popt_)
            pcovs.append(pcov_)
            while mask[-1] == False:
                mask = x < i
                hours.append(i)
                popt_, pcov_ = curve_fit(exp_model, x[mask], y_data[mask], p0=popts[-1])
                popts.append(popt_)
                i += hours_lag
            #Fit using the entire dataset
            popt_, pcov_ = curve_fit(exp_model, x, y_data, p0=popts[-1])
            popts.append(popt_)
            pcovs.append(pcov_)
            hours.append(x[-1])
            print(f'Optimal parameters for {exp_model.__name__} fit of the entire dataset:')
            print(f'{popt_} +/- {np.sqrt(np.diag(pcov_))}')
            print(f'Covariance matrix:\n {pcov_}')
            fig, axs = plot_with_residuals(x, y_data, exp_model, popt_)
            axs[0].set(ylabel=r'$P_{eq}$ [mbar]', xlabel='Time [hours]')
            axs[1].axhline(y=0., color='r', linestyle='-')
            fig.suptitle(f'Fit of {hours[-1]} [hours] with {exp_model.__name__}')
            return hours, popts, pcovs
        except RuntimeError as e:
            print(e)
            plt.figure()
            plt.errorbar(x[mask], y_data[mask], marker='.', label='Dataset')
            plt.plot(x[mask], exp_model(x[mask], *p0), label='Curve with initial parameters')
            plt.grid()
            plt.legend()
            plt.show()
            return None, None, None

def parameters_plots(x, popts):
    """Plots the trend over time of the optimal parameters of the exponential
       model of interest.
    
    Arguments
    ---------
    - x : np.array or array-like
        Array of times
    - popts: list or array-like
        List containing the values of the optimal parameters to be plotted.
    
    """
    #Plotting trend of parameters of the fit model
    n_of_params = len(popts[0])
    fig, axs = plt.subplots(n_of_params)
    #fig.suptitle(fr'Optimal parameters trends')
    for i in range(n_of_params):
        popt_list = [p[i] for p in popts]

        axs[i].errorbar(x, popt_list, marker='d', linestyle='', color='tab:orange')
        axs[i].set(xlabel=r'Time [hours]', ylabel=f'popt[{i}]')
        axs[i].grid(True)
    
    return fig, axs


if __name__ == "__main__":
    #Obtaining argument parser objects
    args = EXPONENTIAL_FITS_ARGPARSER.parse_args()
    #Datafiles are briefly described above their pathfile line.
    #Select the interested one and comment the other paths_to_data, start_times, stop_times

    #Datafiles from 12/02/2024 to 22/02/2024 - AC DME filled without epoxy samples
    #paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']

    #start_times = [['2024-02-12 12:00:35.000']]
    #stop_times = [['2024-02-19 9:00:00.000']]

    

    #Datafiles from 26/02/2024 - AC DME filled, epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples.txt"]
    #paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_DME_with_epoxysamples_40degrees.txt"]
    #paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from2602.txt']
    #paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from0804.txt']

    
    #start_times = [['2024-02-26 15:51:00.000','2024-02-27 8:00:00.000', 
    #                '2024-02-28 8:00:00.000','2024-02-29 8:00:00.000',
    #                '2024-03-01 8:00:00.000','2024-03-02 8:00:00.000',
    #                '2024-03-03 8:00:00.000']]
    #stop_times = [['2024-02-26 22:00:00.000','2024-02-27 22:00:00.000',
    #               '2024-02-28 22:00:00.000','2024-02-29 22:00:00.000',
    #               '2024-03-01 22:00:00.000','2024-03-02 22:00:00.000',
    #               '2024-03-03 22:00:00.000']]

    #Data from 26/2 to 4/4, with 22°C, until gas heating
    start_times = [['2024-02-26 15:50:35.000']]
    stop_times = [['2024-03-15 9:00:00.000']]

    #Data from 26/2 to 4/4, with 22°C (two different time intervals, in the middle
    #temperature was set to 40°C)

    #start_times = [['2024-02-26 15:50:35.000', '2024-03-25 13:00:00.000']]
    #stop_times = [['2024-03-15 9:00:00.000', None]]

    #Data from 4/4, with 22°C, after having rose for some day to 40°C
    #start_times = [['2024-03-26 14:00:00.000']]
    #stop_times = [[None]]

    #Data from 8/4, with 22°C with new epoxy samples
    #start_times = [['2024-04-08 11:35:35.000']]
    #stop_times = [[None]]

    # --------------------------------------------------------------------------

    #Data sampling parameters (from data sampling logbook)
    log_time = 5000e-3 #s
    T_Julabo = 22 #°C

    #Loading the dataset with the first set of epoxy samples inside
    timestamps, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange,\
    P4, P5, t_diffs = LabViewdata_reading(paths_to_data, start_times, stop_times)

    # Including uncertainties in the quantities of interest
    P4 = unumpy.uarray(P4, np.full(len(P4), 0.12))
    T5 = unumpy.uarray(T5,  np.full(len(T5), 0.1))

    # Constructing the statistic for the temperature inside AC 
    # (In first round, this was not monitored directly)
    T_eff = T5+0.16*(T6-T5)/1.16 #°C

    #Computing time in hours
    t_hours = t_diffs/3600 #hours

    #Computing the equivalent pressure, quantity that takes into consideration
    # The temperature variations inside AC
    P_eq = unumpy.nominal_values((((P4*100)/(T_eff+273.15))*(T_Julabo+273.15))/100) #mbar
    dP_eq = unumpy.std_devs((((P4*100)/(T_eff+273.15))*(T_Julabo+273.15))/100)

    #Loading the dataset with the second set of epoxy samples inside
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from0804.txt']
    #Data from 8/4, with 22°C with new epoxy samples
    start_times = [['2024-04-08 11:35:35.000']]
    stop_times = [[None]]
    #Data sampling parameters (from logbook)
    log_time = 5000e-3 #s
    T_Julabo = 22 #°C

    # Selecting the exponential model from argparse argument
    model = funcs[args.func]

    # I DATASET
    #Performing fit for I dataset
    hours, popts, pcovs = iterative_exponential_fit(t_hours, P_eq, model,\
                      p0=args.list, yerr=dP_eq, hours_start=38, hours_lag=24)
    
    # Plotting the fitted trend and the correspondent residuals
    fig, axs=plot_with_residuals(t_hours, P_eq, model, popts[-1], yerr=dP_eq)
    popt1 = popts[-1]
    diag1 = np.sqrt(np.diag(pcovs[-1]))
    chi_2 = (((P_eq - model(t_hours, *popt1))/(dP_eq))**2).sum()
    axs[0].set(ylabel=r'$p_{eq}$ [mbar]')
    axs[0].grid()
    dpopt = diag1
    axs[1].set(xlabel='Time from filling [hours]', ylabel=r'Normalized residuals [# $\sigma_{p}$]')
    axs[1].grid()
    
    # Plotting the trend of optimal parameters over time
    fig, axs = parameters_plots(hours, popts)
    
    print('I DATASET RESULTS:')
    print(f'Asymptotic values for {model.__name__}')
    print(f'asymptotic value = {model(4*popts[-1][-1],*popts[-1])}')
    print(f'4 charasteric times are {(4*popts[-1][-1])/24} days')

    # II DATASET
    #Loading the dataset with the first set of epoxy samples inside
    timestamps, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange,\
    P4, P5, t_diffs = LabViewdata_reading(paths_to_data, start_times, stop_times)

    #Computing time in hours and effective temperature
    t_hours = t_diffs/3600 #hours
    T_eff = T5+0.16*(T6-T5)/1.16 #°C

    #Computing the equivalent pressure
    P_eq = (((P4*100)/(T_eff+273.15))*(T_Julabo+273.15))/100 #mbar
    dP_eq = np.sqrt((77/(P4*100))**2 + (0.05/T_eff)**2) #relative
    dP_eq = P_eq*dP_eq #absolute
    
    #Performing fit for II dataset
    hours, popts = iterative_exponential_fit(t_hours, P_eq, model,\
                      p0=args.list, yerr=dP_eq, hours_start=38, hours_lag=24)
    
    # Plotting the fitted trend and the correspondent residuals
    fig, axs=plot_with_residuals(t_hours, P_eq, model, popts[-1])
    print('II DATASET RESULTS:')   
    print(f'Asymptotic values for {model.__name__}')
    print(f'asymptotic value = {model(4*popts[-1][-1],*popts[-1])}')
    print(f'4 charasteric times are {(4*popts[-1][-1])/24} days')

    # Plotting the trend of optimal parameters over time
    fig, axs = parameters_plots(hours, popts)

    # III DATASET
    #Datafiles from 17/04/2024, 10:47 - AC DME filled, III set of epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from1704.txt']
    start_times = [['2024-04-19 12:21:00.000']]
    stop_times = [[None]]

    #Data sampling parameters (from logbook)
    log_time = 5000e-3 #s
    T_Julabo = 22 #°C
    
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
    dP_eq = np.sqrt((77/(P4*100))**2 + (0.05/T_eff_filtered)**2) #relative
    dP_eq = P_eq*dP_eq #absolute

    #Performing fit for III dataset
    hours, popts = iterative_exponential_fit(t_hours, P_eq, model,\
                      p0=args.list, hours_start=100, hours_lag=24)
    
    axs[0].plot(t_hours, P_eq, marker='.', linestyle='', color='tab:purple', label='Third dataset')
    
    if popts is not None:

        axs[0].plot(t_hours, model(t_hours, *popts[-1]), color='tab:pink', label=fr'Third dataset, $\alpha$ = {popts[-1][2]:.2f}')
        axs[0].legend()
        axs[0].set(xlabel=r't [hours]', ylabel=r'$P_{eq}$ [mbar]')
        res_normalized = (P_eq - model(t_hours, *popts[-1]))/P_eq
        axs[1].plot(t_hours, res_normalized, color='tab:purple', label='Third dataset residuals')
        axs[1].legend()
        axs[1].set(xlabel=r't [hours]', ylabel=r'$\frac{P_{eq} - P(t; P_0, \Delta, \alpha, \tau)}{P_{eq}}$')
    
        fig, axs=plot_with_residuals(t_hours, P_eq, model, popts[-1])
        fig.suptitle('Dataset from 17/04/2024 - Third set of epoxy samples')

        fig, axs = parameters_plots(hours, popts)
        axs[0].set(ylabel=r'$P_0$ [mbar]')
        axs[1].set(ylabel=r'$\Delta_1$ [mbar]')
        axs[2].set(ylabel=r'$\alpha$')
        axs[3].set(ylabel=r'$\tau$ [hours]')
        
        print(f'Asymptotic values for {model.__name__}, third set of epoxy samples')
        print(f'asymptotic value = {model(4*popts[-1][-1],*popts[-1])}')
        print(f'4 charasteric times are {(4*popts[-1][-1])/24} days')

    plt.show()