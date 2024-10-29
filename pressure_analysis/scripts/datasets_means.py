import numpy as np

from loguru import logger

from pressure_analysis.labviewdatareading import LabViewdata_reading

"""This script is used for writing a .txt file with the mean value of a LabView 
    BFS dataset every n hours of sampling.
"""

def writing_txt(output_file_path: str, data: list, T_julabo: float, log_time: float, time_interval: int):
    """Takes as input a Labview BFS dataset and writes a .txt file containing the
    mean every time_interval hours of T2, T5, T6, P4, P5.

    Arguments
    ---------
    - data : list
        List of data read from BFS' LabView software.
    - T_julabo : float
        Julabo temperature set during data sampling.
    - log_time : float
        logging time in s of the LabView data file.
    - time_interval : int
        time interval in hours on which performing the mean.
    """

    #Obtaining data
    timestamps, T0, T1, T2, T3, T4, T5, T6, T7, TJ, P0, P1, P2, P3, PressFullrange,\
    P4, P5, t_diffs = data

    # Computing the number of points in every time interval in order to do 
    # the slicing
    n = int((time_interval-1)*(3600/log_time))
    print(n)
    # Converting the interesting data into numpy arrays ...
    t_diffs = np.array(t_diffs/(3600*24)) #in days
    print(t_diffs)
    T2 = np.array(T2)
    T5 = np.array(T5)
    T6 = np.array(T6)
    P4 = np.array(P4)
    P5 = np.array(P5)
    t_diffs_0 = t_diffs[0] #this is the first hour of data
    T2_0 = T2[0]
    T5_0 = T5[0]
    T6_0 = T6[0]
    P4_0 = P4[0]
    P5_0 = P5[0]
    # ... saving all the other pts point (indicating the initial pressure) ...
    t_diffs = t_diffs[1:]
    T2 = T2[1:]
    T5 = T5[1:]
    T6 = T6[1:]
    P4 = P4[1:]
    P5 = P5[1:]

    # ... reshape and perform RMS and mean on subgroups ...
    # MEAN
    t_diffs = np.mean(t_diffs[:len(t_diffs)//n*n].reshape(-1, n), axis=1)
    print(T2[:len(T2)//n*n].reshape(-1, n))
    print(T2[:len(T2)//n*n].reshape(-1, n)[0].shape)
    T2_mean = np.mean(T2[:len(T2)//n*n].reshape(-1, n), axis=1)
    T5_mean = np.mean(T5[:len(T5)//n*n].reshape(-1, n), axis=1)
    T6_mean = np.mean(T6[:len(T6)//n*n].reshape(-1, n), axis=1)
    P4_mean = np.mean(P4[:len(P4)//n*n].reshape(-1, n), axis=1)
    P5_mean = np.mean(P5[:len(P5)//n*n].reshape(-1, n), axis=1)

    #RMSD
    dT2 = [np.sqrt(np.mean((T2_mean[i] - T2[:len(T2)//n*n].reshape(-1, n)[i])**2).sum()) for i in range(len(T2_mean)-1)]
    dT5 = [np.sqrt(np.mean((T5_mean[i] - T5[:len(T5)//n*n].reshape(-1, n)[i])**2).sum()) for i in range(len(T5_mean)-1)]
    dT6 = [np.sqrt(np.mean((T6_mean[i] - T6[:len(T6)//n*n].reshape(-1, n)[i])**2).sum()) for i in range(len(T6_mean)-1)]
    dP4 = [np.sqrt(np.mean((P4_mean[i] - P4[:len(P4)//n*n].reshape(-1, n)[i])**2).sum()) for i in range(len(P4_mean)-1)]
    dP5 = [np.sqrt(np.mean((P5_mean[i] - P5[:len(P5)//n*n].reshape(-1, n)[i])**2).sum()) for i in range(len(P5_mean)-1)]

    # ... prepending the first point ...
    t_diffs = np.insert(t_diffs, 0, t_diffs_0, axis=0)
    T2_mean = np.insert(T2_mean, 0, T2_0, axis=0)
    T5_mean = np.insert(T5_mean, 0, T5_0, axis=0)
    T6_mean = np.insert(T6_mean, 0, T6_0, axis=0)
    P4_mean = np.insert(P4_mean, 0, P4_0, axis=0)
    P5_mean = np.insert(P5_mean, 0, P5_0, axis=0)
    dT2 = np.insert(dT2, 0, 0.1, axis=0)
    dT5 = np.insert(dT5, 0, 0.1, axis=0)
    dT6 = np.insert(dT6, 0, 0.1, axis=0)
    dP4 = np.insert(dP4, 0, 0.1, axis=0)
    dP5 = np.insert(dP5, 0, 0.1, axis=0)
    
    # Opening .txt file
    #output_file_path = 'first_epoxy_sample_data.txt'
    logger.info(f'Opening output file {output_file_path}...')
    output_file = open(output_file_path, 'w')
    output_file.write('# t [days]   T_Julabo [°C]   T2 [°C]  dT2 [°C]  T5 [°C]  dT5 [°C]  T6 [°C] dT6 [°C]  P4 [mbar]  dP4 [mbar] P5 [mbar]   dP5 [mbar]\n')

    # Writing data row by row...
    for day, t2, dt2, t5, dt5, t6, dt6, p4, dp4, p5, dp5 in zip(t_diffs, T2_mean, dT2, T5_mean, dT5, T6_mean, dT6, P4_mean, dP4, P5_mean, dP5):
        output_file.write(f'{day}   {T_julabo}  {t2:.2f}  {dt2:.2f}  {t5:.2f}  {dt5:.2f}  {t6:.2f}  {dt6:.2f}  {p4:.1f}   {dp4:.2f}  {p5:.1f} {dp5:.2f}\n')
    
    #... and eventually closing the file 
    output_file.close()
    logger.info('Output file closed.')

    return


if __name__ == "__main__":
    # Extrapolating data from LabView
    # Datafiles are briefly described above their pathfile line.
    # Select the interested one and comment the other paths_to_data, start_times, stop_times
    '''
    # Loading dataset without epoxy samples
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_DME_measurements.txt']
    #Data from 26/2 to 4/4, with 22°C, until gas heating
    start_times = [['2024-02-12 15:35:00.000', '2024-02-19 18:30:00.000']]
    stop_times = [['2024-02-19 11:30:00.000', '2024-02-20 12:30:00.000']]
    #Data sampling parameters (from logbook)
    log_time = 5000e-3 #s
    T_Julabo = 22 #°C
    

    empty_AC_dataset = LabViewdata_reading(paths_to_data, start_times, stop_times)
    writing_txt('empty_AC_epoxy_sample_data_with_rms.txt', empty_AC_dataset, T_Julabo, log_time, 24)
    '''

    
    # Loading first epoxy samples set
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from2602.txt']
    #Data from 26/2 to 4/4, with 22°C, until gas heating
    start_times = [['2024-02-26 15:50:35.000']]
    stop_times = [['2024-03-15 9:00:00.000']]
    #Data sampling parameters (from logbook)
    log_time = 5000e-3 #s
    T_Julabo = 22 #°C

    first_dataset = LabViewdata_reading(paths_to_data, start_times, stop_times)
    writing_txt('first_epoxy_sample_data.txt', first_dataset, T_Julabo, log_time, 24)

    #Loading the second dataset
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from0804.txt']
    #Data from 8/4, with 22°C with new epoxy samples
    start_times = [['2024-04-08 11:35:35.000']]
    stop_times = [[None]]
    #Data sampling parameters (from logbook)
    log_time = 5000e-3 #s
    T_Julabo = 22 #°C

    second_dataset = LabViewdata_reading(paths_to_data, start_times, stop_times)
    writing_txt('second_epoxy_sample_data.txt', second_dataset, T_Julabo, log_time, 24)

    #Loading third dataset
    #Datafiles from 17/04/2024, 10:47 - AC DME filled, III set of epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ['/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from1704.txt']
    start_times = [['2024-04-19 12:21:00.000']]
    stop_times = [['2024-06-04 16:00:00.000']]
    #Data sampling parameters (from logbook)
    log_time = 5000e-3 #s
    T_Julabo = 22 #°C

    third_dataset = LabViewdata_reading(paths_to_data, start_times, stop_times)
    writing_txt('third_epoxy_sample_data.txt', third_dataset, T_Julabo, log_time, 24)

    #Loading third dataset
    #Datafiles from 17/04/2024, 10:47 - AC DME filled, III set of epoxy samples inside, T_Julabo = 22°C
    paths_to_data = ["/Users/chiara/Desktop/Thesis_material/Master_thesis/pressure_analysis/Data/merged_measurements_from1704.txt"]
    start_times = [['2024-04-19 12:26:30.000']]
    stop_times = [[None]]
    #Data sampling parameters (from logbook)
    log_time = 5000e-3 #s
    T_Julabo = 22 #°C

    gpd_dataset = LabViewdata_reading(paths_to_data, start_times, stop_times)
    writing_txt('GPD_data.txt', gpd_dataset, T_Julabo, log_time, 24)
    