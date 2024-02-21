import numpy as np
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

"""
    Simple linear fit for computing the slope of the calibration curve of pressure
    sensors used in BFS. 
"""
# Conversion factors 
from_psi_to_mbar = 68.9476 #mbar/psi
from_torr_to_mbar = 1.33322 #mbar/torr

#Calibration data 
# The interesting output is the response to a set of pressures in VDC. 

# 12404 is the MKS sensor, installed as P5 in BFS
p_std_12404 = np.array([0.000, 99.916, 199.919, 399.860, 599.833, 799.673, 999.566]) #Torr
V_uut_12404 = np.array([0.0000, 0.9992, 1.9993, 3.9988, 5.9985, 7.9970, 9.9956]) #V

full_scale_p_12404 = 999.9 #Torr
full_scale_V_12404 = 10 #V

# 12405 is the MKS sensor, installed as P4 in BFS
p_std_12405 = np.array([0.000, 99.916, 199.919, 399.860, 599.833, 799.673, 999.566]) #Torr
V_uut_12405 = np.array([0.0000, 0.9991, 1.9993, 3.9987, 5.9985, 7.9968, 9.9956]) #V
full_scale_p_12405 = 999.9 #Torr
full_scale_V_12405 = 10 #V


# 762479 is the Setra sensor, installed as P1 in BFS
p_std_762479 = np.array([0.5262, 5.0001, 9.9999, 15.0000, 20.0000, 24.9999, 30.0000,\
                         35.0000, 39.9999, 44.9999, 49.9999]) #psi
V_uut_762479 = np.array([0.0526, 0.5001, 1.0001, 1.5002, 2.0002, 2.5002, 3.0002, 3.5001,
                     4.00002, 4.5003, 5.0000]) #V
dV_762479 = np.array([0.0526, 0.5001*0.023, 1.0001*0.006, 1.5002*0.014, 2.0002*0.009,
                      2.5002*0.007, 3.0002*0.006, 3.5001*0.003, 4.00002*0.005,
                      4.5003*0.007, 5.0000*0.001]) #V

# 762478 is the Setra sensor, installed as P0 in BFS
p_std_762478 = np.array([0.5238, 5.0001, 10.0000, 15.0000, 20.0001, 25.0000,
                         30.0000, 35.0000, 39.9999, 45.0000, 50.0000]) #psi
V_uut_762478 = np.array([0.0504, 0.5002, 1.0003, 1.5002, 2.0000, 2.5000, 2.9999, 3.4999,
                         3.9999, 4.4996, 4.9995]) #V


#618969 is the Omega sensor, installed as P3 in BFS
p_std_618969 = np.array([0.00, 25.00, 50.00]) #psi 
V_uut_618969 = np.array([np.mean([-0.042, -0.040]), np.mean([13.737, 13.739]), 27.601]) #mV

#618970 is the Omega sensor, installed as in BFS
p_std_618970 = np.array([0.00, 25.00, 50.00]) #psi 
V_uut_618970 = np.array([np.mean([-0.042, -0.042]), np.mean([15.581, 15.570]), 31.290]) #mV

#618970 is the Omega sensor, installed as in BFS
p_std_618972 = np.array([0.00, 25.00, 50.00]) #psi 
V_uut_618972 = np.array([np.mean([-0.164, -0.164]), np.mean([15.703, 15.707]), 31.684]) #mV


def vdc_line(x, m, q):
    '''Line converting sensor VDC output to pressure values
    '''
    return m*(x + q)

if __name__ == "__main__":
    #Fits and plots for MKS sensors 

    popt12404, pcov12404 = curve_fit(vdc_line, V_uut_12404, p_std_12404*from_torr_to_mbar, p0=[100., 0.])
    print(f'Parameters for MKS 12404 sensor:\
          Slope = {popt12404[0]} +/- {np.sqrt(pcov12404[0][0])},\
          Zero offset = {popt12404[1]} +/- {np.sqrt(pcov12404[1][1])}')
    popt12405, pcov12405 = curve_fit(vdc_line, V_uut_12405, p_std_12405*from_torr_to_mbar, p0=[100., 0.])
    print(f'Parameters for MKS 12405 sensor:\
          Slope = {popt12405[0]} +/- {np.sqrt(pcov12405[0][0])},\
          Zero offset = {popt12405[1]} +/- {np.sqrt(pcov12405[1][1])}')
    fig, axs = plt.subplots(2)
    fig.suptitle(r'Calibration points for MKS sensors and their linear fit')
    z = np.linspace(0, 10, 1000)
    axs[0].errorbar(V_uut_12404, p_std_12404*from_torr_to_mbar, color='red', marker='.', linestyle='', label='MKS, S/N 12404')
    axs[0].plot(z,vdc_line(z, *popt12404))
    axs[0].set(xlabel=r'VDC [V]', ylabel=r'$P_5$ [mbar]')
    axs[0].grid(True)
    axs[1].errorbar(V_uut_12405, p_std_12405*from_torr_to_mbar, color='blue', marker='.', linestyle='', label='MKS S/N 12405')
    axs[1].plot(z,vdc_line(z, *popt12405))
    axs[1].set(xlabel=r'VDC [V]', ylabel=r'$P_4$ [mbar]') #Check the pressure number!
    axs[1].grid(True)
    plt.legend()

    #Fits and plots for Setra sensor
    
    popt762479, pcov762479 = curve_fit(vdc_line, V_uut_762479, p_std_762479*from_psi_to_mbar, p0=[100.*from_psi_to_mbar, 0.])
    m_opt, q_opt = popt762479
    print(f'Parameters for Setra 762479:\
            Slope = {m_opt} +/- {np.sqrt(pcov762479[0][0])},\
            Zero offset = {q_opt} +/- {np.sqrt(pcov762479[1][1])}')
    popt762478, pcov762478 = curve_fit(vdc_line, V_uut_762478, p_std_762478*from_psi_to_mbar, p0=[100.*from_psi_to_mbar, 0.])
    m_opt, q_opt = popt762478
    print(f'Parameters for Setra 762478:\
        Slope = {m_opt} +/- {np.sqrt(pcov762478[0][0])},\
        Zero offset = {q_opt} +/- {np.sqrt(pcov762478[1][1])}')

    fig, axs = plt.subplots(2)
    fig.suptitle(r'Calibration points for Setra sensors and their linear fit')
    z = np.linspace(0, 5, 1000)
    axs[0].errorbar(V_uut_762479, p_std_762479*from_psi_to_mbar, color='red', marker='.', linestyle='', label='Setra, S/N 762479')
    axs[0].plot(z,vdc_line(z, *popt762479))
    axs[0].set(xlabel=r'VDC [V]', ylabel=r'$P_1$ [mbar]')
    axs[0].grid(True)
    plt.legend()
    axs[1].errorbar(V_uut_762478, p_std_762478*from_psi_to_mbar, color='blue', marker='.', linestyle='', label='Setra, S/N 762478')
    axs[1].plot(z,vdc_line(z, *popt762478))
    axs[1].set(xlabel=r'VDC [V]', ylabel=r'$P_0$ [mbar]') #Check the pressure number!
    axs[1].grid(True)
    plt.legend()

    #Fits and plots for Omega sensors
    popt618969, pcov618969 = curve_fit(vdc_line, V_uut_618969*1e-3, p_std_618969*from_psi_to_mbar, p0=[1e5, 0.])
    m_opt, q_opt = popt618969
    print(f'Parameters for Omega 618969:\
            Slope = {m_opt} +/- {np.sqrt(pcov618969[0][0])},\
            Zero offset = {q_opt} +/- {np.sqrt(pcov618969[1][1])}')
    popt618970, pcov618970 = curve_fit(vdc_line, V_uut_618970*1e-3, p_std_618970*from_psi_to_mbar, p0=[1e5, 0.])
    m_opt, q_opt = popt618970
    print(f'Parameters for Omega 618970:\
            Slope = {m_opt} +/- {np.sqrt(pcov618970[0][0])},\
            Zero offset = {q_opt} +/- {np.sqrt(pcov618970[1][1])}')
    popt618972, pcov618972 = curve_fit(vdc_line, V_uut_618972*1e-3, p_std_618972*from_psi_to_mbar, p0=[1e5, 0.])
    m_opt, q_opt = popt618972
    print(f'Parameters for Omega 618972:\
            Slope = {m_opt} +/- {np.sqrt(pcov618972[0][0])},\
            Zero offset = {q_opt} +/- {np.sqrt(pcov618972[1][1])}')


    fig, axs = plt.subplots(3)
    fig.suptitle(r'Calibration points for Omega sensors and their linear fit')
    z = np.linspace(0, 30e-3, 1000)
    axs[0].plot(z,vdc_line(z, *popt618969))
    axs[0].set(xlabel=r'VDC [V]', ylabel=r'$P_3$ [mbar]') #Check the pressure number!
    axs[0].grid(True)
    axs[0].errorbar(V_uut_618969*1e-3, p_std_618969*from_psi_to_mbar, marker='.', linestyle='', label='Omega 618969')
    axs[1].plot(z,vdc_line(z, *popt618970))
    axs[1].set(xlabel=r'VDC [V]', ylabel=r'$P?$ [mbar]') #Check the pressure number!
    axs[1].grid(True)
    axs[1].errorbar(V_uut_618970*1e-3, p_std_618970*from_psi_to_mbar, marker='.', linestyle='', label='Omega 618970')
    axs[2].plot(z,vdc_line(z, *popt618972))
    axs[2].set(xlabel=r'VDC [V]', ylabel=r'$P?$ [mbar]') #Check the pressure number!
    axs[2].grid(True)
    axs[2].errorbar(V_uut_618972*1e-3, p_std_618972*from_psi_to_mbar, marker='.', linestyle='', label='Omega 618972')
    
    plt.legend()
    plt.show()