import numpy as np
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

"""
    Simple linear fit for computing the slope of the calibration curve of pressure
    sensors used in BFS. 
"""

#Calibration data 
V_12404 = np.array([0.0000, 0.9992, 1.9992, 3.9986, 5.9983, 7.9967, 9.9957]) #V
p_uut_12404 = np.array([0.0000, 99.9108, 199.9021, 399.8435, 599.8173, 799.6620, 999.5547]) #Torr
dp_uut_12404 = np.array([0.0000, 99.9108*(0.0050), 199.9021*(0.0086),
                         399.8435*(0.0040), 599.8173*(0.0026), 799.6620*(0.0014),
                         999.5547*(0.0012)]) #Torr
full_scale_p_12404 = 999.9 #Torr
full_scale_V_12404 = 10 #V

from_psi_to_mbar = 68.9476 #mbar/psi
V_762479 = np.array([0.0526, 0.5001, 1.0001, 1.5002, 2.0002, 2.5002, 3.0002, 3.5001,
                     4.00002, 4.5003, 5.0000])
dV_762479 = np.array([0.0526, 0.5001*0.023, 1.0001*0.006, 1.5002*0.014, 2.0002*0.009,
                      2.5002*0.007, 3.0002*0.006, 3.5001*0.003, 4.00002*0.005,
                      4.5003*0.007, 5.0000*0.001])
p_uut_762479 = np.array([0.5262, 5.0001, 9.9999, 15.0000, 20.0000, 24.9999, 30.0000,\
                         35.0000, 39.9999, 44.9999, 49.9999])

def vdc_line(x, m, q):
    return m*(x + q)

if __name__ == "__main__":
    popt, pcov = curve_fit(vdc_line, V_762479, p_uut_762479*from_psi_to_mbar, p0=[100.*from_psi_to_mbar, 0.])
    m_opt, q_opt = popt
    print(f'Slope = {m_opt} +/- {np.sqrt(pcov[0][0])}, Zero offset = {q_opt} +/- {np.sqrt(pcov[1][1])}')

    plt.figure()
    z = np.linspace(0, 5, 1000)
    plt.plot(z,vdc_line(z, *popt))
    plt.xlabel('VDC output [V]')
    plt.ylabel('Pressure [mbar]')
    plt.errorbar(V_762479, p_uut_762479*from_psi_to_mbar, xerr=dV_762479, marker='.', linestyle='')

    plt.show()