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
full_scale_p = 999.9 #Torr
full_scale_V = 10 #V

def line(x, m, q):
    return m*x + q

if __name__ == "__main__":
    popt, pcov = curve_fit(line, V_12404, p_uut_12404, p0=[100., 0.])
    m_opt, q_opt = popt
    print(f'Slope = {m_opt} +/- {np.sqrt(pcov[0][0])}, Zero offset = {q_opt} +/- {np.sqrt(pcov[1][1])}')

    plt.figure()
    z = np.linspace(0, 10, 1000)
    plt.plot(z,line(z, *popt))
    plt.errorbar(V_12404, p_uut_12404, yerr=dp_uut_12404, marker='.', linestyle='')

    plt.show()