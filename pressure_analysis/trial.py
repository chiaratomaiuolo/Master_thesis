import argparse
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit
from uncertainties import unumpy, ufloat

from pressure_analysis.models import expo, alpha_expo_scale, double_expo, empty_AC_exp
from pressure_analysis.labviewdatareading import plot_with_residuals


""" This script is done for preparing a plot for the PM 2024 poster and for my
    Master Thesis
"""

# Loading data
#AC DME filled without epoxy samples
V_gas = 165688 #mm^3
t0, T_Julabo0, T2_0, dT2_0,  T5_0,  dT5_0, T6_0, dT6_0, P4_0, dP4_0, P5_0, dP5_0 = np.loadtxt('empty_AC_epoxy_sample_data_with_rms.txt', unpack=True)
t1, T_Julabo1, T2_0, dT2_1,  T5_1,  dT5_1, T6_1, dT6_1, P4_1, dP4_1, P5_1, dP5_1  = np.loadtxt('first_epoxy_sample_data.txt', unpack=True)
V1 = 8522 #mm**3
S1 = 10972 #mm**2
t2, T_Julabo2, T2_2, dT2_2,  T5_2,  dT5_2, T6_2, dT6_2, P4_2, dP4_2, P5_2, dP5_2 = np.loadtxt('second_epoxy_sample_data.txt', unpack=True)
V2 = 3305 #mm**3
S2 = 10333 #mm**2
t3, T_Julabo3, T2_3, dT2_3,  T5_3,  dT5_3, T6_3, dT6_3, P4_3, dP4_3, P5_3, dP5_3 = np.loadtxt('third_epoxy_sample_data.txt', unpack=True)
V3 = 9239 #mm**3
S3 = 4893 #mm**2
tGPD, T_JulaboGPD, T2_GPD, dT2_GPD, T5_GPD, dT5_GPD, T6_GPD, dT6_GPD, P4_GPD, dP4_GPD, P5_GPD, dP5_GPD = np.loadtxt('GPD_data.txt', unpack=True)

# Defining unumpy.arrays with nominal value and uncertainties
T_Julabo0 = ufloat(22, 0.1)

P4_0 = unumpy.uarray(P4_0, np.full(len(P4_0), 0.12))
T5_0 = unumpy.uarray(T5_0,  np.full(len(T5_0), 0.1))
T5_0 = T5_0+0.16*(T6_0-T5_0)/1.16 #°C

P4_1 = unumpy.uarray(P4_1, np.full(len(P4_1), 0.12))
T5_1 = unumpy.uarray(T5_1,  np.full(len(T5_1), 0.1))
T5_1 = T5_1+0.16*(T6_1-T5_1)/1.16 #°C

P4_2 = unumpy.uarray(P4_2, np.full(len(P4_2), 0.12))
T5_2 = unumpy.uarray(T5_2,  np.full(len(T5_2), 0.1))
T5_2 = T5_2+0.16*(T6_2-T5_2)/1.16 #°C

P4_3 = unumpy.uarray(P4_3, np.full(len(P4_3), 0.12))
T5_3 = unumpy.uarray(T5_3,  np.full(len(T5_3), 0.1))
T5_3 = T5_3+0.16*(T6_3-T5_3)/1.16 #°C

P4_GPD = unumpy.uarray(P4_GPD, np.full(len(P4_GPD), 0.12))
T2_GPD = unumpy.uarray(T2_GPD,  np.full(len(T2_GPD), 0.1))

#Computing the equivalent pressures
P_eq0 = (((P4_0*100)/(T5_0+273.15))*(T_Julabo0+273.15))/100 #mbar
P_eq1 = (((P4_1*100)/(T5_1+273.15))*(T_Julabo1+273.15))/100 #mbar
P_eq2 = (((P4_2*100)/(T5_2+273.15))*(T_Julabo2+273.15))/100 #mbar
P_eq3 = (((P4_3*100)/(T5_3+273.15))*(T_Julabo3+273.15))/100 #mbar
P_eqGPD =  (((P5_GPD*100)/(T2_GPD+273.15))*(T_JulaboGPD+273.15))/100 #mbar

def line(x, m, q):
    return m*x+q

#Fitting the curves
popt0, pcov0 = curve_fit(expo, t0, unumpy.nominal_values(P_eq0), sigma=unumpy.std_devs(P_eq0), p0=[1201., 5., 7.])
diag0 = np.sqrt(np.diag(pcov0))
print(f'Zero dataset optimal parameters: {popt0} +/- {np.sqrt(np.diag(pcov0))}')
popt1, pcov1 = curve_fit(alpha_expo_scale, t1, unumpy.nominal_values(P_eq1), sigma=unumpy.std_devs(P_eq1), p0=[1200., 627., 0.5147, 87])
diag1 = np.sqrt(np.diag(pcov1))
print(f'First dataset optimal parameters: {popt1} +/- {np.sqrt(np.diag(pcov1))}')
popt2, pcov2 = curve_fit(alpha_expo_scale, t2, unumpy.nominal_values(P_eq2), sigma=unumpy.std_devs(P_eq2), p0=[1199., 134., 0.72, 49/24])
diag2 = np.sqrt(np.diag(pcov2))
print(f'Second dataset optimal parameters: {popt2} +/- {np.sqrt(np.diag(pcov2))}')
popt3, pcov3 = curve_fit(alpha_expo_scale, t3, unumpy.nominal_values(P_eq3), sigma=unumpy.std_devs(P_eq3), p0=[1199., 1000., 0.48, 7873/24])
diag3 = np.sqrt(np.diag(pcov3))
print(f'Third dataset optimal parameters: {popt3} +/- {np.sqrt(np.diag(pcov3))}')
poptGPD, pcovGPD = curve_fit(alpha_expo_scale, tGPD, unumpy.nominal_values(P_eqGPD), sigma=unumpy.std_devs(P_eqGPD), p0=[1199., 1000., 0.48, 7873/24])
diagGPD = np.sqrt(np.diag(pcov3))
print(f'Third dataset optimal parameters: {poptGPD} +/- {np.sqrt(np.diag(pcovGPD))}')

#Plotting the dataset without epoxy samples - single exponential
#Computing chi square
chi_2 = (((unumpy.nominal_values(P_eq0) - expo(t0, *popt0))/unumpy.std_devs(P_eq0))**2).sum()
print(f'chi2/ndof for 0 set= {chi_2:.1f}/{len(P_eq0)-3}')
print(f'Asymptotic pressure: {expo(1000, *popt0)}')

t = np.linspace(0, max(t0), 5000)
plt.figure('Fit without epoxy samples')
plt.errorbar(t0, unumpy.nominal_values(P_eq0), yerr=unumpy.std_devs(P_eq0), marker='.', linestyle='')
plt.plot(t, expo(t, *popt0), color=plt.gca().lines[-1].get_color())
plt.xlabel('time from filling [days]')
plt.ylabel(r'$\hat{p}_{\text{eq,24h}}$ [mbar]')
plt.annotate(
    r'$p(t) = p_0 - \Delta_p(1- \exp{-\frac{t}{\tau}})$' + '\n' + f'$p_0={popt0[0]:.2f} \pm {diag0[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_p={popt0[1]:.0f} \pm {diag0[1]:.0f}$ [mbar],' + '\n' + fr'$\tau={popt0[2]:.0f} \pm {diag0[2]:.0f}$ [days]' + '\n' + rf'$\chi^2$/ndof = {chi_2:.1f}/{len(P_eq0)-3}',
    xy=(-0.2, 1192.3), xycoords='data',
    xytext=(0, 0), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle"))
plt.grid()


plt.show()

# Plotting GPD with stretched exponential

#Computing chi square
chi_2 = (((unumpy.nominal_values(P_eqGPD) - alpha_expo_scale(tGPD, *poptGPD))/unumpy.std_devs(P_eqGPD))**2).sum()
print(f'chi2/ndof for 0 set= {chi_2:.1f}/{len(P_eqGPD)-4}')

fig, ax = plt.subplots(2)
t = np.linspace(0, max(tGPD), 5000)
ax[0].errorbar(tGPD, unumpy.nominal_values(P_eqGPD), yerr=unumpy.std_devs(P_eqGPD), marker='.', linestyle='', color='tab:purple')
ax[0].plot(t, alpha_expo_scale(t, *poptGPD), color='tab:purple')
ax[0].set(xlabel='time from filling [days]', ylabel=r'$\hat{p}_{\text{eq,24h}}$ [mbar]')

ax[0].annotate(
    r'$p(t) = p_0 - \Delta_p(1- \exp{-\left(\frac{t}{\tau}\right)^{\alpha}})$' + '\n' + f'$p_0={popt1[0]:.2f} \pm {diag1[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_p={popt1[1]:.0f} \pm {diag1[1]:.0f}$ [mbar],' + '\n' + fr'$\alpha={popt1[2]:.3f} \pm {diag1[2]:.3f}$,'+ '\n' + fr'$\tau={popt1[3]:.1f} \pm {diag1[3]:.1f}$ [days]' + '\n' + rf'$\chi^2$/ndof = {chi_2:.1f}/{len(P_eq1)-4}',
    xy=(10, 800), xycoords='data',
    xytext=(0, 0), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec='tab:purple'),
    arrowprops=dict(arrowstyle="->", ec='tab:purple',
                    connectionstyle="angle"))

ax[0].grid()

res_norm = (unumpy.nominal_values(P_eqGPD) - alpha_expo_scale(tGPD, *poptGPD))/unumpy.std_devs(P_eqGPD)
ax[1].errorbar(tGPD, res_norm, yerr=unumpy.std_devs(P_eqGPD), marker='.', linestyle='', color='tab:purple')
ax[1].set(xlabel='time from filling [days]', ylabel=r'Normalized residuals [# $\sigma_{p}$]')
ax[1].grid()
plt.show()



# Fitting the first sample with the single exponential
popt, pcov = curve_fit(expo, t1, unumpy.nominal_values(P_eq1), sigma=unumpy.std_devs(P_eq1), p0=[1200., 230., 7])
diag = np.sqrt(np.diag(pcov))

#Computing chi square
chi_2 = (((unumpy.nominal_values(P_eq1) - expo(t1, *popt))/unumpy.std_devs(P_eq1))**2).sum()
print(f'chi2/ndof for 0 set= {chi_2:.1f}/{len(P_eq1)-3}')
'''
plt.figure()
t = np.linspace(0, max(t1), 5000)
plt.errorbar(t1, unumpy.nominal_values(P_eq1), yerr=unumpy.std_devs(P_eq1), marker='o', linestyle='', color='tab:blue', label='Data sampled')
plt.plot(t, alpha_expo_scale(t, *popt1), color='tab:blue', label='Stretched exponential fit')
plt.plot(t, expo(t, *popt), color='tab:red', label='Single exponential fit')
plt.tick_params(axis='x', labelsize=20)
plt.tick_params(axis='y', labelsize=20)
plt.xlabel('time from filling [days]', fontsize='20')
plt.ylabel(r'$\hat{p}_{\text{eq,24h}}$ [mbar]', fontsize='20')
plt.grid()
plt.legend(fontsize='20')
'''
# Plt with res
fig, ax = plt.subplots(2)
t = np.linspace(0, max(t1), 5000)
ax[0].errorbar(t1, unumpy.nominal_values(P_eq1), yerr=unumpy.std_devs(P_eq1), marker='o', linestyle='', color='tab:blue', label='Data sampled')
ax[0].plot(t, alpha_expo_scale(t, *popt1), color='tab:blue', label='Stretched exponential fit')
ax[0].plot(t, expo(t, *popt), color='tab:red', label='Single exponential fit')
ax[0].tick_params(axis='x', labelsize=20)
ax[0].tick_params(axis='y', labelsize=20)
ax[0].set_xlabel('time from filling [days]', fontsize=20)
ax[0].set_ylabel(r'$\hat{p}_{\text{eq,24h}}$ [mbar]', fontsize=20)

plt.grid()
ax[0].legend(fontsize='20')

ax[0].grid()

res = (unumpy.nominal_values(P_eq1) - expo(t1, *popt))
ax[1].errorbar(t1, res, yerr=unumpy.std_devs(P_eq1), marker='.', linestyle='', color='tab:orange')
ax[1].set_xlabel('time from filling [days]', fontsize=20)
ax[1].set_ylabel('Residuals [mbar]', fontsize=20)
#ax[1].set(xlabel='time from filling [days]', ylabel='Residuals [mbar]')
ax[1].grid()

'''
plt.annotate(
    r'$p(t) = p_0 - \Delta_p(1- \exp{-\frac{t}{\tau}})$' + '\n' + f'$p_0={popt[0]:.2f} \pm {diag[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_p={popt[1]:.0f} \pm {diag[1]:.0f}$ [mbar],' + '\n' + fr'$\tau={popt[2]:.0f} \pm {diag[2]:.0f}$ [days]' + '\n' + rf'$\chi^2$/ndof = {chi_2:.1f}/{len(P_eq1)-3}',
    xy=(8, 1100), xycoords='data',
    xytext=(0, 0), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec='tab:orange'),
    arrowprops=dict(arrowstyle="->", ec='tab:orange',
                    connectionstyle="angle"), fontsize='20')
plt.grid()
'''
# Res
res = (unumpy.nominal_values(P_eq1) - expo(t1, *popt))
res2 = (unumpy.nominal_values(P_eq1) - alpha_expo_scale(t1, *popt1))
ax[1].errorbar(t1, res, yerr=unumpy.std_devs(P_eq1), marker='o', linestyle='', color='tab:red', label='Single exponential')
ax[1].errorbar(t1, res2, yerr=unumpy.std_devs(P_eq1), marker='^', linestyle='', color='tab:blue', label='Stretched exponential')
ax[1].set(xlabel='time from filling [days]', ylabel='Residuals [mbar]')

ax[1].tick_params(axis='x', labelsize=20)
ax[1].tick_params(axis='y', labelsize=20)
ax[1].legend(fontsize='20')
ax[1].grid()

plt.show()

# Plt with res
fig, ax = plt.subplots(2)
t = np.linspace(0, max(t1), 5000)
ax[0].errorbar(t1, unumpy.nominal_values(P_eq1), yerr=unumpy.std_devs(P_eq1), marker='o', linestyle='', color='tab:blue')
ax[0].plot(t, expo(t, *popt), color='tab:blue')
ax[0].set_xlabel('time from filling [days]', fontsize=15)
ax[0].set_ylabel(r'$\hat{p}_{\text{eq,24h}}$ [mbar]', fontsize=15)
ax[0].tick_params(axis='x', labelsize=15)
ax[0].tick_params(axis='y', labelsize=15)

ax[0].annotate(
    r'$p(t) = p_0 - \Delta_p(1- \exp{-\frac{t}{\tau}})$' + '\n' + f'$p_0={popt[0]:.2f} \pm {diag[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_p={popt[1]:.0f} \pm {diag[1]:.0f}$ [mbar],' + '\n' + fr'$\tau={popt[2]:.0f} \pm {diag[2]:.0f}$ [days]' + '\n' + rf'$\chi^2$/ndof = {chi_2:.1f}/{len(P_eq1)-3}',
    xy=(10, 1100), xycoords='data',
    xytext=(0, 0), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec='tab:blue'),
    arrowprops=dict(arrowstyle="->", ec='tab:blue',
                    connectionstyle="angle"))

ax[0].grid()

res = (unumpy.nominal_values(P_eq1) - expo(t1, *popt))
ax[1].errorbar(t1, res, yerr=unumpy.std_devs(P_eq1), marker='o', linestyle='', color='tab:blue')
ax[1].set_xlabel('time from filling [days]', fontsize=15)
ax[1].set_ylabel('Residuals [mbar]', fontsize=15)
ax[1].tick_params(axis='x', labelsize=15)
ax[1].tick_params(axis='y', labelsize=15)
ax[1].grid()


plt.show()


# Plotting first sample with stretched exponential

#Computing chi square
chi_2 = (((unumpy.nominal_values(P_eq1) - alpha_expo_scale(t1, *popt1))/unumpy.std_devs(P_eq1))**2).sum()
print(f'chi2/ndof for 0 set= {chi_2:.1f}/{len(P_eq1)-4}')

fig, ax = plt.subplots(2)
t = np.linspace(0, max(t1), 5000)
ax[0].errorbar(t1, unumpy.nominal_values(P_eq1), yerr=unumpy.std_devs(P_eq1), marker='.', linestyle='', color='tab:orange')
ax[0].plot(t, alpha_expo_scale(t, *popt1), color='tab:orange')
ax[0].set(xlabel='time from filling [days]', ylabel=r'$\hat{p}_{\text{eq,24h}}$ [mbar]')

ax[0].annotate(
    r'$p(t) = p_0 - \Delta_p(1- \exp{-\left(\frac{t}{\tau}\right)^{\alpha}})$' + '\n' + f'$p_0={popt1[0]:.2f} \pm {diag1[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_p={popt1[1]:.0f} \pm {diag1[1]:.0f}$ [mbar],' + '\n' + fr'$\alpha={popt1[2]:.3f} \pm {diag1[2]:.3f}$,'+ '\n' + fr'$\tau={popt1[3]:.1f} \pm {diag1[3]:.1f}$ [days]' + '\n' + rf'$\chi^2$/ndof = {chi_2:.1f}/{len(P_eq1)-4}',
    xy=(10, 1100), xycoords='data',
    xytext=(0, 0), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec='tab:orange'),
    arrowprops=dict(arrowstyle="->", ec='tab:orange',
                    connectionstyle="angle"))

ax[0].grid()

res = (unumpy.nominal_values(P_eq1) - alpha_expo_scale(t1, *popt1))
ax[1].errorbar(t1, res, yerr=unumpy.std_devs(P_eq1), marker='.', linestyle='', color='tab:orange')
ax[1].set(xlabel='time from filling [days]', ylabel='Residuals [mbar]')
ax[1].grid()
plt.show()

# Plotting second sample with stretched exponential

#Computing chi square
chi_2 = (((unumpy.nominal_values(P_eq2) - alpha_expo_scale(t2, *popt2))/unumpy.std_devs(P_eq2))**2).sum()
print(f'chi2/ndof for 0 set= {chi_2:.1f}/{len(P_eq2)-4}')

fig, ax = plt.subplots(2)
t = np.linspace(0, max(t2), 5000)
ax[0].errorbar(t2, unumpy.nominal_values(P_eq2), yerr=unumpy.std_devs(P_eq2), marker='.', linestyle='', color='tab:green')
ax[0].plot(t, alpha_expo_scale(t, *popt2), color='tab:green')
ax[0].set(xlabel='time from filling [days]', ylabel=r'$\hat{p}_{\text{eq,24h}}$ [mbar]')

ax[0].annotate(
    r'$p(t) = p_0 - \Delta_p(1- \exp{-\left(\frac{t}{\tau}\right)^{\alpha}})$' + '\n' + f'$p_0={popt2[0]:.2f} \pm {diag2[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_p={popt2[1]:.0f} \pm {diag2[1]:.0f}$ [mbar],' + '\n' + fr'$\alpha={popt2[2]:.3f} \pm {diag2[2]:.3f}$,'+ '\n' + fr'$\tau={popt2[3]:.2f} \pm {diag2[3]:.2f}$ [days]' + '\n' + rf'$\chi^2$/ndof = {chi_2:.1f}/{len(P_eq2)-4}',
    xy=(4.5, 1140), xycoords='data',
    xytext=(0, 0), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec='tab:green'),
    arrowprops=dict(arrowstyle="->", ec='tab:green',
                    connectionstyle="angle"))

ax[0].grid()

res = (unumpy.nominal_values(P_eq2) - alpha_expo_scale(t2, *popt2))
ax[1].errorbar(t2, res, yerr=unumpy.std_devs(P_eq2), marker='.', linestyle='', color='tab:green')
ax[1].set(xlabel='time from filling [days]', ylabel=r'Residuals [mbar]')
ax[1].grid()
plt.show()


# Plotting third sample with stretched exponential

#Computing chi square
chi_2 = (((unumpy.nominal_values(P_eq3) - alpha_expo_scale(t3, *popt3))/unumpy.std_devs(P_eq3))**2).sum()
print(f'chi2/ndof for 0 set= {chi_2:.1f}/{len(P_eq3)-4}')

t = np.linspace(0, max(t3), 5000)
fig, ax = plt.subplots(2)
ax[0].errorbar(t3, unumpy.nominal_values(P_eq3), yerr=unumpy.std_devs(P_eq3), marker='.', linestyle='', color='tab:red')
ax[0].plot(t, alpha_expo_scale(t, *popt3), color='tab:red')
ax[0].set(xlabel='time from filling [days]', ylabel=r'$\hat{p}_{\text{eq,24h}}$ [mbar]')

ax[0].annotate(
    r'$p(t) = p_0 - \Delta_p(1- \exp{-\left(\frac{t}{\tau}\right)^{\alpha}})$' + '\n' + f'$p_0={popt3[0]:.2f} \pm {diag3[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_p={popt3[1]:.0f} \pm {diag3[1]:.0f}$ [mbar],' + '\n' + fr'$\alpha={popt3[2]:.3f} \pm {diag3[2]:.3f}$,'+ '\n' + fr'$\tau={popt3[3]:.0f} \pm {diag3[3]:.0f}$ [days]' + '\n' + rf'$\chi^2$/ndof = {chi_2:.1f}/{len(P_eq3)-4}',
    xy=(30, 1130), xycoords='data',
    xytext=(0, 0), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec='tab:red'),
    arrowprops=dict(arrowstyle="->", ec='tab:red',
                    connectionstyle="angle"))

ax[0].grid()

res = (unumpy.nominal_values(P_eq3) - alpha_expo_scale(t3, *popt3))
ax[1].errorbar(t3, res, yerr=unumpy.std_devs(P_eq3), marker='.', linestyle='', color='tab:red')
ax[1].set(xlabel='time from filling [days]', ylabel=r'Residuals [mbar]')
ax[1].grid()
plt.show()


# Summary plot
fig, ax = plt.subplots()
t = np.linspace(0, max(t0), 5000)
ax.errorbar(t0, unumpy.nominal_values(P_eq0), yerr=unumpy.std_devs(P_eq0), marker='.', linestyle='')
ax.plot(t, expo(t, *popt0), color=plt.gca().lines[-1].get_color())
ax.annotate('Empty AC',
            xy=(7, 1200), xycoords='data',
            textcoords='data', va="center",
            bbox=dict(boxstyle="round", fc=plt.gca().lines[-1].get_color(), ec="none", alpha=0.4))

ax.annotate(
    fr'$p_0={popt0[0]:.2f} \pm {diag0[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_p={popt0[1]:.0f} \pm {diag0[1]:.0f}$ [mbar],'+ '\n' + fr'$\alpha=1$ (standard exp),'+ '\n' + fr'$\tau={popt0[2]:.0f} \pm {diag0[2]:.0f}$ [days]',
    xy=(t[-1], expo(t[-1],*popt0)), xycoords='data',
    xytext=(15, 1170), textcoords='data',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle,angleA=-90,angleB=0"))


t = np.linspace(0, max(t1), 5000)
ax.errorbar(t1, unumpy.nominal_values(P_eq1), yerr=unumpy.std_devs(P_eq1), marker='.', linestyle='')
ax.plot(t, alpha_expo_scale(t, *popt1), color=plt.gca().lines[-1].get_color())
ax.annotate(fr'$V_1$ = {V1} mm$^3$' + '\n' + fr'$S_1$ = {S1} mm$^2$' +'\n' + fr'$\frac{{S_1}}{{V_1}}$ = {S1/V1:.2f} $\frac{{1}}{{\text{{mm}}}}$',
            xy=(17, 995), xycoords='data',
            textcoords='data', va="center",
            bbox=dict(boxstyle="round", fc=plt.gca().lines[-1].get_color(), ec="none", alpha=0.4))

ax.annotate(
    fr'$P_0={popt1[0]:.2f} \pm {diag1[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_P={popt1[1]:.0f} \pm {diag1[1]:.0f}$ [mbar],'+ '\n' + fr'$\alpha={popt1[2]:.4f} \pm {diag1[2]:.4f}$,'+ '\n' + fr'$\tau={popt1[3]:.0f} \pm {diag1[3]:.0f}$ [days]',
    xy=(10.1, 1022.8), xycoords='data',
    xytext=(27.5, 1000), textcoords='data',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle,angleA=0,angleB=90"))

t = np.linspace(0, max(t2), 5000)
ax.errorbar(t2, unumpy.nominal_values(P_eq2), yerr=unumpy.std_devs(P_eq2), marker='.', linestyle='')
ax.plot(t, alpha_expo_scale(t, *popt2), color=plt.gca().lines[-1].get_color())

ax.annotate(
    fr'$P_0={popt2[0]:.2f} \pm {diag2[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_P={popt2[1]:.1f} \pm {diag2[1]:.1f}$ [mbar],'+ '\n' + fr'$\alpha={popt2[2]:.3f} \pm {diag2[2]:.3f}$,'+ '\n' + fr'$\tau={popt2[3]:.2f} \pm {diag2[3]:.2f}$ [days]',
    xy=(t[-1], alpha_expo_scale(t[-1], *popt2)), xycoords='data',
    xytext=(15.5, 1110), textcoords='data',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle,angleA=0,angleB=90"))

ax.annotate(fr'$V_2$ = {V2} mm$^3$' + '\n' + fr'$S_2$ = {S2} mm$^2$' +'\n' + fr'$\frac{{S_2}}{{V_2}}$ = {S2/V2:.2f} $\frac{{1}}{{\text{{mm}}}}$',
            xy=(8, 1070), xycoords='data',
            textcoords='data', va="center",
            bbox=dict(boxstyle="round", fc=plt.gca().lines[-1].get_color(), ec="none", alpha=0.4))
t = np.linspace(0, max(t3), 5000)
ax.errorbar(t3, unumpy.nominal_values(P_eq3), yerr=unumpy.std_devs(P_eq3), marker='.', linestyle='')
ax.plot(t, alpha_expo_scale(t, *popt3), color=plt.gca().lines[-1].get_color())

ax.annotate(
    fr'$P_0={popt3[0]:.2f} \pm {diag3[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_P={popt3[1]:.0f} \pm {diag3[1]:.0f}$ [mbar],'+ '\n' + fr'$\alpha={popt3[2]:.3f} \pm {diag3[2]:.3f}$,'+ '\n' + fr'$\tau={popt3[3]:.0f} \pm {diag3[3]:.0f}$ [days]',
    xy=(t[-1], alpha_expo_scale(t[-1], *popt3)), xycoords='data',
    xytext=(35,1110), textcoords='data',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle,angleA=0,angleB=90"))

ax.annotate(fr'$V_3$ = {V3} mm$^3$' + '\n' + fr'$S_3$ = {S3} mm$^2$' +'\n' + fr'$\frac{{S_3}}{{V_3}}$ = {S3/V3:.2f} $\frac{{1}}{{\text{{mm}}}}$',
            xy=(30, 1080), xycoords='data',
            textcoords='data', va="center",
            bbox=dict(boxstyle="round", fc=plt.gca().lines[-1].get_color(), ec="none", alpha=0.4))
ax.set_xlabel('time from filling [days]', fontsize='20')
ax.set_ylabel(r'$\hat{p}_{\text{eq,24h}}$ [mbar]', fontsize='20')

ax.grid()
#plt.errorbar(tGPD[:-1], P_eqGPD[:-1], marker='.', linestyle='')

#Computing V**(1/3)/S**(1/2)
V = np.array([V1, V2, V3])
S = np.array([S1, S2, S3])
adim_metric = (V**(1/3))/(S**(1/2))

'''
plt.figure('V/S')
plt.errorbar(V/S, [popt1[2], popt2[2], popt3[2]], yerr=[diag1[2], diag2[2], diag3[2]], marker='.', linestyle='')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('V/S')
plt.ylabel(r'$\alpha$')
'''

def power_law(x, C, l):
    return C*(x**(l))

def sigmoid(x, x0, k):
    return 100 / (1 + np.exp(-k*(x-x0)))

#PARAMETERS SUMMARY PLOTS

z = np.linspace(min(S/V), max(S/V), 100)
# label=r'$\alpha(\frac{S}{V}) = m\cdot \frac{S}{V} + q$'
fig, axs = plt.subplots(nrows=1, ncols=3)
#plt.subplots_adjust(left=0.3, bottom=0.07, right=None, top=0.99, wspace=None, hspace=0.3)
popt, pcov = curve_fit(line, S/V, [popt1[2], popt2[2], popt3[2]]) #delta p
diag = np.sqrt(np.diag(pcov))
print(popt, np.sqrt(np.diag(pcov)))
axs[0].errorbar(S/V, [popt1[2], popt2[2], popt3[2]], yerr=[diag1[2], diag2[2], diag3[2]], marker='^', linestyle='', color='tab:brown')
#axs[0].errorbar(1/2.122, 0.478, marker='*', linestyle='', color='tab:green', label='GPD 38')
axs[0].plot(z, line(z, *popt), color='tab:purple', label=fr'$m = {popt[0]:.2f} \pm {diag[0]:.2f}$ [mm]' + '\n' + fr'$q = {popt[1]:.2f} \pm {diag[1]:.2f}$')
axs[0].text(0.5, 0.73, fr'$\alpha(\frac{{S}}{{V}}) = m\cdot \frac{{S}}{{V}} + q$', bbox=dict(boxstyle="round", fc='none', ec="black"))
axs[0].grid()
axs[0].set_xlabel(r'$S_{sample}/V_{sample}$ [1/mm]', fontsize='20')
axs[0].set_ylabel(r'$\alpha$', fontsize='20')
axs[0].tick_params(axis='x', labelsize=20)
axs[0].tick_params(axis='y', labelsize=20)
axs[0].legend(loc='lower right')
'''
axs[0].annotate(
    fr'$\alpha(\frac{{S}}{{V}}) = m\cdot \frac{{S}}{{V}} + q$' + '\n' + fr'$m = {popt[0]:.2f} \pm {diag[0]:.2f}$ [mm]' + '\n' + fr'$q = {popt[1]:.2f} \pm {diag[1]:.2f}$',
    xy=(min(S/V), popt1[2]), xycoords='data', size=15,
    xytext=(0, 40), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec='tab:orange')
)
'''


popt, pcov = curve_fit(power_law, S/V, [popt1[3], popt2[3], popt3[3]])
diag = np.sqrt(np.diag(pcov))
print(popt, np.sqrt(np.diag(pcov)))
axs[1].errorbar(S/V, [popt1[3], popt2[3], popt3[3]], yerr=[diag1[3], diag2[3], diag3[3]], marker='^', linestyle='', color='tab:brown')
#axs[1].errorbar(1/2.122, 629, yerr=90, marker='*', linestyle='', color='tab:green', label='GPD 38')
axs[1].plot(z, power_law(z, *popt), color='tab:purple', label= fr'$C = {popt[0]:.0f} \pm {diag[0]:.0f}$ [mm$^2$]' + '\n' + fr'${{\Gamma}} = {popt[1]:.2f} \pm {diag[1]:.2f}$')
axs[1].text(1.7, 440, fr'$\tau(\frac{{S}}{{V}}) = C\cdot \left(\frac{{S}}{{V}}\right)^{{\Gamma}}$', bbox=dict(boxstyle="round", fc='none', ec="black"))
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].grid()
axs[1].set_xlabel(r'$S_{sample}/V_{sample}$ [1/mm]', fontsize='20')
axs[1].set_ylabel(r'$\tau$ [days]', fontsize='20')
axs[1].tick_params(axis='x', labelsize=20)
axs[1].tick_params(axis='y', labelsize=20)
axs[1].legend(loc='lower left')
'''
axs[1].annotate(
    fr'$\tau(\frac{{S}}{{V}}) = C\cdot \left(\frac{{S}}{{V}}\right)^{{\Gamma}}$' + '\n' + fr'$C = {popt[0]:.0f} \pm {diag[0]:.0f}$ [mm$^2$]' + '\n' + fr'${{\Gamma}} = {popt[1]:.2f} \pm {diag[1]:.2f}$',
    xy=(min(S/V), popt1[3]), xycoords='data', size=15,
    xytext=(0, -80), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", color='tab:orange')
)
'''

deltas = np.array([popt1[1], popt2[1], popt3[1]])
P0s = np.array([popt1[0], popt2[0], popt3[0]])
relative_deltas = (deltas/P0s)*(100)
x = np.hstack([V/(V_gas-V), 1/20.208])
print(x)
y = np.hstack([relative_deltas, (245/800)*100])
popt, pcov = curve_fit(power_law, x, y, sigma=y*0.05)
print(popt, np.sqrt(np.diag(pcov)))
z = np.linspace(min(V/(V_gas-V)), max(V/(V_gas-V)), 1000)
axs[2].plot(z, power_law(z, *popt), color='tab:orange', label= fr'$D = {popt[0]:.0f} \pm {diag[0]:.0f}$' + '\n' + fr'${{\xi}} = {popt[1]:.2f} \pm {diag[1]:.2f}$')
axs[2].errorbar(V/(V_gas-V), relative_deltas, marker='^', linestyle='', color='tab:pink')
axs[2].text(0.02, 80, fr'$\frac{{\Delta_p}}{{P_0}}(\frac{{V_{{samples}}}}{{V_{{gas}}}}) = D\cdot \left(\frac{{V_{{samples}}}}{{V_{{gas}}}}\right)^{{\xi}}$', bbox=dict(boxstyle="round", fc='none', ec="black"))
#axs[2].errorbar(1/20.208, (245/800)*100, marker='*', linestyle='', label='GPD 38', color='tab:green')
axs[2].set_xlabel(r'$V_{samples}/V_{gas}$', fontsize='20')
axs[2].set_ylabel(r'$\frac{\Delta_P}{P_0}$ [%]', fontsize='20')
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_yticks(np.linspace(10, 100, 10), minor=True)
axs[2].set_xticks(np.array([0.02, 0.04, 0.06]), minor=False)

axs[2].legend(loc='best')

# Imposta i minor ticks per rendere la griglia più fitta
#ax.tick_params(np.logspace(10, 100, 10), minor=True)

axs[2].grid(True, which='both',axis='both')

#plt.savefig('/Users/chiara/Desktop/params.pdf')

plt.show()