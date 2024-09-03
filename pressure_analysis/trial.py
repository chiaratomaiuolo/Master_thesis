import argparse
import numpy as np 
import matplotlib.pyplot as plt 

from scipy.optimize import curve_fit

from pressure_analysis.models import expo, alpha_expo_scale, double_expo, empty_AC_exp


""" This script is done for preparing a plot for the PM 2024 poster
"""

# Loading data
#AC DME filled without epoxy samples
V_gas = 165688 #mm^3
t0, T_Julabo0, T2_0, T5_0, T6_0, P4_0, P5_0 = np.loadtxt('empty_AC_epoxy_sample_data.txt', unpack=True)

t1, T_Julabo1, T2_1, T5_1, T6_1, P4_1, P5_1  = np.loadtxt('first_epoxy_sample_data.txt', unpack=True)
V1 = 8522 #mm**3
S1 = 10972 #mm**2
t2, T_Julabo2, T2_2, T5_2, T6_2, P4_2, P5_2 = np.loadtxt('second_epoxy_sample_data.txt', unpack=True)
V2 = 3305 #mm**3
S2 = 10333 #mm**2
t3, T_Julabo3, T2_3, T5_3, T6_3, P4_3, P5_3 = np.loadtxt('third_epoxy_sample_data.txt', unpack=True)
V3 = 9239 #mm**3
S3 = 4893 #mm**2
tGPD, T_JulaboGPD, T2_GPD, T5_GPD, T6_GPD, P4_GPD, P5_GPD = np.loadtxt('GPD_data.txt', unpack=True)

#Computing the equivalent pressures
P_eq0 = (((P4_0*100)/(T5_0+273.15))*(T_Julabo0+273.15))/100 #mbar
P_eq1 = (((P4_1*100)/(T5_1+273.15))*(T_Julabo1+273.15))/100 #mbar
P_eq2 = (((P4_2*100)/(T5_2+273.15))*(T_Julabo2+273.15))/100 #mbar
P_eq3 = (((P4_3*100)/(T5_3+273.15))*(T_Julabo3+273.15))/100 #mbar
P_eqGPD =  (((P5_GPD*100)/(T2_GPD+273.15))*(T_JulaboGPD+273.15))/100 #mbar

def line(x, m, q):
    return m*x+q

#Fitting the curves
popt0, pcov0 = curve_fit(expo, t0, P_eq0, p0=[1199., 2., 7.])
diag0 = np.sqrt(np.diag(pcov0))
print(f'Zero dataset optimal parameters: {popt0} +/- {np.sqrt(np.diag(pcov0))}')
popt1, pcov1 = curve_fit(alpha_expo_scale, t1, P_eq1, p0=[1199., 549., 0.53, 1470/24])
diag1 = np.sqrt(np.diag(pcov1))
print(f'First dataset optimal parameters: {popt1} +/- {np.sqrt(np.diag(pcov1))}')
popt2, pcov2 = curve_fit(alpha_expo_scale, t2, P_eq2, p0=[1199., 134., 0.72, 49/24])
diag2 = np.sqrt(np.diag(pcov2))
print(f'Second dataset optimal parameters: {popt2} +/- {np.sqrt(np.diag(pcov2))}')
popt3, pcov3 = curve_fit(alpha_expo_scale, t3, P_eq3, p0=[1199., 1000., 0.48, 7873/24])
diag3 = np.sqrt(np.diag(pcov3))
print(f'Third dataset optimal parameters: {popt3} +/- {np.sqrt(np.diag(pcov3))}')

# Checking pts
fig, ax = plt.subplots()
t = np.linspace(0, max(t0), 5000)
ax.errorbar(t0, P_eq0, marker='.', linestyle='')
ax.plot(t, expo(t, *popt0), color=plt.gca().lines[-1].get_color())
ax.annotate('Empty AC',
            xy=(t1[30], P_eq1[30]), xycoords='data',
            xytext=(45, 9), textcoords='offset points',
            size=18, va="center",
            bbox=dict(boxstyle="round", fc=plt.gca().lines[-1].get_color(), ec="none", alpha=0.4))
ax.annotate(
    fr'$P_0={popt0[0]:.2f} \pm {diag0[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_P={popt0[1]:.0f} \pm {diag0[1]:.0f}$ [mbar],'+ '\n' + fr'$\alpha=1$ (standard exp),'+ '\n' + fr'$\tau={popt0[2]:.0f} \pm {diag0[2]:.0f}$ [days]',
    xy=(t0[-1], P_eq0[-1]), xycoords='data', size=18,
    xytext=(40, -40), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle"))



t = np.linspace(0, max(t1), 5000)
ax.errorbar(t1, P_eq1, marker='.', linestyle='')
ax.plot(t, alpha_expo_scale(t, *popt1), color=plt.gca().lines[-1].get_color())
ax.annotate(fr'$V_1$ = {V1} mm$^3$' +'\n' + fr'$\frac{{S_1}}{{V_1}}$ = {S1/V1:.2f} $\frac{{1}}{{\text{{mm}}}}$',
            xy=(t1[729], P_eq1[729]), xycoords='data',
            xytext=(-160, -20), textcoords='offset points',
            size=18, va="center",
            bbox=dict(boxstyle="round", fc=plt.gca().lines[-1].get_color(), ec="none", alpha=0.4))
ax.annotate(
    fr'$P_0={popt1[0]:.2f} \pm {diag1[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_P={popt1[1]:.0f} \pm {diag1[1]:.0f}$ [mbar],'+ '\n' + fr'$\alpha={popt1[2]:.4f} \pm {diag1[2]:.4f}$,'+ '\n' + fr'$\tau={popt1[3]:.0f} \pm {diag1[3]:.0f}$ [days]',
    xy=(t1[-1], P_eq1[-1]), xycoords='data', size=18,
    xytext=(-20, -10), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle,angleA=0,angleB=-90"))
t = np.linspace(0, max(t2), 5000)
ax.errorbar(t2, P_eq2, marker='.', linestyle='')
ax.plot(t, alpha_expo_scale(t, *popt2), color=plt.gca().lines[-1].get_color())
ax.annotate(
    fr'$P_0={popt2[0]:.2f} \pm {diag2[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_P={popt2[1]:.1f} \pm {diag2[1]:.1f}$ [mbar],'+ '\n' + fr'$\alpha={popt2[2]:.3f} \pm {diag2[2]:.3f}$,'+ '\n' + fr'$\tau={popt2[3]:.2f} \pm {diag2[3]:.2f}$ [days]',
    xy=(t2[-1], P_eq2[-1]), xycoords='data', size=18,
    xytext=(150, -95), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle,angleA=0,angleB=90"))
ax.annotate(fr'$V_2$ = {V2} mm$^3$' +'\n' + fr'$\frac{{S_2}}{{V_2}}$ = {S2/V2:.2f} $\frac{{1}}{{\text{{mm}}}}$',
            xy=(t2[-1], P_eq2[-1]), xycoords='data',
            xytext=(10, 20), textcoords='offset points',
            size=18, va="center",
            bbox=dict(boxstyle="round", fc=plt.gca().lines[-1].get_color(), ec="none", alpha=0.4))
t = np.linspace(0, max(t3), 5000)
ax.errorbar(t3, P_eq3, marker='.', linestyle='')
ax.plot(t, alpha_expo_scale(t, *popt3), color=plt.gca().lines[-1].get_color())
ax.annotate(
    fr'$P_0={popt3[0]:.2f} \pm {diag3[0]:.2f}$ [mbar],' + '\n' + fr'$\Delta_P={popt3[1]:.0f} \pm {diag3[1]:.0f}$ [mbar],'+ '\n' + fr'$\alpha={popt3[2]:.3f} \pm {diag3[2]:.3f}$,'+ '\n' + fr'$\tau={popt3[3]:.0f} \pm {diag3[3]:.0f}$ [days]',
    xy=(t3[-1], P_eq3[-1]), xycoords='data', size=18,
    xytext=(-180, 60), textcoords='offset points',
    bbox=dict(boxstyle="round", fc="1", ec=plt.gca().lines[-1].get_color()),
    arrowprops=dict(arrowstyle="->", ec=plt.gca().lines[-1].get_color(),
                    connectionstyle="angle,angleA=0,angleB=90"))
ax.annotate(fr'$V_3$ = {V3} mm$^3$' +'\n' + fr'$\frac{{S_3}}{{V_3}}$ = {S3/V3:.2f} $\frac{{1}}{{\text{{mm}}}}$',
            xy=(t3[725], P_eq3[725]), xycoords='data',
            xytext=(45, 20), textcoords='offset points',
            size=18, va="center",
            bbox=dict(boxstyle="round", fc=plt.gca().lines[-1].get_color(), ec="none", alpha=0.4))
ax.set_xlabel('time from filling [days]', fontsize = 22)
ax.set_ylabel(r'$P_{eq}$ [mbar]', fontsize = 22)
ax.tick_params(axis='x', labelsize=22)
ax.tick_params(axis='y', labelsize=22)
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


z = np.linspace(min(S/V), max(S/V), 100)
# label=r'$\alpha(\frac{S}{V}) = m\cdot \frac{S}{V} + q$'
fig, axs = plt.subplots(nrows=1, ncols=3)
plt.subplots_adjust(left=0.3, bottom=0.07, right=None, top=0.99, wspace=None, hspace=0.3)
popt, pcov = curve_fit(line, S/V, [popt1[2], popt2[2], popt3[2]])
diag = np.sqrt(np.diag(pcov))
print(popt, np.sqrt(np.diag(pcov)))
axs[0].errorbar(S/V, [popt1[2], popt2[2], popt3[2]], yerr=[diag1[2], diag2[2], diag3[2]], marker='^', linestyle='', color='tab:red')
#axs[0].errorbar(1/2.122, 0.478, marker='*', linestyle='', color='tab:green', label='GPD 38')
axs[0].plot(z, line(z, *popt), color='tab:orange', label=fr'$m = {popt[0]:.2f} \pm {diag[0]:.2f}$ [mm]' + '\n' + fr'$q = {popt[1]:.2f} \pm {diag[1]:.2f}$')
axs[0].text(0.5, 0.6, fr'$\alpha(\frac{{S}}{{V}}) = m\cdot \frac{{S}}{{V}} + q$', fontsize='20')
axs[0].grid()
axs[0].set_xlabel(r'$S_{sample}/V_{sample}$ [1/mm]', fontsize = 20)
axs[0].set_ylabel(r'$\alpha$', fontsize = 20)
axs[0].tick_params(axis='x', labelsize=20)
axs[0].tick_params(axis='y', labelsize=20)
axs[0].legend(fontsize='15', loc='upper left')
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
axs[1].errorbar(S/V, [popt1[3], popt2[3], popt3[3]], yerr=[diag1[3], diag2[3], diag3[3]], marker='^', linestyle='', color='tab:red')
#axs[1].errorbar(1/2.122, 629, yerr=90, marker='*', linestyle='', color='tab:green', label='GPD 38')
axs[1].plot(z, power_law(z, *popt), color='tab:orange', label= fr'$C = {popt[0]:.0f} \pm {diag[0]:.0f}$ [mm$^2$]' + '\n' + fr'${{\Gamma}} = {popt[1]:.2f} \pm {diag[1]:.2f}$')
axs[1].text(0.8, 1000, fr'$\tau(\frac{{S}}{{V}}) = C\cdot \left(\frac{{S}}{{V}}\right)^{{\Gamma}}$', fontsize='20')
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].grid()
axs[1].set_xlabel(r'$S_{sample}/V_{sample}$ [1/mm]', fontsize = 20)
axs[1].set_ylabel(r'$\tau$ [days]', fontsize = 20)
axs[1].tick_params(axis='x', labelsize=20)
axs[1].tick_params(axis='y', labelsize=20)
axs[1].legend(fontsize='15', loc='lower left')
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
popt, pcov = curve_fit(sigmoid, x, y, p0 = [np.median(x), 1])
z = np.linspace(min(V/(V_gas-V)), 0.08, 1000)
#axs[2].plot(z, sigmoid(z, *popt), color='tab:orange')
axs[2].errorbar(V/(V_gas-V), relative_deltas, marker='^', linestyle='', color='tab:blue')
#axs[2].errorbar(1/20.208, (245/800)*100, marker='*', linestyle='', label='GPD 38', color='tab:green')
axs[2].set_xlabel(r'$V_{samples}/V_{gas}$', fontsize = 20)
axs[2].set_ylabel(r'$\frac{\Delta_P}{P_0}$ [%]', fontsize = 20)
axs[2].tick_params(axis='x', labelsize=20)
axs[2].tick_params(axis='y', labelsize=20)
#axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_yticks(np.linspace(10, 100, 10), minor=True)
axs[2].set_xticks(np.array([0.02, 0.04, 0.06]), minor=False)

# Imposta i minor ticks per rendere la griglia più fitta
#ax.tick_params(np.logspace(10, 100, 10), minor=True)

axs[2].grid(True, which='both',axis='both')

plt.savefig('/Users/chiara/Desktop/params.pdf')

plt.show()