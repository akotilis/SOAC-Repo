    # BEGINNING OF THE PYTHON SCRIPT

# https://docs.python.org/2.7/
# IMPORT MODULES
import numpy as N  # http://www.numpy.org
import matplotlib.pyplot as P   # http://matplotlib.org
import math as M  # https://docs.python.org/2/library/math.html
from docx import Document  # https://python-docx.readthedocs.io/en/latest/user/install.html#install


# Length of the simulation and time-step
time_max = 48.0   # length in hours of the simulation
dt = 1.0 # time step in s

# PARAMETER VALUES
A = 0.001 # Pa m^-1
phase = 0.0 # phase of surface pressure gradient in time
lat = 52 # latitude in degress
phi = 0.0 # phase of the pressure gradient 
labda = 0.0  # Rayleigh damping coefficient  #max=0.033

# CONSTANTS
omega = 0.000072792  # angular velocity Earth [s^-1]
pi = M.pi
ro = 1.25 # density kg m^-3
fcor = 2 * omega *  M.sin(lat * pi/180)  # Coriolis parameter
C1= - (A/(fcor*ro)) * ( (M.pow(omega,2) /(M.pow(fcor,2) - M.pow(omega,2))) + 1)
C3 = A * omega /(ro*(M.pow(fcor,2) - M.pow(omega,2)))


#  NUMBER OF TIME STEPS AND TIME-AXIS FOR PLOTTING 
nt_len = int(48 * 3600 / dt) # maximum number of time steps in the integration 
time = N.zeros((nt_len))
time_axis = N.zeros((nt_len))

#  DEFINE time, u, v and the analytical solution, u_ana, as arrays and fill them with zero's 
u = N.zeros((nt_len))      # x-component velocity, numerical solution
v = N.zeros((nt_len))      # y-component velocity, numerical solution
u_ana = N.zeros((nt_len))  # analytical solution x-component velocity
v_ana = N.zeros((nt_len))
# INITIAL CONDITION (t=0) : atmosphere in rest
nt = 0
time[nt] = 0
time_axis[nt] = 0
u[nt] = 0
v[nt] = 0
u_ana[nt] = 0
v_ana[nt] = 0

# TIME LOOP EULER FORWARD SCHEME 
for nt in range(len(time)-1): 
 du = dt * (-(labda * u[nt]) + (fcor*v[nt]) - ((A/ro)* M.cos((omega*time[nt])+phase)))
 dv = dt * (-(labda * v[nt]) - (fcor * u[nt]))
 time[nt+1] = time[nt]+dt 
 u[nt+1] = u[nt] + du
 v[nt+1] = v[nt] + dv	
 u_ana[nt+1] = (C1 * M.sin(fcor * time[nt+1])) + ( C3* M.sin((omega * time[nt+1]) + phase) ) 
 v_ana[nt+1] = (C1 * M.cos(fcor * time[nt+1]) + (M.cos(omega * time[nt+1]) * fcor * A) / (ro*(M.pow(fcor,2)-M.pow(omega, 2))))

for nt in range(len(time)):
 time_axis[nt] = time[nt] / 3600. # time axis in hours
 
# MAKE PLOT of evolution in time of u, v and u_ana
P.plot(time_axis, v_ana, color='black', linestyle = 'dashed')
P.plot(time_axis, u_ana, color='black', linestyle = 'dashed')
P.plot(time_axis, u, color='red')
P.plot(time_axis, v, color='blue')
P.axis([0,time_axis[nt_len-1],-25.0,25.0])  # define axes 
P.xticks(N.arange(0,time_axis[nt_len-1]+0.1,6), fontsize=12) 
P.yticks(N.arange(-25.0,25.0,5), fontsize=12) 
P.xlabel('time [hours]', fontsize=14) # label along x-axes
P.ylabel('velocity [m/s]', fontsize=14) # label along x-axes
P.title('SeabreezeSimulation') # Title at top of plot
P.text(1, 23, 'u (analytical solution) (no damping): black line', fontsize=10, color='black')
P.text(1, 21, 'u (numerical solution): red line (forward time difference scheme)', fontsize=10, color='red')
P.text(1, 19, 'v (numerical solution): blue line (forward time difference scheme)', fontsize=10, color='blue')
P.grid(True)
# P.savefig("SeabreezeSimulation.png") # save plot as png-file
P.show() # show plot on screen


#%%==============================================================
# Importing the table
# !pip install tabula
from tabula.io import read_pdf

tables = read_pdf('ObservationsAnalysisIJmuiden(AnswerProblem1_13AD).pdf')
# time = tables[0][["time[hr],after7/5/1976,00UTC"]]
time = N.array(range(48))
day = tables[0][["day"]]
hour = tables[0][["hour"]]
dpdx = tables[0][["dpdx[Pa/km]"]]*1e-3 #Pa/m
dpdy = tables[0][["dpdy[Pa/km]"]]*1e-3 #Pa/m
u0 = tables[0][["u0[m/s]"]]
v0 = tables[0][["v0[m/s]"]]
dpdx_m = tables[0][["dpdx_m[Pa/km]"]]*1e-3 #Pa/m
u0_m = tables[0][["u0_m[m/s]"]]
v0_m = tables[0][["v0_m[m/s]"]]
#%%==================================================================
# Plotting the observation
P.figure()
P.plot(time, u0, color="red", label="Measurements of $u_0$")
P.plot(time, v0, color="blue", label="Measurements of $v_0$")
P.axis([0, time[-1], -10, 10])
P.xticks(N.arange(0,time[-1], 6))
P.xlabel('time [hours]', fontsize=14) # label along x-axes
P.ylabel('velocity [m/s]', fontsize=14) # label along x-axes
P.title("Measurements at IJmuiden")
P.legend()
P.grid()
P.tight_layout()

P.figure()
P.scatter(time[:24], dpdx[:24], label="Data")
P.xlabel('time [hours]', fontsize=14) # label along x-axes
P.ylabel('dpdx [Pa/m]', fontsize=14)
P.legend()


#%%
# Least square fit
from scipy.optimize import curve_fit

def f(x, a, b):
    return a * N.cos(omega * x *3600 + b)

xdata = N.arange(0,24,1)
ydata = dpdx_m[24:].to_numpy()

params, cov = curve_fit(f, xdata, ydata.ravel())  #ydata.ravel() converts array from 2D to 1D
print(params)

P.scatter(xdata, ydata)
P.plot(xdata, f(xdata, params[0], params[1]), '--', color='red')
#%%


# END OF THE PYTHON SCRIPT
