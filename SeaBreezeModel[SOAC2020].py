    # BEGINNING OF THE PYTHON SCRIPT

# https://docs.python.org/2.7/
# IMPORT MODULES
import numpy as N  # http://www.numpy.org
import matplotlib.pyplot as P   # http://matplotlib.org
import math as M  # https://docs.python.org/2/library/math.html

#%%===========================================================================
# Importing the table
# !pip install tabula    #uncomment if the library is not installed
#=============================================================================
from tabula.io import read_pdf

tables = read_pdf('ObservationsAnalysisIJmuiden(AnswerProblem1_13AD).pdf')
# time = tables[0][["time[hr],after7/5/1976,00UTC"]]
time_data = N.array(range(48))
day = tables[0][["day"]]
hour = tables[0][["hour"]]
dpdx = tables[0][["dpdx[Pa/km]"]]*1e-3 #Pa/m
dpdy = tables[0][["dpdy[Pa/km]"]]*1e-3 #Pa/m
u_data = tables[0][["u0[m/s]"]]
v_data = tables[0][["v0[m/s]"]]
dpdx_m = tables[0][["dpdx_m[Pa/km]"]]*1e-3 #Pa/m
u0_m = tables[0][["u0_m[m/s]"]]
v0_m = tables[0][["v0_m[m/s]"]]
#%%===========================================================================

# Length of the simulation and time-step
time_max = 48.0   # length in hours of the simulation
dt = 30.0 # time step in s

# PARAMETER VALUES
A = 0.001 # Pa m^-1
phase = 0.0 # phase of surface pressure gradient in time
lat = 52.47 # latitude in degress
phi = 0.0 # phase of the pressure gradient 
labda = 0.0  # Rayleigh damping coefficient  #max=0.033

# CONSTANTS
omega = 0.000072792  # angular velocity Earth [s^-1]
pi = M.pi
ro = 1.25 # density kg m^-3
fcor = 2 * omega *  M.sin(lat * pi/180)  # Coriolis parameter
C1= - (A/(fcor*ro)) * ( (M.pow(omega,2) /(M.pow(fcor,2) - M.pow(omega,2))) + 1)
C3 = A * omega /(ro*(M.pow(fcor,2) - M.pow(omega,2)))
# C3=0

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
# v_ana[nt] = 0

# TIME LOOP EULER FORWARD SCHEME 
for nt in range(len(time)-1): 
 du = dt * (-(labda * u[nt]) + (fcor*v[nt]) - ((A/ro)* M.cos((omega*time[nt])+phase)))
 dv = dt * (-(labda * v[nt]) - (fcor * u[nt]))
 time[nt+1] = time[nt]+dt 
 u[nt+1] = u[nt] + du
 v[nt+1] = v[nt] + dv	
 u_ana[nt+1] = (C1 * M.sin(fcor * time[nt+1])) + ( C3* M.sin((omega * time[nt+1]) + phase) ) 
 # v_ana[nt+1] = (C1 * M.cos(fcor * time[nt+1]) + (M.cos(omega * time[nt+1]) * fcor * A) / (ro*(M.pow(fcor,2)-M.pow(omega, 2))))

for nt in range(len(time)):
 time_axis[nt] = time[nt] / 3600. # time axis in hours
 
# MAKE PLOT of evolution in time of u, v and u_ana
P.figure(figsize = (8,4))
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



#%%===========================================================================
# Plotting the observation
P.figure(figsize = (8,4))
P.plot(time_data, u_data, color="red", label="Measurements of $u_0$")
P.plot(time_data, v_data, color="blue", label="Measurements of $v_0$")
P.axis([0, time_data[-1], -10, 10])
P.xticks(N.arange(0,time_data[-1], 6))
P.xlabel('time [hours]', fontsize=14) # label along x-axes
P.ylabel('velocity [m/s]', fontsize=14) # label along x-axes
P.title("Measurements at IJmuiden")
P.legend()
P.grid()
P.tight_layout()

# P.figure(figsize = (8,4))
# P.scatter(time[:24], dpdx[:24], label="Data")
# P.xlabel('time [hours]', fontsize=14) # label along x-axes
# P.ylabel('dpdx [Pa/m]', fontsize=14)
# P.legend()


#%%===========================================================================
# Least square fit
from scipy.optimize import curve_fit

def f(x, a, b, B):
   return  a * N.cos(omega * x *3600 + b) + B    #x*3600 will turn hours to seconds

xdata = N.arange(0,24,1)   #create array of hours of measurements
ydata = dpdx_m[24:].to_numpy()

params, cov = curve_fit(f, xdata, ydata.ravel())  #ydata.ravel() converts array from 2D to 1D
print('A=%.5e, phi = %.3f, B=%.5e' %(params[0], params[1], params[2]))

P.figure(figsize = (8,4))
P.scatter(xdata, ydata, label="Data")
P.plot(xdata, f(xdata, params[0], params[1], params[2]), '--', color='red', label="Fitted curve")
P.title("Data and fitted curve")
P.ylabel("dp/dx [Pa/m]")
P.xlabel("time[hours]")
P.legend()
#%%===========================================================================
# Comparison of the model with the data with the values of A and phi

angle_data = (N.arctan2(v_data, u_data)*180/N.pi).to_numpy()  #observed angle of wind direction in degrees
angle_data = -angle_data   #change to clockwise rotation
angle_data[angle_data<0] = angle_data[angle_data<0] + 360     #change the angle range to 0-360

A, phase = params[0], params[1]
u[nt]=0
v[nt]=0
labda = 0.0  # Rayleigh damping coefficient

for nt in range(len(time)-1): 
 du = dt * (-(labda * u[nt]) + (fcor*v[nt]) - ((A/ro)* M.cos((omega*time[nt])+phase)))
 dv = dt * (-(labda * v[nt]) - (fcor * u[nt]))
 time[nt+1] = time[nt]+dt 
 u[nt+1] = u[nt] + du
 v[nt+1] = v[nt] + dv	
 u_ana[nt+1] = (C1 * M.sin(fcor * time[nt+1])) + ( C3* M.sin((omega * time[nt+1]) + phase) ) 
 # v_ana[nt+1] = (C1 * M.cos(fcor * time[nt+1]) + (M.cos(omega * time[nt+1]) * fcor * A) / (ro*(M.pow(fcor,2)-M.pow(omega, 2))))

angle_num = N.arctan2(v, u)*180/N.pi
angle_num = -angle_num
angle_num[angle_num<0] = angle_num[angle_num<0] + 360     #change the angle range to 0-360


P.figure(figsize=(8,4))
P.scatter(time_data, angle_data, color="black")
P.xlabel('time [hours]', fontsize=14)
P.ylabel('Wind direction($\degree$) [m/s]', fontsize=14)
P.xticks(N.arange(0, time_data[-1]+2, 2))
P.plot(time_axis[1:-1], angle_num[1:-1], color="red")   #1553
P.tight_layout()
#%%===========================================================================
# Estimation of B and vg

def fB(x, B):
   return  A* N.cos(omega * x *3600 + phase) + B


param = curve_fit(fB, xdata, ydata.ravel())
print('B=%.5e, v_g = %.3f' %(param[0], param[0]/(ro*fcor)))

#%%===========================================================================

# Calculating ug
# dpdy = dpdy.to_numpy()
ug = N.zeros(len(dpdy))
for i in range(len(ug)):
   ug[i] = -dpdy[i]/(ro*fcor)
    
print('u_g = %.3f' %(N.mean(ug)))
"""
#%%===========================================================================
# Running the model for A,phi,B

A, phase, B = params[0], params[1], param[0]
# u[nt] = ug
ug = N.mean(ug)
vg = B/(ro*fcor)
labda = 0.0

for nt in range(len(time)-1): 
 du = dt * (-(labda * ug) + (fcor*vg) - ((A/ro)* M.cos((omega*time[nt])+phase)))
 dv = dt * (-(labda * vg) - (fcor * ug))
 time[nt+1] = time[nt]+dt 
 u[nt+1] = u[nt] + du
 v[nt+1] = v[nt] + dv	
 u_ana[nt+1] = (C1 * M.sin(fcor * time[nt+1])) + ( C3* M.sin((omega * time[nt+1]) + phase) ) 
 v_ana[nt+1] = (C1 * M.cos(fcor * time[nt+1]) + (M.cos(omega * time[nt+1]) * fcor * A) / (ro*(M.pow(fcor,2)-M.pow(omega, 2))))


angle_num1 = N.arctan2(v, u) * 180/N.pi
angle_num1 = -angle_num1
angle_num1[angle_num1<0] = angle_num1[angle_num1<0] + 360    #change the angle range to 0-360

P.figure(figsize=(8,4))
P.scatter(time_data, angle_data, color="black")
P.xlabel('time [hours]', fontsize=14)
P.ylabel('Wind direction($\degree$) [m/s]', fontsize=14)
P.xticks(N.arange(0, time_data[-1]+2, 2))
P.plot(time_axis[1:-1], angle_num[1:-1], color="red")
P.plot(time_axis[1:-1], angle_num1[1:-1], color="red")    #1553
P.tight_layout()
"""
#%%
# Testing scheme

# Running the model for A,phi,B

A, phase, B = params[0], params[1], param[0]
u_num = N.zeros((nt_len))
v_num = N.zeros((nt_len))
nt = 0 
u_num[nt] = 0
v_num[nt] = 0
ug = N.mean(ug)
vg = B/(ro*fcor)

for nt in range(len(time)-1):
   u_star = u_num[nt] + (fcor*(v_num[nt] - vg)) - (((A*N.cos(omega*time[nt]+phase))/ro) - labda * u_num[nt]) * dt
   v_star = v_num[nt] - (fcor*(u_num[nt] - ug) + labda*v_num[nt]) * dt
   time[nt+1] = time[nt] + dt
   u_num[nt+1] = u_num[nt] + (fcor*(v_star-vg) - ((A*N.cos(omega*time[nt+1]+phase))/ro) - labda*u_star) * dt
   v_num[nt+1] = v_num[nt] - (fcor*(u_star-ug) + labda*v_star) * dt
    
    
angle_num1 = N.arctan2(v_num, u_num) * 180/N.pi
angle_num1 = -angle_num1
angle_num1[angle_num1<0] = angle_num1[angle_num1<0] + 360    #change the angle range to 0-360

P.figure(figsize=(8,4))
P.scatter(time_data, angle_data, color="black", label='Measurements')
P.xlabel('time [hours]', fontsize=14)
P.ylabel('Wind direction($\degree$) [m/s]', fontsize=14)
P.xticks(N.arange(0, time_data[-1]+2, 2))
P.plot(time_axis[1:-1], angle_num[1:-1], color="red", label='Model without ug,vg')
P.plot(time_axis[1:-1], angle_num1[1:-1], color="orange", label='Model with constant ug,vg')   #1553
P.legend()
P.tight_layout()
#%%===========================================================================
# Realistic geostrophic wind

B = N.zeros(len(dpdx_m))
for i in range(len(dpdx_m)):
    B[i] = dpdx_m[24][i] - A*N.cos(omega*xdata*3600+phase)
    

ug = N.zeros(len(dpdy))
for i in range(len(ug)):
    ug[i] = -dpdy[i]/(ro*fcor)
    
    
#%%
# Wind plots 

fig, ax = P.subplots(2,1, figsize = (8,6), sharex = True)

P.sca(ax[0])
P.plot(time_data, u_data, label = 'Measurements')
P.plot(time_axis, u, label = 'Model without ug, vg')
P.plot(time_axis, u_num, label = 'Model with constant ug, vg')
P.xticks(N.arange(0, time_data[-1]+2, 2))
# P.xlabel('time [hours]', fontsize=14)
P.ylabel('u [m/s]', fontsize=14)

P.sca(ax[1])
P.plot(time_data, v_data, label = 'Measurements')
P.plot(time_axis, v, label = 'Model without ug, vg')
P.plot(time_axis, v_num, label = 'Model with constant ug, vg')
P.ylabel('v [m/s]', fontsize=14)
P.xlabel('time [hours]', fontsize=14)
P.legend()
P.tight_layout()
P.show
"""
get the constant ug,vg line - leaast square fit to get values for B,ug,vg??
get the realistic ug,vg line - line221
correct the plot - discard the vertical lines
"""
#%% Model Simulation 

# Length of the simulation and time-step
time_max = 48.0   # length in hours of the simulation
dt = 30.0 # time step in s
lat = 52
A = params[0]
phase = params[1]
B = 0

# CONSTANTS
omega = 0.000072792  # angular velocity Earth [s^-1]
pi = M.pi
ro = 1.25 # density kg m^-3
fcor = 2 * omega *  M.sin(lat * pi/180)  # Coriolis parameter
u_data = tables[0][["u0[m/s]"]]
v_data = tables[0][["v0[m/s]"]]
u_data = u_data.to_numpy()
v_data = v_data.to_numpy()

def SBmodel(u0, v0, labda, cd, D):
    
    print('Inputs: u0=',u0,',v0=',v0,'lambda=',labda, 'cd=',N.round(cd,5),'dp/dy=',D)
    
    nt_len = int(48 * 3600 / dt) # maximum number of time steps in the integration 
    u = N.zeros((nt_len))      # x-component velocity, numerical solution
    v = N.zeros((nt_len))      # y-component velocity, numerical solution
  
    # INITIAL CONDITION (t=0) : atmosphere in rest
    nt = 0
    time[nt] = 0
    time_axis[nt] = 0
    u[nt] = u0
    v[nt] = v0
    
    

    # TIME LOOP EULER FORWARD SCHEME 
    for nt in range(len(time)-1): 
     du = dt * (-(labda * u[nt] + cd * N.abs(u[nt])*u[nt]) + (fcor*v[nt]) - ((A/ro)* M.cos((omega*time[nt])+phase))-B/ro)
     dv = dt * (-(labda * v[nt] + cd * N.abs(v[nt])*v[nt]) - (fcor * u[nt]) - (D/ro))
     time[nt+1] = time[nt]+dt 
     u[nt+1] = u[nt] + du
     v[nt+1] = v[nt] + dv	
    
    for nt in range(len(time)):
     time_axis[nt] = time[nt] / 3600. # time axis in hours
     
     # MAKE PLOT of evolution in time of u, v and u_ana
    fig, ax = P.subplots(1,2, figsize = (16,6))
    
    P.sca(ax[0])
    P.plot(time_axis, u, color='#8B008B', label='u numerical solution')
    P.scatter(N.arange(0, 48), u_data, color='black', label='u measured', marker='s')
    P.xlabel('time [hours]', fontsize=14)
    P.ylabel('u [m/s]', fontsize=14)
    P.xticks(N.arange(0,time_axis[nt_len-1]+0.1,6), fontsize=12) 
    P.title('Seabreeze simulation for u')
    P.legend()
    P.grid()
    P.tight_layout()
    if labda==0 and cd==0:
        P.ylim(-20,20)
    else:
        P.ylim(-8,8)
    
    P.sca(ax[1])
    P.plot(time_axis, v, color='#FF8C00', label='v numerical solution')
    P.scatter(N.arange(0, 48), v_data, color='black', label='v measured', marker='s')
    P.xlabel('time [hours]', fontsize=14)
    P.ylabel('v [m/s]', fontsize=14)
    P.xticks(N.arange(0,time_axis[nt_len-1]+0.1,6), fontsize=12) 
    P.title('Seabreeze simulation for v')
    P.legend()
    P.grid()
    P.tight_layout()
    if labda==0 and cd==0:
        P.ylim(-20,20)
    else:
        P.ylim(-8,8)
    P.show()
        
    # fig.suptitle(r'$\lambda$={}; $C_D$={}; D={}; u= {}m/s; v={}m/s'.format(labda, N.round(cd,5), N.round(D,5), N.round(u0,2), N.round(v0,2), fontsize=16))
   
# Comparing the outcome with the observations
   
    # Standard deviation
    std_u_data = N.std(u_data)
    std_v_data = N.std(v_data)
    std_u = N.std(u)
    std_v = N.std(v)

    #Correlation coefficient
    step = int(3600/dt)
    u_sim = u[0::step]
    v_sim = v[0::step]
    r_u = N.corrcoef(u_data, u_sim, rowvar = False)[0][1]
    r_v = N.corrcoef(v_data, v_sim, rowvar = False)[0][1]

    # Root mean square(RMS) difference
    def rmse(predictions, targets):
        return N.sqrt(((predictions - targets) ** 2).mean())
    
    rmsd_u = rmse(u_sim, u_data)
    rmsd_v = rmse(v_sim, v_data)
    

    # Centred pattern RMS
    crms_u = N.sqrt(std_u_data**2+std_u**2-2*std_u_data*std_u*r_u)
    crms_v = N.sqrt(std_v_data**2+std_v**2-2*std_v_data*std_v*r_v)
    
    print('correlation between u_sim and observed u',r_u)
    print('correlation between v_sim and observed v',r_v)
    print('standard deviation  u_numerical= ', std_u)
    print('standard deviation  v_numerical= ', std_v)
    print('rmsd u_numerical =', rmsd_u)
    print('rmsd v_numerical =', rmsd_v)
    print('centered rms u_numerical=', crms_u)
    print('centered rms v_numerical=', crms_v)

    return u_sim, v_sim

# END OF THE PYTHON SCRIPT
