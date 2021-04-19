    # BEGINNING OF THE PYTHON SCRIPT

# https://docs.python.org/2.7/
# IMPORT MODULES
import numpy as N  # http://www.numpy.org
import matplotlib.pyplot as P   # http://matplotlib.org
import math as M  # https://docs.python.org/2/library/math.html

# Length of the simulation and time-step
time_max = 48.0   # length in hours of the simulation
dt = 30.0 # time step in s

# PARAMETER VALUES
A = 0.001 # Pa m^-1
phase = 0.0 # phase of surface pressure gradient in time
lat = 52 # latitude in degress
phi = 0.0 # phase of the pressure gradient 
labda = 0.0  # Rayleigh damping coefficient

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

# INITIAL CONDITION (t=0) : atmosphere in rest
nt = 0
time[nt] = 0
time_axis[nt] = 0
u[nt] = 0
v[nt] = 0
u_ana[nt] = 0

# TIME LOOP EULER FORWARD SCHEME 
for nt in range(len(time)-1): 
 du = dt * (-(labda * u[nt]) + (fcor*v[nt]) - ((A/ro)* M.cos((omega*time[nt])+phase)))
 dv = dt * (-(labda * v[nt]) - (fcor * u[nt]))
 time[nt+1] = time[nt]+dt 
 u[nt+1] = u[nt] + du
 v[nt+1] = v[nt] + dv	
 u_ana[nt+1] = (C1 * M.sin(fcor * time[nt+1])) + ( C3* M.sin((omega * time[nt+1]) + phase) ) 

for nt in range(len(time)):
 time_axis[nt] = time[nt] / 3600. # time axis in hours
 
# MAKE PLOT of evolution in time of u, v and u_ana
P.plot(time_axis, u_ana, color='black')
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
P.savefig("SeabreezeSimulation.png") # save plot as png-file
P.show() # show plot on screen

# END OF THE PYTHON SCRIPT
