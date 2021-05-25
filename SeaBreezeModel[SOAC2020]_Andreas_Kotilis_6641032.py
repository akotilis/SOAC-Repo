    # BEGINNING OF THE PYTHON SCRIPT

# https://docs.python.org/2.7/
# IMPORT MODULES
import numpy as N  # http://www.numpy.org
import matplotlib.pyplot as P   # http://matplotlib.org
import math as M  # https://docs.python.org/2/library/math.html
from tabulate import tabulate

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
P.figure(figsize = (8,4))
P.plot(time_axis, v_ana, color='black', linestyle = 'dashed',markersize=.5, label='analytical solution')
P.plot(time_axis, u_ana, color='black', linestyle = 'dashed')
P.plot(time_axis, u, color='#8B008B', label= 'u (numerical solution)', lw=1)
P.plot(time_axis, v, color='#FF8C00', label= 'v (numerical solution)', lw=1)
P.axis([0,time_axis[nt_len-1],-25.0,25.0])  # define axes 
P.xticks(N.arange(0,time_axis[nt_len-1]+0.1,6), fontsize=12) 
P.yticks(N.arange(-25.0,25.0,5), fontsize=12) 
P.xlabel('time [hours]', fontsize=14) # label along x-axes
P.ylabel('velocity [m/s]', fontsize=14) # label along x-axes
P.title('SeabreezeSimulation') # Title at top of plot
# P.text(1, 23, 'u (analytical solution) (no damping): black line', fontsize=10, color='black')
# P.text(1, 21, 'u (numerical solution): red line (forward time difference scheme)', fontsize=10, color='red')
# P.text(1, 19, 'v (numerical solution): blue line (forward time difference scheme)', fontsize=10, color='blue')
P.grid(True)
P.legend()
P.tight_layout()
# P.savefig("SeabreezeSimulation.png") # save plot as png-file
P.show() # show plot on screen



#%%===========================================================================
# Plotting the observation
#=============================================================================
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
# Best fit curve
#=============================================================================
from scipy.optimize import curve_fit

def f(x, a, b, B):
   return  a * N.cos(omega * x *3600 + b) + B    #x*3600 will turn hours to seconds

xdata = N.arange(0,24,1)   #create array of hours of measurements
ydata = dpdx_m[24:].to_numpy()

params, cov = curve_fit(f, xdata, ydata.ravel())  #ydata.ravel() converts array from 2D to 1D
print('A=%.5e Pa/m, phi = %.3f degrees, B=%.5e Pa/m' %(params[0], (params[1]*180)/N.pi, params[2]))

P.figure(figsize = (8,4))
P.scatter(xdata, ydata, label="Data")
P.plot(xdata, f(xdata, params[0], params[1], params[2]), '--', color='red', label="Fitted curve")
P.title("Data and fitted curve")
P.ylabel("dp/dx [Pa/m]", fontsize=14)
P.xlabel("time[hours]", fontsize=14)
P.legend()
P.grid()
P.tight_layout()


#%%==========================================================================
# Model Simulation  
#=============================================================================
# Length of the simulation and time-step
time_max = 48.0   # length in hours of the simulation
dt = 30.0 # time step in s
lat = 52
A = params[0]
phase = params[1]
# B = 0
B = params[2]
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
   fig.suptitle(r' u= {}m/s; v={}m/s; $\lambda$={}; $C_D$={}; D={};'.format(u0, v0, labda, cd, D, fontsize=16)) 
   P.sca(ax[0])
   P.plot(time_axis, u, color='#8B008B', label='u numerical solution')
   P.scatter(N.arange(0, 48), u_data, color='black', label='u measured', marker='s')
   P.xlabel('time [hours]', fontsize=14)
   P.ylabel('u [m/s]', fontsize=14)
   P.xticks(N.arange(0,time_axis[nt_len-1]+0.1,6), fontsize=12) 
   P.title('Seabreeze simulation for u')
   P.ylim(-20,20)
   P.legend()
   P.grid()
   P.tight_layout()
   
   P.sca(ax[1])
   P.plot(time_axis, v, color='#FF8C00', label='v numerical solution')
   P.scatter(N.arange(0, 48), v_data, color='black', label='v measured', marker='s')
   P.xlabel('time [hours]', fontsize=14)
   P.ylabel('v [m/s]', fontsize=14)
   P.xticks(N.arange(0,time_axis[nt_len-1]+0.1,6), fontsize=12) 
   P.title('Seabreeze simulation for v')
   P.ylim(-20,20)
   P.legend()
   P.grid()
   P.tight_layout()
   P.show()
        
  
   
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
    
   
   result_table = [['r_u', r_u], ['r_v', r_v],
                   ['$\sigma_u$', std_u], ['$\sigma_v$', std_v],
                   ['E_u', rmsd_u], ['E_v',rmsd_v],
                   ["E'_u", crms_u], ["E'_v", crms_v]]
   print('Inputs: u0=',u0,',v0=',v0,'lambda=',labda, 'cd=',N.round(cd,5),'dp/dy=',D)
   print(tabulate(result_table, headers=['Parameters', 'Value'], tablefmt=str))
   
   return u_sim, v_sim
#%%===========================================================================
# Simulations ran
#=============================================================================
# SBmodel(-9.039, -7.5849, 0, 0, 0)
usim1, vsim1 = SBmodel(-9.039, -7.5849, 0, 0, 0)
usim2, vsim2 = SBmodel(-9.039, -7.5849, 6e-05, 0, 0)
usim3, vsim3 = SBmodel(3.9, -3.3, 6e-05, 0, 0)
usim4, vsim4 = SBmodel(3.9, -3.3, 0, 1e-05, 0)
D = N.mean(dpdy)
usim5, vsim5 = SBmodel(3.9, -3.3, 1e-04, 0, 0.00015)




#%%=========================================================================== 
# Taylor Diagrams (credits to https://gist.github.com/ycopin/3342888)
#=============================================================================
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist import floating_axes
from mpl_toolkits.axisartist import grid_finder


def taylor_diagram(data, fig, location=111):
   trans = PolarAxes.PolarTransform()
    
    # Correlation labels
   r1_locs = N.hstack((N.arange(1,10)/10.0,[0.95,0.99])) 
   t1_locs = N.arccos(r1_locs)        
   gl1 = grid_finder.FixedLocator(t1_locs)    
   tf1 = grid_finder.DictFormatter(dict(zip(t1_locs, map(str,r1_locs))))
    
    # SD grlidlines. They are vertical to the corr. coef. lines and they have values between 0 and 1, with steps of 0.25
   r2_locs = N.arange(0,5,0.5)
   r2_labels = ['0 ', '0.5 ', '1 ', '1.5 ', '2','2.5','3','3.5','4','4.5','5']
   gl2 = grid_finder.FixedLocator(r2_locs)
   tf2 = grid_finder.DictFormatter(dict(zip(r2_locs, map(str,r2_labels))))

   ghelper = floating_axes.GridHelperCurveLinear(trans,extremes=(0,N.pi/2,0,5),
                                                  grid_locator1=gl1,tick_formatter1=tf1,
                                                  grid_locator2=gl2,tick_formatter2=tf2)
    
   ax = floating_axes.FloatingSubplot(fig, location, grid_helper=ghelper)
   fig.add_subplot(ax)
   ax.axis["top"].set_axis_direction("bottom")   #Angle axis
   ax.axis["top"].toggle(ticklabels=True, label=True)
   ax.axis["top"].major_ticklabels.set_axis_direction("top")
   ax.axis["top"].label.set_axis_direction("top")
   ax.axis["top"].label.set_text("Correlation coefficient")

   ax.axis["left"].set_axis_direction("bottom")  #x-axis
   ax.axis["left"].label.set_text("Standard deviation")

   ax.axis["right"].set_axis_direction("top")   # y-axis
   ax.axis["right"].toggle(ticklabels=True)
   ax.axis["right"].major_ticklabels.set_axis_direction("left")

   ax.axis["bottom"].set_visible(False)         
   ax.grid()
   
   polar_ax = ax.get_aux_axes(trans)   
   t = N.linspace(0,N.pi/2)
    
    # Create the black dashed line. I use it as a reference line for the observation's SD. The closer the model's SD is to this line, the better.
   r = N.zeros_like(t) + N.std(data)
   polar_ax.plot(t,r,'k--')
    
   rs, ts = N.meshgrid(N.linspace(0, 5), N.linspace(0, N.pi/2))
    # Compute centered RMS difference
   rms = N.sqrt(N.std(data)**2 + rs**2 - 2*N.std(data)*rs*N.cos(ts))
    
   contours = polar_ax.contour(ts, rs, rms, 6)
   P.clabel(contours, inline=1, fontsize=10, fmt='%.2f')
    
   return polar_ax

def plot_taylor(axes, refsample, sample, *args, **kwargs):
   std = N.std(sample)
   corr = N.corrcoef(refsample, sample) 
   theta = N.arccos(corr[0,1])
   t,r = theta,std
   d = axes.plot(t,r, *args, **kwargs) 
   return d

#%%===========================================================================
# Taylor diagram for the zonal wind speed
#=============================================================================
from sklearn.metrics import mean_squared_error

fig = P.figure(figsize=(8,8))
ax1 = taylor_diagram(u_data, fig, 111)


# plot_taylor(ax1, u_data.ravel(), usim1, color = 'purple', marker = 'o', alpha = 1, linestyle = 'none',
#                 markersize = 10,
#                  label='1. No friction: ' + str(N.round(N.sqrt(mean_squared_error(u_data, usim1)),2)))

plot_taylor(ax1, u_data.ravel(), usim2, color = 'purple', marker = '*', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='1. Rayleigh 7 May: ' + str(N.round(N.sqrt(mean_squared_error(u_data, usim2)),2)))

plot_taylor(ax1, u_data.ravel(), usim3, color = 'orange', marker = '*', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='2. Rayleigh 6 May: ' + str(N.round(N.sqrt(mean_squared_error(u_data, usim3)),2)))

plot_taylor(ax1, u_data.ravel(), usim4, color = 'blue', marker = '*', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='3. Drag coefficient.: ' + str(N.round(N.sqrt(mean_squared_error(u_data, usim4)),2)))

plot_taylor(ax1, u_data.ravel(), usim5, color = 'green', marker = '*', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='4. Geostrophic wind: ' + str(N.round(N.sqrt(mean_squared_error(u_data, usim5)),2)))

plot_taylor(ax1, u_data.ravel(), u_data.ravel(), color = 'red', marker = 'o', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='Observed: ' + str(N.round(N.sqrt(mean_squared_error(u_data, u_data)),2)))


ax1.legend(bbox_to_anchor=(1.1, 1.05),borderaxespad=0)
ax1.get_legend().set_title('RMSE')
P.suptitle('Zonal wind speed (u)', fontsize = 20)

#%%===========================================================================
# Taylor diagram for the meridional wind speed
#=============================================================================
fig = P.figure(figsize=(8,8))
ax1 = taylor_diagram(v_data, fig, 111)

# plot_taylor(ax1, u_data.ravel(), usim1, color = 'purple', marker = 'o', alpha = 1, linestyle = 'none',
#                 markersize = 10,
#                  label='1. No friction: ' + str(N.round(N.sqrt(mean_squared_error(u_data, usim1)),2)))

plot_taylor(ax1, v_data.ravel(), vsim2, color = 'purple', marker = '*', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='1. Rayleigh 7 May: ' + str(N.round(N.sqrt(mean_squared_error(v_data, vsim2)),2)))

plot_taylor(ax1, v_data.ravel(), vsim3, color = 'orange', marker = '*', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='2. Rayleigh 6 May: ' + str(N.round(N.sqrt(mean_squared_error(v_data, vsim3)),2)))

plot_taylor(ax1, v_data.ravel(), vsim4, color = 'blue', marker = '*', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='3. Drag coefficient.: ' + str(N.round(N.sqrt(mean_squared_error(v_data, vsim4)),2)))

plot_taylor(ax1, v_data.ravel(), vsim5, color = 'green', marker = '*', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='4. Geostrophic wind: ' + str(N.round(N.sqrt(mean_squared_error(v_data, vsim5)),2)))

plot_taylor(ax1, v_data.ravel(), v_data.ravel(), color = 'red', marker = 'o', alpha = 1, linestyle = 'none',
                markersize = 10,
                 label='Observed: ' + str(N.round(N.sqrt(mean_squared_error(v_data, v_data)),2)))


ax1.legend(bbox_to_anchor=(1.1, 1.05),borderaxespad=0)
ax1.get_legend().set_title('RMSE')
P.suptitle('Meridional wind speed (v)', fontsize = 20)


# END OF THE PYTHON SCRIPT
