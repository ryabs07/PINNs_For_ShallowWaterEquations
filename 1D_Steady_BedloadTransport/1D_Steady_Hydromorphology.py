import deepxde as dde
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import cmocean
import pandas as pd
import imageio.v2 as imageio
from io import BytesIO

#This code solves the steady hydromorphodynamics problem presented in https://arxiv.org/abs/1112.1582 using PINN. The analytical results are compared to the results from PINN model. 

dde.config.set_default_float("float64")
# Set default font size
mpl.rcParams['font.size'] = 9

# Activating text rendering by LaTex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# Get the current directory (where the script is located)
current_dir = os.path.dirname(os.path.abspath("1D_Steady_Hydromorphology.py"))

# Check if the 'Plots' folder exists in the parent folder
plots_folder_path = os.path.join(current_dir, "Plots")
if not os.path.exists(plots_folder_path):
    # If 'Plots' folder doesn't exist, create it
    os.makedirs(plots_folder_path)
 
#function definition for initial bottom topography
def z_initial(x):
    u = ((alpha * x + beta) / A) ** (1/3)
    return C - (u*u*u + 2*g*q)/(2*g*u)

#Configuration for the simulation
x_start = 0 #Starting point of domain
x_end = 7 #End point of domain
XL = x_end - x_start
time_start = 0 #Starting time of simulation
time_end = 50 #Ending time of simulation
g = 9.81  # Specify the value for gravity constant g
q = 1       #Specific discharge

#Parameters for the problem statement
alpha = 0.005
beta = 0.005
A = 0.005
C = 1

# Preparation of Input for Prediction using PINN
total_points = 500 #number of discretized points in x-direction
x = np.linspace(x_start, x_end, total_points)
num_time_steps = 100 #Number of timesteps
t = np.linspace(time_start, time_end, num_time_steps)
X, T = np.meshgrid(x,t) 
input = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

#Define the simplified 1D Shallow Water Equations with Exner Equation
def pde(a, b):
    #a is a 2D input array where a[:,0:1] represents spatial domain in x direction and a[:,1:2] represents the temporal domain
    #a is the input of the neural net
    #b is a the output array where y[:,0:1] represents bed elevation (topography)
    #b is the output of the neural net
    x, t = a[:, 0:1], a[:, 1:2]
    z = b[:, 0:1]
    #velocity as a function of space for steady hydrodynamics
    u = torch.pow((alpha * x + beta) / A, 1/3)
    h = q / u
    huu = q*q/h
    qb = A * u * u * u
    #define the gradients using automatic differentiation
    dh_dx = dde.grad.jacobian(h, a, i=0, j=0)
    dhuu_dx = dde.grad.jacobian(huu, a, i=0, j=0)
    dz_dx = dde.grad.jacobian(z, a, i=0, j=0)
    dz_dt = dde.grad.jacobian(z, a, i=0, j=1)
    dqb_dx = dde.grad.jacobian(qb, a, i=0, j=0)
    #momentum equation for shallow waters
    momentum_x = dhuu_dx + g * h * (dh_dx + dz_dx)
    #Exner Equation of sediment mass balance for hydromorphodynamics
    mass_balance_bedload =  dz_dt + dqb_dx
    return [momentum_x, mass_balance_bedload]

#Definition of geometry of domain in both space and time
geom = dde.geometry.Interval(x_start, x_end)
timedomain = dde.geometry.TimeDomain(time_start, time_end)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#Define initial condition of topography for PINN
ic_z = dde.icbc.IC(
    geomtime, lambda x: z_initial(x[:, 0:1]), lambda _, on_initial: on_initial, component=0
)

#Assign the number of points in domain for the training of neural net to satisfy PDEs and initial conditions
data = dde.data.TimePDE(geomtime, pde, [ic_z], num_domain=5000, num_initial=500, anchors = input)
#Definition of feed forward neural network architecture
#[input layer dimension] + [hiden layer dimension]*[number of hidden layers] + [output layer  dimension] , "activation function", "initializer of weights"
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot uniform")
model = dde.Model(data, net)
checkpoint = dde.callbacks.ModelCheckpoint('Models/Model.ckpt', verbose=1, save_better_only=True, period = 500)

#Compile the neural net with Adam optimizer, set learning rate and loss weight to  momentum equation, sediment mass balance equation, initial condition of h and initial condition of u respectively
model.compile("adam", lr=3e-4, loss_weights= [1e2, 1e2, 1e4])
pde_resampler = dde.callbacks.PDEPointResampler(period=100)
losshistory, train_print_timestep = model.train(iterations=15000, callbacks=[pde_resampler, checkpoint])#
dde.saveplot(losshistory, train_print_timestep, issave=True, isplot=True)
# Save the trained model
model.save("1stStageAdamOptimized.h5")

#Retrain the model using L-BFGS optimizer for fine tuining
checkpoint = dde.callbacks.ModelCheckpoint('Models/Model.ckpt', verbose=1, save_better_only=True, period = 10)
dde.optimizers.set_LBFGS_options(maxiter= 20000, gtol= 1e-5)
model.compile("L-BFGS", loss_weights= [1e2, 1e2, 1e4]) #
losshistory, train_print_timestep = model.train(callbacks=[pde_resampler, checkpoint])#
dde.saveplot(losshistory, train_print_timestep, issave=True, isplot=True)
# Save the trained model
model.save("2ndStage_LBFGS_Optimzed.h5")

### Post Processing 
output = model.predict(input) #passing the input gridpoints and timestamps through optimized neural network

# Extracting bottom topography from the result
z_PINN = (output[:, 0].reshape((-1, total_points)).T)

#save the output arrays of PINN in npy format
np.save('z.npy', z_PINN)

u = ((alpha * x + beta) / A) ** (1/3)
h = q / u
z_initial = z_initial(x)
#analytical solution of development of bottom topography
z_analytical = z_initial[:, np.newaxis] - alpha * t
x_plot = x
wsl = z_PINN + h[:, np.newaxis]
wsl_analytical = z_analytical + h[:, np.newaxis]

print_timesteps = [0, int(num_time_steps/4), int(3*num_time_steps/4), num_time_steps-1]
dt = time_end / (num_time_steps-1)
#print_timestep = -1 # print_timestep = 0 for printing initial condition, print_timestep = -1 for printing final condition
# # Create the figure for selected timesteps
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for timestep in range(num_time_steps):
    # Clear the axis
    ax.clear()
    # Plot the water depth at the current timestep
    timestamp = (timestep) * dt
    if t[timestep] == print_timesteps[0] or timestep ==  print_timesteps[1] or timestep ==  print_timesteps[2] or timestep ==  print_timesteps[3]:
        ax.plot(x_plot, wsl_analytical[:, timestep], label='WSL analytical')
        ax.plot(x_plot, wsl[:, timestep], linestyle='--', label='WSL PINN')
        ax.plot(x_plot, z_analytical[:, timestep], label='topography analytical')
        ax.plot(x_plot, z_PINN[:, timestep], linestyle='--', label='topography PINN')
        # Fill the area between the curves
        ax.fill_between(x_plot, z_analytical[:, timestep], wsl_analytical[:, timestep], color='skyblue', alpha=0.5)
        ax.fill_between(x_plot, -0.4, z_analytical[:, timestep], color='sandybrown', hatch = '...', alpha=0.7)
        # Set the axis labels and title
        ax.set_xlabel('x-distance [m]')
        ax.set_ylabel('elevation [m]')
        ax.set_title('1D Steady Bedload Transport, ' + f'Time: {timestamp:.2f} s')
        ax.set_xlim([x_start, x_end])
        ax.set_ylim([-0.4, 1.4])
        ax.legend(loc='upper right', fontsize=5)  # Add legend
        ax.grid(True)
        plt.tight_layout()
        filename = os.path.join(plots_folder_path,'SteadyBedLoadTransport_' + "{:.2f}".format(t[timestep]))  
        # Save the current figure to the file
        plt.savefig(filename + '.png', format='png')  # Fixed filename argument
        plt.savefig(filename + '.eps', format='eps')   
# Show the current figure
plt.show()

# Post Processing for the movie clip
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(4.5, 3.5))
# Create a list to store the generated frames
frames = []
# Loop through each timestep
for timestep in range(num_time_steps):
    # Clear the axis
    ax.clear()
    
    # Plot the water depth at the current timestep
    timestamp = (timestep) * dt
    ax.plot(x_plot, wsl_analytical[:, timestep], label='WSL analytical')
    ax.plot(x_plot, wsl[:, timestep], linestyle='--', label='WSL PINN')
    ax.plot(x_plot, z_analytical[:, timestep], label='topography analytical')
    ax.plot(x_plot, z_PINN[:, timestep], linestyle='--', label='topography PINN')
    # Fill the area between the curves
    ax.fill_between(x_plot, z_PINN[:, timestep], wsl[:, timestep], color='skyblue', alpha=0.5)
    ax.fill_between(x_plot, -0.4, z_PINN[:, timestep], color='sandybrown', hatch = '...', alpha=0.7)
    
    # Set the axis labels and title
    ax.set_xlabel('x-distance [m]')
    ax.set_ylabel('Water depth $h$ [m]')
    ax.set_title('1D Steady Bedload Transport, ' + f'Time: {timestamp:.2f} s')
    ax.set_xlim([x_start, x_end])
    ax.set_ylim([-0.4, 1.2])
    ax.legend(loc='upper right', fontsize=5)  # Add legend
    ax.grid(True)
  
    
    # Create an in-memory file object
    img_buffer = BytesIO()
    
    # Save the current figure to the in-memory file object
    plt.savefig(img_buffer, format='png')
    
    # Read the contents of the in-memory file object and add it to the list of frames
    img_buffer.seek(0)
    img_data = img_buffer.getvalue()
    img = imageio.imread(img_data, format='PNG')
    frames.append(img)
    
    # Clear the in-memory file object for the next iteration
    img_buffer.close()
    
# Save the list of frames as an MP4 file
# (adjust the file name and parameters as needed)
mp4_filename = '1D_SteadyBedLoadTransport.mp4'
imageio.mimsave(mp4_filename, frames, fps=7)


data = np.loadtxt("loss.dat", skiprows=1)
 # Extract columns of interest
iterations = data[:, 0]
mass_loss = data[:, 1]
momentum_loss = data[:, 2]
h_ic_loss= data[:, 3]
u_ic_loss = data[:, 4]
total_loss = mass_loss + momentum_loss + h_ic_loss + u_ic_loss

fig, axes = plt.subplots(figsize=(5.5, 3.5))
plt.plot(iterations, total_loss, linewidth=1.5, color='darkorange')
# Customize the plot
plt.xlabel('Iterations')
plt.ylabel('Total Loss')
title = r'$ \mathcal{L}_{Total} = 10^2 \times \mathcal{L}_{Momentum_x} + 10^2 \times \mathcal{L}_{BedLoadMassBalance} + 10^4 \times \mathcal{L}_{IC_{topography}}$'
axes.set_title(title, loc='center', wrap=True, fontsize = 7)
# Set y-axis to log scale
plt.yscale('log')
# Set x-axis limits from the minimum to maximum value of 'iterations'
#plt.xlim(min(iterations), max(iterations))
# Add vertical dashed line at iterations = 30000
plt.axvline(x=15000, linestyle='--', color='gray')
# Add text on the left side of the line (adam optimizer)
plt.text(13000, 1e2, 'Adam optimizer', rotation=0, va='center', ha='right')
# Add text on the right side of the line (L-BFGS optimizer)
plt.text(17000, 1e2, 'L-BFGS optimizer', rotation=0, va='center', ha='left')
filename = os.path.join(plots_folder_path,"total_loss")
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps') 
# Show the plot
plt.show()
