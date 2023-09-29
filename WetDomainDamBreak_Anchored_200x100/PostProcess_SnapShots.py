import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import scipy.io
from io import BytesIO
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import os
import cmocean

# Set default font size
mpl.rcParams['font.size'] = 7

# Activating text rendering by LaTex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    
})


# Get the current directory (where the script is located)
current_dir = os.path.dirname(os.path.abspath("PostProcess_SnapShots.py"))

# Check if the 'Plots' folder exists in the parent folder
plots_folder_path = os.path.join(current_dir, "Plots")
if not os.path.exists(plots_folder_path):
    # If 'Plots' folder doesn't exist, create it
    os.makedirs(plots_folder_path)


def nash_sutcliffe_efficiency(observed, simulated):
    """Calculate the Nash-Sutcliffe Efficiency (NSE)"""
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse

def calculate_mean_rel_error(h, h_analytical):
    abs_diff = np.abs(h - h_analytical)
    rel_diff = abs_diff / np.abs(h_analytical)
    mre_per_timestep = np.mean(rel_diff, axis=0)
    return mre_per_timestep
def calculate_rmse(h, h_analytical):
    squared_diff = (h - h_analytical) ** 2
    mse_per_timestep = np.mean(squared_diff, axis=0)
    rmse_per_timestep = np.sqrt(mse_per_timestep)
    return rmse_per_timestep
def calculate_l2_error(h, h_analytical):
    squared_diff = (h - h_analytical) ** 2
    l2_error_per_timestep = np.sqrt(np.mean(squared_diff, axis=0))
    return l2_error_per_timestep


# Import the array of h and u from analytical solution (Ritter's Solution)
h = np.load('h.npy')
u = np.load('u.npy')

h_analytical = pd.read_csv('h_analytical.csv', header=None)
u_analytical = pd.read_csv('u_analytical.csv', header=None)
# Convert DataFrame to arrays
h_analytical = h_analytical.values
u_analytical = u_analytical.values

absolute_error_h = np.abs(h - h_analytical)
absolute_error_u = np.abs(u - u_analytical)

mean_relative_error = calculate_mean_rel_error(h, h_analytical) 
rmse = calculate_rmse(h, h_analytical)
l2_error = calculate_l2_error(h, h_analytical) 
x_start = 0
x_end = 100
time_start = 0
time_end = 8

print_timesteps = [0, 25, 75, 99]

total_points = len(h[:,0])
num_time_steps = len(h[0,:])
dt = time_end / (num_time_steps-1)
x = np.linspace(x_start, x_end, total_points)
t = np.linspace(time_start, time_end, num_time_steps)
X, T = np.meshgrid(x,t) 
input = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))



h_flat = h.flatten()
h_analytical_flat = h_analytical.flatten()
u_flat = u.flatten()
u_analytical_flat = u_analytical.flatten()


# Calculate RMSE and NSE for water depth
mse_h = mean_squared_error(h_analytical_flat, h_flat)
rmse_h = np.sqrt(mse_h)
nse_h = nash_sutcliffe_efficiency(h_analytical_flat, h_flat)

# Calculate RMSE and NSE for velocity
mse_u = mean_squared_error(u_analytical_flat, u_flat)
rmse_u = np.sqrt(mse_u)
nse_u = nash_sutcliffe_efficiency(u_analytical_flat, u_flat)

# Define the path and filename for the output file
output_file = 'evaluation_results.txt'

# Write the results to the output file
with open(output_file, 'w') as file:
    file.write("RMSE for water depth: " + str(rmse_h) + "\n")
    file.write("Nash-Sutcliffe Efficiency (NSE) for water depth: " + str(nse_h) + "\n")
    file.write("RMSE for velocity: " + str(rmse_u) + "\n")
    file.write("Nash-Sutcliffe Efficiency (NSE) for velocity: " + str(nse_u) + "\n")

print("Results written to", output_file)

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
title = r'$ \mathcal{L}_{Total} = 10^2 \times \mathcal{L}_{MassBalance} + 10^2 \times \mathcal{L}_{Momentum} + 10^5 \times \mathcal{L}_{IC\ depth} + 10^4 \times \mathcal{L}_{IC\ velocity}$'
axes.set_title(title, loc='center', wrap=True, fontsize = 7)
# Set y-axis to log scale
plt.yscale('log')
# Set x-axis limits from the minimum to maximum value of 'iterations'
#plt.xlim(min(iterations), max(iterations))
# Add vertical dashed line at iterations = 30000
plt.axvline(x=35000, linestyle='--', color='gray')
# Add text on the left side of the line (adam optimizer)
plt.text(33000, 1e4, 'Adam optimizer', rotation=0, va='center', ha='right')
# Add text on the right side of the line (L-BFGS optimizer)
plt.text(37000, 1e4, 'L-BFGS optimizer', rotation=0, va='center', ha='left')
filename = os.path.join(plots_folder_path,"total_loss")
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps') 
# Show the plot
plt.show()

# Create the figure for selected timesteps
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Loop through each timestep
for timestep in range(h.shape[1]):
    # Clear the axis
    ax.clear()
    timestamp = (timestep) * dt
    if t[timestep] == print_timesteps[0] or timestep ==  print_timesteps[1] or timestep ==  print_timesteps[2] or timestep ==  print_timesteps[3]:
        
        # Plot the water depth at the current timestep
        ax.plot(x, h_analytical[:, timestep], label='Analytical')
        ax.plot(x, h[:, timestep], linestyle='--', label='PINN')
        
        # Fill the area between the curves
        ax.fill_between(x, 0, h[:, timestep], color='skyblue', alpha=0.5)
        # Set the axis labels and title
        ax.set_xlabel('x-distance [m]')
        ax.set_ylabel(r'water depth $h$ [m]')
        ax.set_title(f'time: {timestamp:.2f} s')
        ax.set_xlim([x_start, x_end])
        ax.set_ylim([0, 1.2])
        ax.legend(fontsize = 7)  # Add legend
        # Add grid to the plot
        ax.grid(True)
        plt.tight_layout()
        filename = os.path.join(plots_folder_path,'h_' + "{:.2f}".format(t[timestep]))
       
        # Save the current figure to the file
        plt.savefig(filename + '.png', format='png')  # Fixed filename argument
        plt.savefig(filename + '.eps', format='eps') 
        
# Show the current figure
plt.show()

# Create the figure for selected timesteps
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Loop through each timestep
for timestep in range(u.shape[1]):
    # Clear the axis
    ax.clear()
    timestamp = (timestep) * dt
    if t[timestep] == print_timesteps[0] or timestep ==  print_timesteps[1] or timestep ==  print_timesteps[2] or timestep ==  print_timesteps[3]:
        
        # Plot the water depth at the current timestep
        ax.plot(x, u_analytical[:, timestep], label='Analytical')
        ax.plot(x, u[:, timestep], linestyle='--', label='PINN')
        
        # Fill the area between the curves
        ax.fill_between(x, 0, u[:, timestep], color='skyblue', alpha=0.5)
        # Set the axis labels and title
        ax.set_xlabel('x-distance [m]')
        ax.set_ylabel(r'velocity $u$ [m]')
        ax.set_title(f'time: {timestamp:.2f} s')
        ax.set_xlim([x_start, x_end])
        ax.set_ylim([0, 1.2])
        ax.legend(fontsize = 7)  # Add legend
        # Add grid to the plot
        ax.grid(True)
        plt.tight_layout()
        filename = os.path.join(plots_folder_path,'u_' + "{:.2f}".format(t[timestep]))
       
        # Save the current figure to the file
        plt.savefig(filename + '.png', format='png')  # Fixed filename argument
        plt.savefig(filename + '.eps', format='eps')   
# Show the current figure
plt.show()


# Create a 2D plot with a color map
plt.imshow(h.T, aspect='auto', extent=[x.min(), x.max(), t.min(), t.max()], origin='lower', cmap='cmo.amp', vmin=0, vmax=h.max())
# Add colorbar for reference
plt.colorbar(label='Water Depth $h$ [m]')
# Add labels and title
plt.xlabel('x-distance [m]')
plt.ylabel('Time [s]')
plt.title('$h$ from PINN over space and time')
filename = os.path.join(plots_folder_path,'h_space_time')
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps')   
# Show the plot
plt.show()

# Create a 2D plot with a color map
plt.imshow(h_analytical.T, aspect='auto', extent=[x.min(), x.max(), t.min(), t.max()], origin='lower', cmap='cmo.amp', vmin=0, vmax=h_analytical.max())
# Add colorbar for reference
plt.colorbar(label='Water Depth $h$ [m]')
# Add labels and title
plt.xlabel('x-distance [m]')
plt.ylabel('Time [s]')
plt.title('$h$ from analytical solution over space and time')
filename = os.path.join(plots_folder_path,'h_analytical_space_time')
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps')
# Show the plot
plt.show()

# Create a 2D plot with a color map
plt.imshow(absolute_error_h.T, aspect='auto', extent=[x.min(), x.max(), t.min(), t.max()], origin='lower', cmap='cmo.amp', vmin=0, vmax=h.max())
# Add colorbar for reference
plt.colorbar(label='Absolute error in Water Depth $h$ [m]')
# Add labels and title
plt.xlabel('x-distance [m]')
plt.ylabel('Time [s]')
plt.title(' $|h - h_{analytical}|$ over space and time')
filename = os.path.join(plots_folder_path,'abs_error_h')
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps')
# Show the plot
plt.show()

# Create a 2D plot with a color map
plt.imshow(u.T, aspect='auto', extent=[x.min(), x.max(), t.min(), t.max()], origin='lower', cmap='cmo.amp', vmin=0, vmax=u_analytical.max())
# Add colorbar for reference
plt.colorbar(label='water velocity $u$ [m/s]')
# Add labels and title
plt.xlabel('x-distance [m]')
plt.ylabel('Time [s]')
plt.title('$u$ from PINN over space and time')
filename = os.path.join(plots_folder_path,'u_space_time')
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps')   
# Show the plot
plt.show()

# Create a 2D plot with a color map
plt.imshow(u_analytical.T, aspect='auto', extent=[x.min(), x.max(), t.min(), t.max()], origin='lower', cmap='cmo.amp', vmin=0, vmax=u_analytical.max())
# Add colorbar for reference
plt.colorbar(label='water velocity $u$ [m/s]')
# Add labels and title
plt.xlabel('x-distance [m]')
plt.ylabel('Time [s]')
plt.title('$u$ from analytical solution over space and time')
filename = os.path.join(plots_folder_path,'u_analytical_space_time')
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps')
# Show the plot
plt.show()

# Create a 2D plot with a color map
plt.imshow(absolute_error_u.T, aspect='auto', extent=[x.min(), x.max(), t.min(), t.max()], origin='lower', cmap='cmo.amp', vmin=0, vmax=h.max())
# Add colorbar for reference
plt.colorbar(label='Absolute error in velocity $u$ [m/s]')
# Add labels and title
plt.xlabel('x-distance [m]')
plt.ylabel('Time [s]')
plt.title(' $|u - u_{analytical}|$ over space and time')
filename = os.path.join(plots_folder_path,'abs_error_u')
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps')
# Show the plot
plt.show()