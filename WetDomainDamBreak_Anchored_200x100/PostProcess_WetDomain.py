# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:26:22 2023

@author: DTL-Hiwi
"""
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import scipy.io
import pandas as pd
from io import BytesIO
from sklearn.metrics import mean_squared_error
import matplotlib as mpl

# Set default font size
mpl.rcParams['font.size'] = 8



def nash_sutcliffe_efficiency(observed, simulated):
    """Calculate the Nash-Sutcliffe Efficiency (NSE)"""
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse = 1 - (numerator / denominator)
    return nse


# Import the array of h and u from analytical solution (Ritter's Solution)
h = np.load('h.npy')
u = np.load('u.npy')

h_analytical = pd.read_csv('h_analytical.csv', header=None)
u_analytical = pd.read_csv('u_analytical.csv', header=None)
# Convert DataFrame to arrays
h_analytical = h_analytical.values
u_analytical = u_analytical.values
x_start = 0
x_end = 100
time_start = 0
time_end = 8

total_points = len(h[:,0])
num_time_steps = len(h[0,:])
dt = time_end / num_time_steps
x = np.linspace(x_start, x_end, total_points)
t = np.linspace(time_start, time_end, num_time_steps)
X, T = np.meshgrid(x,t) 
input = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))


### Post Processing
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Create a list to store the generated frames
frames = []

# Loop through each timestep
for timestep in range(h.shape[1]):
    # Clear the axis
    ax.clear()
    
    # Plot the water depth at the current timestep
    
    ax.plot(x, h_analytical[:, timestep], label='Analytical')
    ax.plot(x, h[:, timestep], linestyle='--', label='PINN')
    
    # Fill the area between the curves
    ax.fill_between(x, 0, h[:, timestep], color='skyblue', alpha=0.5)
   # ax.fill_between(x, 0, h_values_transpose[:, timestep], color='lightgreen', alpha=0.5)
    
    timestamp = (timestep+1) * dt
    # Set the axis labels and title
    ax.set_xlabel('x-distance [m]')
    ax.set_ylabel('Water depth h [m]')
    ax.set_title(f'Time: {timestamp:.2f} s')
    ax.set_xlim([x_start, x_end])
    ax.set_ylim([0, 1.2])
    ax.legend()  # Add legend
    
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
mp4_filename = 'water_depth_animation.mp4'
imageio.mimsave(mp4_filename, frames, fps=10)

# Show the final animation
plt.show()


#Plot for velocity

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(4.5, 3.5))

# Create a list to store the generated frames
frames = []

# Loop through each timestep
for timestep in range(u.shape[1]):
    # Clear the axis
    ax.clear()
    
    # Plot the water depth at the current timestep
    
    ax.plot(x, u_analytical[:, timestep], label='Analytical')
    ax.plot(x, u[:, timestep], linestyle='--', label='PINN')
    
    timestamp = (timestep+1) * dt
    # Set the axis labels and title
    ax.set_xlabel('x-distance [m]')
    ax.set_ylabel('Water velocity u [m/s]')
    ax.set_title(f'Time: {timestamp:.2f} s')
    ax.set_xlim([x_start, x_end])
    #ax.set_ylim([0.8*np.min(u_analytical), 1.2*np.max(u_analytical)])
    ax.set_ylim([0, 1.5])
    ax.legend()  # Add legend
    
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
mp4_filename = 'water_velocity_animation.mp4'
imageio.mimsave(mp4_filename, frames, fps=10)

# Show the final animation
plt.show()


# h_flat = h.flatten()
# h_analytical_flat = h_analytical.flatten()
# u_flat = u.flatten()
# u_analytical_flat = u_analytical.flatten()


# # Calculate RMSE and NSE for water depth
# mse_h = mean_squared_error(h_analytical_flat, h_flat)
# rmse_h = np.sqrt(mse_h)
# nse_h = nash_sutcliffe_efficiency(h_analytical_flat, h_flat)

# # Calculate RMSE and NSE for velocity
# mse_u = mean_squared_error(u_analytical_flat, u_flat)
# rmse_u = np.sqrt(mse_u)
# nse_u = nash_sutcliffe_efficiency(u_analytical_flat, u_flat)

# # Define the path and filename for the output file
# output_file = 'evaluation_results.txt'

# # Write the results to the output file
# with open(output_file, 'w') as file:
#     file.write("RMSE for water depth: " + str(rmse_h) + "\n")
#     file.write("Nash-Sutcliffe Efficiency (NSE) for water depth: " + str(nse_h) + "\n")
#     file.write("RMSE for velocity: " + str(rmse_u) + "\n")
#     file.write("Nash-Sutcliffe Efficiency (NSE) for velocity: " + str(nse_u) + "\n")

# print("Results written to", output_file)