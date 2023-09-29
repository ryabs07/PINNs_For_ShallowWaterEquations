import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from io import BytesIO
import imageio.v2 as imageio

def equation_h(h, x):
     return (hu_initial**2 / (2 * h**2)) + g * (h + 0.1 + 0.1 * np.exp(-((x  - 5)**2))) - 6.386

def h_initial_func(x_array):
    h_values = np.zeros_like(x_array)  # Create an array to store the results

    for i, x in enumerate(x_array):
        h_values[i] = fsolve(equation_h, h_u_guess, args=(x,))[0]
    
    return h_values 
def u_initial_func(x_array):
    u_values = np.zeros_like(x_array)  # Create an array to store the results

    for i, x in enumerate(x_array):
        h_value = fsolve(equation_h, h_u_guess, args=(x,))[0]
        u_values[i] = hu_initial / h_value
    
    return u_values 


def z_initial_func(x):
    return ( 0.1 + 0.1 * np.exp(-((x - 5)**2)) ) 

# Initial guess for u
h_u_guess = 1.0

#Configuration for the simulation
x_start = 0 #Starting point of domain
x_end = 10 #End point of domain
time_start = 0 #Starting time of simulation
time_end = 10 #Ending time of simulation
g = 9.81  # Specify the value for gravity constant g

hu_initial = 0.5 #initial specific discharge
Ag = 0.005 
m = 3
#porosity = 0.1
eta = 1 #1 / (1-porosity)

h_boundary = 0.5
u_left_boundary = hu_initial / h_boundary

# Preparation of Input for Prediction using PINN
total_points = 500 #number of discretized points in x-direction
x = np.linspace(x_start, x_end, total_points)
num_time_steps = 300 #Number of timesteps
t = np.linspace(time_start, time_end, num_time_steps)
X, T = np.meshgrid(x,t) 
input = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

# Training PINN on additional fixed points near dune
x_anchor = np.linspace(0.35, 0.75, 300)
t_anchor = np.linspace(time_start, time_end, 300)
X_anchor, T_anchor = np.meshgrid(x_anchor,t_anchor) 
input_anchor = np.hstack((X_anchor.flatten()[:,None], T_anchor.flatten()[:,None]))
 

#Define the simplified 1D shallow water + exner equations
def pde(x, y):
    h, u, z = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    #get required partial derivates
    dh_dx = dde.grad.jacobian(h, x, i=0, j=0)
    dh_dt = dde.grad.jacobian(h, x, i=0, j=1)
    du_dx = dde.grad.jacobian(u, x, i=0, j=0)
    du_dt = dde.grad.jacobian(u, x, i=0, j=1)
    dz_dx = dde.grad.jacobian(z, x, i=0, j=0)
    dz_dt = dde.grad.jacobian(z, x, i=0, j=1)
    
    qb = Ag * u ** (m) #bed load flux
    dqb_dx = dde.grad.jacobian(qb, x, i=0, j=0)
    
    mass_balance = dh_dt  + ( h * du_dx + u * dh_dx ) #SWE mass balance
    momentum_x = du_dt  + u * du_dx + g * dh_dx + g * dz_dx #SWE momentum eqn 
    exner_eqn =  dz_dt +  eta * dqb_dx #exner equation for bed load transport
    
    return [mass_balance, momentum_x, exner_eqn]

#Definition of geometry of domain in both spcae and time
geom = dde.geometry.Interval(x_start, x_end)
timedomain = dde.geometry.TimeDomain(time_start, time_end)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#Define initial condition of water depth, velocity and topography
ic_h = dde.icbc.IC(
    geomtime, lambda x: h_initial_func(x[:, 0:1]), lambda _, on_initial: on_initial, component=0
)
ic_u = dde.icbc.IC(
    geomtime, lambda x: u_initial_func(x[:, 0:1]), lambda _, on_initial: on_initial, component=1
)
ic_z = dde.icbc.IC(
    geomtime, lambda x: z_initial_func(x[:, 0:1]), lambda _, on_initial: on_initial, component=2
)

def boundary_l(x, on_boundary):
    return on_boundary and np.isclose(x[0], x_start) #define left boundary
def boundary_r(x, on_boundary):
    return on_boundary and np.isclose(x[0], x_end) #define right boundary

#Define boundary conditions for water depth and velocity
bc_l_h = dde.icbc.DirichletBC(geomtime, lambda x: h_boundary, boundary_l, component=0)
bc_r_h = dde.icbc.DirichletBC(geomtime, lambda x: h_boundary, boundary_r, component=0)
bc_l_u = dde.icbc.DirichletBC(geomtime, lambda x: u_left_boundary, boundary_l, component=1)
bc_r_u = dde.icbc.DirichletBC(geomtime, lambda x: u_left_boundary, boundary_r, component=1)

#Assign the number of points in domain for the training of neural net to satisfy PDEs and initial conditions
data = dde.data.TimePDE(geomtime, pde, [ic_h, ic_u, ic_z, bc_l_h, bc_r_h, bc_l_u, bc_r_u], num_domain=25000,
                        num_initial = 1500, num_boundary=1500, anchors = input)
#Definition of feed forward neural network architecture
#[input layer dimension] + [hiden layer dimension]*[number of hidden layers] + [output layer  dimension] , "activation function", "initializer of weights"
net = dde.nn.FNN([2] + [40] * 6 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)

#define weights for PDEs, Initial Conditions and Boundary Conditions
weights = [1e4, 1e4, 1e6, 1e6, 1e6, 1e6, 1e3, 1e3, 1e3, 1e3]

#save model every 500 iterations
checkpoint = dde.callbacks.ModelCheckpoint('Models/Model.ckpt', verbose=1, save_better_only=True, period = 500)
#Compile the neural net with Adam optimizer, set learning rate and loss weight to mass balance equation, momentum equation, initial condition of h and initial condition of u respectively
model.compile("adam", lr=3e-4, loss_weights= weights ) #
#random resampling of training points every 10 iterations
pde_resampler = dde.callbacks.PDEPointResampler(period=10)
#train with adam optimizer
losshistory, train_print_timestep = model.train(iterations=20000, callbacks=[pde_resampler, checkpoint], display_every=100)#
dde.saveplot(losshistory, train_print_timestep, issave=True, isplot=True)
# Save the trained model
model.save("1stStageAdamOptimized.h5")
output = model.predict(input) #passing the input gridpoints and timestamps through optimized neural network

x_plot = x 
# Extracting water depth 'h', velocity 'u' and topography 'z' from the result
h_PINN = (output[:, 0].reshape((-1, total_points)).T)
u_PINN = (output[:, 1].reshape((-1, total_points)).T)
z_PINN = (output[:, 2].reshape((-1, total_points)).T)
wsl_PINN = h_PINN + z_PINN

state = 0 #initial state
state_final = -1 #final state

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(x_plot, wsl_PINN[:,state], label='initial WSL')
plt.plot(x_plot, z_PINN[:,state], label='initial topography')
plt.plot(x_plot, z_PINN[:,state_final], '--', label='final topography')
plt.plot(x_plot, wsl_PINN[:,state_final], '--', label='final WSL')
plt.fill_between(x_plot, z_PINN[:,state_final], wsl_PINN[:, state_final], color='skyblue', alpha=0.5)
plt.fill_between(x_plot, 0, z_PINN[:,state_final], color='sandybrown', hatch = '...', alpha=0.7)
plt.xlabel('x-distance [m]')
plt.ylabel('elevation [m]')
plt.title('initial vs. final (t=10.0 s) water surface level and topography')
plt.xlim(0, 10)
plt.ylim(0, 0.8)
plt.legend()
plt.grid(True)
# Save the plot as a PNG file
plt.savefig('Plots/1D_UnsteadyErodableBump.png', dpi=400, bbox_inches='tight')
# Save the plot as an EPS file
plt.savefig('Plots/1D_UnsteadyErodableBump.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()

#retrain with L-BFGS for fine tuning the model
checkpoint = dde.callbacks.ModelCheckpoint('Models/Model.ckpt', verbose=1, save_better_only=True, period = 10)
dde.optimizers.set_LBFGS_options(maxiter= 50000, gtol= 1e-5)
model.compile("L-BFGS", loss_weights= weights) #
losshistory, train_print_timestep = model.train(callbacks=[pde_resampler, checkpoint], display_every=100)
dde.saveplot(losshistory, train_print_timestep, issave=True, isplot=True)
# Save the trained model
model.save("2ndStage_LBFGS_Optimzed.h5")

### Post Processing 
output = model.predict(input) #passing the input gridpoints and timestamps through optimized neural network

# Extracting water depth 'h', velocity 'u' and topography 'z' from the result
h_PINN = (output[:, 0].reshape((-1, total_points)).T)
u_PINN = (output[:, 1].reshape((-1, total_points)).T)
z_PINN = (output[:, 2].reshape((-1, total_points)).T)
wsl_PINN = h_PINN + z_PINN

state = 0 
state_final = -1

# Plotting
plt.figure(figsize=(6, 4))
plt.plot(x_plot, wsl_PINN[:,state], label='initial WSL')
plt.plot(x_plot, z_PINN[:,state], label='initial topography')
plt.plot(x_plot, z_PINN[:,state_final], '--', label='final topography')
plt.plot(x_plot, wsl_PINN[:,state_final], '--', label='final WSL')
plt.fill_between(x_plot, z_PINN[:,state_final], wsl_PINN[:, state_final], color='skyblue', alpha=0.5)
plt.fill_between(x_plot, 0, z_PINN[:,state_final], color='sandybrown', hatch = '...', alpha=0.7)
plt.xlabel('x-distance [m]')
plt.ylabel('elevation [m]')
plt.title('initial vs. final (t=10.0 s) water surface level and topography')
plt.xlim(0, 10)
plt.ylim(0, 0.8)
plt.legend()
plt.grid(True)
# Save the plot as a PNG file
plt.savefig('Plots/1D_UnsteadyErodableBump.png', dpi=400, bbox_inches='tight')
# Save the plot as an EPS file
plt.savefig('Plots/1D_UnsteadyErodableBump.eps', format='eps', dpi=300, bbox_inches='tight')
plt.show()


#make a movie clip from initial condition until the final condition
dt = time_end / (num_time_steps-1) # * T_scale
# Post Processing for the movie clip
# Set up the figure and axis
fig, ax = plt.subplots(figsize=(6, 4))
# Create a list to store the generated frames
frames = []
# Loop through each timestep
for timestep in range(num_time_steps):
    # Clear the axis
    ax.clear()
    
    # Plot the water depth at the current timestep
    timestamp = (timestep) * dt
    
    ax.plot(x_plot, wsl_PINN[:, timestep], label='WSL PINN')
    #ax.plot(x_plot, u_PINN[:, timestep], linestyle='--', label='velocity PINN')
    
    ax.plot(x_plot, z_PINN[:, timestep], label='topography PINN')
    # Fill the area between the curves
    ax.fill_between(x_plot, z_PINN[:, timestep], wsl_PINN[:, timestep], color='skyblue', alpha=0.5)
    ax.fill_between(x_plot, -0.4, z_PINN[:, timestep], color='sandybrown', hatch = '...', alpha=0.7)
    
    # Set the axis labels and title
    ax.set_xlabel('x-distance [m]')
    ax.set_ylabel('Elevation [m]')
    ax.set_title('1D Unteady Bedload Transport, ' + f'Time: {timestamp:.2f} s')
    #ax.set_xlim([x_start * L_scale, x_end * L_scale])
    ax.set_xlim([x_start , x_end ])
    ax.set_ylim([0, 0.8])
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
mp4_filename = '1D_UnsteadyBedLoadTransport.mp4'
imageio.mimsave(mp4_filename, frames, fps=30)


data = np.loadtxt("loss.dat", skiprows=1)
  # Extract columns of interest
iterations = data[:, 0]
mass_loss = data[:, 1]
momentum_loss = data[:, 2]
h_ic_loss= data[:, 3]
u_ic_loss = data[:, 4]
z_ic_loss = data[:, 5]
h_bc_left_loss= data[:, 6]
h_bc_right_loss= data[:, 7]
u_bc_left_loss= data[:, 8]
u_bc_right_loss= data[:, 9]

total_loss = mass_loss + momentum_loss + h_ic_loss + u_ic_loss + z_ic_loss + h_bc_left_loss + h_bc_right_loss + u_bc_left_loss + u_bc_right_loss
bc_l_h, bc_r_h, bc_l_u, bc_r_u
fig, axes = plt.subplots(figsize=(5.5, 3.5))
plt.plot(iterations, total_loss, linewidth=1.5, color='darkorange')
# Customize the plot
plt.xlabel('Iterations')
plt.ylabel('Total Loss')
#title = r'$ \mathcal{L}_{Total} = 10^2 \times \mathcal{L}_{Momentum_x} + 10^2 \times \mathcal{L}_{BedLoadMassBalance} + 10^4 \times \mathcal{L}_{IC_{topography}}$'
axes.set_title(title, loc='center', wrap=True, fontsize = 7)
# Set y-axis to log scale
plt.yscale('log')
# Set x-axis limits from the minimum to maximum value of 'iterations'
#plt.xlim(min(iterations), max(iterations))
# Add vertical dashed line at iterations = 30000
plt.axvline(x=20000, linestyle='--', color='gray')
# Add text on the left side of the line (adam optimizer)
plt.text(17000, 1e2, 'Adam optimizer', rotation=0, va='center', ha='right')
# Add text on the right side of the line (L-BFGS optimizer)
plt.text(23000, 1e2, 'L-BFGS optimizer', rotation=0, va='center', ha='left')
filename = 'Plots/total_loss'
# Save the current figure to the file
plt.savefig(filename + '.png', format='png')  # Fixed filename argument
plt.savefig(filename + '.eps', format='eps') 
# Show the plot
plt.show()

