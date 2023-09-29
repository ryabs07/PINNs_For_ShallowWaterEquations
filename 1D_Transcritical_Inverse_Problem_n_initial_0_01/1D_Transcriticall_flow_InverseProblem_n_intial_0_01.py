import deepxde as dde
import matplotlib.pyplot as plt
import pandas as pd

dde.config.set_default_float("float64")

g = 9.81  # Specify the value for gravity constant g

# Read the first 8 columns of the Excel file, skipping the first row (header)
data = pd.read_excel('1D_MacDo_fluvial_mannings.xlsx')
num_columns_to_import = 8
selected_columns = data.iloc[:, :num_columns_to_import]
# Convert the selected DataFrame to a NumPy array
analytic_dataset = selected_columns.values
observe_x, h, u, z = analytic_dataset[:,0].reshape(len(analytic_dataset[:,0]), 1), analytic_dataset[:,1].reshape(len(analytic_dataset[:,0]), 1), analytic_dataset[:,2].reshape(len(analytic_dataset[:,0]), 1), analytic_dataset[:,3].reshape(len(analytic_dataset[:,0]), 1)


n_analytical = 0.0328  #true value of mannings roughness coeff. 

observe_h = dde.icbc.PointSetBC(observe_x, h, component=0)
observe_u = dde.icbc.PointSetBC(observe_x, u, component=1)
observe_z = dde.icbc.PointSetBC(observe_x, z, component=2)

#Define the simplified steady 1D Shallow Water Equations
def pde(a, b):

    x = a[:, 0:1]
    h, u, z = b[:, 0:1], b[:, 1:2], b[:, 2:3]
       
    dh_dx = dde.grad.jacobian(b, a, i=0, j=0)
    du_dx = dde.grad.jacobian(b, a, i=1, j=0)
    dz_dx = dde.grad.jacobian(z, a, i=0, j=0)

    mass_balance = h * du_dx + u * dh_dx 
    momentum_x = u * du_dx + g * dh_dx  + g * dz_dx + ( ( g * u ** 2 * n_pinn **2 ) / ( h ** (4/3) ) ) 
    return [mass_balance, momentum_x]

#Definition of geometry of domain in space
geom = dde.geometry.Interval(observe_x[0], observe_x[-1])
#Assign the neural network to satisfy observed values of water depth, velocity and topography
#anchor the sampling points to exact discretized spatial points from observed data
data = dde.data.TimePDE(geom, pde, [observe_h, observe_u, observe_z], num_domain=0, anchors = observe_x)
#Definition of feed forward neural network architecture
#[input layer dimension] + [hiden layer dimension]*[number of hidden layers] + [output layer  dimension] , "activation function", "initializer of weights"
net = dde.nn.FNN([1] + [30] * 4 + [3], "tanh", "Glorot uniform")
model = dde.Model(data, net)
#save the model at every 500th iteration
checkpoint = dde.callbacks.ModelCheckpoint('Models/Model.ckpt', verbose=1, save_better_only=True, period = 500)
#assign weights to PDE1, PDE2, observed h, u, z in the total loss function
weights = [1e3,1e3,1e3,1e3,1e3]

#initialize the value of mannings roughness for PINN in a realistic range
n_pinn = dde.Variable(0.010) 
#Compile the neural net with Adam optimizer, set learning rate and loss weight, and set n_pinn as trainable variable
model.compile("adam", lr=1e-4, loss_weights = weights, external_trainable_variables=n_pinn) #, loss_weights=[1e4, 1e4, 1e6, 1e6, 1e6, 1e6, 1e6]
variable = dde.callbacks.VariableValue(n_pinn, period=50, filename="Mannings_coeff.dat")
#train the model with Adam optimizer for 50000 iterations
losshistory, train_state = model.train(iterations=50000, callbacks=[checkpoint, variable], display_every=50)#
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
# Save the trained model
model.save("AdamOptimized.h5")

### Post Processing , pass the spatial co-ordinates through optimized neural net
output = model.predict(observe_x) #passing the input gridpoints and timestamps through optimized neural network

# Extracting water depth 'h' and velocity 'u' and topography 'z' from the result
h_pinn = output[:, 0].reshape(len(analytic_dataset[:,0]), 1)
u_pinn = output[:, 1].reshape(len(analytic_dataset[:,0]), 1)
z_pinn = output[:, 2].reshape(len(analytic_dataset[:,0]), 1)

h_error = h - h_pinn
u_error = u - u_pinn
z_error = z - z_pinn

h_error_relative = h_error / h
u_error_relative = u_error / u
z_error_relative = z_error / z


# Create the figure for visualization of problem statement
fig, ax = plt.subplots(figsize=(6, 2.5))
ax.plot(observe_x[:,0], h[:,0] + z[:,0], label='WSL')
ax.plot(observe_x[:,0], z[:,0],color='peru', label='topography')
ax.plot(observe_x[:,0], h_pinn[:,0] + z_pinn[:,0], '--', label='WSL_PINN')
ax.plot(observe_x[:,0], z_pinn[:,0], '--', color='red',label='topography_PINN')
# Fill the area between the curves
ax.fill_between(observe_x[:,0], z[:,0], h[:,0] + z[:,0], color='skyblue', alpha=0.5)
ax.fill_between(observe_x[:,0], -0.2, z[:,0], color='sandybrown', hatch = '////', alpha=0.7)
ax.set_xlim(0,100)
ax.set_ylim(min(z)-0.2, 1.2*max(h+z))
# Set the axis labels and title
ax.set_xlabel('x-distance [m]')
ax.set_ylabel('elevation [m]')
ax.set_title('1D Steady Transcritical Flow')
ax.legend(fontsize = 'small')  # Add legend
ax.grid(True)
plt.show()

# Plotting evolution of n_pinn
mannings_evolution = ((pd.read_csv('Mannings_coeff.dat', delimiter = ' ', header=None)).iloc[:, :]).values
iterations = mannings_evolution[:,0]
mannings_n = mannings_evolution[:,1]
mannings_n = [float(s.strip('[]')) for s in mannings_n]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(iterations, mannings_n, color = 'orangered', label=' manning\'s roughness from PINN')
ax.set_ylim(min(mannings_n), 0.05)
ax.set_xlabel('iterations')
ax.set_ylabel('manning\'s roughness $n$')
plt.axhline(y=0.0328, color='blue', linestyle='--', label='true manning\'s roughness')
ax.grid(True)
ax.legend(fontsize='small')
# Create the title and subtitle
title = "Optimization of Manning's roughness $n$ from Inverse PINN"
subtitle = f"$n_{{actual}} = {n_analytical}$,  $n_{{PINN\,final\,iteration}} = {mannings_n[-1]}$"
full_title = f"{title}\n{subtitle}"
ax.set_title(full_title)
plt.show()
