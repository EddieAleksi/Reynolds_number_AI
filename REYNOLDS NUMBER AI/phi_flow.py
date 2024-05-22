from phi.flow import *
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
import numpy as np

plt.rcParams['animation.html'] = 'jshtml'  # Configuring animation for Jupyter

run_time = 150
buoyancy_factor = 1.0  # Corrected spelling
time_step = 1.0
viscosity = 0.01

# Initializing grids
domain_bounds = Box(x=100, y=100)
velocity = StaggeredGrid(0, extrapolation.ZERO, x=200, y=200, bounds=domain_bounds)
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=200, y=200, bounds=domain_bounds)

# Inflow configuration
inflow = 0.2 * CenteredGrid(SoftGeometryMask(Sphere(x=25, y=15, radius=5)), extrapolation.ZERO,
                            bounds=smoke.bounds, resolution=smoke.resolution)

def step(prior_velocity, prior_smoke, dt=time_step):
    smoke_next = advect.mac_cormack(prior_smoke, prior_velocity, dt) + inflow
    buoyancy_force = smoke_next * (0.0, buoyancy_factor) @ velocity
    velocity_temp = advect.semi_lagrangian(prior_velocity, prior_velocity, dt) + buoyancy_force * dt
    velocity_temp = diffuse.explicit(velocity_temp, viscosity, dt)
    velocity_next, pressure = fluid.make_incompressible(velocity_temp)
    return velocity_next, smoke_next

velocity, smoke = step(velocity, smoke)  # Initial step
plt.imshow(np.asarray(smoke.values.numpy('y,x')), origin='lower', cmap='magma')

# Storing steps for visualization
steps = [[smoke.values, velocity.values.vector[0], velocity.values.vector[1]]]

for t in tqdm(range(run_time)):
    velocity, smoke = step(velocity, smoke)
    steps.append([smoke.values, velocity.values.vector[0], velocity.values.vector[1]])

# Visualization of multiple time steps
fig, axes = plt.subplots(5, 5, figsize=(20, 12))
m = 0
for i in range(5):
    for j in range(5):
        axes[i, j].imshow(steps[m][0].numpy('y,x'), origin='lower', cmap='magma')
        axes[i, j].set_title(f"d at t = {m}")
        m += 5  # Correct increment
plt.show()
