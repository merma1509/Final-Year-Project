import numpy as np
from NBody import NBodySimulator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

names = ["m0", 'm1', "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]
X = np.array([
    [ 1.81899E+08,  9.83630E+08, -1.58778E+08, -1.12474E+01,  7.54876E+00,  2.68723E-01],
    [-5.67576E+10, -2.73592E+10,  2.89173E+09,  1.16497E+04, -4.14793E+04, -4.45952E+03],
    [ 4.28480E+10,  1.00073E+11, -1.11872E+09, -3.22930E+04,  1.36960E+04,  2.05091E+03],
    [-1.43778E+11, -4.00067E+10, -1.38875E+07,  7.65151E+03, -2.87514E+04,  2.08354E+00],
    [-1.14746E+11, -1.96294E+11, -1.32908E+09,  2.18369E+04, -1.01132E+04, -7.47957E+02],
    [-5.66899E+11, -5.77495E+11,  1.50755E+10,  9.16793E+03, -8.53244E+03, -1.69767E+02],
    [ 8.20513E+10, -1.50241E+12,  2.28565E+10,  9.11312E+03,  4.96372E+02, -3.71643E+02],
    [ 2.62506E+12,  1.40273E+12, -2.87982E+10, -3.25937E+03,  5.68878E+03,  6.32569E+01],
    [ 4.30300E+12, -1.24223E+12, -7.35857E+10,  1.47132E+03,  5.25363E+03, -1.42701E+02],
    [ 1.65554E+12, -4.73503E+12,  2.77962E+10,  5.24541E+03,  6.38510E+02, -1.60709E+03]
])
masses = np.array([1.98854E+30, 3.30200E+23, 4.86850E+24, 5.97219E+24, 6.41850E+23,
                  1.89813E+27, 5.68319E+26, 8.68103E+25, 1.02410E+26, 1.30700E+22])

n = 5 # Number of planets to remove from the end
names = names[0:-n]
X = X[:-n]
masses = masses[0:-n]

# Time step is an Hour
SolarSystem = NBodySimulator(Xi=X, masses=masses)
T, dt = 2 * 365 * 24 * 60 ** 2, 60 ** 2     # 2 Years, 1 hour step
sim_results = SolarSystem.simulations_run(T, dt)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection='3d')
colors = ["yellow", "gray", "orange", "deepskyblue", "red", "chocolate", 
          "bisque", "lightcyan", "blue", "lightgray"]


# Initialize plots
lines = [ax.plot([], [], [], color=colors[i], linewidth=1.5)[0] for i in range(len(names))]
points = [ax.plot([], [], [], 'o', ms=10, mfc=colors[c], mec=colors[c] , mew=0.5, label=names[c])[0]
           for c in range(len(colors[:-n]))]

# Set axis labels
ax.set_title("Dynamics of N-Body Systems")
ax.set_xlabel('X-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Z-Axis')
ax.grid(True)
ax.legend(loc='upper left')

# Animation function
def update(frame):
    # Get min and max values dynamically
    x_min, x_max = sim_results[:, :, 0].min(), sim_results[:, :, 0].max()
    y_min, y_max = sim_results[:, :, 1].min(), sim_results[:, :, 1].max()
    z_min, z_max = sim_results[:, :, 2].min(), sim_results[:, :, 2].max()

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    for i, line in enumerate(lines):
        line.set_data(sim_results[:frame, i, 0], sim_results[:frame, i, 1])
        line.set_3d_properties(sim_results[:frame, i, 2])

        # Update moving points
        points[i].set_data([sim_results[frame, i, 0]], [sim_results[frame, i, 1]])
        points[i].set_3d_properties([sim_results[frame, i, 2]])

    return lines + points

# Create animation
ani = FuncAnimation(fig, update, frames=len(sim_results), interval=10, blit=True)
# ani.save('nbody.gif', writer='pillow')

plt.show()
