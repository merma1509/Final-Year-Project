import numpy as np
from matplotlib import pyplot as plt
from NBody import NBodySimulator

def earth_stable_orbit(r):
    G = 6.67408e-11  # Gravitational constant
    massE = 5.974e24  # Earth's mass in Kg
    rE = 6.3781e6  # Earth's radius in m
    return np.sqrt(G * massE / (r + rE))

# Earth and Satellite Properties
massE, rE = 5.974e24, 6.378e6 
rS = 760e3  
masses = [massE, 250e3] 
T, dt = 500 * 60, 1 

def run_simulation(velocity_factor):
    """
    Run simulation with a given velocity factor.
    """
    X = np.array([
        [0, 0, 0, 0], 
        [rE + rS, 0, 0, earth_stable_orbit(rS) * velocity_factor] 
    ])
    sim = NBodySimulator(Xi=X, masses=masses)
    return sim.simulations_run(T, dt), sim.energies

# Run simulations
unperturbed_results, unperturbed_energies = run_simulation(1.0)
perturbed_results, perturbed_energies = run_simulation(1.1)
stabilized_results, stabilized_energies = run_simulation(1.0) 

# Generate Earth Circle for Plotting
theta = np.linspace(0, 2 * np.pi, 150)
earth_x = rE * np.cos(theta)
earth_y = rE * np.sin(theta)

# Set up Figure and Axes
fig, ax = plt.subplots(figsize=(10, 5))

# Plot Orbits
for results, label, style in zip([unperturbed_results, perturbed_results, stabilized_results],
                                 ["Unperturbed", "Perturbed", "Stabilized"],
                                 ['dashed', 'dotted', 'dashed']):
    ax.plot(results[:, 1, 0], results[:, 1, 1], linestyle=style, label=label)
ax.plot(earth_x, earth_y, linestyle='dashed', label='Earth Surface')
ax.set_aspect('equal')
ax.legend()
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")
ax.set_title("Orbits")

'''# Energy Plot
for energies, label in zip([unperturbed_energies, perturbed_energies, stabilized_energies],
                           ["Unperturbed", "Perturbed", "Stabilized"]):
    energy_times = np.arange(0, energies.shape[0] * dt, dt) / 60
    ax2.plot(energy_times, energies[:, 2], label=label)
ax2.set_xlabel("Time (Minutes)")
ax2.set_ylabel("Total Energy (J)")
ax2.legend()
ax2.set_title("Energy Conservation")

# Distance from Initial Position
iters = unperturbed_results.shape[0]
times = np.arange(0, iters * dt, dt) / 60
dists_unperturbed = [np.linalg.norm(unperturbed_results[i, 1, :2] - unperturbed_results[0, 1, :2]) for i in range(iters)]
dists_perturbed = [np.linalg.norm(perturbed_results[i, 1, :2] - perturbed_results[0, 1, :2]) for i in range(iters)]
dists_stabilized = [np.linalg.norm(stabilized_results[i, 1, :2] - stabilized_results[0, 1, :2]) for i in range(iters)]

ax3.plot(times, dists_unperturbed, label="Unperturbed")
ax3.plot(times, dists_perturbed, label="Perturbed")
ax3.plot(times, dists_stabilized, label="Stabilized")
ax3.set_xlabel("Time (Minutes)")
ax3.set_ylabel("Distance from Initial Position (m)")
ax3.legend()
ax3.set_title("Deviation from Initial Position")'''

plt.tight_layout()
plt.show()
