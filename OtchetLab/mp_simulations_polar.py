from mp_simulations import MaterialPointMotion
import numpy as np
import matplotlib.pyplot as plt

class Polar(MaterialPointMotion):
    def __init__(self, k, m, c, alpha=0.5, beta=1.0, A=0.001, omega=1.0):
        """
        Initializes the Polar class for material point motion.
        """
        super().__init__(k, m, c, alpha, beta, A, omega)

    def plot_trajectories_in_polar(self, theta_span, Y0, theta_eval):
        """
        Plots the trajectories in polar coordinates for unperturbed, perturbed, and stabilized systems.
        """
        try:
            # Solve unperturbed system
            sol_unperturbed = self.solve_system(self.unperturbed_system, theta_span, Y0, theta_eval)

            # Solve perturbed system
            sol_perturbed = self.solve_system(self.perturbed_system, theta_span, Y0, theta_eval)

            # Stabilized system (using nominal solution from unperturbed system)
            u_nominal_func = lambda theta: np.interp(theta, sol_unperturbed.t, sol_unperturbed.y[0])
            sol_stabilized = self.solve_system(self.stabilized_system, theta_span, Y0, theta_eval, args=(u_nominal_func,))

            # Extract results
            u_unperturbed = sol_unperturbed.y[0]
            u_perturbed = sol_perturbed.y[0]
            u_stabilized = sol_stabilized.y[0]

            # Compute radial distances
            r_unperturbed = 1 / u_unperturbed
            r_perturbed = 1 / u_perturbed
            r_stabilized = 1 / u_stabilized

            # Plot results in polar coordinates
            plt.figure(figsize=(15, 10))
            ax = plt.subplot(111, projection='polar')

            ax.plot(theta_eval, r_unperturbed, label='Ideal Trajectory')
            ax.plot(theta_eval, r_perturbed, label='Perturbed Trajectory', linestyle='--')
            ax.plot(theta_eval, r_stabilized, label='Stabilized Trajectory', linestyle='-.')

            ax.set_title('Orbital Trajectories in Polar Coordinates')
            ax.grid(True)
            ax.legend()

            plt.show()

        except AttributeError as e:
            print(f"Error: {e}. Ensure that all required methods are implemented.")

# Parameters
k = 1.0
m = 1.0
c = 1.0
u0 = 0.1
v0 = 0.0
Y0 = [u0, v0]
theta_span = (0, 8 * np.pi)
theta_eval = np.linspace(*theta_span, 1000)

# Create a Polar object and plot trajectories
polar_plotter = Polar(k, m, c)
polar_plotter.plot_trajectories_in_polar(theta_span, Y0, theta_eval)
