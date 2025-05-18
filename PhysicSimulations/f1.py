import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

class OrbitalSystem:
    def __init__(self):
        # Constants
        self.G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
        self.M = 5.972e24     # Mass of Earth (kg)
        self.m = 1e3          # Mass of satellite (kg)
        self.r0 = 7.0e6       # Initial radial distance (m)
        self.theta0 = 0.0     # Initial angular position (rad)
        self.v0 = 7.5e3       # Initial tangential velocity (m/s)

        # Perturbation parameters (initial values)
        self.Fr = 10.1        # Radial perturbation force (m/s^2)
        self.Ftheta = 10.1    # Tangential perturbation force (m/s^2)

        # Stabilization parameters
        self.lambda1 = 0.5    # Stabilization coefficient 1 (velocity term)
        self.lambda2 = 1.0    # Stabilization coefficient 2 (position term)

        # Initial conditions: [r, pr, theta, ptheta]
        self.p_r0 = self.m * 0.01          # Initial radial momentum (dr/dt = 0)
        self.p_theta0 = self.m * self.r0 * self.v0  # Initial angular momentum
        self.y0 = [self.r0, self.p_r0, self.theta0, self.p_theta0]

        # Time parameters
        self.t_span = (0, 10000)  # Simulation time (s)
        self.t_eval = np.linspace(self.t_span[0], self.t_span[1], 5000)  # Time points for evaluation

    # Unperturbed system
    def unperturbed_system(self, t, y):
        r, pr, theta, ptheta = y
        dr_dt = pr / self.m
        dpr_dt = (ptheta**2) / (self.m * r**3) - (self.G * self.M * self.m) / (r**2)
        dtheta_dt = ptheta / (self.m * r**2)
        dptheta_dt = 0  # Angular momentum is conserved
        return [dr_dt, dpr_dt, dtheta_dt, dptheta_dt]

    # Perturbed system
    def perturbed_system(self, t, y):
        r, pr, theta, ptheta = y
        dr_dt = pr / self.m
        dpr_dt = (ptheta**2) / (self.m * r**3) - (self.G * self.M * self.m) / (r**2) + self.update_perturbations(1000 * t)[0]
        dtheta_dt = ptheta / (self.m * r**2)
        dptheta_dt = self.update_perturbations(1000 * t)[1] * r  # Tangential perturbation
        return [dr_dt, dpr_dt, dtheta_dt, dptheta_dt]

    # Stabilized system
    def stabilized_system(self, t, y):
        r, pr, theta, ptheta = y
        dr_dt = pr / self.m
        dpr_dt = (ptheta**2) / (self.m * r**3) - (self.G * self.M * self.m) / (r**2) + self.update_perturbations(1000 * t)[0] - 2 * self.lambda1 * (dr_dt + self.lambda2**2 * (r - self.r0))
        dtheta_dt = ptheta / (self.m * r**2)
        dptheta_dt = self.update_perturbations(1000 * t)[1] * r - 2 * self.lambda1 * (dtheta_dt + self.lambda2**2 * (theta - self.theta0))
        return [dr_dt, dpr_dt, dtheta_dt, dptheta_dt]

    # Solve the systems
    def solve_systems(self):
        self.unperturbed_sol = solve_ivp(self.unperturbed_system, self.t_span, self.y0, t_eval=self.t_eval, method='LSODA')
        self.perturbed_sol = solve_ivp(self.perturbed_system, self.t_span, self.y0, t_eval=self.t_eval, method='LSODA')
        self.stabilized_sol = solve_ivp(self.stabilized_system, self.t_span, self.y0, t_eval=self.t_eval, method='LSODA')

    # Convert polar to Cartesian coordinates
    def polar_to_cartesian(self, r, theta):
        return r * np.cos(theta), r * np.sin(theta)

    # Prepare data for plotting
    def prepare_data(self):
        self.r_unperturbed, self.theta_unperturbed = self.unperturbed_sol.y[0], self.unperturbed_sol.y[2]
        self.r_perturbed, self.theta_perturbed = self.perturbed_sol.y[0], self.perturbed_sol.y[2]
        self.r_stabilized, self.theta_stabilized = self.stabilized_sol.y[0], self.stabilized_sol.y[2]

        self.x_unperturbed, self.y_unperturbed = self.polar_to_cartesian(self.r_unperturbed, self.theta_unperturbed)
        self.x_perturbed, self.y_perturbed = self.polar_to_cartesian(self.r_perturbed, self.theta_perturbed)
        self.x_stabilized, self.y_stabilized = self.polar_to_cartesian(self.r_stabilized, self.theta_stabilized)

    # Update perturbation coefficients dynamically
    def update_perturbations(self, t):
        # Example: Increase perturbations over time
        self.Fr = 10.1 * (1 + 0.1 * np.sin(t))  # Oscillating perturbation
        self.Ftheta = 10.1 * (1 + 0.1 * np.cos(t))  # Oscillating perturbation
        return self.Fr, self.Ftheta

    # Animate the systems
    def animate(self):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        # Plot for unperturbed system
        axs[0, 0].set_xlim(-1.5 * self.r0, 1.5 * self.r0)
        axs[0, 0].set_ylim(-1.5 * self.r0, 1.5 * self.r0)
        axs[0, 0].plot(0, 0, color='yellow', marker='o', markersize=15, label='Earth')
        axs[0, 0].set_aspect("equal")
        axs[0, 0].set_title("Unperturbed System")
        axs[0, 0].set_xlabel("x (m)")
        axs[0, 0].set_ylabel("y (m)")
        axs[0, 0].grid()

        # Plot for perturbed system
        axs[0, 1].set_xlim(-1.5 * self.r0, 1.5 * self.r0)
        axs[0, 1].set_ylim(-1.5 * self.r0, 1.5 * self.r0)
        axs[0, 1].plot(0, 0, color='yellow', marker='o', markersize=15, label='Earth')
        axs[0, 1].set_aspect("equal")
        axs[0, 1].set_title("Perturbed System")
        axs[0, 1].set_xlabel("x (m)")
        axs[0, 1].set_ylabel("y (m)")
        axs[0, 1].grid()

        # Plot for stabilized system
        axs[1, 0].set_xlim(-1.5 * self.r0, 1.5 * self.r0)
        axs[1, 0].set_ylim(-1.5 * self.r0, 1.5 * self.r0)
        axs[1, 0].plot(0, 0, color='yellow', marker='o', markersize=15, label='Earth')
        axs[1, 0].set_aspect("equal")
        axs[1, 0].set_title("Stabilized System")
        axs[1, 0].set_xlabel("x (m)")
        axs[1, 0].set_ylabel("y (m)")
        axs[1, 0].grid()

        # Plot for combined system
        axs[1, 1].set_xlim(-1.5 * self.r0, 1.5 * self.r0)
        axs[1, 1].set_ylim(-1.5 * self.r0, 1.5 * self.r0)
        axs[1, 1].plot(0, 0, color='yellow', marker='o', markersize=15, label='Earth')
        axs[1, 1].set_aspect("equal")
        axs[1, 1].set_title("Combined Systems")
        axs[1, 1].set_xlabel("x (m)")
        axs[1, 1].set_ylabel("y (m)")
        axs[1, 1].grid()

        # Initialize lines for animation
        line_unperturbed, = axs[0, 0].plot([], [], "b--", lw=1, label="Unperturbed")
        point_unperturbed, = axs[0, 0].plot([], [], "ro", markersize=4, label="Satellite")

        line_perturbed, = axs[0, 1].plot([], [], "r--", lw=1, label="Perturbed")
        point_perturbed, = axs[0, 1].plot([], [], "bo", markersize=4, label="Satellite")

        line_stabilized, = axs[1, 0].plot([], [], "g-", lw=1, label="Stabilized")
        point_stabilized, = axs[1, 0].plot([], [], "mo", markersize=4, label="Satellite")

        line_combined1, = axs[1, 1].plot([], [], "b--", lw=1, label="Unperturbed")
        line_combined2, = axs[1, 1].plot([], [], "r--", lw=1, label="Perturbed")
        line_combined3, = axs[1, 1].plot([], [], "g-", lw=1, label="Stabilized")

        # Add legends
        for ax in axs.flat:
            ax.legend()

        # Initialization function for animation
        def init():
            line_unperturbed.set_data([], [])
            point_unperturbed.set_data([], [])
            line_perturbed.set_data([], [])
            point_perturbed.set_data([], [])
            line_stabilized.set_data([], [])
            point_stabilized.set_data([], [])
            line_combined1.set_data([], [])
            line_combined2.set_data([], [])
            line_combined3.set_data([], [])
            return (line_unperturbed, point_unperturbed, line_perturbed, point_perturbed,
                    line_stabilized, point_stabilized, line_combined1, line_combined2, line_combined3)

        # Update function for animation
        def update(frame):
            # Update perturbation coefficients dynamically
            self.update_perturbations(self.t_eval[frame])

            # Update unperturbed system
            line_unperturbed.set_data(self.x_unperturbed[:frame], self.y_unperturbed[:frame])
            point_unperturbed.set_data([self.x_unperturbed[frame]], [self.y_unperturbed[frame]])

            # Update perturbed system
            line_perturbed.set_data(self.x_perturbed[:frame], self.y_perturbed[:frame])
            point_perturbed.set_data([self.x_perturbed[frame]], [self.y_perturbed[frame]])

            # Update stabilized system
            line_stabilized.set_data(self.x_stabilized[:frame], self.y_stabilized[:frame])
            point_stabilized.set_data([self.x_stabilized[frame]], [self.y_stabilized[frame]])

            # Update combined system
            line_combined1.set_data(self.x_unperturbed[:frame], self.y_unperturbed[:frame])
            line_combined2.set_data(self.x_perturbed[:frame], self.y_perturbed[:frame])
            line_combined3.set_data(self.x_stabilized[:frame], self.y_stabilized[:frame])

            return (line_unperturbed, point_unperturbed, line_perturbed, point_perturbed,
                    line_stabilized, point_stabilized, line_combined1, line_combined2, line_combined3)

        # Create animation
        ani = FuncAnimation(fig, update, frames=len(self.t_eval), init_func=init, blit=True, interval=10)

        plt.tight_layout()
        plt.show()


# Run the simulation
orbital_system = OrbitalSystem()
orbital_system.solve_systems()
orbital_system.prepare_data()
orbital_system.animate()