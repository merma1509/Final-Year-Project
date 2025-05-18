from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np


class OrbitalSimulation:
    def __init__(self, earth_mass=1, scale_factor=1, perturbation=0):
        """
        Initializes the orbital simulation parameters.
        :param earth_mass: Mass of the earth (arbitrary units)
        :param scale_factor: Scale factor for rendering
        :param perturbation: Perturbation factor for the orbit
        """
        self.G = 1  # 6.67e-11                     # Gravitational constant, normalized
        self.earth_mass = earth_mass
        self.scale_factor = scale_factor
        self.perturbation = perturbation

        # Initial conditions: [r, dr/dt, theta, dtheta/dt]
        self.initial_conditions = [1 + perturbation, 0, 0, 1] 

    def equations(self, t, state):
        """
        Differential equations for the orbital system.
        :param t: Time (unused explicitly, system is time-invariant)
        :param state: State variables [r, dr/dt, theta, dtheta/dt]
        :return: Derivatives [dr/dt, radial_acceleration, dtheta/dt, angular_acceleration]
        """
        r, dr_dt, theta, dtheta_dt = state
        r_accel = -self.G * self.earth_mass / r**2 + r * dtheta_dt**2
        theta_accel = -2 * dr_dt * dtheta_dt / r
        return [dr_dt, r_accel, dtheta_dt, theta_accel]

    def solve_orbit(self, duration=5e1, steps=5000):
        """
        Solves the orbital differential equations.
        :param duration: Simulation duration (arbitrary units)
        :param steps: Number of time steps
        :return: Time array and solution array
        """
        t_span = [0, duration]
        t_eval = np.linspace(t_span[0], t_span[1], 2000)
        solution = solve_ivp(
            self.equations,
            t_span,
            self.initial_conditions,
            t_eval=t_eval,
            method='RK45'
        )
        return solution.t, solution.y


class OrbitalGraphics:
    def __init__(self, sim_unperturbed, sim_perturbed):
        """
        Initializes the graphics for orbital simulation.
        :param sim_unperturbed: Unperturbed OrbitalSimulation object
        :param sim_perturbed: Perturbed OrbitalSimulation object
        """
        self.sim_unperturbed = sim_unperturbed
        self.sim_perturbed = sim_perturbed
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.line_unperturbed, = self.ax.plot([], [], 'g--', label='Unperturbed Orbit')
        self.line_perturbed, = self.ax.plot([], [], 'r--', label='Perturbed Orbit')
        self.satellite, = self.ax.plot([], [], 'b*', markersize=10, label='Satellite')
        self.earth, = self.ax.plot([0], [0], 'ro', markersize=16, label='Earth')

    def init_plot(self):
        """
        Initializes the plot for animation.
        :return: Artists to be initialized
        """
        axis_limit = 1.5 / self.sim_unperturbed.scale_factor  
        self.ax.set_xlim(-axis_limit, axis_limit)
        self.ax.set_ylim(-axis_limit, axis_limit)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.legend()
        self.ax.set_title('Orbital Simulation with Perturbation')
        return self.line_unperturbed, self.line_perturbed, self.satellite, self.earth

    def update_plot(self, frame):
        """
        Updates the animation frame.
        :param frame: Current frame index
        :return: Artists to be updated
        """
        r_u = self.solution_unperturbed[1][0][frame]
        theta_u = self.solution_unperturbed[1][2][frame]
        x_u = r_u * np.cos(theta_u)
        y_u = r_u * np.sin(theta_u)

        r_p = self.solution_perturbed[1][0][frame]
        theta_p = self.solution_perturbed[1][2][frame]
        x_p = r_p * np.cos(theta_p)
        y_p = r_p * np.sin(theta_p)

        # Update Earth's position
        self.satellite.set_data([x_u], [y_u])

        # Update trajectories
        trajectory_x_u = self.solution_unperturbed[1][0][:frame + 1] * np.cos(self.solution_unperturbed[1][2][:frame + 1])
        trajectory_y_u = self.solution_unperturbed[1][0][:frame + 1] * np.sin(self.solution_unperturbed[1][2][:frame + 1])
        self.line_unperturbed.set_data(trajectory_x_u, trajectory_y_u)

        trajectory_x_p = self.solution_perturbed[1][0][:frame + 1] * np.cos(self.solution_perturbed[1][2][:frame + 1])
        trajectory_y_p = self.solution_perturbed[1][0][:frame + 1] * np.sin(self.solution_perturbed[1][2][:frame + 1])
        self.line_perturbed.set_data(trajectory_x_p, trajectory_y_p)

        return self.line_unperturbed, self.line_perturbed, self.satellite, self.earth

    def run(self):
        """
        Runs the orbital simulation and animation.
        """
        t_u, y_u = self.sim_unperturbed.solve_orbit()
        self.solution_unperturbed = (t_u, y_u)

        t_p, y_p = self.sim_perturbed.solve_orbit()
        self.solution_perturbed = (t_p, y_p)

        anim = FuncAnimation(
            self.fig,
            self.update_plot,
            init_func=self.init_plot,
            frames=len(self.solution_unperturbed[0]),
            interval=10,
            blit=True
        )
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    sim_unperturbed = OrbitalSimulation()
    sim_perturbed = OrbitalSimulation(perturbation=-.1)
    graphics = OrbitalGraphics(sim_unperturbed, sim_perturbed)
    graphics.run()
