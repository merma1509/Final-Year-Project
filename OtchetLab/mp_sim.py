import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MaterialPointMotion:
    def __init__(self, k=1.0, m=1.0, c=1.0, alpha=0.5, beta=1.0, A=0.001, omega=1.0):
        """
        Initializes the MaterialPointMotion class with the given parameters.

        :param k: Gravitational constant times central mass
        :param m: Mass of the material point
        :param c: Angular momentum
        :param alpha: Baumgarte damping coefficient
        :param beta: Baumgarte stiffness coefficient
        :param A: Amplitude of the perturbation
        :param omega: Frequency of the perturbation
        """
        self.k = k
        self.m = m
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.A = A
        self.omega = omega

    def unperturbed_system(self, theta, Y):
        """
        Defines the unperturbed orbital dynamics system.

        :param theta: The angular position (theta)
        :param Y: A list of [u, v] where u is the radial distance and v is the radial velocity
        :return: The derivatives of [u, v] (du/dtheta, dv/dtheta)
        """
        u, v = Y
        du_dtheta = v
        dv_dtheta = (self.k * self.m**2) / self.c**2 - u
        return [du_dtheta, dv_dtheta]

    def perturbed_system(self, theta, Y):
        """
        Defines the orbital dynamics with periodic perturbations.

        :param theta: The angular position (theta)
        :param Y: A list of [u, v] where u is the radial distance and v is the radial velocity
        :return: The derivatives of [u, v] (du/dtheta, dv/dtheta)
        """
        u, v = Y
        du_dtheta = v
        dv_dtheta = (self.k * self.m**2) / self.c**2 
        - u +  np.exp(-0.8 * theta) + np.exp(-0.2 * theta)
        return [du_dtheta, dv_dtheta]


    def solve_system(self, system, theta_span, Y0, t_eval, **kwargs):
        """
        Solves the orbital dynamics system numerically using the Runge-Kutta method.

        :param system: The system to solve (unperturbed, perturbed, or stabilized)
        :param theta_span: The range of theta (start, end)
        :param Y0: Initial conditions for [u, v]
        :param t_eval: The points in time where the solution should be evaluated
        :param kwargs: Additional arguments for the system function (e.g., u_nominal_func)
        :return: The solution of the system
        """
        return solve_ivp(system, theta_span, Y0, t_eval=t_eval, method='RK45', args=kwargs.get("args", ()))

    def simulate(self, theta_span=(0, 8 * np.pi), num_points=1000):
        """
        Simulate the system and compute the solutions for the unperturbed, perturbed, and stabilized systems.

        :param theta_span: The range of theta (start, end)
        :param num_points: The number of points to evaluate the solution at
        :return: A tuple containing the time points and the results of each system
        """
        # Initial conditions
        u0, v0 = 0.1, 0.0
        Y0 = [u0, v0]
        theta_eval = np.linspace(*theta_span, num_points)

        # Solve each system
        sol_unperturbed = self.solve_system(self.unperturbed_system, theta_span, Y0, theta_eval)
        sol_perturbed = self.solve_system(self.perturbed_system, theta_span, Y0, theta_eval)

        # Return results
        return theta_eval, sol_unperturbed.y[0], sol_perturbed.y[0]

    def animate_polar(self, theta_vals, u_unperturbed, u_perturbed):
        """
        Creates an animation of the polar trajectories of the unperturbed, perturbed, and stabilized systems.

        :param theta_vals:    The theta values (angular position)
        :param u_unperturbed: The unperturbed radial distances
        :param u_perturbed:   The perturbed radial distances
        :param u_stabilized:  The radial distances with Baumgarte stabilization
        """
        def safe_reciprocal(u, max_r=100):
            """Ensure r = 1/u is stable and bounded."""
            r = np.where(np.abs(u) > 1e-6, 1 / u, max_r)
            return np.clip(r, 0, max_r) 

        # Calculate radial distances
        r_unperturbed = safe_reciprocal(u_unperturbed)
        r_perturbed = safe_reciprocal(u_perturbed)

        # Plot setup
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(12, 10))
        ax.set_title('Polar Trajectories of Material Point')
        ax.grid(True)
        ax.set_ylim(0, 13)

        # Initialize plot elements
        line_unperturbed, = ax.plot([], [], label='Unperturbed', color='blue')
        line_perturbed, = ax.plot([], [], label='Perturbed', color='green', linestyle='--')
        ax.legend()

        # Animation update function
        def update(frame):
            line_unperturbed.set_data(theta_vals[:frame], r_unperturbed[:frame])
            line_perturbed.set_data(theta_vals[:frame], r_perturbed[:frame])
            return line_unperturbed, line_perturbed

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=len(theta_vals), interval=30, blit=True)
        ani.save('polar_trajectories.gif', writer='pillow')
        plt.show()


if __name__ == "__main__":
    material_point = MaterialPointMotion(k=1.0, m=1.0, c=1.0, alpha=0.5, beta=1.0, A=0.001, omega=1.0)
    theta_vals, u_unperturbed, u_perturbed = material_point.simulate()
    material_point.animate_polar(theta_vals, u_unperturbed, u_perturbed)
