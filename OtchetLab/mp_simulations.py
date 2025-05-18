import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class MaterialPointMotion:
    def __init__(self, k=1.0, m=1.0, c=1.0, alpha=0.5, beta=0.2, A=0.002, omega=1.0):
        """
        Initializes the MaterialPointMotion class with the given parameters.
        Parameters:
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
        Parameters:
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
        Parameters:
            :param theta: The angular position (theta)
            :param Y: A list of [u, v] where u is the radial distance and v is the radial velocity
            :return: The derivatives of [u, v] (du/dtheta, dv/dtheta)
        """
        u, v = Y
        du_dtheta = v
        dv_dtheta = (self.k * self.m**2) / self.c**2 - u + self.A * np.sin(self.omega * theta)
        return [du_dtheta, dv_dtheta]

    def stabilized_system(self, theta, Y, u_nominal_func):
        """
        Defines the orbital dynamics with Baumgarte stabilization.
        Parameters:
            :param theta: The angular position (theta)
            :param Y: A list of [u, v] where u is the radial distance and v is the radial velocity
            :param u_nominal_func: A function that returns the nominal solution (u_nominal) for a given theta
            :return: The derivatives of [u, v] (du/dtheta, dv/dtheta)
        """
        u, v = Y
        du_dtheta = v
        dv_dtheta = ((self.k * self.m**2) / self.c**2
                     - u
                     + self.A * np.sin(self.omega * theta)
                     - (u - u_nominal_func(theta))
                     - 2 * self.alpha * v
                     - self.beta**2 * (u - u_nominal_func(theta)))
        return [du_dtheta, dv_dtheta]

    def solve_system(self, system, theta_span, Y0, t_eval, **kwargs):
        """
        Solves the orbital dynamics system numerically using the Runge-Kutta method.
        Parameters:
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

        Parameters:
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

        # Stabilized system using nominal solution from unperturbed system
        u_nominal_func = lambda theta: np.interp(theta, sol_unperturbed.t, sol_unperturbed.y[0])
        sol_stabilized = self.solve_system(self.stabilized_system, theta_span, Y0, theta_eval, args=(u_nominal_func,))

        # Return results
        return theta_eval, sol_unperturbed.y[0], sol_perturbed.y[0], sol_stabilized.y[0]

    def visualize(self, theta_vals, u_unperturbed, u_perturbed, u_stabilized):
        """
        Visualizes the trajectories of the unperturbed, perturbed, and stabilized systems.

        Parameters:

            :param theta_vals:    The theta values (angular position)
            :param u_unperturbed: The unperturbed radial distances
            :param u_perturbed:   The perturbed radial distances
            :param u_stabilized:  The radial distances with Baumgarte stabilization
        """
        # Convert radial distances to Cartesian coordinates
        def polar_to_cartesian(r, theta):
            return r * np.cos(theta), r * np.sin(theta)

        r_unperturbed = 1 / u_unperturbed
        r_perturbed = 1 / u_perturbed
        r_stabilized = 1 / u_stabilized

        x_unperturbed, y_unperturbed = polar_to_cartesian(r_unperturbed, theta_vals)
        x_perturbed, y_perturbed = polar_to_cartesian(r_perturbed, theta_vals)
        x_stabilized, y_stabilized = polar_to_cartesian(r_stabilized, theta_vals)

        # Plot results
        plt.figure(figsize=(15, 10))
        plt.plot(x_unperturbed, y_unperturbed, label='Unperturbed')
        plt.plot(x_perturbed, y_perturbed, label='Perturbed', linestyle='--')
        plt.plot(x_stabilized, y_stabilized, label='Stabilized', linestyle='-.')
        plt.plot(0, 0, 'ro', label='Central Body')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Material Point Trajectories with Baumgarte Stabilization')
        plt.grid(True)
        plt.legend()
        plt.axis('equal')
        plt.show()


if __name__ == "__main__":
    material_point = MaterialPointMotion(k=1.0, m=1.0, c=1.0, alpha=0.5, beta=0.2, A=0.001, omega=1.0)
    theta_vals, u_unperturbed, u_perturbed, u_stabilized = material_point.simulate()
    material_point.visualize(theta_vals, u_unperturbed, u_perturbed, u_stabilized)
