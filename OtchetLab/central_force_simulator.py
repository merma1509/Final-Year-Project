import numpy as np
from scipy.integrate import solve_ivp


class CentralForceSimulator:
    """
    Class for simulating central force motion with perturbation and stabilization.
    """

    def __init__(self, k=1.0, h=1.0, epsilon=0.1, lambda1=0.5, lambda2=0.2):
        """
        Initializes the simulator with the provided parameters.

        Parameters:
            k       (float): The constant for unperturbed system.
            h       (float): The scaling factor for the system.
            epsilon (float): The strength of the perturbation.
            lambda1 (float): The damping factor for the first stabilization term.
            lambda2 (float): The damping factor for the second stabilization term.
        """
        self.k = k
        self.h = h
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # sinusoidal perturbation
        self.g = lambda u: np.sin(u)

    def unperturbed(self, t, y):
        """
        Returns the differential equations for the unperturbed system.

        Parameters:
            t (float): Time.
            y  (list): List of state variables [u, u'].

        Returns:
            list: Derivatives of state variables.
        """
        u, u_prime = y
        dudt = u_prime
        du_primedt = -u + self.k / self.h**2
        return [dudt, du_primedt]
    

    def perturbed(self, t, y):
        """
        Returns the differential equations for the perturbed system.

        Parameters:
            t (float): Time.
            y  (list): List of state variables [u, u'].

        Returns:
            list: Derivatives of state variables.
        """
        u, u_prime = y
        dudt = u_prime
        du_primedt = -u + self.k / self.h**2 + (self.epsilon / self.h**2) * self.g(1 / u)
        return [dudt, du_primedt]

    def stabilized(self, t, y):
        """
        Returns the differential equations for the stabilized system.

        Parameters:
            t (float): Time.
            y  (list): List of state variables [u, u'].

        Returns:
            list: Derivatives of state variables.
        """
        u, u_prime = y
        dudt = u_prime
        du_primedt = (
            - u
            - self.lambda1 * u_prime
            - self.lambda2 * u
            + self.k / self.h**2
            + (self.epsilon / self.h**2) * self.g(1 / u)
        )
        return [dudt, du_primedt]

    def solve_trajectory(self, func, theta_range, u0, u_prime0, num_points=1000):
        """
        Solves the trajectory using the given differential function.

        Parameters:
            func     (callable): The differential equation function.
            theta_range (tuple): The range of angles (start, end).
            u0          (float): Initial condition for u.
            u_prime0    (float): Initial condition for u'.
            num_points    (int): Number of points in the solution.

        Returns:
            tuple: x and y coordinates of the trajectory.
        """
        theta_eval = np.linspace(*theta_range, num_points)
        sol = solve_ivp(func, theta_range, [u0, u_prime0], t_eval=theta_eval, method="RK45")
        r = 1 / sol.y[0]
        x = r * np.cos(sol.t)
        y = r * np.sin(sol.t)
        return x, y
