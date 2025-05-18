import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.animation as animation


class CentralForceSimulator:
    def __init__(self, k=1.0, h=1.0, epsilon=0.1, lambda1=0.5, lambda2=0.2):
        self.k = k
        self.h = h
        self.epsilon = epsilon
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # sinusoidal perturbation
        self.g = lambda u: np.sin(u)

    def unperturbed(self, t, y):
        u, u_prime = y
        dudt = u_prime
        du_primedt = -u + self.k / self.h**2
        return [dudt, du_primedt]

    def perturbed(self, t, y):
        u, u_prime = y
        dudt = u_prime
        du_primedt = -u + self.k / self.h**2 + (self.epsilon / self.h**2) * self.g(1 / u)
        return [dudt, du_primedt]

    def stabilized(self, t, y):
        u, u_prime = y
        dudt = u_prime
        du_primedt = (
            -u
            - self.lambda1 * u_prime
            - self.lambda2 * u
            + self.k / self.h**2
            + (self.epsilon / self.h**2) * self.g(1 / u)
        )
        return [dudt, du_primedt]

    def solve_trajectory(self, func, theta_range, u0, u_prime0, num_points=1000):
        theta_eval = np.linspace(*theta_range, num_points)
        sol = solve_ivp(func, theta_range, [u0, u_prime0], t_eval=theta_eval, method="RK45")
        r = 1 / sol.y[0]
        x = r * np.cos(sol.t)
        y = r * np.sin(sol.t)
        return x, y

    def animate_trajectories(self, x_unperturbed, y_unperturbed, x_perturbed, y_perturbed, x_stabilized, y_stabilized):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Trajectories: Unperturbed, Perturbed, and Stabilized")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid()

        # Initialize lines for each trajectory
        line_unperturbed, = ax.plot([], [], linestyle="--", alpha=.5, label="Unperturbed")
        line_perturbed, = ax.plot([], [], alpha=.7, label="Perturbed")
        line_stabilized, = ax.plot([], [], alpha=.9, label="Stabilized")

        # Initialize scatter markers for each moving body
        marker_unperturbed, = ax.plot([], [], 'o', color="red", markersize=7)
        marker_perturbed, = ax.plot([], [], 'o', color="orange", markersize=7)
        marker_stabilized, = ax.plot([], [], 'o', color="black", markersize=7)

        ax.legend()

        # Determine bounds for axes
        x_min = min(x_unperturbed.min(), x_perturbed.min(), x_stabilized.min())
        x_max = max(x_unperturbed.max(), x_perturbed.max(), x_stabilized.max())
        y_min = min(y_unperturbed.min(), y_perturbed.min(), y_stabilized.min())
        y_max = max(y_unperturbed.max(), y_perturbed.max(), y_stabilized.max())
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        # Update function for animation
        def update(frame):
            # Update trajectory lines
            line_unperturbed.set_data(x_unperturbed[:frame], y_unperturbed[:frame])
            line_perturbed.set_data(x_perturbed[:frame], y_perturbed[:frame])
            line_stabilized.set_data(x_stabilized[:frame], y_stabilized[:frame])

            # Update marker positions
            marker_unperturbed.set_data([x_unperturbed[frame]], [y_unperturbed[frame]])
            marker_perturbed.set_data([x_perturbed[frame]], [y_perturbed[frame]])
            marker_stabilized.set_data([x_stabilized[frame]], [y_stabilized[frame]])

            return line_unperturbed, line_perturbed, line_stabilized, marker_unperturbed, marker_perturbed, marker_stabilized

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(x_unperturbed),
            interval=50,
            blit=True,
        )

        return ani


# Parameters
k = 1.0
h = 1.0
epsilon = 0.2
lambda1 = 0.5
lambda2 = 0.2
theta_range = (0, 20 * np.pi)
u0 = 1.0
u_prime0 = 0.0
num_points = 2000

# Initialize simulator
simulator = CentralForceSimulator(k=k, h=h, epsilon=epsilon, lambda1=lambda1, lambda2=lambda2)

# Solve for trajectories
x_unperturbed, y_unperturbed = simulator.solve_trajectory(simulator.unperturbed, theta_range, 1 / u0, u_prime0, num_points)
x_perturbed, y_perturbed = simulator.solve_trajectory(simulator.perturbed, theta_range, 1 / u0, u_prime0, num_points)
x_stabilized, y_stabilized = simulator.solve_trajectory(simulator.stabilized, theta_range, 1 / u0, u_prime0, num_points)

# Animate trajectories
ani = simulator.animate_trajectories(x_unperturbed, y_unperturbed, x_perturbed, y_perturbed, x_stabilized, y_stabilized)

# Show animation
plt.show()
