import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class EllipticalMotionWithStabilization:
    def __init__(self, a, b, omega, real_roots, complex_roots, initial_conditions, alpha, beta):
        self.a = a                         # Semi-major axis
        self.b = b                         # Semi-minor axis
        self.omega = omega                 # Angular velocity
        self.real_roots = real_roots
        self.complex_roots = complex_roots
        self.initial_conditions = initial_conditions
        self.alpha = alpha                 # stabilization  (damping)
        self.beta = beta                   # stabilization  (stiffness)

    def unperturbed_trajectory(self, t):
        """Calculate unperturbed elliptical trajectory."""
        x = self.a * np.cos(self.omega * t) + self.initial_conditions["unperturbed"]["x0"]
        y = self.b * np.sin(self.omega * t)
        return x, y

    def perturbed_trajectory(self, t):
        """Calculate perturbed trajectory"""
        x_unperturbed, y_unperturbed = self.unperturbed_trajectory(t)
        C1, C2 = self.initial_conditions["perturbed"]["C1"], self.initial_conditions["perturbed"]["C2"]

        # Real roots perturbation
        alpha_real = C1 * np.exp(self.real_roots[0] * t) + C2 * np.exp(self.real_roots[1] * t)

        # Complex roots perturbation
        real_part = np.real(self.complex_roots[0])
        imag_part = np.imag(self.complex_roots[0])
        alpha_complex = np.exp(real_part * t) * (C1 * np.cos(imag_part * t) + C2 * np.sin(imag_part * t))

        # Total perturbation
        alpha = alpha_real + alpha_complex
        x_perturbed = x_unperturbed + alpha
        y_perturbed = y_unperturbed

        return x_perturbed, y_perturbed

    def stabilized_trajectory(self, t):
        """Calculate stabilized trajectory using Baumgarte stabilization."""
        x_perturbed, y_perturbed = self.perturbed_trajectory(t)

        # Constraint calculation: \phi(x, y) = (x^2 / a^2) + (y^2 / b^2) - 1
        constraint = (x_perturbed**2 / self.a**2) + (y_perturbed**2 / self.b**2) - 1

        # Constraint derivative: \dot{\phi}
        constraint_derivative = (
            (2 * x_perturbed / self.a**2) * self.omega * -np.sin(self.omega * t) + 
            (2 * y_perturbed / self.b**2) * self.omega * np.cos(self.omega * t)
        )

        # Stabilization term: \ddot{\phi} + 2\alpha\dot{\phi} + \beta\phi = 0
        stabilization_term = -self.alpha * constraint_derivative - self.beta * constraint

        # Apply stabilization terms to the perturbed trajectory
        x_stabilized = x_perturbed + stabilization_term * (2 * x_perturbed / self.a**2)
        y_stabilized = y_perturbed + stabilization_term * (2 * y_perturbed / self.b**2)

        return x_stabilized, y_stabilized


# Ellipse parameters
a = 5
b = 3
omega = 1
real_roots = [-0.8, -0.2]
complex_roots = [-0.8 - 0.6j, -0.8 + 0.6j]

# Stabilization coefficients: Damping and Stiffness term
alpha = 1.0   
beta = 0.0    

# Initial conditions
initial_conditions = {
    "unperturbed": {"x0": 9, "y0": 0, "vx0": 0, "vy0": 1},
    "perturbed": {"x0": 10, "y0": 0, "vx0": 0, "vy0": 1, "C1": -0.16, "C2": -1},
}

# Time array
t = np.linspace(0, 20*np.pi, 2000)

# Instantiate the motion class with stabilization
motion = EllipticalMotionWithStabilization(a, b, omega, real_roots, complex_roots, initial_conditions, alpha, beta)

# Calculate trajectories
x_unperturbed, y_unperturbed = motion.unperturbed_trajectory(t)
x_perturbed, y_perturbed = motion.perturbed_trajectory(t)
x_stabilized, y_stabilized = motion.stabilized_trajectory(t)

# Create the figure and axis for the animation
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(-5, 20)
ax.set_ylim(-5, 5)

# ax.plot(0, 0, 'ro', markersize=8, label='Central Body')
ax.set_title("Невозмущенное, Возмущенное и Стабилизированное Движение Материальной Точки")
ax.set_xlabel("x (см)")
ax.set_ylabel("y (см)")
ax.grid()
ax.axis("equal")

# Plot static elements
line_unperturbed,  = ax.plot([], [], label="Невозмущенное", color="blue", alpha=0.8)
line_stabilized, = ax.plot([], [], label="Возмущенное", color="green", alpha=0.7)
line_perturbed,  = ax.plot([], [], label="Стабилизированное", color="red", alpha=0.5)
point_unperturbed, = ax.plot([], [], "o", color="black")
point_perturbed, = ax.plot([], [], "o", color="green")
point_stabilized, = ax.plot([], [], "o", color="blue")
ax.legend()

# Animation update function
def update(frame):
    line_unperturbed.set_data(x_unperturbed[:frame], y_unperturbed[:frame])
    line_perturbed.set_data(x_perturbed[:frame], y_perturbed[:frame])
    point_unperturbed.set_data([x_unperturbed[frame]], [y_unperturbed[frame]])
    point_perturbed.set_data([x_perturbed[frame]], [y_perturbed[frame]])
    point_stabilized.set_data([x_stabilized[frame]], [y_stabilized[frame]])
    line_stabilized.set_data(x_stabilized[:frame], y_stabilized[:frame])
    return point_stabilized, line_stabilized, point_perturbed,line_perturbed, line_unperturbed, point_unperturbed

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)
# ani.save('ellipse2_sim.gif', writer='pillow')

# Show the animation
# plt.savefig('ell_sim2.png')
plt.show()
