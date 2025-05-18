import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')

class EllipticalMotion:
    def __init__(self, a, b, omega):
        self.a = a  # Semi-major axis
        self.b = b  # Semi-minor axis
        self.omega = omega  # Angular velocity

    def unperturbed_trajectory(self, t):
        """Calculate the unperturbed elliptical trajectory."""
        x = self.a * np.cos(self.omega * t)
        y = self.b * np.sin(self.omega * t)
        return x, y

    def perturbed_trajectory(self, t, initial_conditions, roots):
        """Calculate the perturbed trajectory."""
        C1, C2 = initial_conditions

        # Perturbation due to real roots
        if len(roots) == 2 and np.isreal(roots).all():
            alpha = C1 * np.exp(roots[0] * t) + C2 * np.exp(roots[1] * t)
        
        # Perturbation due to complex roots
        elif len(roots) == 2 and not np.isreal(roots).all():
            real_part = np.real(roots[0])
            imag_part = np.imag(roots[0])
            alpha = np.exp(real_part * t) * (C1 * np.cos(imag_part * t) + C2 * np.sin(imag_part * t))
        else:
            raise ValueError("Unsupported root configuration.")

        # Combine unperturbed and perturbed motion
        x_unperturbed, y_unperturbed = self.unperturbed_trajectory(t)
        x_perturbed = x_unperturbed + alpha
        y_perturbed = y_unperturbed
        return x_perturbed, y_perturbed

# Parameters for the ellipse
a = 5
b = 3
omega = 1

# Perturbation parameters
perturb_params_real = {
    'roots': [-0.1, -0.05],  # Real roots
    'initial_conditions': (0.5, 0.3),
}

perturb_params_complex = {
    'roots': [-0.8 + 0.6j, -0.8 - 0.6j],  # Complex roots
    'initial_conditions': (0.5, 0.3),
}

# Time array
t = np.linspace(0, 50, 5000)

# Instantiate the motion class
motion = EllipticalMotion(a, b, omega)

# Calculate trajectories
x_unperturbed, y_unperturbed = motion.unperturbed_trajectory(t)
x_perturbed_real, y_perturbed_real = motion.perturbed_trajectory(
    t, perturb_params_real['initial_conditions'], perturb_params_real['roots']
)
x_perturbed_complex, y_perturbed_complex = motion.perturbed_trajectory(
    t, perturb_params_complex['initial_conditions'], perturb_params_complex['roots']
)

# Animation setup
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-10, 10)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid()

line1, = ax.plot([], [], 'b-', label='Unperturbed Motion')
line2, = ax.plot([], [], 'r--', label='Perturbed Motion (Real Roots)')
line3, = ax.plot([], [], 'g-.', label='Perturbed Motion (Complex Roots)')
point1, = ax.plot([], [], 'b', alpha=0.8)
point2, = ax.plot([], [], 'r', alpha=0.6)
point3, = ax.plot([], [], 'g', alpha=0.7)
ax.legend()

# Animation function
def update(frame):
    line1.set_data(x_unperturbed[:frame + 1], y_unperturbed[:frame + 1])
    line2.set_data(x_perturbed_real[:frame + 1], y_perturbed_real[:frame + 1])
    line3.set_data(x_perturbed_complex[:frame + 1], y_perturbed_complex[:frame + 1])
    point1.set_data([x_unperturbed[frame]], [y_unperturbed[frame]])
    point2.set_data([x_perturbed_real[frame]], [y_perturbed_real[frame]])
    point3.set_data([x_perturbed_complex[frame]], [y_perturbed_complex[frame]])
    return line1, line2, line3, point1, point2, point3

ani = FuncAnimation(fig, update, frames=len(t), interval=10, blit=True)

plt.ioff()  # Disable interactive mode
plt.show()
