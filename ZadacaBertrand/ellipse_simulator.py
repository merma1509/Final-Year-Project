import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('TkAgg')

class EllipseSimulation:
    def __init__(self, a, b, a0, a1, x0, y0, dx0, dy0, time_span, time_points):
        self.a = a
        self.b = b
        self.p = b**2 / a
        self.e = np.sqrt(a**2 - b**2) / a
        self.a0 = a0
        self.a1 = a1
        self.x0 = x0
        self.y0 = y0
        self.dx0 = dx0
        self.dy0 = dy0
        self.time_span = time_span
        self.time_points = time_points
        self.c0 = self.x0 * self.dy0 - self.y0 * self.dx0

    def equations(self, t, state):
        x, y, dx, dy = state

        c0 =  (x * dy - y * dx)
        r = np.sqrt(x**2 + y**2)

        R0 = c0**2 / r**3 - (self.a0 + c0**2 / (self.p * r**3)) * (r - self.e * x - self.p)
        R = R0 - self.a1 * ((x / r - self.e) * dx + y * dy / r)

        ddx = -x / self.p * R
        ddy = -y / self.p * R
        return [dx, dy, ddx, ddy]

    def solve(self):
        # Solve the differential equations for both motions
        solution = solve_ivp(
            self.equations, self.time_span, 
            [self.x0, self.y0, self.dx0, self.dy0],
            t_eval=self.time_points, method='LSODA'
        )

        # Extract trajectories for perturbed motion
        x_perturbed, y_perturbed = solution.y[0], solution.y[1]
        return x_perturbed, y_perturbed


class Animator:
    def __init__(self, x_unperturbed, y_unperturbed, x_perturbed, y_perturbed, time_points):
        self.x_unperturbed = x_unperturbed
        self.y_unperturbed = y_unperturbed
        self.x_perturbed = x_perturbed
        self.y_perturbed = y_perturbed
        self.time_points = time_points
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.set_xlim(-12, 12)
        self.ax.set_ylim(-12, 12)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Траекторий невозмущенного и возмущенного движения точки")
        self.ax.grid(True)
        self.ax.axis('equal')

        self.line_unperturbed, = self.ax.plot([], [], alpha=.89, label="Невозмущенное", color="blue", linestyle='-.' )
        self.line_perturbed, = self.ax.plot([], [],  alpha=0.5, label="Возмущенное", color="red", linestyle="--")
        self.point_unperturbed, = self.ax.plot([], [], 'bo', markersize=6)
        self.point_perturbed, = self.ax.plot([], [], 'ro', markersize=6)

    def update(self, frame):
        # Update the trajectory lines
        self.line_unperturbed.set_data(self.x_unperturbed[:frame], self.y_unperturbed[:frame])
        self.line_perturbed.set_data(self.x_perturbed[:frame], self.y_perturbed[:frame])

        # Update the moving points
        self.point_unperturbed.set_data([self.x_unperturbed[frame]], [self.y_unperturbed[frame]])
        self.point_perturbed.set_data([self.x_perturbed[frame]], [self.y_perturbed[frame]])

        return self.line_unperturbed, self.line_perturbed, self.point_unperturbed, self.point_perturbed

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, frames=len(self.time_points), interval=50, blit=True)
        # ani.save('ellipse_simulator.gif', writer='pillow')
        self.ax.legend()
        plt.ioff()
        plt.show()


def main():
    # Perturbation coefficients and initial conditions
    a0_perturbed, a1_perturbed = -1, -1.6
    time_span = (0, 20 * np.pi)
    time_points = np.linspace(time_span[0], 20 * np.pi, 1000)

    # Initial conditions for the perturbed motion
    x0_perturbed, y0_perturbed, dx0_perturbed, dy0_perturbed = [10.0, 0.0, 0.0, 1.0]

    # EllipseSimulation class instance creation for perturbed motion
    simulation = EllipseSimulation(a=5, b=3, a0=a0_perturbed, a1=a1_perturbed, 
                                  x0=x0_perturbed, y0=y0_perturbed, 
                                  dx0=dx0_perturbed, dy0=dy0_perturbed,
                                  time_span=time_span, time_points=time_points)

    # Get the perturbed motion solution
    x_perturbed, y_perturbed = simulation.solve()

    # For unperturbed motion, use the same parameters but with different coefficients
    simulation_unperturbed = EllipseSimulation(a=5, b=3, a0=-1, a1=-1.6, 
                                              x0=9.0, y0=0.0, dx0=0.0, dy0=1.0, 
                                              time_span=time_span, time_points=time_points)
    x_unperturbed, y_unperturbed = simulation_unperturbed.solve()

    # Create an Animator instance and start animation
    animator = Animator(x_unperturbed, y_unperturbed, x_perturbed, y_perturbed, time_points)
    animator.animate()


if __name__ == "__main__":
    main()
