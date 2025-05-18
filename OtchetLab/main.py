import numpy as np
from central_force_simulator import CentralForceSimulator
from animation_handler import AnimationHandler
import matplotlib.pyplot as plt


def main():
    # Parameters
    k = 1.0
    h = 1.0
    epsilon = 0.002
    lambda1 = -0.16
    lambda2 = -1
    theta_range = (0, 20 * np.pi)
    u0 = 1.0
    u_prime0 = 0.0
    num_points = 2000

    # Initialize simulator and animation handler
    simulator = CentralForceSimulator(k=k, h=h, epsilon=epsilon, lambda1=lambda1, lambda2=lambda2)
    animator = AnimationHandler()

    # Solve for trajectories
    x_unperturbed, y_unperturbed = simulator.solve_trajectory(simulator.unperturbed, theta_range, 1 / u0, u_prime0, num_points)
    x_perturbed, y_perturbed = simulator.solve_trajectory(simulator.perturbed, theta_range, 1 / u0, u_prime0, num_points)
    x_stabilized, y_stabilized = simulator.solve_trajectory(simulator.stabilized, theta_range, 1 / u0, u_prime0, num_points)

    # Animate trajectories
    ani = animator.animate_trajectories(x_unperturbed, y_unperturbed, x_perturbed, y_perturbed, x_stabilized, y_stabilized)

    # Show the animation
    # ani.save("central_force_trajectories.gif", writer='pillow', fps=30)
    plt.show()


if __name__ == "__main__":
    main()
