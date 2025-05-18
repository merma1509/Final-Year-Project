import matplotlib.pyplot as plt
import matplotlib.animation as animation


class AnimationHandler:
    """
    Class for handling the animation of central force trajectories.
    """

    def animate_trajectories(self, x_unperturbed, y_unperturbed, x_perturbed, y_perturbed, x_stabilized, y_stabilized):
        """
        Animates the trajectories of the unperturbed, perturbed, and stabilized systems.

        Parameters:
            x_unperturbed (array): x coordinates of the unperturbed trajectory.
            y_unperturbed (array): y coordinates of the unperturbed trajectory.
            x_perturbed   (array): x coordinates of the perturbed trajectory.
            y_perturbed   (array): y coordinates of the perturbed trajectory.
            x_stabilized  (array): x coordinates of the stabilized trajectory.
            y_stabilized  (array): y coordinates of the stabilized trajectory.

        Returns:
            ani: The animation object.
        """
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title("Trajectories: Unperturbed, Perturbed, and Stabilized")
        ax.scatter(0, 0, color="red", label="Central Body", zorder=5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid()

        # Initialize lines for each trajectory
        line_unperturbed, = ax.plot([], [], alpha=.7 , linestyle="--", label="Unperturbed")
        line_perturbed, = ax.plot([], [], alpha=.5, label="Perturbed")
        line_stabilized, = ax.plot([], [], alpha=.6, label="Stabilized")

        # Initialize scatter markers for each moving body (single points initially)
        marker_unperturbed, = ax.plot([], [], 'o', color="blue", markersize=7)
        marker_perturbed, = ax.plot([], [], 'o', color="orange", markersize=7)
        marker_stabilized, = ax.plot([], [], 'o', color="green", markersize=7)

        ax.legend()

        # Determine bounds for axes
        x_min = min(x_unperturbed.min(), x_perturbed.min(), x_stabilized.min())
        x_max = max(x_unperturbed.max(), x_perturbed.max(), x_stabilized.max())
        y_min = min(y_unperturbed.min(), y_perturbed.min(), y_stabilized.min())
        y_max = max(y_unperturbed.max(), y_perturbed.max(), y_stabilized.max())
        ax.set_xlim(x_min - 0.1, x_max + 0.1)
        ax.set_ylim(y_min - 0.1, y_max + 0.1)

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
