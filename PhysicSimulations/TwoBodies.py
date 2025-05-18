from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
import numpy as np

# Class to Simulate Two Body Systems
class TwoBodySimulation:
    def __init__(self, body_1st=1.9891e30 , body_2nd=5.97219e24, body_3rd=3.85e2, height=36000, eccentricity=0.0, timestep=.01):
        self.sun_mass = body_1st           # Mass of the Sun in Kg
        self.earth_mass = body_2nd         # Mass of the Earth in Kg
        self.satellite_mass = body_3rd     # Mass of the Artificial Satellite in Kg
        self.mass_ratio = self.satellite_mass / self.earth_mass
        self.eccentricity = eccentricity
        self.timestep = timestep
        self.satellite_height = height     # The Altitude of the satellite in km

        # Initial conditions
        self.state = {
            "u": [1.0, 0.0, 0.0, self._initial_velocity()],
            "positions": [{"x": 0, "y": self.satellite_height}, {"x": 0, "y": self.satellite_height}]  
        }

    def _initial_velocity(self):
        """Calculate initial velocity based on mass ratio and eccentricity."""
        return np.sqrt((1 + 1) * (1 + self.eccentricity))

    def derivatives(self, u):
        """Calculate the derivatives for the equations of motion."""
        r = np.array(u[:2])       # Position vector
        v = np.array(u[2:])       # Velocity vector
        rr = np.linalg.norm(r)    # Distance between bodies

        drdt = v
        dvdt = -(1 + 1) * r / (rr ** 3)
        return np.concatenate((drdt, dvdt))

    def runge_kutta_step(self, u):
        """Perform one step of Runge-Kutta integration."""
        h = self.timestep
        k1 = self.derivatives(u)
        k2 = self.derivatives(u + 0.5 * h * k1)
        k3 = self.derivatives(u + 0.5 * h * k2)
        k4 = self.derivatives(u + h * k3)
        return u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def update(self):
        """Update the simulation state for one timestep."""
        self.state["u"] = self.runge_kutta_step(self.state["u"])
        self._calculate_positions()

    def _calculate_positions(self):
        """Calculate positions of the two bodies based on the current state."""
        r = 4                     # Distance between the bodies
        m12 = 1 + 1
        a1 = (1 / m12) * r
        a2 = (1 / m12) * r

        u = self.state["u"]
        self.state["positions"][0] = {"x": -a2 * u[0], "y": -a2 * u[1]}
        self.state["positions"][1] = {"x": a1 * u[0], "y": a1 * u[1]}

# Visualization
class OrbitalAnimator:
    def __init__(self, simulation, num_steps=2000, background_image=None):
        self.simulation = simulation
        self.num_steps = num_steps
        self.fig, self.ax = plt.subplots(figsize=(20, 10))
        self.earth, = self.ax.plot([], [], "bo", label="Body1")                   # Body 1 marker
        self.sun, = self.ax.plot([], [], "yo", label="Body2", markersize=15)      # Body 2 marker
        self.earth_trail, = self.ax.plot([], [], "g--", label="Body1 Trajectory")
        self.sun_trail, = self.ax.plot([], [], "r--", label="Bogy2 Trajectory")
        self.background_image = background_image

        # Position lists for trajectories
        self.earth_positions = []
        self.sun_positions = []

    def _init_plot(self):
        if self.background_image:
            img = plt.imread(self.background_image)
            self.ax.imshow(img, extent=(-15, 15, -8, 8), aspect='auto', zorder=1, alpha=0.5)

        self.ax.plot(0, 0, '+', markersize=15)
        self.ax.set_title('Two Body System Simulations')
        self.ax.set_xlim(-15, 15)
        self.ax.set_ylim(-8, 8)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.legend()
        return self.earth, self.sun, self.earth_trail, self.sun_trail

    def _update_plot(self, frame):
        self.simulation.update()
        pos = self.simulation.state["positions"]
        earth_pos = pos[0]
        sun_pos = pos[1]

        # Append current positions to trails
        self.earth_positions.append((earth_pos["x"], earth_pos["y"]))
        self.sun_positions.append((sun_pos["x"], sun_pos["y"]))

        # Unpack positions
        earth_x, earth_y = zip(*self.earth_positions)
        sun_x, sun_y = zip(*self.sun_positions)

        # Update markers and trails
        self.earth.set_data([earth_pos["x"]], [earth_pos["y"]])
        self.sun.set_data([sun_pos["x"]], [sun_pos["y"]])
        self.earth_trail.set_data(earth_x, earth_y)
        self.sun_trail.set_data(sun_x, sun_y)

        return self.earth, self.sun, self.earth_trail, self.sun_trail

    def animate(self):
        anim = FuncAnimation(
            self.fig, self._update_plot, init_func=self._init_plot,
            frames=self.num_steps, interval=20, blit=True
        )
        # anim.save('twobodysimulation.gif', writer='pillow')
        plt.grid(True)
        plt.show()

# Run the simulation
if __name__ == "__main__":
    simulation = TwoBodySimulation(eccentricity=0.70, body_3rd=5.97219e24, height=0)
    animator = OrbitalAnimator(simulation)
    animator.animate()