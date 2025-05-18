from itertools import combinations
from math import ceil
import numpy as np
import copy

class NBodySimulator:
    def __init__(self, Xi=None, masses=None,  G=6.67408e-11):
        self.Xi = Xi                  # Initial system state matrix
        self.masses = masses          # Body Masses list
        self.G = G
        self.sim_results = None       # Where we will store the results of Simulation and we can't define it without simulations tart
        self.energies = None          # Where we will store the KE, Etot, and PE for each simulation results

    
    def getting_energies(self, X):
        """
        Parameters:
            - X (matrix): The current state to determine the energies KE, PE and Etot
            - masses (array): array of masses corresponding to the bodies
            - G: Gravitational constant
        Returns:
            - KE: Total Kinetic Energy of the System in Joules
            - PE: Total Potential Energy of the System in Joules
            - E : Total Energy equals PE + KE
        """

        N, D = X.shape           # Get the Number of Bodies, and the dimensionality of the Matrix
        D = D // 2               # 2D or 3D Dimensions  
        R = X[:, :D]             # Positions Submatrix
        V = X[:, D:]             # Velocities Submatrix

        # Kinetic Energy
        KE = 0
        for i in range(N):
            KE += 0.5 * self.masses[i] * np.linalg.norm(V[i]) ** 2          # KE = 1/2 * mass * v ** 2

        # Potential Energy
        PE = 0
        for telo_i, telo_j in self.pairs:
            r = np.linalg.norm(R[telo_j] - R[telo_i])       # Distance between the bodies in considerations
            PE -= self.masses[telo_i] * self.masses[telo_j] / r
        PE *= self.G

        Etot = KE + PE            # Total Energy of the System

        return KE, PE, Etot
    
    def getting_state_derivatives(self, X):
        """
        Parameters:
            - X (matrix): The current state to determine the state derivative for
        Returns:
            - Xdot (matrix): The state derivative for the input system state
        """

        N, D = X.shape           # Get the Number of Bodies, and the dimensionality of the Matrix
        D = D // 2               # 2D or 3D
        R = X[:, :D]             # Positions Submatrix
        V = X[:, D:]             # Velocities Submatrix

        Xdot = np.zeros_like(X) # The placeholder matrix Xdot with same size as X
        Xdot[:, :D] = V         # Fill in the velocities from the system state

        # We Iterate Over Pairs and Fill out the Accelartion, the self.pairs gets defined when we start the simulation
        for telo_i, telo_j in self.pairs:   # telo_i, telo_j are the indices of the bodies
            # Get Vector from telo_i => telo_j and its magnitude
            r1, r2 = R[telo_i], R[telo_j]
            r_vec = r2 - r1
            r = np.linalg.norm(r_vec)

            # Find Force from telo_i => telo_j and their corresponding accelerations
            F = self.G * self.masses[telo_i] * self.masses[telo_j] * r_vec / r ** 3
            a1 = F / self.masses[telo_i]
            a2 = -F / self.masses[telo_j]

            # Apply accelerations to telo_i and telo_j
            Xdot[telo_j, D:] += a1
            Xdot[telo_j, D:] += a2

        return Xdot
    
    def rk4_integrator(self, X, dt, evaluate):
        """
        Parameters:
            - X (Matrix): Current state of the system
            - dt (float): Integration Timestep
            - evaluate (func): Function that will return the system state derivatives
        Returns:
            - X: Updated system state one timestamp later
        """
        k1 = evaluate(X)
        k2 = evaluate(X + 0.5 * k1 * dt)
        k3 = evaluate(X + 0.5 * k2 * dt)
        k4 = evaluate(X + k3 * dt)

        X_new = (k1 + 2*k2 + 2*k3 + k4) * (1/6.)
        return X + X_new * dt
    

    def simulations_run(self, T, dt):
        """
        Parameters:
            - T (int): Total runtime of the simulation
            - dt (float): Timestep for integration
        Returns:
            - sim_results (Matrix): Matrix of the simulation results of system states
        """

        # Check to ensure initial conditions and masses have been set
        assert self.Xi is not None
        assert self.masses is not None

        # Setting Up Simulation Parameters
        iters = ceil(T / dt)     # Number of Simulation Iterations

        N, D = self.Xi.shape
        self.sim_results = np.zeros((iters + 1, N, D))
        self.sim_results[0] = self.Xi                 # First Simulation results is our initial conditions
        self.pairs = list(combinations(range(N), 2))  # Force Pair Indexes

        # Init Energies
        self.energies = np.zeros((iters+1, 3))
        KE, PE, E = self.getting_energies(self.Xi)
        self.energies[0] = np.array([KE, PE, E])

        # Run Simulations Iterations
        X = copy.deepcopy(self.Xi)        # Copy as to not modify Xi
        for i in range(iters):
            X = self.rk4_integrator(X, dt, self.getting_state_derivatives)     # Get new system state
            self.sim_results[i + 1] = X                                        # Store new system state
            KE, PE, E = self.getting_energies(X)                               # Get new system state's energies
            self.energies[i + 1] = np.array([KE, PE, E])                       # Store new Energy
        return self.sim_results


