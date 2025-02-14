from abc import ABC, abstractmethod
import jax.numpy as jnp

class PDESolver(ABC):
    """Base class for PDE solvers."""
    
    def __init__(self, Lx, Ly, T, Mx, My, Nt, gamma):
        """
        Initialize solver for 2D problems.
        
        Args:
            Lx (float): Domain length in x direction
            Ly (float): Domain length in y direction
            T (float): Time span
            Mx (int): Number of spatial points in x
            My (int): Number of spatial points in y
            Nt (int): Number of time points
            gamma (float): Diffusion coefficient
        """
        self.Lx = Lx
        self.Ly = Ly
        self.T = T
        self.Mx = Mx
        self.My = My
        self.Nt = Nt
        self.gamma = gamma
        
        # Grid spacing
        self.dx = Lx / (Mx - 1)
        self.dy = Ly / (My - 1)
        self.dt = T / (Nt - 1)

    @abstractmethod
    def solve_2d(self, initial_condition, boundary_conditions):
        """
        Solve 2D heat equation.
        
        Args:
            initial_condition: Function that takes (x, y) and returns initial temperature
            boundary_conditions: Dictionary with boundary conditions
            
        Returns:
            tuple: (u, x, y, t) arrays containing solution and grid points
        """
        pass
    
    def create_mesh(self, dim=1):
        """Create mesh grids."""
        x = jnp.linspace(0, self.Lx, self.Mx)
        y = jnp.linspace(0, self.Ly, self.My)
        t = jnp.linspace(0, self.T, self.Nt)
        
        if dim == 1:
            return x, t
        elif dim == 2:
            X, Y = jnp.meshgrid(x, y)
            return X, Y, t 