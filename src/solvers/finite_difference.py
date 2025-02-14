import jax.numpy as jnp
from .base_solver import PDESolver

class FiniteDifferenceSolver(PDESolver):
    """Finite Difference Method solver for PDEs."""
    
    def __init__(self, Lx, Ly, T, Mx, My, Nt, gamma):
        super().__init__(Lx, Ly, T, Mx, My, Nt, gamma)
        self.dx = Lx / (Mx - 1)
        self.dy = Ly / (My - 1)
        self.dt = T / (Nt - 1)
        # Check stability condition (CFL)
        stability = gamma * self.dt * (1/self.dx**2 + 1/self.dy**2)
        assert stability <= 0.5, f"Stability condition not met: {stability} > 0.5"

    def solve(self, initial_condition, boundary_conditions):
        """Alias for solve_1d for backward compatibility"""
        return self.solve_2d(initial_condition, boundary_conditions)

    def solve_2d(self, initial_condition, boundary_conditions):
        """
        Solve 2D heat equation using explicit finite difference method.
        
        Args:
            initial_condition: Function that takes (x, y) and returns initial temperature
            boundary_conditions: Dictionary with boundary conditions
        
        Returns:
            tuple: (u, x, y, t) where u is solution array and others are grid points
        """
        # Create spatial and temporal grids
        x = jnp.linspace(0, self.Lx, self.Mx)
        y = jnp.linspace(0, self.Ly, self.My)
        t = jnp.linspace(0, self.T, self.Nt)
        X, Y = jnp.meshgrid(x, y)
        
        # Initialize solution array
        u = jnp.zeros((self.Nt, self.My, self.Mx))
        
        # Set initial condition
        u = u.at[0].set(initial_condition(X, Y))
        
        # Get boundary condition function
        bc_func = boundary_conditions['dirichlet']
        
        # Compute finite difference coefficients
        rx = self.gamma * self.dt / (self.dx * self.dx)
        ry = self.gamma * self.dt / (self.dy * self.dy)
        
        # Time stepping
        for n in range(self.Nt-1):
            # Apply boundary conditions
            bc_val = bc_func(t[n])
            u = u.at[n, 0, :].set(bc_val)    # Bottom boundary
            u = u.at[n, -1, :].set(bc_val)   # Top boundary
            u = u.at[n, :, 0].set(bc_val)    # Left boundary
            u = u.at[n, :, -1].set(bc_val)   # Right boundary
            
            # Update interior points
            for i in range(1, self.My-1):
                for j in range(1, self.Mx-1):
                    u = u.at[n+1, i, j].set(
                        u[n, i, j] + 
                        rx * (u[n, i, j+1] - 2*u[n, i, j] + u[n, i, j-1]) +
                        ry * (u[n, i+1, j] - 2*u[n, i, j] + u[n, i-1, j])
                    )
            
            # Apply boundary conditions for next time step
            bc_val = bc_func(t[n+1])
            u = u.at[n+1, 0, :].set(bc_val)    # Bottom boundary
            u = u.at[n+1, -1, :].set(bc_val)   # Top boundary
            u = u.at[n+1, :, 0].set(bc_val)    # Left boundary
            u = u.at[n+1, :, -1].set(bc_val)   # Right boundary
        
        return u, x, y, t