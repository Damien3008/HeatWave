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
        
        # Get boundary condition type and function
        bc_type = boundary_conditions['type']
        if bc_type == 'dirichlet':
            bc_func = boundary_conditions['function']
        elif bc_type == 'neumann':
            bc_func = boundary_conditions['function']
        elif bc_type == 'mixed':
            mixed_bcs = boundary_conditions
        else:
            raise ValueError(f"Boundary condition type {bc_type} not supported in FDM solver")
        
        # Compute finite difference coefficients
        rx = self.gamma * self.dt / (self.dx * self.dx)
        ry = self.gamma * self.dt / (self.dy * self.dy)
        
        # Time stepping
        for n in range(self.Nt-1):
            # Apply boundary conditions based on type
            if bc_type == 'dirichlet':
                bc_val = bc_func(t[n])
                u = u.at[n, 0, :].set(bc_val)    # Bottom boundary
                u = u.at[n, -1, :].set(bc_val)   # Top boundary
                u = u.at[n, :, 0].set(bc_val)    # Left boundary
                u = u.at[n, :, -1].set(bc_val)   # Right boundary
            elif bc_type == 'neumann':
                flux = bc_func(t[n])
                # Apply Neumann BC using finite differences
                u = u.at[n, 0, 1:-1].set(u[n, 1, 1:-1] - flux * self.dy)    # Bottom
                u = u.at[n, -1, 1:-1].set(u[n, -2, 1:-1] + flux * self.dy)  # Top
                u = u.at[n, 1:-1, 0].set(u[n, 1:-1, 1] - flux * self.dx)    # Left
                u = u.at[n, 1:-1, -1].set(u[n, 1:-1, -2] + flux * self.dx)  # Right
            elif bc_type == 'mixed':
                self._apply_mixed_bc(u, n, mixed_bcs)
            
            # Update interior points
            for i in range(1, self.My-1):
                for j in range(1, self.Mx-1):
                    u = u.at[n+1, i, j].set(
                        u[n, i, j] + 
                        rx * (u[n, i, j+1] - 2*u[n, i, j] + u[n, i, j-1]) +
                        ry * (u[n, i+1, j] - 2*u[n, i, j] + u[n, i-1, j])
                    )
            
            # Apply boundary conditions for next time step
            if bc_type == 'dirichlet':
                bc_val = bc_func(t[n+1])
                u = u.at[n+1, 0, :].set(bc_val)    # Bottom boundary
                u = u.at[n+1, -1, :].set(bc_val)   # Top boundary
                u = u.at[n+1, :, 0].set(bc_val)    # Left boundary
                u = u.at[n+1, :, -1].set(bc_val)   # Right boundary
        
        return u, x, y, t

    def _apply_mixed_bc(self, u, n, mixed_bcs):
        """Apply mixed boundary conditions."""
        for side, bc in [('left', mixed_bcs['left']), 
                        ('right', mixed_bcs['right']),
                        ('top', mixed_bcs['top']), 
                        ('bottom', mixed_bcs['bottom'])]:
            if bc['type'] == 'dirichlet':
                if side == 'left':
                    u = u.at[n, :, 0].set(bc['value'])
                elif side == 'right':
                    u = u.at[n, :, -1].set(bc['value'])
                elif side == 'top':
                    u = u.at[n, -1, :].set(bc['value'])
                else:  # bottom
                    u = u.at[n, 0, :].set(bc['value'])
            else:  # neumann
                flux = bc['value']
                if side == 'left':
                    u = u.at[n, 1:-1, 0].set(u[n, 1:-1, 1] - flux * self.dx)
                elif side == 'right':
                    u = u.at[n, 1:-1, -1].set(u[n, 1:-1, -2] + flux * self.dx)
                elif side == 'top':
                    u = u.at[n, -1, 1:-1].set(u[n, -2, 1:-1] + flux * self.dy)
                else:  # bottom
                    u = u.at[n, 0, 1:-1].set(u[n, 1, 1:-1] - flux * self.dy)
        return u

    def solve(self, initial_condition, boundary_conditions):
        """Alias for solve_2d for backward compatibility"""
        return self.solve_2d(initial_condition, boundary_conditions)