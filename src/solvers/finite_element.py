import jax.numpy as jnp
from .base_solver import PDESolver

class FiniteElementSolver(PDESolver):
    """Finite Element Method solver for PDEs."""
    
    def __init__(self, Lx, Ly, T, Mx, My, Nt, gamma):
        super().__init__(Lx, Ly, T, Mx, My, Nt, gamma)
        # Initialize FEM specific parameters
        self.dx = Lx / (Mx - 1)
        self.dy = Ly / (My - 1)
        self.dt = T / (Nt - 1)

    def solve_2d(self, initial_condition, boundary_conditions):
        """
        Solve 2D heat equation using finite element method.
        
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
        
        # Create element matrices
        M_e = jnp.array([[4, 2, 2, 1],
                         [2, 4, 1, 2],
                         [2, 1, 4, 2],
                         [1, 2, 2, 4]]) * (self.dx * self.dy / 36)
        
        K_e = jnp.array([[2, -2, -1, 1],
                         [-2, 2, 1, -1],
                         [-1, 1, 2, -2],
                         [1, -1, -2, 2]]) * (self.dy/(6*self.dx))
        K_e += jnp.array([[2, 1, -2, -1],
                          [1, 2, -1, -2],
                          [-2, -1, 2, 1],
                          [-1, -2, 1, 2]]) * (self.dx/(6*self.dy))
        
        # Assemble global matrices
        M = jnp.zeros((self.Mx * self.My, self.Mx * self.My))
        K = jnp.zeros((self.Mx * self.My, self.Mx * self.My))
        
        for i in range(self.My - 1):
            for j in range(self.Mx - 1):
                nodes = jnp.array([
                    i * self.Mx + j,
                    i * self.Mx + (j + 1),
                    (i + 1) * self.Mx + j,
                    (i + 1) * self.Mx + (j + 1)
                ])
                
                # Assembly using JAX operations
                idx = nodes[:, None]
                M = M.at[idx, nodes].add(M_e)
                K = K.at[idx, nodes].add(K_e)
        
        # Time stepping matrix
        A = jnp.linalg.inv(M + self.gamma * self.dt * K)
        
        # Time stepping
        for n in range(self.Nt-1):
            # Get current solution vector
            u_vec = u[n].reshape(-1)
            
            # Apply boundary conditions
            bc_val = bc_func(t[n])
            boundary_nodes = self._get_boundary_nodes()
            u_vec = u_vec.at[boundary_nodes].set(bc_val)
            
            # Solve system
            u_new = A @ (M @ u_vec)
            
            # Reshape and store solution
            u = u.at[n+1].set(u_new.reshape(self.My, self.Mx))
            
            # Apply boundary conditions for next step
            bc_val = bc_func(t[n+1])
            u = u.at[n+1, 0, :].set(bc_val)    # Bottom boundary
            u = u.at[n+1, -1, :].set(bc_val)   # Top boundary
            u = u.at[n+1, :, 0].set(bc_val)    # Left boundary
            u = u.at[n+1, :, -1].set(bc_val)   # Right boundary
        
        return u, x, y, t
    
    def _get_boundary_nodes(self):
        """Get indices of boundary nodes."""
        bottom = jnp.arange(self.Mx)
        top = jnp.arange(self.Mx) + (self.My - 1) * self.Mx
        left = jnp.arange(self.My) * self.Mx
        right = jnp.arange(self.My) * self.Mx + (self.Mx - 1)
        return jnp.concatenate([bottom, top, left, right])

    def solve(self, initial_condition, boundary_conditions):
        """Alias for solve_2d for backward compatibility"""
        return self.solve_2d(initial_condition, boundary_conditions)