import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os
from src.solvers import FiniteDifferenceSolver, FiniteElementSolver, SpectralSolver
from src.utils.config import load_config, get_solver, get_initial_condition

def create_heat_animation(u, x, y, t, config):
    """Create animation of heat diffusion process."""
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=config['visualization']['animation']['fps'], 
                       metadata=dict(artist='Me'),
                       bitrate=1800)
    except Exception as e:
        print(f"Error setting up animation writer: {e}")
        print("Please make sure ffmpeg is installed:")
        print("  - On Mac: brew install ffmpeg")
        print("  - On Ubuntu: sudo apt-get install ffmpeg")
        print("  - On Windows: conda install ffmpeg")
        return

    viz_config = config['visualization']
    anim_config = viz_config['animation']
    
    # Create figure
    fig = plt.figure(figsize=viz_config['figsize'])
    
    # Setup subplots
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    
    # Create mesh grid
    X, Y = jnp.meshgrid(x, y)
    
    # Create descriptive title
    solver_type = config['solver_type'].upper()
    ic_type = config['initial_condition']['type'].capitalize()
    bc_type = config['boundary_conditions']['type'].capitalize()
    gamma = config['physics']['gamma']
    
    title = f'Heat Equation Solution ({solver_type} Method)\n'
    title += f'Initial: {ic_type}, BC: {bc_type}, γ={gamma:.3f}'
    
    # Set title
    fig.suptitle(title, fontsize=12)
    
    # Select frames to animate
    frame_indices = range(0, len(t), anim_config['skip_frames'])
    times = t[::anim_config['skip_frames']]
    
    # Find global min/max for consistent colorbar
    vmin = jnp.min(u)
    vmax = jnp.max(u)
    
    # Create initial surface plot
    surf = ax1.plot_surface(X, Y, u[0], cmap=viz_config['cmap'])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('Temperature')
    
    # Create initial contour plot
    contour = ax2.contourf(X, Y, u[0], levels=viz_config['levels'], 
                          cmap=viz_config['cmap'], vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(contour, ax=ax2)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    
    # Function to update frames
    def update(frame_idx):
        # Clear axes but keep colorbar
        ax1.clear()
        ax2.clear()
        
        # Get current solution
        u_current = u[frame_idx * anim_config['skip_frames']]
        t_current = times[frame_idx]
        
        # Update 3D surface plot
        surf = ax1.plot_surface(X, Y, u_current, cmap=viz_config['cmap'],
                              vmin=vmin, vmax=vmax)
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('Temperature')
        ax1.set_title(f't = {t_current:.2f}')
        
        # Set consistent view angle for 3D plot
        ax1.view_init(elev=30, azim=45)
        ax1.set_zlim([vmin, vmax])
        
        # Update contour plot
        contour = ax2.contourf(X, Y, u_current, levels=viz_config['levels'], 
                              cmap=viz_config['cmap'], vmin=vmin, vmax=vmax)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f't = {t_current:.2f}')
        
        # Maintain consistent axes limits
        ax2.set_xlim([x[0], x[-1]])
        ax2.set_ylim([y[0], y[-1]])
        
        return surf, contour

    # Create animation with tight layout
    plt.tight_layout()
    anim = animation.FuncAnimation(
        fig, update, frames=len(frame_indices),
        interval=anim_config['interval'], blit=False
    )
    
    # Save animation
    try:
        filename = anim_config['filename']
        print(f"Saving animation to {filename}...")
        anim.save(filename, writer=writer)
        print("Animation saved successfully!")
    except Exception as e:
        print(f"Error saving animation: {e}")
    finally:
        plt.close()

def plot_static_solution(u, x, y, t, config):
    """Plot static 2D solution at final time."""
    viz_config = config['visualization']
    fig = plt.figure(figsize=viz_config['figsize'])
    
    # 3D surface plot
    if viz_config.get('surface_plot', True):
        ax1 = fig.add_subplot(121, projection='3d')
        X, Y = jnp.meshgrid(x, y)
        surf = ax1.plot_surface(X, Y, u[-1], cmap=viz_config['cmap'])
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_zlabel('Temperature')
        ax1.set_title(f'Surface plot at t = {t[-1]:.2f}')
    
    # Contour plot
    if viz_config.get('contour_plot', True):
        ax2 = fig.add_subplot(122)
        X, Y = jnp.meshgrid(x, y)
        contour = ax2.contourf(X, Y, u[-1], levels=viz_config['levels'], 
                              cmap=viz_config['cmap'])
        plt.colorbar(contour)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Contour plot at t = {t[-1]:.2f}')
    
    # Create descriptive title
    solver_type = config['solver_type'].upper()
    ic_type = config['initial_condition']['type'].capitalize()
    bc_type = config['boundary_conditions']['type'].capitalize()
    gamma = config['physics']['gamma']
    
    title = f'Heat Equation Solution ({solver_type} Method)\n'
    title += f'Initial: {ic_type}, BC: {bc_type}, γ={gamma:.6f}, t={t[-1]:.2f}'
    
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.show()

def get_boundary_conditions(config):
    """Create boundary conditions from config."""
    bc_config = config['boundary_conditions']
    bc_type = bc_config['type'].lower()
    
    if bc_type == "dirichlet":
        value = bc_config['dirichlet']['value']
        return {
            'type': 'dirichlet',
            'function': lambda t: value
        }
    
    elif bc_type == "neumann":
        flux = bc_config['neumann']['flux']
        return {
            'type': 'neumann',
            'function': lambda t: flux
        }
    
    elif bc_type == "periodic":
        return {
            'type': 'periodic'
        }
    
    elif bc_type == "mixed":
        mixed_bcs = bc_config['mixed']
        return {
            'type': 'mixed',
            'left': mixed_bcs['left'],
            'right': mixed_bcs['right'],
            'top': mixed_bcs['top'],
            'bottom': mixed_bcs['bottom']
        }
    
    elif bc_type == "time_dependent":
        td_config = bc_config['time_dependent']
        if td_config['function'] == 'sin':
            A, f = td_config['amplitude'], td_config['frequency']
            return {
                'type': 'dirichlet',
                'function': lambda t: A * jnp.sin(2 * jnp.pi * f * t)
            }
        elif td_config['function'] == 'cos':
            A, f = td_config['amplitude'], td_config['frequency']
            return {
                'type': 'dirichlet',
                'function': lambda t: A * jnp.cos(2 * jnp.pi * f * t)
            }
        elif td_config['function'] == 'linear':
            A = td_config['amplitude']
            return {
                'type': 'dirichlet',
                'function': lambda t: A * t
            }
    
    raise ValueError(f"Unknown boundary condition type: {bc_type}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Solve 2D heat equation with various methods.')
    parser.add_argument('--config', type=str, default='config/solver_config.yaml',
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    print(f"Using configuration from: {args.config}")
    
    # Available solvers
    solvers = {
        'finite_difference': FiniteDifferenceSolver,
        'finite_element': FiniteElementSolver,
        'spectral': SpectralSolver
    }
    
    # Initialize solver from config
    solver = get_solver(config, solvers)
    print(f"Using solver: {config['solver_type']}")
    
    # Get initial and boundary conditions
    initial_condition = get_initial_condition(config)
    boundary_conditions = get_boundary_conditions(config)
    
    # Solve the equation
    print("Solving heat equation...")
    u, x, y, t = solver.solve_2d(initial_condition, boundary_conditions)
    
    # Visualize solution based on config
    viz_type = config['visualization']['type'].lower()
    if viz_type == 'animation':
        print("Creating animation...")
        create_heat_animation(u, x, y, t, config)
    else:
        print("Creating static plot...")
        plot_static_solution(u, x, y, t, config)

if __name__ == '__main__':
    main() 