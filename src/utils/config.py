import os
import yaml
import jax.numpy as jnp

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file (relative or absolute)
    
    Returns:
        dict: Configuration dictionary
    """
    # If it's a relative path, make it relative to the current working directory
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)
    
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_solver(config, solvers):
    """Get solver instance based on configuration."""
    solver_type = config['solver_type'].lower()
    if solver_type not in solvers:
        raise ValueError(f"Unknown solver type: {solver_type}")
    
    # Extract parameters from config
    domain = config['domain']
    disc = config['discretization']
    phys = config['physics']
    
    return solvers[solver_type](
        domain['Lx'], domain['Ly'], domain['T'],
        disc['Mx'], disc['My'], disc['Nt'],
        phys['gamma']
    )

def get_initial_condition(config):
    """Create initial condition function from config."""
    ic_config = config['initial_condition']
    ic_type = ic_config['type'].lower()
    
    if ic_type == 'gaussian':
        def initial_condition(x, y):
            return ic_config['amplitude'] * jnp.exp(
                -(((x - ic_config['center_x'])**2 + 
                   (y - ic_config['center_y'])**2) / 
                  (2 * ic_config['width']**2))
            )
    elif ic_type == 'sinusoidal':
        def initial_condition(x, y):
            return ic_config['amplitude'] * jnp.sin(jnp.pi * x / config['domain']['Lx']) * \
                   jnp.sin(jnp.pi * y / config['domain']['Ly'])
    elif ic_type == 'constant':
        def initial_condition(x, y):
            return ic_config['amplitude'] * jnp.ones_like(x)
    elif ic_type == 'custom':
        def initial_condition(x, y):
            # Initialize with zeros
            temperature = jnp.zeros_like(x)
            # Add contribution from each source
            for source in ic_config['sources']:
                temperature += source['amplitude'] * jnp.exp(
                    -(((x - source['x'])**2 + 
                       (y - source['y'])**2) / 
                      (2 * source['width']**2))
                )
            return temperature
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")
    
    return initial_condition 