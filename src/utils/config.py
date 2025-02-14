import yaml
import jax.numpy as jnp
from .paths import get_config_path

def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file, relative to config directory
    """
    full_path = get_config_path(config_path)
    with open(full_path, 'r') as f:
        return yaml.safe_load(f)

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
    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")
    
    return initial_condition 