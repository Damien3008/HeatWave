import os

def get_project_root():
    """Get the absolute path to the project root directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

def get_config_path(config_name="solver_config.yaml"):
    """Get the absolute path to a config file."""
    return os.path.join(get_project_root(), "config", config_name) 