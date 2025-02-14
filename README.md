<div align="center">

# HeatWave: A High-Performance 2D Heat Equation Solver with Multiple Numerical Methods
### Leveraging JAX for GPU-Accelerated Heat Diffusion Simulations

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4.13-green.svg)](https://github.com/google/jax)
[![License: GNU](https://img.shields.io/badge/License-GNU-yellow.svg)](https://opensource.org/licenses/GNU)

<img src="asset/periodic_heat_wave.gif" alt="Periodic Heat Wave" width="800"/>

*Visualization of a periodic heat wave with sinusoidal boundary conditions using the spectral method*

</div>

## 📝 Overview

HeatWave is a high-performance numerical solver for heat equations, leveraging JAX for GPU acceleration and automatic differentiation. This project demonstrates the practical application of numerical methods in computational physics using modern high-performance computing techniques.

### Simulation Parameters
- Gaussian initial condition
- Zero temperature boundary conditions
- Diffusion coefficient γ = 0.1
- Grid size: 50x50
- Time span: T = 30.0

## 🔥 Key Features

### Multiple Numerical Methods
- **Finite Difference Method (FDM)**
  - Explicit time-stepping scheme
  - Second-order central differences
  - Stability condition: CFL ≤ 0.5

- **Finite Element Method (FEM)**
  - Galerkin formulation
  - Linear elements
  - Mass and stiffness matrix assembly

- **Spectral Method**
  - Fast Fourier Transform (FFT) based
  - Exponential time differencing
  - Highly accurate for smooth solutions

### Boundary Conditions
- **Dirichlet**: Fixed temperature values at boundaries
- **Neumann**: Fixed heat flux at boundaries
- **Mixed**: Different conditions for each boundary
- **Time-dependent**: Oscillating or varying boundary conditions

### Example Configurations

1. **Heat Diffusion** (`config/solver_config.yaml`)
   - Basic heat diffusion example
   - Gaussian initial condition
   - Dirichlet boundary conditions
   ```yaml
   initial_condition:
     type: "gaussian"
     amplitude: 1.0
     center_x: 2.5
     center_y: 2.5
     width: 1.0
   ```

2. **Periodic Heat Wave** (`config/periodic_heat_wave.yaml`)
   - Time-dependent boundary conditions
   - Sinusoidal temperature oscillation
   - High-resolution grid (100x100)
   ```yaml
   boundary_conditions:
     type: "time_dependent"
     time_dependent:
       enabled: true
       function: "sin"
       amplitude: 0.8
       frequency: 0.5
   ```

3. **Four Heat Sources** (`config/four_sources.yaml`)
   - Multiple Gaussian heat sources
   - Symmetric configuration
   - Zero-temperature boundaries
   ```yaml
   initial_condition:
     type: "custom"
     sources: [
       {"x": 2.5, "y": 2.5, "amplitude": 1.0, "width": 0.8},
       {"x": 2.5, "y": 7.5, "amplitude": 1.0, "width": 0.8},
       {"x": 7.5, "y": 2.5, "amplitude": 1.0, "width": 0.8},
       {"x": 7.5, "y": 7.5, "amplitude": 1.0, "width": 0.8}
     ]
   ```

4. **Four Sources with Periodic Heat** (`config/four_sources_periodic.yaml`)
   - Four oscillating heat sources
   - Time-dependent periodic boundaries
   - Dynamic wave-like patterns
   ```yaml
   initial_condition:
     type: "custom"
     sources: [
       {"x": 2.5, "y": 2.5, "amplitude": 2.0, "width": 0.5},
       {"x": 2.5, "y": 7.5, "amplitude": 2.0, "width": 0.5},
       {"x": 7.5, "y": 2.5, "amplitude": 2.0, "width": 0.5},
       {"x": 7.5, "y": 7.5, "amplitude": 2.0, "width": 0.5}
     ]
   boundary_conditions:
     type: "time_dependent"
     time_dependent:
       enabled: true
       function: "sin"
       amplitude: 1.0
       frequency: 2.0
   ```

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/heatwave.git
cd heatwave

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

The `-e` flag installs the package in "editable" or "development" mode, which means:
- Changes to the source code take effect immediately
- No need to reinstall after code changes
- Package is installed in your Python environment with references to the source code

### Usage
1. Choose a configuration file from the examples or create your own:
```bash
python examples/heat_equation_example.py --config config/<YOUR_CONFIG_FILE>.yaml
```

2. Modify parameters in the config file to experiment with different scenarios:
```yaml
solver_type: "spectral"  # Options: spectral, finite_difference, finite_element
domain:
  Lx: 5.0
  Ly: 5.0
  T: 5.0
```

## 📁 Project Structure
```
heatwave/
├── config/
│   └── solver_config.yaml     # Configuration file
├── examples/
│   └── heat_equation_example.py
├── src/
│   ├── solvers/
│   │   ├── base_solver.py
│   │   ├── finite_difference.py
│   │   ├── finite_element.py
│   │   └── spectral.py
│   └── utils/
│       └── config.py
└── asset/
    └── report.pdf            # Detailed project report
```

## 📜 License
This project is licensed under the GNU License - see the [LICENSE](LICENSE) file for details.

---
<div align="center">
<i>This project was developed as part of an engineering school curriculum.</i>
</div>