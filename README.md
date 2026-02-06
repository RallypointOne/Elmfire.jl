# Elmfire.jl

[![Build Status](https://github.com/RallypointOne/Elmfire.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/RallypointOne/Elmfire.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs Workflow](https://github.com/RallypointOne/Elmfire.jl/actions/workflows/docs.yml/badge.svg)](https://github.com/RallypointOne/Elmfire.jl/actions/workflows/docs.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://rallypointone.github.io/Elmfire.jl/)

A Julia implementation of the ELMFIRE wildfire spread model. This package provides tools for simulating fire behavior including surface fire spread, crown fire, ember spotting, and more.

## Features

- **Rothermel Surface Fire Model**: Industry-standard fire spread rate calculations
- **Level-Set Solver**: Accurate fire front propagation using narrow-band level-set methods
- **Crown Fire**: Coupled surface-crown fire modeling with canopy properties
- **Spotting/Ember Transport**: Lognormal ember distribution and transport physics
- **Weather**: Time-varying weather with spatial interpolation
- **Geospatial I/O**: Read/write GeoTIFF rasters, compute slope/aspect from DEMs
- **Monte Carlo Ensembles**: Probabilistic fire spread with burn probability maps
- **Parallel Execution**: Multi-threaded ensemble simulations
- **WUI Models**: Wildland-urban interface structure ignition
- **Suppression Models**: Containment line construction and resource allocation

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/joshday/Elmfire.jl")
```

## Quick Start

```julia
using Elmfire

# Create a 100x100 grid with 30-ft cells
state = FireState(100, 100, 30.0)

# Load standard fuel models (FBFM 1-40)
fuel_table = create_standard_fuel_table()

# Set weather conditions
weather = ConstantWeather(
    wind_speed_mph = 15.0,
    wind_direction = 270.0,  # Wind from the west
    M1 = 0.06,               # 1-hr dead fuel moisture (6%)
    M10 = 0.08,              # 10-hr dead fuel moisture
    M100 = 0.10,             # 100-hr dead fuel moisture
    MLH = 0.60,              # Live herbaceous moisture
    MLW = 0.90               # Live woody moisture
)

# Ignite the center of the grid
ignite!(state, 50, 50, 0.0)

# Run simulation for 30 minutes on fuel model 1 (short grass)
simulate_uniform!(state, 1, fuel_table, weather, 0.0, 0.0, 0.0, 30.0)

# Get results
burned_acres = get_burned_area_acres(state)
```

## Crown Fire Simulation

```julia
# Enable crown fire with canopy properties
config = SimulationConfig{Float64}(
    enable_crown_fire = true,
    foliar_moisture = 100.0
)

simulate_full_uniform!(
    state, 1, fuel_table, weather, 0.0, 0.0, 0.0, 30.0;
    canopy_cbd = 0.15,  # Canopy bulk density (kg/m³)
    canopy_cbh = 3.0,   # Canopy base height (m)
    canopy_cc = 0.7,    # Canopy cover (70%)
    canopy_ch = 18.0,   # Canopy height (m)
    config = config
)
```

## Time-Varying Weather

```julia
# Create weather grids for different times
grid1 = WeatherGrid{Float64}(weather_morning, 1, 1, 1e6)
grid2 = WeatherGrid{Float64}(weather_afternoon, 1, 1, 1e6)

# Create time series (weather changes at 60 minutes)
times = [0.0, 60.0]
weather_series = WeatherTimeSeries{Float64}([grid1, grid2], times)
weather_interp = WeatherInterpolator(weather_series, ncols, nrows, cellsize)

# Run with time-varying weather
simulate_full!(state, fuel_ids, fuel_table, weather_interp, slope, aspect, 0.0, 120.0)
```

## Ensemble Simulations

```julia
# Configure perturbations for Monte Carlo runs
perturb_config = PerturbationConfig{Float64}(
    wind_speed_std = 3.0,
    wind_direction_std = 15.0,
    moisture_std = 0.02
)

ensemble_config = EnsembleConfig{Float64}(
    n_members = 100,
    perturbation = perturb_config
)

# Run ensemble (multi-threaded)
result = run_ensemble!(ensemble_config, base_state, fuel_ids, fuel_table,
                       weather_interp, slope, aspect, 0.0, 60.0)

# Get burn probability map
burn_prob = result.burn_probability
```

## Geospatial Data

```julia
# Read landscape data from GeoTIFF files
landscape = read_landscape(
    fuel_path = "fuels.tif",
    dem_path = "elevation.tif"
)

# Write fire perimeter as GeoJSON
write_fire_perimeter(state, landscape.metadata, "perimeter.geojson")

# Write output raster
write_geotiff("burned.tif", state.burned, landscape.metadata)
```

## Level Set PDE Solver

Fire front propagation is computed by solving the level set equation:

```
∂φ/∂t + F|∇φ| = 0
```

Where φ is the signed distance function (φ < 0 inside fire, φ > 0 outside) and F is the local fire spread rate. The numerical scheme uses:

- **2nd-order Runge-Kutta** time integration
- **Superbee flux limiter** for gradient calculation (prevents oscillations)
- **Narrow band method** for efficiency (only computes near the fire front)
- **CFL-adaptive timestep** for numerical stability

## Examples

See the `examples/` directory for complete examples:

- `basic_simulation.jl` - Getting started with fire simulations
- `spotting_example.jl` - Ember transport and spot fire ignition
- `animation.jl` - Animated fire spread visualization
- `marshall_fire/` - Case study with real-world data

## Documentation

Tutorials are available in `docs/tutorials/`:

- Weather Effects
- Fuel Models
- Terrain Effects
- Crown Fire
- Ensemble Analysis

## References

- Rothermel, R.C. (1972). A mathematical model for predicting fire spread in wildland fuels.
- Anderson, H.E. (1983). Predicting wind-driven wild land fire size and shape.
- Scott, J.H. & Reinhardt, E.D. (2001). Assessing crown fire potential.

## License

MIT License
