# Elmfire.jl Examples

This folder contains example scripts demonstrating Elmfire.jl capabilities.

## Prerequisites

Install required packages:
```julia
using Pkg
Pkg.add("Plots")
```

## Examples

### 1. `basic_simulation.jl`
Basic fire spread simulation comparing:
- No wind (circular spread)
- With wind (elliptical spread)
- Crown fire effects

Outputs:
- `burned_comparison.png` - Side-by-side burned areas
- `time_of_arrival.png` - When each cell burned
- `fireline_intensity.png` - Fire intensity comparison
- `spread_rate.png` - Rate of spread maps
- `summary.png` - 4-panel overview

### 2. `spotting_example.jl`
Demonstrates ember spotting (firebrands):
- Comparison with and without spotting enabled
- Shows how spot fires accelerate fire spread
- Configurable spotting parameters

Outputs:
- `spotting_comparison.png` - Burned area comparison
- `spotting_toa.png` - Time of arrival comparison
- `spotting_contours.png` - Fire progression contours

### 3. `animation.jl`
Creates animated visualizations:
- GIF of fire spread over time
- Time of arrival animation
- Perimeter evolution plot

Outputs:
- `fire_spread.gif` - Animated fire progression
- `fire_toa.gif` - Time of arrival animation
- `perimeter_evolution.png` - Perimeter at different times

## Running Examples

From the Julia REPL:
```julia
cd("path/to/Elmfire/examples")
include("basic_simulation.jl")
```

Or from the command line:
```bash
julia --project=.. basic_simulation.jl
```

Output files are saved to the `output/` subdirectory.
