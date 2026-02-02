# ELMFIRE Julia Rewrite Assessment

## Executive Summary

**ELMFIRE** (Eulerian Level Set Model of FIRE spread) is an operational wildland fire simulation system written primarily in Fortran (~13,000 LOC). Rewriting it in Julia would be a **substantial but feasible project**, likely requiring 6-12 months of dedicated effort for a small team with expertise in both numerical methods and fire science.

---

## What is ELMFIRE?

ELMFIRE is a physics-based wildfire spread simulation system used for:

- Real-time fire spread forecasting (deployed via the Pyrecast project for US fires)
- Historical fire reconstruction
- Landscape-scale burn probability estimation
- Wildland-Urban Interface (WUI) fire modeling

The scientific foundation is published in [Fire Safety Journal](https://doi.org/10.1016/j.firesaf.2013.08.014).

---

## Current Codebase Overview

| Metric | Value |
|--------|-------|
| Primary Language | Fortran (53.9%) |
| Supporting Languages | Shell (36.9%), Python (9%) |
| Core Source Files | 13 Fortran modules |
| Lines of Code (Fortran) | ~13,000 |
| Repository Size | 57.7 MB |
| License | EPL-2.0 |
| Active Development | Yes (last commit Jan 2025) |

### Key Fortran Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `elmfire_io.f90` | 2,544 | GeoTIFF/VRT/BSQ I/O, MPI parallel I/O |
| `elmfire_level_set.f90` | 2,126 | Core level set PDE solver |
| `elmfire_subs.f90` | 1,550 | Utilities, interpolation, coordinate transforms |
| `elmfire_spotting.f90` | 1,087 | Ember transport, spotting models |
| `elmfire_post.f90` | 1,047 | Statistical post-processing |
| `elmfire_spread_rate.f90` | 661 | Rothermel model, crown fire physics |
| `elmfire_vars.f90` | 620 | Global variables, fuel model tables |
| Others | ~3,300 | Init, config, ignition, suppression, calibration |

---

## Core Algorithms Requiring Implementation

### 1. Level Set Method (Highest Complexity)

The heart of ELMFIRE is solving the hyperbolic PDE:

```
∂φ/∂t + Ux(∂φ/∂x) + Uy(∂φ/∂y) = 0
```

Key numerical features:
- Narrow-band formulation (Sethian, 1996)
- Second-order Runge-Kutta time integration
- Superbee flux limiters for stability
- CFL-constrained adaptive time stepping

**Julia Consideration**: Packages like `LevelSetMethods.jl` or `DifferentialEquations.jl` could provide foundations, but the specific narrow-band implementation with fire-specific boundary conditions would need custom work.

### 2. Fire Spread Physics (High Complexity)

- **Rothermel Surface Fire Model**: Reaction intensity, spread rate with wind/slope
- **Elliptical Fire Shape**: Huygens principle with length/width ratios
- **Crown Fire Transition**: Cruz (2005) model linking surface to crown fire
- **Slope Corrections**: Terrain aspect effects on spread velocity

**Julia Consideration**: These are well-documented empirical models that translate straightforwardly to Julia, but require careful validation against reference implementations.

### 3. MPI Parallelization (High Complexity)

- Domain decomposition for large landscapes
- MPI shared memory windows for fuel/weather data
- Parallel I/O for large raster datasets

**Julia Consideration**: `MPI.jl` provides MPI bindings, but the shared memory window patterns in ELMFIRE are sophisticated. Julia's native parallelism (threads, distributed) might offer simpler alternatives for some use cases.

### 4. Geospatial I/O (Medium-High Complexity)

- GeoTIFF reading/writing
- VRT (GDAL virtual raster) support
- BSQ with ENVI headers
- Coordinate reference system handling

**Julia Consideration**: `ArchGDAL.jl` and `Rasters.jl` handle most formats. The challenge is matching ELMFIRE's specific I/O patterns and ensuring perfect compatibility with existing data pipelines.

### 5. Ember Transport & Spotting (Medium Complexity)

- Lagrangian particle tracking
- Eulerian grid-based ember density
- Multiple spotting distance distributions (lognormal, Sardoy, Himoto)

**Julia Consideration**: Particle tracking is natural in Julia. `DifferentialEquations.jl` could handle trajectory integration.

### 6. Suppression & WUI Models (Medium Complexity)

- Directional containment strategies
- Hamada building separation model
- UMD/UCB radiation models for structure ignition

### 7. Monte Carlo Ensemble System (Low-Medium Complexity)

- Randomized ignition locations
- Weather perturbations
- Ensemble statistics (burn probability, percentiles)

**Julia Consideration**: Julia excels at this. `Distributions.jl` and native parallelism make Monte Carlo simulations elegant.

---

## Dependencies to Replace

| Fortran/System Dep | Julia Equivalent | Maturity |
|--------------------|------------------|----------|
| MPI (OpenMPI) | `MPI.jl` | Mature |
| GDAL | `ArchGDAL.jl`, `Rasters.jl` | Mature |
| Fortran namelist config | `TOML.jl` or custom | Easy |
| gfortran/ifort | Native Julia | N/A |

---

## Advantages of Julia Rewrite

1. **Unified Language**: Replace Fortran + Shell + Python with pure Julia
2. **Modern Tooling**: Package management, testing, documentation all built-in
3. **Easier Extensibility**: Adding new fire models or physics would be simpler
4. **Interactive Development**: REPL-based exploration and debugging
5. **Differentiability**: Automatic differentiation for calibration/optimization via `Zygote.jl`
6. **GPU Potential**: Easy GPU acceleration via `CUDA.jl` for level set solver
7. **Better Visualization**: Direct plotting integration
8. **Community**: Growing scientific Julia ecosystem

---

## Challenges & Risks

### Technical Challenges

1. **Numerical Fidelity**: The level set method must produce identical fire perimeters. Subtle differences in floating-point handling or flux limiters could cause divergent results.

2. **Performance Parity**: Fortran is highly optimized for numerical computing. Julia should match or exceed performance, but requires careful implementation.

3. **MPI Complexity**: ELMFIRE's MPI patterns (shared memory windows, complex domain decomposition) are non-trivial to replicate.

4. **Validation Burden**: Extensive testing against ELMFIRE reference cases needed.

5. **I/O Compatibility**: Must read/write identical file formats for pipeline integration.

### Project Risks

1. **Domain Expertise Required**: Understanding fire science is essential for correct implementation
2. **Moving Target**: ELMFIRE is actively developed; a rewrite would need to track upstream changes
3. **Operational Deployment**: Pyrecast uses ELMFIRE operationally; a Julia version would need equal reliability

---

## Estimated Effort

### By Component

| Component | Estimated Effort | Complexity |
|-----------|------------------|------------|
| Level set solver | 3-4 weeks | High |
| Fire spread physics (Rothermel, crown) | 2-3 weeks | Medium-High |
| Geospatial I/O | 2-3 weeks | Medium-High |
| MPI parallelization | 3-4 weeks | High |
| Spotting/ember transport | 1-2 weeks | Medium |
| WUI models | 1 week | Medium |
| Suppression | 1 week | Low-Medium |
| Monte Carlo framework | 1 week | Low |
| Configuration system | 1 week | Low |
| Testing & validation | 4-6 weeks | High |
| Documentation | 2 weeks | Low |

### Total Estimate

- **Minimum Viable Implementation**: 3-4 months (core functionality, single-threaded)
- **Full Feature Parity**: 6-9 months (all features, parallelization)
- **Production Ready**: 9-12 months (validation, optimization, documentation)

This assumes 1-2 developers with Julia experience and access to fire science expertise.

---

## Recommended Approach

### Phase 1: Core Engine (Months 1-3)
1. Implement Rothermel spread model with unit tests
2. Build level set solver on a fixed grid
3. Single-fire simulation with constant wind
4. Validate against ELMFIRE test cases

### Phase 2: Full Physics (Months 4-6)
1. Add crown fire model
2. Implement spotting/ember transport
3. Weather interpolation
4. Variable terrain/fuels

### Phase 3: Operational Features (Months 7-9)
1. Monte Carlo ensemble support
2. Parallel execution (threads first, MPI optional)
3. Full geospatial I/O compatibility
4. WUI and suppression models

### Phase 4: Production (Months 10-12)
1. Performance optimization
2. Comprehensive validation suite
3. Documentation and tutorials
4. Package registration

---

## Existing Julia Ecosystem Resources

| Package | Relevance |
|---------|-----------|
| `DifferentialEquations.jl` | PDE solving infrastructure |
| `Rasters.jl` | Geospatial raster handling |
| `ArchGDAL.jl` | GDAL bindings |
| `MPI.jl` | Distributed computing |
| `CUDA.jl` | GPU acceleration potential |
| `Distributions.jl` | Monte Carlo sampling |
| `Makie.jl` | Visualization |
| `Documenter.jl` | Documentation generation |

---

## Conclusion

Rewriting ELMFIRE in Julia is a **significant but worthwhile undertaking**. The ~13,000 lines of Fortran are dense with numerical methods and fire physics, making this more than a simple translation exercise.

**Key factors favoring the rewrite:**
- Julia's performance matches Fortran for numerical code
- Unified language eliminates Shell/Python glue code
- Modern package ecosystem provides building blocks
- Potential for GPU acceleration and automatic differentiation

**Key challenges:**
- Level set method requires careful numerical implementation
- MPI patterns are sophisticated
- Extensive validation needed for operational use
- Fire science domain expertise essential

A dedicated team could achieve feature parity in 6-12 months, with the resulting code likely being more maintainable and extensible than the current Fortran implementation.

---

## References

- ELMFIRE Repository: https://github.com/lautenberger/elmfire
- Scientific Paper: https://doi.org/10.1016/j.firesaf.2013.08.014
- ELMFIRE Documentation: https://elmfire.io/
