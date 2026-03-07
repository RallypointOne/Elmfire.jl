# Elmfire.jl vs. Original ELMFIRE Fortran: Physics & Numerics Comparison

## 1. Rothermel Surface Fire Spread Model

**Verdict: Faithful reproduction**

Both implementations use identical core equations:

| Component | Formula | Match? |
|-----------|---------|--------|
| Reaction intensity | `IR = Γ'·WN·η_s·HOC·η_m(M/Mex)` | Identical |
| Moisture damping | `η = 1 - 2.59r + 5.11r² - 3.52r³` | Identical |
| Propagating flux ratio | `ξ = exp((0.792+0.681√σ)(0.1+β)) / (192+0.2595σ)` | Identical |
| Reaction velocity | `Γ' = Γ'_peak·(β/β_op)^A·exp(A(1-β/β_op))` | Identical |
| Wind factor | `φ_w = C·(β/β_op)^(-E)·w^B` | Identical |
| Slope factor | `φ_s = 5.275·β^(-0.3)·tan²θ` | Identical |
| Spread rate | `R = R₀·(1 + φ_s + φ_w)` | Identical |
| Wind limit | `w_limited = min(w, 0.9·IR)` | Identical |
| Residence time | `τ = 384/σ` | Identical |
| Heat of pre-ignition | `Q_ig = 250 + 1116·M` | Identical |
| Optimal packing | `β_op = 3.348/σ^0.8189` | Identical |
| All A/B/C/E coefficients | `A=133/σ^0.7913`, etc. | Identical |

Constants also match: `ρ_p=32`, `st=0.055`, `SE=0.01`, `η_s=0.174/SE^0.19`, `σ_10hr=109`, `σ_100hr=30`.

**One difference in fuel model processing:**

- Fortran pre-computes a **2D fuel model table** indexed by `[fuel_id, live_moisture_class]` (304 models × 91 moisture classes from 30–120%), recalculating dynamic herbaceous partitioning for each class.
- Julia computes fuel models on-the-fly for a single moisture class, or uses `FuelModelArray` (same 2D structure) for GPU.

## 2. Elliptical Fire Spread

**Verdict: Matching** (resolved)

**Length-to-breadth ratio** — identical (Anderson 1982):

```
LB = 0.936·exp(0.2566·U) + 0.461·exp(-0.1548·U) - 0.397
```

- Fortran: caps at configurable `MAX_LOW`
- Julia: caps at 8

**Velocity at arbitrary angle** — both now use the Richards (1990) ellipse parametric equation:

```
A = 0.5·(V_head + V_back)
B = A / LB
denom = sqrt(A²·cos²θ + B²·sin²θ)
DYDT = A²·cosθ/denom + 0.5·(V_head - V_back)
DXDT = B²·sinθ/denom
```

Julia implements this via `velocity_components(es::EllipticalSpread, ...)` in `rothermel.jl`, used by both CPU and GPU code paths.

## 3. Level Set Numerics

**Verdict: Matching** (resolved)

| Aspect | Fortran | Julia | Match? |
|--------|---------|-------|--------|
| PDE | `∂φ/∂t + (ux·∂φ/∂x + uy·∂φ/∂y) = 0` | Same | Identical |
| Time integration | RK2 (Heun's method) | RK2 (Heun's method) | Identical |
| Flux limiter | Half-superbee: `max(0, max(min(r/2,1), min(r,1/2)))` | Same formula | Identical |
| Gradient stencil | 4-point upwind per direction | Same | Identical |
| Reinitialization | None (clamp φ to [-100, 100]) | None (clamp to [-1000, 1000]) | Same approach |
| Target CFL | 0.4 | 0.45 | Close match |
| **Band thickness** | **2 cells** | **5 cells** | **Different** |
| Padding | 3 cells from edges | 2-cell explicit padding | Similar |
| dt_max | 600 seconds | 10 minutes | Same |

**Band thickness difference:** Julia's default of 5 cells vs. Fortran's 2 means Julia tracks more cells around the fire front. This is less efficient but more robust against fast-moving fires outrunning the band.

**Narrow band data structures differ:**

- Fortran: doubly-linked list of NODE structs with 50+ fields per node
- Julia: `BitMatrix` + `Vector{CartesianIndex}` with swap-and-compact removal

## 4. Wind Adjustment Factor

**Verdict: Matching** (resolved)

Both implementations now have:

- **Open/unsheltered:** logarithmic profile based on fuel bed depth: `WAF = 1.83 / ln((20 + 0.36·H) / (0.13·H))`
- **Canopy-sheltered:** `WAF = (1/ln((20+0.36H)/(0.13H))) × 0.555/√(f·H)` where `f = CC·crown_ratio/3`

Julia's `wind_adjustment_factor(fuel_bed_depth, canopy_cover, canopy_height)` is used in `simulate_full!` when canopy data is available, falling back to the unsheltered formula otherwise.

Note: Fortran uses a pre-computed 2D lookup table for performance; Julia computes on-the-fly.

## 5. Crown Fire Model

**Verdict: Faithful reproduction**

| Component | Formula | Match? |
|-----------|---------|--------|
| Critical FLI (Van Wagner) | `I_crit = (0.01·CBH·(460+26·FMC))^1.5` | Identical |
| Crown spread (Cruz 2005) | `CROSA = 11.02·WS₁₀^0.9·CBD^0.19·exp(-17·M1)` | Identical |
| Wind conversion | `WS₁₀(km/h) = WS₂₀ft(mph) × 1.609/0.87` | Identical |
| Critical spread rate | `R₀ = 3/CBD` (m/min → ft/min) | Identical |
| Crown activity coefficient | `CAC = CROSA/R₀` | Identical |
| Active (CAC>1) | `CROS = CROSA` | Identical |
| Passive (CAC≤1) | `CROS = CROSA·exp(-CAC)` | Identical |
| Canopy HPUA | `CBD × depth × 12000 kJ/m²` | Identical |
| Critical canopy cover | Required for active crown fire | Same logic |

## 6. Spotting / Ember Transport

**Verdict: Same physics models, wind-aware transport** (improved)

**Lognormal distribution** — identical parameterization.

**Sardoy (2008) model** — all constants match:

- Physical: `ρ=1.1`, `C_p=1.0`, `T=300K`, `g=9.81`
- `L_c = (I·1000/(ρ·C_p·T·√g))^0.67`
- Froude transition at `Fr = 1`
- Low-Fr: `μ=1.47·(I^0.54/u^0.55)+1.14`, `σ=0.86·(u^0.44/I^0.21)+0.19`
- High-Fr: `μ=1.32·I^0.26·u^0.11-0.02`, `σ=4.95/(I^0.01·u^0.02)-3.48`
- Spanwise: `σ_span = 0.92·L_c`

**Transport comparison:**

| Aspect | Fortran | Julia |
|--------|---------|-------|
| Transport | Full trajectory advection through wind field | Step-by-step wind-field advection (when WeatherInterpolator provided) |
| Himoto model | Structure fire spotting for IFBFM=91 | Not implemented |
| Eulerian CDF | CDF-based continuous distribution per grid cell | Not implemented |
| Landing | DEM-aware, checks terrain intersection | Grid bounds + unburned check only |
| Wind perturbation | Through trajectory integration | ±8° random offset at launch |

Julia's `transport_ember` now supports wind-field advection: when a `WeatherInterpolator` is provided (automatic in `simulate_full!`), embers are stepped through the spatially varying wind field at ~1-cell resolution. Without an interpolator, falls back to direct placement using source-cell wind.

**Remaining differences:** Julia does not implement DEM-aware landing (terrain intersection), the Himoto urban fire model, or the Eulerian CDF approach.

## 7. Weather Interpolation

**Verdict: Matching** (resolved)

| Aspect | Fortran | Julia |
|--------|---------|-------|
| Temporal | Linear interpolation between time steps | Identical |
| Spatial | Bilinear interpolation | Bilinear interpolation |
| Wind direction | Decompose to (u,v), interpolate, recombine | Decompose to (sin,cos), interpolate, recombine — identical approach |
| Update intervals | Independent per variable | All variables updated together |

Julia's `get_weather_at` now uses `bilinear_interp` for all scalar weather fields and `bilinear_interp_wind_direction` (sin/cos decomposition) for wind direction, producing smooth spatial gradients matching the Fortran approach.

## 8. Fire Acceleration & Diurnal Adjustment

**Verdict: Matching** (resolved)

| Aspect | Fortran | Julia |
|--------|---------|-------|
| Acceleration factor | `1 - exp(-t/τ)` buildup to steady-state | Identical (`acceleration_factor`) |
| Diurnal adjustment | Multiplicative factor on spread rate by time of day | Implemented (`DiurnalConfig`, `diurnal_adjustment`) |

Both features are applied as multiplicative factors on the spread rate after Rothermel + dampening calculation.

## 9. Features in Fortran Not in Julia

| Feature | Description |
|---------|-------------|
| Himoto structure fire spotting | Urban firebrand model |
| DEM-aware ember landing | Terrain intersection for spotting |
| Independent update intervals | Per-variable weather refresh rates |

## 10. Features in Julia Not in Fortran

| Feature | Description |
|---------|-------------|
| Multi-precision (Float32/Float64) | Generic type parameter throughout |
| GPU acceleration | KernelAbstractions.jl extension |
| Spread rate dampening modes | WIND_SPEED_CAP, ABSOLUTE_CAP, LINEAR_DAMPENING |
| Monte Carlo ensemble framework | Parallel ensemble simulations with perturbation configs |
| WUI building ignition models | Radiative heat flux, Hamada urban spread |
| Suppression/containment models | Resource allocation, containment lines |

## 11. Summary of Remaining Differences

1. **Band thickness** — Julia uses 5 cells vs. Fortran's 2. More robust but less efficient. This is a tuning parameter, not a physics difference.

2. **Ember landing** — Julia does not check terrain intersection for firebrand landing. Minor for flat terrain, potentially significant in mountainous areas.

3. **Himoto urban fire model** — Fortran supports structure fire spotting for urban fuel models; Julia does not implement this.
