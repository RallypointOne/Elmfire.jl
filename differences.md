# Elmfire.jl vs. ELMFIRE: Physics & Numerics Comparison

Reference implementation: the ELMFIRE Fortran source vendored at `elmfire/`
(`elmfire/build/source/*.f90`) and its technical reference
(`elmfire/docs/tech_ref.rst`). There is no ELMFIRE Python package — the Python in
that repository is cloudfire orchestration and input generation, not physics.

Values called out below are pinned by `test/test_elmfire_reference.jl`.

## 1. Rothermel Surface Fire Spread Model

**Verdict: faithful**

Verified term by term against `elmfire_init.f90:786-880` (coefficient
pre-computation) and `elmfire_spread_rate.f90:13-132` (per-cell evaluation).

| Component | Formula | Match? |
|-----------|---------|--------|
| Reaction intensity | `IR = Γ'·WN·η_s·HOC·η_m(M/Mex)` | Identical |
| Moisture damping | `η = 1 - 2.59r + 5.11r² - 3.52r³` | Identical |
| Propagating flux ratio | `ξ = exp((0.792+0.681√σ)(0.1+β)) / (192+0.2595σ)` | Identical |
| Reaction velocity | `Γ' = Γ'_peak·(β/β_op)^A·exp(A(1-β/β_op))` | Identical |
| Wind factor | `φ_w = C·(β/β_op)^(-E)·w^B` | Identical |
| Slope factor | `φ_s = 5.275·β^(-0.3)·tan²θ` | Identical |
| Head-fire spread rate | `R = R₀·(1 + φ_s + φ_w)` | Identical |
| Wind limit | `w_limited = min(w, 0.9·IR)` | Identical |
| Max slope factor | `φ_s ≤ φ_w(0.9·IR)` | Identical |
| Residence time | `τ = 384/σ` | Identical |
| Heat of pre-ignition | `Q_ig = 250 + 1116·M` | Identical |
| Optimal packing | `β_op = 3.348/σ^0.8189` | Identical |
| Dynamic herbaceous transfer | linear 30–120%, 1-hr SAV re-weighted | Identical |
| Live moisture of extinction | `Mex_live·(1 - Σ/(Σ_denom·Mex_dead)) - 0.226` | Identical |

Constants match: `ρ_p=32`, `S_T=0.055`, `S_E=0.01`, `η_s=0.174/S_E^0.19`,
`σ_10hr=109`, `σ_100hr=30`. Fuel model data for FBFM 1–13 and NB matches
`fuel_models.csv` exactly; `load_fuel_models` reads that file directly.

Both implementations build a 2D fuel model table indexed by
`[fuel_id, live_moisture_class]` over classes 30–120.

## 2. Direction of Maximum Spread

**Verdict: matching**

Wind and slope factors combine as vectors, not scalars
(`elmfire_level_set.f90:1760-1800`, tech ref §"Elliptical dimensions"):

```
φ_x = α[φ_w·sin(θ_w - π) + φ_s·sin(θ_a - π)]
φ_y = α[φ_w·cos(θ_w - π) + φ_s·cos(θ_a - π)]
|V_DMS| = V_s0·(α + |φ|)
```

`spread_direction` returns `|φ|` and the unit heading; `simulate!`,
`simulate_full!`, `simulate_with_suppression!` and the GPU kernels all use it.
Slope therefore steers the fire, and opposing wind and slope partially cancel.

## 3. Elliptical Fire Spread

**Verdict: matching**

**Effective mid-flame wind speed** — recovered by inverting Rothermel Eq. 47 from
the *combined* factor, so it carries slope as well as wind
(`elmfire_level_set.f90:1808`):

```
U_mf,e = (|φ| / (C(β/β_op)^-E))^(1/B)          [ft/min]
```

capped at `0.9·I_R` unless the cell is crowning. `effective_wind_speed`
implements the inversion; `wsmfeff_coeff` and `B_inverse` are pre-computed per
fuel model and read from the ILH=30 table entry, as ELMFIRE does.

**Length-to-breadth** (Anderson 1982 as modified by Finney):

```
L/B = min(0.936·exp(0.2566·U) + 0.461·exp(-0.1548·U) - 0.397, MAX_LOW)
U   = U_mf,e × 5.07955e-3          (ELMFIRE's WSMFEFF_LOW_MULT)
```

`elliptical_spread` takes `U_mf,e` in **ft/min** and applies the scaling
internally. `lb_cap` defaults to 8, matching `MAX_LOW`.

**Backing rate** (`elmfire_level_set.f90:1813-1817`):

```
V_back/V_head = (L/B - √((L/B)² - 1)) / (L/B + √((L/B)² - 1))
```

equivalently `(1-e)/(1+e)` with `e = √((L/B)² - 1)/(L/B)`.

**Velocity at arbitrary angle** — Richards (1990), relative to the direction of
maximum spread:

```
A = 0.5·(V_DMS + V_back);  B = A / (L/B);  ω = θ_n - θ_DMS
DYDT = A²·cos(ω)/√(A²cos²ω + B²sin²ω) + 0.5·(V_DMS - V_back)
DXDT = B²·sin(ω)/√(A²cos²ω + B²sin²ω)
```

then rotated into map coordinates.

**Slope projection** — velocities above are parallel to the local slope and are
projected onto the horizontal grid (`slope_projection_factors`):

```
U_x/U_x,∥ = 1 - |sin θ_a|(1 - cos γ)
U_y/U_y,∥ = 1 - |cos θ_a|(1 - cos γ)
```

## 4. Fireline Intensity and Flame Length

**Verdict: matching**

`I = I_R·τ·√(U_x,∥² + U_y,∥²)·0.3048` uses the **local** perimeter spread rate
before slope projection, so intensity is highest at the head and lowest at the
back (tech ref §"Variation in fireline intensity along the fire perimeter").
Byram flame length `L_f = (0.0775/0.3048)·I^0.46` matches
`elmfire_level_set.f90:745`.

## 5. Level Set Numerics

**Verdict: matching**

| Aspect | Fortran | Julia | Match? |
|--------|---------|-------|--------|
| PDE | `∂φ/∂t + (ux·∂φ/∂x + uy·∂φ/∂y) = 0` | Same | Identical |
| Time integration | RK2 (Heun's method) | RK2 (Heun's method) | Identical |
| Flux limiter | Half-superbee: `max(0, max(min(r/2,1), min(r,1/2)))` | Same | Identical |
| Gradient stencil | 4-point upwind per direction | Same | Identical |
| φ clamp (stage 1) | `[-100, 100]` | `[-100, 100]` | Identical |
| Normal vectors | central differences | Same | Identical |
| Reinitialization | None | None | Same |
| Target CFL | 0.4 | 0.4 | Identical |
| dt_max | 600 s | 10 min | Identical |
| **Band thickness** | **2 cells** | **5 cells** | **Different** |

**Band thickness** is a tuning parameter, not physics. Julia tracks more cells
around the front: less efficient, but more robust against a fast front outrunning
the band.

The Fortran `LIMIT_GRADIENTS` declares `PHIEAST=1.0` (and friends) in a
declaration statement, which makes them implicitly `SAVE`d, so a degenerate cell
(`|Δ_loc| ≤ 1e-30`) reuses whatever the previous call left behind. Julia falls
back to first-order upwind there instead. This is a deliberate deviation from a
latent Fortran bug.

Narrow band data structures differ: Fortran uses a doubly-linked list of `NODE`
structs, Julia a `BitMatrix` plus a packed `Vector{CartesianIndex}`.

## 6. Wind Adjustment Factor

**Verdict: matching** (`elmfire_init.f90:614-651`)

- **Unsheltered:** `WAF = 1.36·(ln(1.36/0.13) - 1) / ln((20 + 0.36·H)/(0.13·H))`
  with the flame-height-to-fuel-bed ratio taken as 1, as in BEHAVE and FARSITE.
  Returns 0 for a vanishing fuel bed. Unclamped, as in ELMFIRE.
- **Canopy-sheltered:** `WAF = (1/ln((20+0.36H)/(0.13H))) × 0.555/√(f·H)` where
  `f = 0.3333·CC·crown_ratio`. `crown_ratio` is a keyword argument defaulting to
  1.0, matching ELMFIRE's `CROWN_RATIO` namelist default, and is exposed through
  `SimulationConfig`.

Fortran uses a pre-computed 2D lookup table for performance; Julia computes on
the fly and caches per fuel id.

## 7. Crown Fire Model

**Verdict: matching**

| Component | Formula | Match? |
|-----------|---------|--------|
| Critical FLI (Van Wagner) | `I_crit = (0.01·CBH·(460+26·FMC))^1.5` | Identical |
| Crown spread (Cruz 2005) | `CROSA = 11.02·WS₁₀^0.9·CBD^0.19·exp(-0.17·100·M1)` | Identical |
| Wind conversion | `WS₁₀(km/h) = WS₂₀ft(mph) × 1.609/0.87` | Identical |
| Critical spread rate | `R₀ = 3/CBD` (m/min → ft/min) | Identical |
| Crown activity coefficient | `CAC = CROSA/R₀` | Identical |
| Active (CAC>1) | `CROS = CROSA` | Identical |
| Passive (CAC≤1) | `CROS = CROSA·exp(-CAC)` | Identical |
| Canopy HPUA | `CBD × (CH - CBH) × 12000 kJ/m²` | Identical |
| Canopy FLI | `HPUA_canopy × V_local × 5.08e-3` | Identical |
| Spread rate limit | 250 ft/min (`CROWN_FIRE_SPREAD_RATE_LIMIT`) | Identical |
| Critical canopy cover | 0.39 (`CRITICAL_CANOPY_COVER`) | Identical |

Crown fire enters the level set through the wind factor,
`φ_w ← max(φ_w,surface, φ_w,crown)` without the acceleration factor, so it changes
the spread direction and the ellipse shape as well as the magnitude, and it lifts
the `0.9·I_R` cap on the effective wind speed. As in ELMFIRE, the ellipse is
evaluated at most twice per cell per step: once assuming surface fire, and again
if the resulting local intensity crosses the crowning threshold.

## 8. Fire Acceleration & Diurnal Adjustment

**Verdict: matching**

| Aspect | Fortran | Julia |
|--------|---------|-------|
| Acceleration factor | `1 - exp(-t/τ)`, disabled for `τ ≤ 30 s` | Identical (τ in minutes, disabled for `τ ≤ 0.5`) |
| Application | scales both `φ_s` and `φ_w`, and the `α` term in `V_s0(α + \|φ\|)` | Identical |
| Diurnal adjustment | multiplies `V_s0` | Identical |

## 9. Spotting / Ember Transport

**Verdict: same physics models; transport differs**

**Lognormal distribution** — identical parameterization.

**Sardoy (2008)** — all constants match `elmfire_spotting.f90:160-207`:

- Physical: `ρ=1.1`, `C_p=1.0`, `T=300K`, `g=9.81`
- `L_c = (I·1000/(ρ·C_p·T·√g))^0.67`, Froude transition at `Fr = 1`
- Low-Fr: `μ=1.47·(I^0.54/u^0.55)+1.14`, `σ=0.86·(u^0.44/I^0.21)+0.19`
- High-Fr: `μ=1.32·I^0.26·u^0.11-0.02`, `σ=4.95/(I^0.01·u^0.02)-3.48`
- Spanwise: `σ_span = 0.92·L_c`; `μ` clipped at 5

| Aspect | Fortran | Julia |
|--------|---------|-------|
| Transport | Full trajectory advection through wind field | Step-by-step wind-field advection (when a `WeatherInterpolator` is provided) |
| Himoto model | Structure fire spotting for IFBFM=91 | Not implemented |
| Eulerian CDF | CDF-based continuous distribution per grid cell | Not implemented |
| Landing | DEM-aware, checks terrain intersection | Grid bounds + unburned check only |
| Wind perturbation | Through trajectory integration | ±8° random offset at launch |

## 10. Weather Interpolation

**Verdict: matching**

| Aspect | Fortran | Julia |
|--------|---------|-------|
| Temporal | Linear between time steps | Identical |
| Spatial | Bilinear | Bilinear |
| Wind direction | Decompose to components, interpolate, recombine | Identical approach (sin/cos) |
| Update intervals | Independent per variable | All variables updated together |

## 11. Non-Burnable Fuels

**Verdict: matching** (`elmfire_init.f90:156-166`)

Fuel codes 90–100, codes ≤ 0, and 256 are non-burnable, as is any model with no
1-hour fuel load. `get_fuel_model_or_nonburnable` resolves absent non-burnable
codes to model 256, matching ELMFIRE's startup fill, so raw LANDFIRE rasters
carrying 91–99 load without remapping.

## 12. Features in ELMFIRE Not in Elmfire.jl

| Feature | Description |
|---------|-------------|
| Himoto structure fire spotting | Urban firebrand model |
| Eulerian CDF spotting | Continuous per-cell ember deposition |
| DEM-aware ember landing | Terrain intersection for firebrands |
| Hamada / UMD-UCB building spread | In-model WUI spread for IFBFM=91 |
| Independent update intervals | Per-variable weather refresh rates |

## 13. Features in Elmfire.jl Not in ELMFIRE

| Feature | Description |
|---------|-------------|
| Multi-precision (Float32/Float64) | Generic type parameter throughout |
| GPU acceleration | KernelAbstractions.jl extension |
| Spread rate dampening modes | `WIND_SPEED_CAP`, `ABSOLUTE_CAP`, `LINEAR_DAMPENING` |
| Monte Carlo ensemble framework | Parallel ensembles with perturbation configs |
| WUI building ignition models | Radiative heat flux, Hamada urban spread |
| Suppression/containment models | Resource allocation, containment lines |

## 14. Remaining Differences

1. **Band thickness** — 5 cells vs. 2. Tuning, not physics.
2. **Ember landing** — no terrain intersection check. Minor on flat terrain,
   potentially significant in mountainous areas.
3. **Urban fire models** — Himoto spotting and the in-model building spread paths
   are not implemented; Elmfire.jl carries its own WUI models instead.
4. **Weather refresh** — all variables are interpolated on one schedule rather
   than per-variable intervals.
5. **Degenerate flux limiter cells** — first-order upwind rather than reusing a
   `SAVE`d value.
