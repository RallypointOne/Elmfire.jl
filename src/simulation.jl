#-----------------------------------------------------------------------------#
#                     Fire Simulation State and Loop
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Wind and Terrain Helpers
#-----------------------------------------------------------------------------#

"""
    wind_adjustment_factor(fuel_bed_depth::T) -> T

Calculate unsheltered wind adjustment factor (WAF) from 20-ft wind to mid-flame wind.

Uses: WAF = 1.83 / ln((20 + 0.36*H) / (0.13*H)) where H is the fuel bed depth in feet.
"""
function wind_adjustment_factor(fuel_bed_depth::T) where {T<:AbstractFloat}
    if fuel_bed_depth < T(0.1)
        return T(0.1)
    end

    H = fuel_bed_depth
    waf = T(1.83) / log((T(20) + T(0.36)*H) / (T(0.13)*H))
    return clamp(waf, T(0.1), one(T))
end


"""
    wind_adjustment_factor(fuel_bed_depth::T, canopy_cover::T, canopy_height::T) -> T

Calculate sheltered wind adjustment factor (WAF) under forest canopy.

Uses the Albini & Baughman (1979) sheltered formula when canopy cover and height
are non-trivial, otherwise falls back to the unsheltered formula.

# Arguments
- `fuel_bed_depth`: Fuel bed depth (ft)
- `canopy_cover`: Canopy cover fraction (0-1)
- `canopy_height`: Canopy height (m)
"""
function wind_adjustment_factor(fuel_bed_depth::T, canopy_cover::T, canopy_height::T) where {T<:AbstractFloat}
    # Fall back to unsheltered if no meaningful canopy
    if canopy_cover < T(1e-4) || canopy_height < T(1e-4)
        return wind_adjustment_factor(fuel_bed_depth)
    end

    # Convert canopy height to feet
    hft = canopy_height / ft_to_m(T)

    # Ratio of mid-canopy wind to 20-ft wind (log profile)
    numer = T(20) + T(0.36) * hft
    denom = T(0.13) * hft
    if denom < T(0.01)
        return wind_adjustment_factor(fuel_bed_depth)
    end
    uhou20ph = one(T) / log(numer / denom)

    # Crown ratio assumed 1/3 (fraction of canopy occupied by crown)
    crown_ratio = one(T) / T(3)
    f = T(0.3333) * canopy_cover * crown_ratio
    ucouh = T(0.555) / sqrt(f * hft)

    waf = uhou20ph * ucouh
    return clamp(waf, T(0.1), one(T))
end


"""
    acceleration_factor(t::T, tau::T) -> T

Compute the fire acceleration factor at time `t` with time constant `tau` (minutes).

The acceleration factor ramps from 0 at ignition to 1 at steady state following
`1 - exp(-t/τ)`. This models the observed behavior that fires initially spread
slower than the Rothermel steady-state prediction.

When `tau ≤ 0.5` (minutes), acceleration is disabled and returns 1.0.
"""
@inline function acceleration_factor(t::T, tau::T) where {T<:AbstractFloat}
    if tau <= T(0.5)
        return one(T)
    end
    ratio = t / tau
    if ratio > T(7)
        return one(T)
    end
    return one(T) - exp(-ratio)
end


"""
    DiurnalConfig{T<:AbstractFloat}

Configuration for diurnal (day/night) spread rate adjustment.

During the burn period (daytime), the adjustment factor is 1.0 (full spread).
Outside the burn period (nighttime), spread rate is multiplied by `overnight_factor`.

# Fields
- `forecast_start_hour`: Simulation start hour in local time (0-24)
- `sunrise_hour`: Local sunrise hour (default 6.0)
- `sunset_hour`: Local sunset hour (default 20.0)
- `burn_period_center_frac`: Fraction of daylight for burn period center (default 0.667)
- `burn_period_length`: Duration of active burn period in hours (default 10.0)
- `overnight_factor`: Spread rate multiplier outside burn period (default 0.1)
"""
struct DiurnalConfig{T<:AbstractFloat}
    forecast_start_hour::T
    sunrise_hour::T
    sunset_hour::T
    burn_period_center_frac::T
    burn_period_length::T
    overnight_factor::T
end

Base.eltype(::DiurnalConfig{T}) where {T} = T

function DiurnalConfig{T}(;
    forecast_start_hour::T = T(12),
    sunrise_hour::T = T(6),
    sunset_hour::T = T(20),
    burn_period_center_frac::T = T(0.667),
    burn_period_length::T = T(10),
    overnight_factor::T = T(0.1)
) where {T<:AbstractFloat}
    DiurnalConfig{T}(forecast_start_hour, sunrise_hour, sunset_hour,
        burn_period_center_frac, burn_period_length, overnight_factor)
end

DiurnalConfig(; kwargs...) = DiurnalConfig{Float64}(; kwargs...)

"""
    diurnal_adjustment(config::DiurnalConfig{T}, t::T) -> T

Compute diurnal spread rate adjustment factor at simulation time `t` (minutes).

Returns 1.0 during the burn period (daytime) and `config.overnight_factor` outside it.
"""
@inline function diurnal_adjustment(config::DiurnalConfig{T}, t::T) where {T<:AbstractFloat}
    # Convert simulation time (minutes) to hour of day
    hour_of_day = config.forecast_start_hour + t / T(60)
    hour_of_day = mod(hour_of_day, T(24))

    # Compute burn period window
    daylight = config.sunset_hour - config.sunrise_hour
    center = config.sunrise_hour + config.burn_period_center_frac * daylight
    half_len = config.burn_period_length / T(2)
    start_hour = center - half_len
    stop_hour = center + half_len

    # Check if current hour falls within burn period
    if start_hour >= zero(T) && stop_hour <= T(24)
        # No wrap-around
        if hour_of_day >= start_hour && hour_of_day <= stop_hour
            return one(T)
        end
    else
        # Handle wrap-around (burn period crosses midnight)
        start_mod = mod(start_hour, T(24))
        stop_mod = mod(stop_hour, T(24))
        if hour_of_day >= start_mod || hour_of_day <= stop_mod
            return one(T)
        end
    end

    return config.overnight_factor
end


#-----------------------------------------------------------------------------#
#                     Terrain
#-----------------------------------------------------------------------------#

"""
    calculate_tanslp2(slope_degrees::T) -> T

Calculate tan²(slope) from slope in degrees.
"""
function calculate_tanslp2(slope_degrees::T) where {T<:AbstractFloat}
    slope_rad = slope_degrees * pio180(T)
    return tan(slope_rad)^2
end


#-----------------------------------------------------------------------------#
#                     Fire State
#-----------------------------------------------------------------------------#

"""
    FireState{T<:AbstractFloat, A<:AbstractMatrix{T}}

Complete state of a fire simulation. Parameterized on element type `T` and array type `A`
to support both CPU (`Matrix{T}`) and GPU arrays.
"""
mutable struct FireState{T<:AbstractFloat, A<:AbstractMatrix{T}}
    # Level set field (with 2-cell padding for stencil operations)
    phi::A
    phi_old::A

    # Output fields (no padding, matches grid dimensions)
    time_of_arrival::A
    burned::BitMatrix
    spread_rate::A               # Final spread rate (ft/min)
    fireline_intensity::A        # Fireline intensity (kW/m)
    flame_length::A              # Flame length (ft)

    # Velocity components (with padding)
    ux::A
    uy::A

    # Narrow band tracking
    narrow_band::NarrowBand

    # Grid parameters
    ncols::Int           # Number of columns (without padding)
    nrows::Int           # Number of rows (without padding)
    cellsize::T          # Cell size (ft)
    xllcorner::T         # X coordinate of lower-left corner
    yllcorner::T         # Y coordinate of lower-left corner

    # Padding for stencil operations
    padding::Int
end

"""Type alias for CPU-backed `FireState` using standard `Matrix{T}` arrays."""
const CPUFireState{T} = FireState{T, Matrix{T}}

Base.eltype(::FireState{T}) where {T} = T

# Outer constructor that infers A from array arguments
function FireState{T}(
    phi::A, phi_old::A,
    time_of_arrival::A, burned::BitMatrix,
    spread_rate::A, fireline_intensity::A, flame_length::A,
    ux::A, uy::A,
    narrow_band::NarrowBand,
    ncols::Int, nrows::Int, cellsize::T, xllcorner::T, yllcorner::T,
    padding::Int
) where {T<:AbstractFloat, A<:AbstractMatrix{T}}
    return FireState{T, A}(
        phi, phi_old,
        time_of_arrival, burned, spread_rate, fireline_intensity, flame_length,
        ux, uy,
        narrow_band,
        ncols, nrows, cellsize, xllcorner, yllcorner, padding
    )
end


"""
    FireState{T}(ncols, nrows, cellsize; xllcorner=zero(T), yllcorner=zero(T), padding=2, band_thickness=5)

Create a new fire simulation state with specified precision.

# Arguments
- `ncols`: Number of grid columns
- `nrows`: Number of grid rows
- `cellsize`: Cell size in feet
- `xllcorner`: X coordinate of lower-left corner (default 0.0)
- `yllcorner`: Y coordinate of lower-left corner (default 0.0)
- `padding`: Boundary padding for stencil operations (default 2)
- `band_thickness`: Narrow band half-width (default 5)
"""
function FireState{T}(
    ncols::Int, nrows::Int, cellsize::T;
    xllcorner::T = zero(T),
    yllcorner::T = zero(T),
    padding::Int = 2,
    band_thickness::Int = 5
) where {T<:AbstractFloat}
    # Padded dimensions
    nx_pad = ncols + 2*padding
    ny_pad = nrows + 2*padding

    # Initialize level set with large positive value (unburned)
    phi = fill(T(100), nx_pad, ny_pad)
    phi_old = fill(T(100), nx_pad, ny_pad)

    # Initialize output fields (no padding)
    time_of_arrival = fill(-one(T), ncols, nrows)
    burned = falses(ncols, nrows)
    spread_rate = zeros(T, ncols, nrows)
    fireline_intensity = zeros(T, ncols, nrows)
    flame_length = zeros(T, ncols, nrows)

    # Initialize velocity components (with padding)
    ux = zeros(T, nx_pad, ny_pad)
    uy = zeros(T, nx_pad, ny_pad)

    # Initialize narrow band
    narrow_band = NarrowBand(nx_pad, ny_pad, band_thickness)

    return FireState{T}(
        phi, phi_old,
        time_of_arrival, burned, spread_rate, fireline_intensity, flame_length,
        ux, uy,
        narrow_band,
        ncols, nrows, cellsize, xllcorner, yllcorner, padding
    )
end

# Default to Float64 for backwards compatibility
function FireState(
    ncols::Int, nrows::Int, cellsize::Float64;
    xllcorner::Float64 = 0.0,
    yllcorner::Float64 = 0.0,
    padding::Int = 2,
    band_thickness::Int = 5
)
    FireState{Float64}(ncols, nrows, cellsize;
        xllcorner=xllcorner, yllcorner=yllcorner,
        padding=padding, band_thickness=band_thickness)
end


"""
    Base.copy(state::FireState{T}) -> FireState{T}

Create a deep copy of a FireState for thread-safe parallel execution.
"""
function Base.copy(state::FireState{T}) where {T<:AbstractFloat}
    FireState{T}(
        copy(state.phi),
        copy(state.phi_old),
        copy(state.time_of_arrival),
        copy(state.burned),
        copy(state.spread_rate),
        copy(state.fireline_intensity),
        copy(state.flame_length),
        copy(state.ux),
        copy(state.uy),
        NarrowBand(state.ncols + 2*state.padding, state.nrows + 2*state.padding, state.narrow_band.band_thickness),
        state.ncols,
        state.nrows,
        state.cellsize,
        state.xllcorner,
        state.yllcorner,
        state.padding
    )
end


"""
    reset!(state::FireState{T})

Reset a FireState to initial conditions for reuse in ensemble simulations.
"""
function reset!(state::FireState{T}) where {T<:AbstractFloat}
    # Reset level set fields
    fill!(state.phi, T(100))
    fill!(state.phi_old, T(100))

    # Reset output fields
    fill!(state.time_of_arrival, -one(T))
    fill!(state.burned, false)
    fill!(state.spread_rate, zero(T))
    fill!(state.fireline_intensity, zero(T))
    fill!(state.flame_length, zero(T))

    # Reset velocity fields
    fill!(state.ux, zero(T))
    fill!(state.uy, zero(T))

    # Reset narrow band
    fill!(state.narrow_band.is_active, false)
    state.narrow_band.n_active = 0
    fill!(state.narrow_band.ever_tagged, false)

    return nothing
end


"""
    grid_to_padded(state::FireState, ix::Int, iy::Int) -> Tuple{Int, Int}

Convert grid coordinates to padded array coordinates.
"""
@inline function grid_to_padded(state::FireState, ix::Int, iy::Int)
    return (ix + state.padding, iy + state.padding)
end


"""
    padded_to_grid(state::FireState, px::Int, py::Int) -> Tuple{Int, Int}

Convert padded array coordinates to grid coordinates.
"""
@inline function padded_to_grid(state::FireState, px::Int, py::Int)
    return (px - state.padding, py - state.padding)
end


#-----------------------------------------------------------------------------#
#                     Ignition
#-----------------------------------------------------------------------------#

"""
    ignite!(state::FireState{T}, ix::Int, iy::Int, t::T)

Ignite a cell at grid coordinates (ix, iy) at time t.
Sets up the level set as a signed distance function near the ignition point.
"""
function ignite!(state::FireState{T}, ix::Int, iy::Int, t::T) where {T<:AbstractFloat}
    px, py = grid_to_padded(state, ix, iy)

    # Set level set to negative (inside fire) - distance to boundary is ~half cell
    state.phi[px, py] = -T(0.5) * state.cellsize
    state.phi_old[px, py] = -T(0.5) * state.cellsize

    # Mark as burned
    state.burned[ix, iy] = true
    state.time_of_arrival[ix, iy] = t

    # Initialize signed distance for nearby cells (approximate)
    # This ensures proper gradients at the fire front
    for di in -3:3, dj in -3:3
        if di == 0 && dj == 0
            continue
        end
        npx, npy = px + di, py + dj
        if 1 <= npx <= size(state.phi, 1) && 1 <= npy <= size(state.phi, 2)
            # Distance from center (in grid units) * cellsize - half cell
            dist = sqrt(T(di^2 + dj^2)) * state.cellsize - T(0.5) * state.cellsize
            # Only update if this gives a smaller (closer to fire) value
            if dist < state.phi[npx, npy]
                state.phi[npx, npy] = dist
            end
        end
    end

    # Add to narrow band
    nx_pad = state.ncols + 2*state.padding
    ny_pad = state.nrows + 2*state.padding
    tag_band!(state.narrow_band, CartesianIndex(px, py), nx_pad, ny_pad, state.padding)

    return nothing
end


"""
    ignite_point!(state::FireState{T}, x::T, y::T, t::T)

Ignite a cell at world coordinates (x, y) at time t.
"""
function ignite_point!(state::FireState{T}, x::T, y::T, t::T) where {T<:AbstractFloat}
    # Convert world coordinates to grid indices
    ix = floor(Int, (x - state.xllcorner) / state.cellsize) + 1
    iy = floor(Int, (y - state.yllcorner) / state.cellsize) + 1

    if 1 <= ix <= state.ncols && 1 <= iy <= state.nrows
        ignite!(state, ix, iy, t)
    end

    return nothing
end


"""
    ignite_circle!(state::FireState{T}, center_x::Int, center_y::Int, radius_cells::T, t::T)

Ignite all cells within a circle of given radius (in grid cells) centered at (center_x, center_y).
"""
function ignite_circle!(state::FireState{T}, center_x::Int, center_y::Int, radius_cells::T, t::T) where {T<:AbstractFloat}
    r2 = radius_cells^2

    for ix in 1:state.ncols
        for iy in 1:state.nrows
            if T(ix - center_x)^2 + T(iy - center_y)^2 <= r2
                ignite!(state, ix, iy, t)
            end
        end
    end

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Spread Rate Dampening
#-----------------------------------------------------------------------------#

"""
    SpreadRateDampeningMode

Configurable strategy for limiting fire spread rates at extreme wind speeds.

The Rothermel model's power-law wind factor (`phi_w = C * wsmf^B`) overpredicts
spread rates at high wind speeds. These modes provide physically motivated dampening.

### Values
- `NO_DAMPENING`: Status quo — only Rothermel's 0.9*IR wind limit applies
- `WIND_SPEED_CAP`: Andrews et al. (2013) — cap ROS at midflame wind speed (ft/min)
- `ABSOLUTE_CAP`: Hard cap at a user-specified maximum spread rate (ft/min)
- `LINEAR_DAMPENING`: Transition phi_w from power-law to linear above a threshold
"""
@enum SpreadRateDampeningMode begin
    NO_DAMPENING
    WIND_SPEED_CAP
    ABSOLUTE_CAP
    LINEAR_DAMPENING
end

"""
    SpreadRateDampeningConfig{T<:AbstractFloat}

Configuration for spread rate dampening.

### Fields
- `mode::SpreadRateDampeningMode`: Dampening strategy
- `max_spread_rate::T`: Maximum spread rate for `ABSOLUTE_CAP` mode (ft/min, default 250.0)
- `linear_threshold::T`: Wind speed threshold for `LINEAR_DAMPENING` mode (ft/min, default 400.0)

### Examples
```julia
# No dampening (default)
cfg = SpreadRateDampeningConfig{Float64}()

# Cap spread rate at midflame wind speed
cfg = SpreadRateDampeningConfig{Float64}(WIND_SPEED_CAP)

# Hard cap at 200 ft/min
cfg = SpreadRateDampeningConfig{Float64}(ABSOLUTE_CAP, max_spread_rate=200.0)

# Linear dampening above 300 ft/min midflame wind
cfg = SpreadRateDampeningConfig{Float64}(LINEAR_DAMPENING, linear_threshold=300.0)
```
"""
struct SpreadRateDampeningConfig{T<:AbstractFloat}
    mode::SpreadRateDampeningMode
    max_spread_rate::T
    linear_threshold::T
end

function SpreadRateDampeningConfig{T}(
    mode::SpreadRateDampeningMode = NO_DAMPENING;
    max_spread_rate::T = T(250),
    linear_threshold::T = T(400)
) where {T<:AbstractFloat}
    SpreadRateDampeningConfig{T}(mode, max_spread_rate, linear_threshold)
end

SpreadRateDampeningConfig(mode::SpreadRateDampeningMode = NO_DAMPENING; kwargs...) =
    SpreadRateDampeningConfig{Float64}(mode; kwargs...)

Base.eltype(::SpreadRateDampeningConfig{T}) where {T} = T

"""
    apply_spread_rate_dampening(
        velocity::T, wsmf::T, vs0::T, phis::T, phiw::T,
        phiwterm::T, B::T,
        config::SpreadRateDampeningConfig{T}
    ) -> T

Apply spread rate dampening to a computed fire spread velocity.

Called after `surface_spread_rate()` and before `elliptical_spread()`.

### Arguments
- `velocity`: Spread rate from Rothermel model (ft/min)
- `wsmf`: Midflame wind speed (ft/min)
- `vs0`: Base spread rate without wind/slope (ft/min)
- `phis`: Slope factor
- `phiw`: Wind factor (power-law)
- `phiwterm`: Fuel model wind coefficient (C * (beta/betaop)^(-E))
- `B`: Fuel model wind exponent (0.02526 * sigma^0.54)
- `config`: Dampening configuration
"""
@inline function apply_spread_rate_dampening(
    velocity::T, wsmf::T, vs0::T, phis::T, phiw::T,
    phiwterm::T, B::T,
    config::SpreadRateDampeningConfig{T}
) where {T<:AbstractFloat}
    mode = config.mode
    if mode == NO_DAMPENING
        return velocity
    elseif mode == WIND_SPEED_CAP
        return min(velocity, wsmf)
    elseif mode == ABSOLUTE_CAP
        return min(velocity, config.max_spread_rate)
    else  # LINEAR_DAMPENING
        threshold = config.linear_threshold
        if wsmf <= threshold
            return velocity
        end
        # phi_w at threshold (power-law)
        phiw_at_threshold = phiwterm * threshold^B
        # Derivative of phi_w at threshold: d(phiwterm * w^B)/dw = phiwterm * B * w^(B-1)
        dphiw_dw = phiwterm * B * threshold^(B - one(T))
        # Linear extrapolation beyond threshold
        phiw_linear = phiw_at_threshold + dphiw_dw * (wsmf - threshold)
        return vs0 * (one(T) + phis + phiw_linear)
    end
end


#-----------------------------------------------------------------------------#
#                     Main Simulation Loop
#-----------------------------------------------------------------------------#

"""
    simulate!(
        state::FireState{T},
        fuel_ids::AbstractMatrix{Int},
        fuel_table::FuelModelTable{T},
        weather::ConstantWeather{T},
        slope::AbstractMatrix{T},
        aspect::AbstractMatrix{T},
        t_start::T,
        t_stop::T;
        dt_initial::T = one(T),
        target_cfl::T = T(0.45),
        dt_max::T = T(10),
        spread_rate_adj::T = one(T),
        accel_time_constant::T = zero(T),
        diurnal::Union{Nothing, DiurnalConfig{T}} = nothing,
        callback::Union{Nothing, Function} = nothing
    )

Run the fire simulation from t_start to t_stop.

# Arguments
- `state`: Fire state (modified in place)
- `fuel_ids`: Matrix of fuel model IDs for each cell
- `fuel_table`: Fuel model lookup table
- `weather`: Weather conditions
- `slope`: Slope in degrees for each cell
- `aspect`: Aspect direction in degrees for each cell
- `t_start`: Start time (minutes)
- `t_stop`: Stop time (minutes)
- `dt_initial`: Initial timestep (minutes, default 1.0)
- `target_cfl`: Target CFL number (default 0.45)
- `dt_max`: Maximum timestep (minutes, default 10.0)
- `spread_rate_adj`: Spread rate adjustment factor (default 1.0, multiplies base spread rate)
- `accel_time_constant`: Fire acceleration time constant (minutes, default 0 = disabled).
  When > 0.5, spread rate is multiplied by `1 - exp(-t/τ)`, modeling fire buildup from
  ignition to steady-state.
- `diurnal`: Optional `DiurnalConfig` for day/night spread rate adjustment (default nothing = disabled)
- `callback`: Optional callback function(state, t, dt, iteration) called each timestep
"""
function simulate!(
    state::FireState{T},
    fuel_ids::AbstractMatrix{Int},
    fuel_table::FuelModelTable{T},
    weather::ConstantWeather{T},
    slope::AbstractMatrix{T},
    aspect::AbstractMatrix{T},
    t_start::T,
    t_stop::T;
    dt_initial::T = one(T),
    target_cfl::T = T(0.45),
    dt_max::T = T(10),
    spread_rate_adj::T = one(T),
    accel_time_constant::T = zero(T),
    diurnal::Union{Nothing, DiurnalConfig{T}} = nothing,
    dampening::SpreadRateDampeningConfig{T} = SpreadRateDampeningConfig{T}(),
    callback::CB = nothing
) where {T<:AbstractFloat, CB}
    t = t_start
    dt = dt_initial
    iteration = 0

    # Pre-compute wind direction in radians and trig
    wind_dir_rad = weather.wind_direction * pio180(T)
    wind_to_x = -sin(wind_dir_rad)
    wind_to_y = -cos(wind_dir_rad)

    # Wind speed conversion: 20-ft to ft/min
    # 1 mph = 88 ft/min
    ws20_ftpmin = weather.wind_speed_20ft * T(88)

    # Live moisture class (30-120)
    live_moisture_class = clamp(round(Int, T(100) * weather.MLH), 30, 120)

    # Pre-compute static slope grid
    tanslp2_grid = calculate_tanslp2.(slope)

    # Pre-build fast fuel lookup vector (O(1) by fuel_id)
    unique_fids = unique(fuel_ids)
    max_fid = maximum(unique_fids)
    fuel_vec = Vector{FuelModel{T}}(undef, max_fid)
    waf_cache = Vector{T}(undef, max_fid)
    for fid in unique_fids
        fm = get_fuel_model(fuel_table, fid, live_moisture_class)
        fuel_vec[fid] = fm
        waf_cache[fid] = wind_adjustment_factor(fm.delta)
    end

    # Pre-allocate buffers
    cells_to_tag = CartesianIndex{2}[]
    cached_velocity = zeros(T, state.ncols, state.nrows)
    cached_flin = zeros(T, state.ncols, state.nrows)
    nx_pad = state.ncols + 2*state.padding
    ny_pad = state.nrows + 2*state.padding

    while t < t_stop
        iteration += 1

        # Get active cells (zero-allocation view)
        active_cells = get_active_cells(state.narrow_band)

        if isempty(active_cells)
            break  # No more active fire
        end

        # Stage 1: Compute spread rates and velocities for active cells
        for idx in active_cells
            px, py = idx[1], idx[2]
            ix, iy = padded_to_grid(state, px, py)

            # Skip if out of bounds
            if ix < 1 || ix > state.ncols || iy < 1 || iy > state.nrows
                continue
            end

            # Skip already burned cells for velocity calculation
            if state.burned[ix, iy]
                state.ux[px, py] = zero(T)
                state.uy[px, py] = zero(T)
                continue
            end

            # Get fuel model (O(1) vector lookup)
            fuel_id = fuel_ids[ix, iy]
            fm = fuel_vec[fuel_id]

            # Skip non-burnable
            if isnonburnable(fm)
                state.ux[px, py] = zero(T)
                state.uy[px, py] = zero(T)
                continue
            end

            # Use cached wind adjustment factor
            waf = waf_cache[fuel_id]
            wsmf = ws20_ftpmin * waf  # Mid-flame wind speed (ft/min)

            # Use pre-computed slope factor
            tanslp2 = tanslp2_grid[ix, iy]

            # Calculate spread rate
            result = surface_spread_rate(
                fm,
                weather.M1, weather.M10, weather.M100,
                weather.MLH, weather.MLW,
                wsmf, tanslp2;
                adj = spread_rate_adj
            )

            # Apply spread rate dampening
            dampened_velocity = apply_spread_rate_dampening(
                result.velocity, wsmf, result.vs0, result.phis, result.phiw,
                fm.phiwterm, fm.B, dampening
            )

            # Apply acceleration and diurnal factors
            dampened_velocity *= acceleration_factor(t, accel_time_constant)
            if diurnal !== nothing
                dampened_velocity *= diurnal_adjustment(diurnal, t)
            end

            # Cache for burn recording
            cached_velocity[ix, iy] = dampened_velocity
            cached_flin[ix, iy] = result.flin

            # Compute normal vector to fire front
            normal_x, normal_y = compute_normal(state.phi, px, py, state.cellsize)

            # Calculate velocity components using elliptical spread
            # Midflame wind speed in mph = 20-ft wind speed × WAF (dimensionless)
            effective_ws_mph = weather.wind_speed_20ft * waf
            es = elliptical_spread(dampened_velocity, effective_ws_mph)

            # Richards (1990) ellipse velocity decomposition
            ux, uy = velocity_components(es, wind_dir_rad, normal_x, normal_y)
            state.ux[px, py] = ux
            state.uy[px, py] = uy
        end

        # Compute CFL timestep (only after a few iterations)
        if iteration > 5
            dt = compute_cfl_timestep(
                state.ux, state.uy,
                active_cells,
                state.cellsize,
                dt;
                target_cfl = target_cfl,
                dt_max = dt_max
            )
        end

        # Don't overshoot t_stop
        if t + dt > t_stop
            dt = t_stop - t
        end

        # Perform RK2 level set integration
        for idx in active_cells
            state.phi_old[idx] = state.phi[idx]
        end

        rk2_step!(state.phi, state.phi_old, state.ux, state.uy, active_cells, dt, state.cellsize, 1)
        rk2_step!(state.phi, state.phi_old, state.ux, state.uy, active_cells, dt, state.cellsize, 2)

        # Update burned cells and narrow band
        empty!(cells_to_tag)

        for idx in active_cells
            px, py = idx[1], idx[2]
            ix, iy = padded_to_grid(state, px, py)

            # Skip if out of bounds
            if ix < 1 || ix > state.ncols || iy < 1 || iy > state.nrows
                continue
            end

            # Check if cell just burned (phi crossed from positive to negative)
            if state.phi[px, py] <= zero(T) && !state.burned[ix, iy]
                state.burned[ix, iy] = true
                state.time_of_arrival[ix, iy] = t + dt

                # Use cached spread rate (computed earlier this iteration)
                state.spread_rate[ix, iy] = cached_velocity[ix, iy]
                state.fireline_intensity[ix, iy] = cached_flin[ix, iy]

                # Flame length (Byram): Lf = 0.0775 * I^0.46 (ft)
                flin_val = cached_flin[ix, iy]
                if flin_val > zero(T)
                    state.flame_length[ix, iy] = (T(0.0775) / ft_to_m(T)) * flin_val^T(0.46)
                end

                # Tag surrounding cells
                push!(cells_to_tag, idx)
            end
        end

        # Expand narrow band around newly burned cells
        for idx in cells_to_tag
            tag_band!(state.narrow_band, idx, nx_pad, ny_pad, state.padding)
        end

        # Remove isolated cells from narrow band
        untag_isolated!(state.narrow_band, state.phi, state.burned, state.padding)

        t += dt

        # Optional callback
        if callback !== nothing
            callback(state, t, dt, iteration)
        end
    end

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Uniform Conditions Simulation
#-----------------------------------------------------------------------------#

"""
    simulate_uniform!(
        state::FireState{T},
        fuel_id::Int,
        fuel_table::FuelModelTable{T},
        weather::ConstantWeather{T},
        slope_deg::T,
        aspect_deg::T,
        t_start::T,
        t_stop::T;
        kwargs...
    )

Run simulation with uniform fuel, slope, and aspect across the domain.
"""
function simulate_uniform!(
    state::FireState{T},
    fuel_id::Int,
    fuel_table::FuelModelTable{T},
    weather::ConstantWeather{T},
    slope_deg::T,
    aspect_deg::T,
    t_start::T,
    t_stop::T;
    kwargs...
) where {T<:AbstractFloat}
    fuel_ids = fill(fuel_id, state.ncols, state.nrows)
    slope = fill(slope_deg, state.ncols, state.nrows)
    aspect = fill(aspect_deg, state.ncols, state.nrows)

    simulate!(state, fuel_ids, fuel_table, weather, slope, aspect, t_start, t_stop; kwargs...)
end


#-----------------------------------------------------------------------------#
#                     Output Functions
#-----------------------------------------------------------------------------#

"""
    get_fire_perimeter(state::FireState) -> Vector{Tuple{Int, Int}}

Get the grid coordinates of cells on the fire perimeter (burned cells adjacent to unburned).
"""
function get_fire_perimeter(state::FireState)
    perimeter = Tuple{Int,Int}[]

    for ix in 1:state.ncols
        for iy in 1:state.nrows
            if state.burned[ix, iy]
                # Check if any neighbor is unburned
                for (dx, dy) in ((1,0), (-1,0), (0,1), (0,-1))
                    nx, ny = ix + dx, iy + dy
                    if 1 <= nx <= state.ncols && 1 <= ny <= state.nrows
                        if !state.burned[nx, ny]
                            push!(perimeter, (ix, iy))
                            break
                        end
                    end
                end
            end
        end
    end

    return perimeter
end


"""
    get_burned_area(state::FireState{T}) -> T

Get the total burned area in square feet.
"""
function get_burned_area(state::FireState{T}) where {T<:AbstractFloat}
    burned_cells = count(state.burned)
    return T(burned_cells) * state.cellsize^2
end


"""
    get_burned_area_acres(state::FireState{T}) -> T

Get the total burned area in acres.
"""
function get_burned_area_acres(state::FireState{T}) where {T<:AbstractFloat}
    # 1 acre = 43560 ft²
    return get_burned_area(state) / T(43560)
end


#-----------------------------------------------------------------------------#
#                     Phase 2: Extended Simulation
#-----------------------------------------------------------------------------#

"""
    SimulationConfig{T<:AbstractFloat}

Configuration for full simulation with crown fire, spotting, and weather interpolation.
"""
struct SimulationConfig{T<:AbstractFloat}
    enable_crown_fire::Bool
    enable_spotting::Bool
    crown_fire_adj::T              # Crown fire adjustment factor
    critical_canopy_cover::T       # Minimum CC for active crown fire
    foliar_moisture::T             # Foliar moisture content (%)
    spotting_params::Union{Nothing, SpottingParameters{T}}
    use_sardoy::Bool               # Use Sardoy model for spotting
end

Base.eltype(::SimulationConfig{T}) where {T} = T

function SimulationConfig{T}(;
    enable_crown_fire::Bool = false,
    enable_spotting::Bool = false,
    crown_fire_adj::T = one(T),
    critical_canopy_cover::T = T(0.4),
    foliar_moisture::T = T(100),
    spotting_params::Union{Nothing, SpottingParameters{T}} = nothing,
    use_sardoy::Bool = false
) where {T<:AbstractFloat}
    SimulationConfig{T}(
        enable_crown_fire, enable_spotting,
        crown_fire_adj, critical_canopy_cover, foliar_moisture,
        spotting_params, use_sardoy
    )
end

# Default Float64 constructor
SimulationConfig(; kwargs...) = SimulationConfig{Float64}(; kwargs...)


"""
    CanopyGrid{T<:AbstractFloat}

Grid of canopy properties for each cell.
"""
struct CanopyGrid{T<:AbstractFloat}
    cbd::Matrix{T}      # Canopy bulk density (kg/m³)
    cbh::Matrix{T}      # Canopy base height (m)
    cc::Matrix{T}       # Canopy cover (fraction)
    ch::Matrix{T}       # Canopy height (m)
end

Base.eltype(::CanopyGrid{T}) where {T} = T

"""
    CanopyGrid{T}(ncols, nrows)

Create an empty canopy grid (no canopy).
"""
function CanopyGrid{T}(ncols::Int, nrows::Int) where {T<:AbstractFloat}
    CanopyGrid{T}(
        zeros(T, ncols, nrows),
        zeros(T, ncols, nrows),
        zeros(T, ncols, nrows),
        zeros(T, ncols, nrows)
    )
end

"""
    CanopyGrid{T}(ncols, nrows, cbd, cbh, cc, ch)

Create a uniform canopy grid.
"""
function CanopyGrid{T}(
    ncols::Int, nrows::Int,
    cbd::T, cbh::T, cc::T, ch::T
) where {T<:AbstractFloat}
    CanopyGrid{T}(
        fill(cbd, ncols, nrows),
        fill(cbh, ncols, nrows),
        fill(cc, ncols, nrows),
        fill(ch, ncols, nrows)
    )
end

"""
    get_canopy_properties(grid::CanopyGrid{T}, ix::Int, iy::Int) -> CanopyProperties{T}

Get canopy properties for a specific cell.
"""
function get_canopy_properties(grid::CanopyGrid{T}, ix::Int, iy::Int) where {T<:AbstractFloat}
    CanopyProperties{T}(
        grid.cbd[ix, iy],
        grid.cbh[ix, iy],
        grid.cc[ix, iy],
        grid.ch[ix, iy]
    )
end


"""
    simulate_full!(
        state::FireState{T},
        fuel_ids::AbstractMatrix{Int},
        fuel_table::FuelModelTable{T},
        weather_interp::WeatherInterpolator{T},
        slope::AbstractMatrix{T},
        aspect::AbstractMatrix{T},
        t_start::T,
        t_stop::T;
        canopy::Union{Nothing, CanopyGrid{T}} = nothing,
        config::SimulationConfig{T} = SimulationConfig{T}(),
        dt_initial::T = one(T),
        target_cfl::T = T(0.45),
        dt_max::T = T(10),
        spread_rate_adj::T = one(T),
        accel_time_constant::T = zero(T),
        diurnal::Union{Nothing, DiurnalConfig{T}} = nothing,
        callback::Union{Nothing, Function} = nothing,
        rng::AbstractRNG = Random.default_rng()
    )

Run full fire simulation with crown fire, spotting, and weather interpolation.

# Arguments
- `state`: Fire state (modified in place)
- `fuel_ids`: Matrix of fuel model IDs for each cell
- `fuel_table`: Fuel model lookup table
- `weather_interp`: Weather interpolator for spatially/temporally varying weather
- `slope`: Slope in degrees for each cell
- `aspect`: Aspect direction in degrees for each cell
- `t_start`: Start time (minutes)
- `t_stop`: Stop time (minutes)
- `canopy`: Optional canopy properties grid (required if crown fire enabled)
- `config`: Simulation configuration (crown fire, spotting settings)
- `dt_initial`: Initial timestep (minutes, default 1.0)
- `target_cfl`: Target CFL number (default 0.45)
- `dt_max`: Maximum timestep (minutes, default 10.0)
- `spread_rate_adj`: Spread rate adjustment factor (default 1.0, multiplies base spread rate)
- `accel_time_constant`: Fire acceleration time constant (minutes, default 0 = disabled).
  When > 0.5, spread rate is multiplied by `1 - exp(-t/τ)`, modeling fire buildup from
  ignition to steady-state.
- `diurnal`: Optional `DiurnalConfig` for day/night spread rate adjustment (default nothing = disabled)
- `callback`: Optional callback function(state, t, dt, iteration) called each timestep
- `rng`: Random number generator for stochastic processes

# Returns
- `spot_tracker`: SpotFireTracker with any remaining pending spot fires
"""
function simulate_full!(
    state::FireState{T},
    fuel_ids::AbstractMatrix{Int},
    fuel_table::FuelModelTable{T},
    weather_interp::WeatherInterpolator{T},
    slope::AbstractMatrix{T},
    aspect::AbstractMatrix{T},
    t_start::T,
    t_stop::T;
    canopy::Union{Nothing, CanopyGrid{T}} = nothing,
    config::SimulationConfig{T} = SimulationConfig{T}(),
    dt_initial::T = one(T),
    target_cfl::T = T(0.45),
    dt_max::T = T(10),
    spread_rate_adj::T = one(T),
    accel_time_constant::T = zero(T),
    diurnal::Union{Nothing, DiurnalConfig{T}} = nothing,
    dampening::SpreadRateDampeningConfig{T} = SpreadRateDampeningConfig{T}(),
    callback::CB = nothing,
    rng::AbstractRNG = Random.default_rng()
) where {T<:AbstractFloat, CB}
    # Validate configuration
    if config.enable_crown_fire && canopy === nothing
        error("Canopy grid required when crown fire is enabled")
    end

    if config.enable_spotting && config.spotting_params === nothing
        error("Spotting parameters required when spotting is enabled")
    end

    t = t_start
    dt = dt_initial
    iteration = 0

    # Pre-compute static slope grid
    tanslp2_grid = calculate_tanslp2.(slope)

    # Pre-allocate cells_to_tag buffer
    cells_to_tag = CartesianIndex{2}[]
    nx_pad = state.ncols + 2*state.padding
    ny_pad = state.nrows + 2*state.padding

    # Initialize spot fire tracker if spotting enabled
    spot_tracker = if config.enable_spotting
        SpotFireTracker{T}(ignition_delay = one(T))
    else
        nothing
    end

    # Track crown fire type per cell (0=none, 1=passive, 2=active)
    crown_fire_type = zeros(Int, state.ncols, state.nrows)

    while t < t_stop
        iteration += 1

        # Process spot fire ignitions
        if spot_tracker !== nothing
            ignitions = get_ready_ignitions!(spot_tracker, t)
            for (ix, iy) in ignitions
                if !state.burned[ix, iy]
                    ignite!(state, ix, iy, t)
                end
            end
        end

        # Get active cells
        active_cells = get_active_cells(state.narrow_band)

        if isempty(active_cells)
            break  # No more active fire
        end

        # Stage 1: Compute spread rates and velocities for active cells
        for idx in active_cells
            px, py = idx[1], idx[2]
            ix, iy = padded_to_grid(state, px, py)

            # Skip if out of bounds
            if ix < 1 || ix > state.ncols || iy < 1 || iy > state.nrows
                continue
            end

            # Skip already burned cells for velocity calculation
            if state.burned[ix, iy]
                state.ux[px, py] = zero(T)
                state.uy[px, py] = zero(T)
                continue
            end

            # Get weather at this cell and time
            w = get_weather_at(weather_interp, ix, iy, t)

            # Get fuel model
            live_moisture_class = clamp(round(Int, T(100) * w.mlh), 30, 120)
            fuel_id = fuel_ids[ix, iy]
            fm = get_fuel_model(fuel_table, fuel_id, live_moisture_class)

            # Skip non-burnable
            if isnonburnable(fm)
                state.ux[px, py] = zero(T)
                state.uy[px, py] = zero(T)
                continue
            end

            # Calculate wind adjustment factor and mid-flame wind speed
            waf = if canopy !== nothing
                wind_adjustment_factor(fm.delta, canopy.cc[ix, iy], canopy.ch[ix, iy])
            else
                wind_adjustment_factor(fm.delta)
            end
            ws20_ftpmin = w.ws * T(88)  # mph to ft/min
            wsmf = ws20_ftpmin * waf

            # Use pre-computed slope factor
            tanslp2 = tanslp2_grid[ix, iy]

            # Calculate surface spread rate
            surface_result = surface_spread_rate(
                fm,
                w.m1, w.m10, w.m100,
                w.mlh, w.mlw,
                wsmf, tanslp2;
                adj = spread_rate_adj
            )

            # Crown fire calculation
            local velocity::T
            local flin_total::T
            local cft::Int = 0

            if config.enable_crown_fire && canopy !== nothing
                canopy_props = get_canopy_properties(canopy, ix, iy)
                crown_result = crown_spread_rate(
                    canopy_props,
                    surface_result.flin,
                    w.ws,
                    w.m1,
                    surface_result.vs0;
                    crown_fire_adj = config.crown_fire_adj,
                    critical_canopy_cover = config.critical_canopy_cover,
                    foliar_moisture = config.foliar_moisture
                )
                velocity = combined_spread_rate(surface_result, crown_result)
                flin_total = combined_fireline_intensity(surface_result, crown_result, fm)
                cft = crown_result.crown_fire_type
            else
                velocity = surface_result.velocity
                flin_total = surface_result.flin
            end

            # Apply spread rate dampening
            velocity = apply_spread_rate_dampening(
                velocity, wsmf, surface_result.vs0, surface_result.phis, surface_result.phiw,
                fm.phiwterm, fm.B, dampening
            )

            # Apply acceleration and diurnal factors
            velocity *= acceleration_factor(t, accel_time_constant)
            if diurnal !== nothing
                velocity *= diurnal_adjustment(diurnal, t)
            end

            # Compute normal vector to fire front
            normal_x, normal_y = compute_normal(state.phi, px, py, state.cellsize)

            # Calculate velocity components using elliptical spread
            # Midflame wind speed in mph = 20-ft wind speed × WAF (dimensionless)
            effective_ws_mph = w.ws * waf
            es = elliptical_spread(velocity, effective_ws_mph)

            # Wind direction in radians
            wind_dir_rad = w.wd * pio180(T)

            # Richards (1990) ellipse velocity decomposition
            ux, uy = velocity_components(es, wind_dir_rad, normal_x, normal_y)

            state.ux[px, py] = ux
            state.uy[px, py] = uy
        end

        # Compute CFL timestep (only after a few iterations)
        if iteration > 5
            dt = compute_cfl_timestep(
                state.ux, state.uy,
                active_cells,
                state.cellsize,
                dt;
                target_cfl = target_cfl,
                dt_max = dt_max
            )
        end

        # Don't overshoot t_stop
        if t + dt > t_stop
            dt = t_stop - t
        end

        # Perform RK2 level set integration
        for idx in active_cells
            state.phi_old[idx] = state.phi[idx]
        end

        rk2_step!(state.phi, state.phi_old, state.ux, state.uy, active_cells, dt, state.cellsize, 1)
        rk2_step!(state.phi, state.phi_old, state.ux, state.uy, active_cells, dt, state.cellsize, 2)

        # Update burned cells, narrow band, and generate spotting
        empty!(cells_to_tag)

        for idx in active_cells
            px, py = idx[1], idx[2]
            ix, iy = padded_to_grid(state, px, py)

            # Skip if out of bounds
            if ix < 1 || ix > state.ncols || iy < 1 || iy > state.nrows
                continue
            end

            # Check if cell just burned (phi crossed from positive to negative)
            if state.phi[px, py] <= zero(T) && !state.burned[ix, iy]
                state.burned[ix, iy] = true
                state.time_of_arrival[ix, iy] = t + dt

                # Get weather and fuel for this cell
                w = get_weather_at(weather_interp, ix, iy, t + dt)
                live_moisture_class = clamp(round(Int, T(100) * w.mlh), 30, 120)
                fuel_id = fuel_ids[ix, iy]
                fm = get_fuel_model(fuel_table, fuel_id, live_moisture_class)

                if !isnonburnable(fm)
                    waf = if canopy !== nothing
                        wind_adjustment_factor(fm.delta, canopy.cc[ix, iy], canopy.ch[ix, iy])
                    else
                        wind_adjustment_factor(fm.delta)
                    end
                    ws20_ftpmin = w.ws * T(88)
                    wsmf = ws20_ftpmin * waf
                    tanslp2 = tanslp2_grid[ix, iy]

                    surface_result = surface_spread_rate(
                        fm,
                        w.m1, w.m10, w.m100,
                        w.mlh, w.mlw,
                        wsmf, tanslp2;
                        adj = spread_rate_adj
                    )

                    # Crown fire calculation for recording
                    local flin_total::T
                    local cft::Int = 0

                    if config.enable_crown_fire && canopy !== nothing
                        canopy_props = get_canopy_properties(canopy, ix, iy)
                        crown_result = crown_spread_rate(
                            canopy_props,
                            surface_result.flin,
                            w.ws,
                            w.m1,
                            surface_result.vs0;
                            crown_fire_adj = config.crown_fire_adj,
                            critical_canopy_cover = config.critical_canopy_cover,
                            foliar_moisture = config.foliar_moisture
                        )
                        flin_total = combined_fireline_intensity(surface_result, crown_result, fm)
                        cft = crown_result.crown_fire_type
                        crown_fire_type[ix, iy] = cft
                        state.spread_rate[ix, iy] = combined_spread_rate(surface_result, crown_result)
                    else
                        flin_total = surface_result.flin
                        state.spread_rate[ix, iy] = surface_result.velocity
                    end

                    state.fireline_intensity[ix, iy] = flin_total

                    # Flame length (Byram): Lf = 0.0775 * I^0.46 (ft)
                    if flin_total > zero(T)
                        state.flame_length[ix, iy] = (T(0.0775) / ft_to_m(T)) * flin_total^T(0.46)
                    end

                    # Generate spot fires if enabled
                    if config.enable_spotting && spot_tracker !== nothing && config.spotting_params !== nothing
                        spot_fires = generate_spot_fires(
                            ix, iy,
                            flin_total,
                            w.ws,
                            w.wd,
                            cft,
                            config.spotting_params,
                            state.cellsize,
                            state.ncols, state.nrows,
                            t + dt,
                            state.burned;
                            weather_interp = weather_interp,
                            use_sardoy = config.use_sardoy,
                            rng = rng
                        )
                        if !isempty(spot_fires)
                            add_spot_fires!(spot_tracker, spot_fires)
                        end
                    end
                end

                # Tag surrounding cells
                push!(cells_to_tag, idx)
            end
        end

        # Expand narrow band around newly burned cells
        for idx in cells_to_tag
            tag_band!(state.narrow_band, idx, nx_pad, ny_pad, state.padding)
        end

        # Remove isolated cells from narrow band
        untag_isolated!(state.narrow_band, state.phi, state.burned, state.padding)

        t += dt

        # Optional callback
        if callback !== nothing
            callback(state, t, dt, iteration)
        end
    end

    return spot_tracker
end


"""
    simulate_full_uniform!(
        state::FireState{T},
        fuel_id::Int,
        fuel_table::FuelModelTable{T},
        weather::ConstantWeather{T},
        slope_deg::T,
        aspect_deg::T,
        t_start::T,
        t_stop::T;
        canopy_cbd::T = zero(T),
        canopy_cbh::T = zero(T),
        canopy_cc::T = zero(T),
        canopy_ch::T = zero(T),
        config::SimulationConfig{T} = SimulationConfig{T}(),
        kwargs...
    )

Run full simulation with uniform conditions across the domain.
"""
function simulate_full_uniform!(
    state::FireState{T},
    fuel_id::Int,
    fuel_table::FuelModelTable{T},
    weather::ConstantWeather{T},
    slope_deg::T,
    aspect_deg::T,
    t_start::T,
    t_stop::T;
    canopy_cbd::T = zero(T),
    canopy_cbh::T = zero(T),
    canopy_cc::T = zero(T),
    canopy_ch::T = zero(T),
    config::SimulationConfig{T} = SimulationConfig{T}(),
    kwargs...
) where {T<:AbstractFloat}
    # Create uniform grids
    fuel_ids = fill(fuel_id, state.ncols, state.nrows)
    slope = fill(slope_deg, state.ncols, state.nrows)
    aspect = fill(aspect_deg, state.ncols, state.nrows)

    # Create constant weather interpolator
    weather_interp = create_constant_interpolator(weather, state.ncols, state.nrows, state.cellsize)

    # Create canopy grid if crown fire enabled
    canopy = if config.enable_crown_fire
        CanopyGrid{T}(state.ncols, state.nrows, canopy_cbd, canopy_cbh, canopy_cc, canopy_ch)
    else
        nothing
    end

    simulate_full!(
        state, fuel_ids, fuel_table, weather_interp, slope, aspect,
        t_start, t_stop;
        canopy = canopy,
        config = config,
        kwargs...
    )
end


#-----------------------------------------------------------------------------#
#                     GPU Simulation (requires ElmfireKAExt)
#-----------------------------------------------------------------------------#

"""
    simulate_gpu!(
        state::FireState{T},
        fuel_ids::AbstractMatrix{Int},
        fuel_array::FuelModelArray{T},
        weather::ConstantWeather{T},
        slope::AbstractMatrix{T},
        aspect::AbstractMatrix{T},
        t_start::T,
        t_stop::T;
        kwargs...
    )

GPU-accelerated fire simulation. Requires loading KernelAbstractions and Adapt:

    using KernelAbstractions, Adapt

Uses KernelAbstractions.jl kernels for velocity calculation, CFL reduction, and
RK2 level set integration. The narrow band is managed on the CPU, with an active
mask uploaded to GPU each timestep.

See also: [`simulate!`](@ref), [`simulate_gpu_uniform!`](@ref)
"""
function simulate_gpu! end

"""
    simulate_gpu_uniform!(
        state::FireState{T},
        fuel_id::Int,
        fuel_array::FuelModelArray{T},
        weather::ConstantWeather{T},
        slope_deg::T,
        aspect_deg::T,
        t_start::T,
        t_stop::T;
        kwargs...
    )

GPU-accelerated simulation with uniform fuel, slope, and aspect. Requires loading
KernelAbstractions and Adapt.

See also: [`simulate_gpu!`](@ref)
"""
function simulate_gpu_uniform! end
