#-----------------------------------------------------------------------------#
#                     Fire Simulation State and Loop
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Wind and Terrain Helpers
#-----------------------------------------------------------------------------#

"""
    wind_adjustment_factor(fuel_bed_depth::T) -> T

Calculate wind adjustment factor (WAF) from 20-ft wind to mid-flame wind.

Uses the unsheltered (no canopy) formula: WAF = 1.83 / ln((20 + 0.36*H) / (0.13*H))
where H is the fuel bed depth in feet.

For sheltered conditions (with canopy), use a lookup table or more complex formula.
"""
function wind_adjustment_factor(fuel_bed_depth::T) where {T<:AbstractFloat}
    if fuel_bed_depth < T(0.1)
        return T(0.1)
    end

    H = fuel_bed_depth
    waf = T(1.83) / log((T(20) + T(0.36)*H) / (T(0.13)*H))
    return clamp(waf, T(0.1), one(T))
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
        target_cfl::T = T(0.9),
        dt_max::T = T(10),
        spread_rate_adj::T = one(T),
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
- `target_cfl`: Target CFL number (default 0.9)
- `dt_max`: Maximum timestep (minutes, default 10.0)
- `spread_rate_adj`: Spread rate adjustment factor (default 1.0, multiplies base spread rate)
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
    target_cfl::T = T(0.9),
    dt_max::T = T(10),
    spread_rate_adj::T = one(T),
    callback::Union{Nothing, Function} = nothing
) where {T<:AbstractFloat}
    t = t_start
    dt = dt_initial
    iteration = 0

    # Pre-compute wind direction in radians
    wind_dir_rad = weather.wind_direction * pio180(T)

    # Wind speed conversion: 20-ft to ft/min
    # 1 mph = 88 ft/min
    ws20_ftpmin = weather.wind_speed_20ft * T(88)

    # Live moisture class (30-120)
    live_moisture_class = clamp(round(Int, T(100) * weather.MLH), 30, 120)

    while t < t_stop
        iteration += 1

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

            # Get fuel model
            fuel_id = fuel_ids[ix, iy]
            fm = get_fuel_model(fuel_table, fuel_id, live_moisture_class)

            # Skip non-burnable
            if isnonburnable(fm)
                state.ux[px, py] = zero(T)
                state.uy[px, py] = zero(T)
                continue
            end

            # Calculate wind adjustment factor
            waf = wind_adjustment_factor(fm.delta)
            wsmf = ws20_ftpmin * waf  # Mid-flame wind speed (ft/min)

            # Calculate slope factor
            tanslp2 = calculate_tanslp2(slope[ix, iy])

            # Calculate spread rate
            result = surface_spread_rate(
                fm,
                weather.M1, weather.M10, weather.M100,
                weather.MLH, weather.MLW,
                wsmf, tanslp2;
                adj = spread_rate_adj
            )

            # Compute normal vector to fire front
            normal_x, normal_y = compute_normal(state.phi, px, py, state.cellsize)

            # Calculate velocity components using elliptical spread
            # Get effective windspeed for ellipse calculation
            effective_ws_mph = weather.wind_speed_20ft * waf / T(1.47)  # Convert ft/min to mph

            es = elliptical_spread(result.velocity, effective_ws_mph)

            # Calculate velocity components
            ux, uy = velocity_components(
                es.head, es.back,
                wind_dir_rad,
                normal_x, normal_y
            )

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
        # Stage 1
        for idx in active_cells
            state.phi_old[idx] = state.phi[idx]
        end

        rk2_step!(state.phi, state.phi_old, state.ux, state.uy, active_cells, dt, state.cellsize, 1)
        rk2_step!(state.phi, state.phi_old, state.ux, state.uy, active_cells, dt, state.cellsize, 2)

        # Update burned cells and narrow band
        cells_to_tag = CartesianIndex{2}[]

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

                # Record spread properties
                fuel_id = fuel_ids[ix, iy]
                fm = get_fuel_model(fuel_table, fuel_id, live_moisture_class)
                waf = wind_adjustment_factor(fm.delta)
                wsmf = ws20_ftpmin * waf
                tanslp2 = calculate_tanslp2(slope[ix, iy])

                result = surface_spread_rate(
                    fm,
                    weather.M1, weather.M10, weather.M100,
                    weather.MLH, weather.MLW,
                    wsmf, tanslp2;
                    adj = spread_rate_adj
                )

                state.spread_rate[ix, iy] = result.velocity
                state.fireline_intensity[ix, iy] = result.flin

                # Flame length (Byram): Lf = 0.0775 * I^0.46 (ft)
                if result.flin > zero(T)
                    state.flame_length[ix, iy] = (T(0.0775) / ft_to_m(T)) * result.flin^T(0.46)
                end

                # Tag surrounding cells
                push!(cells_to_tag, idx)
            end
        end

        # Expand narrow band around newly burned cells
        nx_pad = state.ncols + 2*state.padding
        ny_pad = state.nrows + 2*state.padding
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
        target_cfl::T = T(0.9),
        dt_max::T = T(10),
        spread_rate_adj::T = one(T),
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
- `target_cfl`: Target CFL number (default 0.9)
- `dt_max`: Maximum timestep (minutes, default 10.0)
- `spread_rate_adj`: Spread rate adjustment factor (default 1.0, multiplies base spread rate)
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
    target_cfl::T = T(0.9),
    dt_max::T = T(10),
    spread_rate_adj::T = one(T),
    callback::Union{Nothing, Function} = nothing,
    rng::AbstractRNG = Random.default_rng()
) where {T<:AbstractFloat}
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
            waf = wind_adjustment_factor(fm.delta)
            ws20_ftpmin = w.ws * T(88)  # mph to ft/min
            wsmf = ws20_ftpmin * waf

            # Calculate slope factor
            tanslp2 = calculate_tanslp2(slope[ix, iy])

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

            # Compute normal vector to fire front
            normal_x, normal_y = compute_normal(state.phi, px, py, state.cellsize)

            # Calculate velocity components using elliptical spread
            effective_ws_mph = w.ws * waf / T(1.47)
            es = elliptical_spread(velocity, effective_ws_mph)

            # Wind direction in radians
            wind_dir_rad = w.wd * pio180(T)

            # Calculate velocity components
            ux, uy = velocity_components(
                es.head, es.back,
                wind_dir_rad,
                normal_x, normal_y
            )

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
        cells_to_tag = CartesianIndex{2}[]

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
                    waf = wind_adjustment_factor(fm.delta)
                    ws20_ftpmin = w.ws * T(88)
                    wsmf = ws20_ftpmin * waf
                    tanslp2 = calculate_tanslp2(slope[ix, iy])

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
        nx_pad = state.ncols + 2*state.padding
        ny_pad = state.nrows + 2*state.padding
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
