#-----------------------------------------------------------------------------#
#                     Ember Spotting Model
#-----------------------------------------------------------------------------#
#
# Implements ember (firebrand) transport for spot fire ignition.
#
# Supports multiple spotting distance distributions:
# - Lognormal (default ELMFIRE)
# - Sardoy (2008) physically-based model
# - Himoto model for urban fires
#
# References:
# - Sardoy et al. (2008) "Modeling transport and combustion of firebrands
#   from burning trees"
# - Himoto & Tanaka (2008) "Development and validation of a physics-based
#   urban fire spread model"
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Spotting Parameters
#-----------------------------------------------------------------------------#

"""
    SpottingParameters{T<:AbstractFloat}

Parameters controlling ember generation and transport.
"""
struct SpottingParameters{T<:AbstractFloat}
    mean_distance::T              # Mean spotting distance (m)
    normalized_variance::T        # Normalized variance of distance distribution
    ws_exponent::T                # Wind speed exponent for distance scaling
    flin_exponent::T              # Fireline intensity exponent for distance scaling
    nembers_max::Int              # Maximum number of embers per cell per timestep
    surface_spotting_percent::T   # Probability of spotting from surface fire (%)
    crown_spotting_percent::T     # Probability of spotting from crown fire (%)
    pign::T                       # Probability of ignition upon landing (%)
    min_distance::T               # Minimum spotting distance (m)
    max_distance::T               # Maximum spotting distance (m)
end

Base.eltype(::SpottingParameters{T}) where {T} = T

function SpottingParameters{T}(;
    mean_distance::T = T(100),
    normalized_variance::T = T(0.5),
    ws_exponent::T = T(1.0),
    flin_exponent::T = T(0.5),
    nembers_max::Int = 10,
    surface_spotting_percent::T = T(1),
    crown_spotting_percent::T = T(10),
    pign::T = T(50),
    min_distance::T = T(10),
    max_distance::T = T(2000)
) where {T<:AbstractFloat}
    SpottingParameters{T}(
        mean_distance, normalized_variance, ws_exponent, flin_exponent,
        nembers_max, surface_spotting_percent, crown_spotting_percent,
        pign, min_distance, max_distance
    )
end

# Default Float64 constructor
SpottingParameters() = SpottingParameters{Float64}()


"""
    SpotFire{T<:AbstractFloat}

A spot fire ignition location.
"""
struct SpotFire{T<:AbstractFloat}
    ix::Int       # Grid x index
    iy::Int       # Grid y index
    time::T       # Time of ignition (minutes)
    distance::T   # Distance from source (m)
end

Base.eltype(::SpotFire{T}) where {T} = T


#-----------------------------------------------------------------------------#
#                     Lognormal Distribution Functions
#-----------------------------------------------------------------------------#

"""
    lognormal_params(mean::T, normalized_variance::T) -> Tuple{T, T}

Convert mean and normalized variance to lognormal μ and σ parameters.

The normalized variance is defined as variance / mean².
"""
function lognormal_params(mean::T, normalized_variance::T) where {T<:AbstractFloat}
    # σ² = ln(1 + CV²) where CV = σ/μ for the lognormal
    # μ_ln = ln(mean) - σ²/2
    variance = mean * normalized_variance
    sigma_sq = log(one(T) + variance / (mean * mean))
    sigma = sqrt(sigma_sq)
    mu = log(mean * mean / sqrt(variance + mean * mean))
    return (mu, sigma)
end


"""
    sample_lognormal(mu::T, sigma::T, rng::AbstractRNG) -> T

Sample from a lognormal distribution with given μ and σ.
"""
function sample_lognormal(mu::T, sigma::T, rng::AbstractRNG) where {T<:AbstractFloat}
    # Box-Muller transform for normal sample
    u1 = rand(rng, T)
    u2 = rand(rng, T)

    # Avoid log(0)
    u1 = max(u1, T(1e-10))

    z = sqrt(T(-2) * log(u1)) * cos(T(2) * pi_val(T) * u2)
    return exp(mu + sigma * z)
end

# Version with default RNG
sample_lognormal(mu::T, sigma::T) where {T<:AbstractFloat} =
    sample_lognormal(mu, sigma, Random.default_rng())


#-----------------------------------------------------------------------------#
#                     Sardoy Model
#-----------------------------------------------------------------------------#

"""
    sardoy_parameters(ws::T, flin::T) -> NTuple{4,T}

Calculate spotting distance distribution parameters using the Sardoy (2008) model.

# Arguments
- `ws`: 20-ft wind speed (mph)
- `flin`: Fireline intensity (kW/m)

# Returns
- `(mu_dist, sigma_dist, mu_spanwise, sigma_spanwise)`: Lognormal parameters
  for downwind and crosswind distributions
"""
function sardoy_parameters(ws::T, flin::T) where {T<:AbstractFloat}
    # Physical constants
    RHO_INF = T(1.1)    # Air density (kg/m³)
    C_PG = T(1.0)       # Air heat capacity (kJ/kg-K)
    T_INF = T(300)      # Ambient temperature (K)
    G = T(9.81)         # Gravitational acceleration (m/s²)

    # Convert units
    u_wind = T(0.447) * max(T(1e-3), abs(ws)) / T(0.87)  # Convert to 10-m wind (m/s)
    i = max(flin, T(1e-6)) / T(1000)  # Fireline intensity (MW/m)

    # Characteristic length scale
    lc = (i * T(1000) / (RHO_INF * C_PG * T_INF * sqrt(G)))^T(0.67)

    # Froude number
    fr = u_wind / sqrt(G * lc)

    local mu_dist, sigma_dist

    if fr <= one(T)
        # Low Froude number regime
        mu_dist = (i^T(0.54)) / max(u_wind^T(0.55), T(1e-5))
        mu_dist = T(1.47) * mu_dist + T(1.14)

        sigma_dist = (u_wind^T(0.44)) / max(i^T(0.21), T(1e-5))
        sigma_dist = T(0.86) * sigma_dist + T(0.19)
    else
        # High Froude number regime
        mu_dist = i^T(0.26) * u_wind^T(0.11)
        mu_dist = T(1.32) * mu_dist - T(0.02)

        sigma_dist = one(T) / max(i^T(0.01), T(1e-5)) / max(u_wind^T(0.02), T(1e-5))
        sigma_dist = T(4.95) * sigma_dist - T(3.48)
    end

    mu_spanwise = zero(T)
    sigma_spanwise = T(0.92) * lc

    # Clamp mu_dist to reasonable range
    mu_dist = min(mu_dist, T(5))
    sigma_dist = max(sigma_dist, T(0.1))

    return (mu_dist, sigma_dist, mu_spanwise, sigma_spanwise)
end


#-----------------------------------------------------------------------------#
#                     Ember Generation
#-----------------------------------------------------------------------------#

"""
    compute_num_embers(
        params::SpottingParameters{T},
        flin::T,
        crown_fire_type::Int,
        dt::T,
        cellsize::T;
        rng::AbstractRNG = Random.default_rng()
    ) -> Int

Calculate the number of embers to generate for a burning cell.
"""
function compute_num_embers(
    params::SpottingParameters{T},
    flin::T,
    crown_fire_type::Int,
    dt::T,
    cellsize::T;
    rng::AbstractRNG = Random.default_rng()
) where {T<:AbstractFloat}
    # Determine spotting percentage based on fire type
    spot_percent = if crown_fire_type >= 1
        params.crown_spotting_percent
    else
        params.surface_spotting_percent
    end

    # Random check if spotting occurs
    if rand(rng, T) * T(100) > spot_percent
        return 0
    end

    # Scale number of embers with fireline intensity
    # Base rate proportional to FLIN and cell area
    ember_rate = flin * cellsize / T(1000)  # Embers per minute per MW/m

    nembers_real = ember_rate * dt

    # Stochastic rounding
    nembers = floor(Int, nembers_real)
    if rand(rng, T) < (nembers_real - nembers)
        nembers += 1
    end

    return min(nembers, params.nembers_max)
end


#-----------------------------------------------------------------------------#
#                     Ember Transport
#-----------------------------------------------------------------------------#

"""
    transport_ember(
        x0::T, y0::T,
        ws20::T,
        wd20::T,
        spotting_distance::T,
        cellsize::T,
        ncols::Int, nrows::Int;
        weather_interp::Union{Nothing, WeatherInterpolator{T}} = nothing,
        time_now::T = zero(T),
        rng::AbstractRNG = Random.default_rng()
    ) -> Tuple{Int, Int, T}

Transport a single ember from source location to landing location.

When `weather_interp` is provided, the ember is advected step-by-step through
the spatially varying wind field. Otherwise, uses source-cell wind only.

# Arguments
- `x0, y0`: Source location (in grid units, 1-based)
- `ws20`: 20-ft wind speed at source (mph)
- `wd20`: Wind direction at source (degrees, FROM convention)
- `spotting_distance`: Transport distance (m)
- `cellsize`: Cell size (ft)
- `ncols, nrows`: Grid dimensions
- `weather_interp`: Optional weather interpolator for wind-field advection
- `time_now`: Current simulation time (minutes), used with weather_interp
- `rng`: Random number generator

# Returns
- `(ix, iy, dist)`: Landing grid indices and actual distance traveled
"""
function transport_ember(
    x0::T, y0::T,
    ws20::T,
    wd20::T,
    spotting_distance::T,
    cellsize::T,
    ncols::Int, nrows::Int;
    weather_interp::Union{Nothing, WeatherInterpolator{T}} = nothing,
    time_now::T = zero(T),
    rng::AbstractRNG = Random.default_rng()
) where {T<:AbstractFloat}
    # Add small random perturbation to initial wind direction (±8°)
    wd_perturb = (rand(rng, T) - T(0.5)) * T(16)

    if weather_interp === nothing
        # Simple direct placement using source-cell wind
        wd_to_rad = (wd20 + T(180) + wd_perturb) * pio180(T)
        if wd_to_rad > T(2) * pi_val(T)
            wd_to_rad -= T(2) * pi_val(T)
        end

        dist_cells = spotting_distance / ft_to_m(T) / cellsize
        dx = dist_cells * sin(wd_to_rad)
        dy = dist_cells * cos(wd_to_rad)

        ix = clamp(round(Int, x0 + dx), 1, ncols)
        iy = clamp(round(Int, y0 + dy), 1, nrows)

        return (ix, iy, spotting_distance)
    end

    # Wind-field advection: step along the wind field
    cell_m = cellsize * ft_to_m(T)  # cell size in meters
    step_m = max(cell_m, T(10))     # step size in meters (~1 cell)
    remaining = spotting_distance
    cx, cy = x0, y0

    while remaining > zero(T)
        ds = min(step_m, remaining)

        # Get wind at current position
        cix = clamp(round(Int, cx), 1, ncols)
        ciy = clamp(round(Int, cy), 1, nrows)
        w = get_weather_at(weather_interp, cix, ciy, time_now)

        wd_to_rad = (w.wd + T(180) + wd_perturb) * pio180(T)
        if wd_to_rad > T(2) * pi_val(T)
            wd_to_rad -= T(2) * pi_val(T)
        end

        ds_cells = ds / ft_to_m(T) / cellsize
        cx += ds_cells * sin(wd_to_rad)
        cy += ds_cells * cos(wd_to_rad)
        remaining -= ds

        # Early exit if out of bounds
        if cx < T(0.5) || cx > T(ncols) + T(0.5) || cy < T(0.5) || cy > T(nrows) + T(0.5)
            break
        end
    end

    ix = clamp(round(Int, cx), 1, ncols)
    iy = clamp(round(Int, cy), 1, nrows)

    return (ix, iy, spotting_distance)
end


"""
    generate_spot_fires(
        ix0::Int, iy0::Int,
        flin::T,
        ws20::T,
        wd20::T,
        crown_fire_type::Int,
        params::SpottingParameters{T},
        cellsize::T,
        ncols::Int, nrows::Int,
        time_now::T,
        burned::BitMatrix;
        weather_interp = nothing,
        use_sardoy::Bool = false,
        rng::AbstractRNG = Random.default_rng()
    ) -> Vector{SpotFire{T}}

Generate potential spot fire locations from a burning cell.

# Arguments
- `ix0, iy0`: Source cell grid indices
- `flin`: Fireline intensity (kW/m)
- `ws20`: 20-ft wind speed (mph)
- `wd20`: Wind direction (degrees, FROM convention)
- `crown_fire_type`: 0 = surface, 1 = passive crown, 2 = active crown
- `params`: Spotting parameters
- `cellsize`: Cell size (ft)
- `ncols, nrows`: Grid dimensions
- `time_now`: Current simulation time (minutes)
- `burned`: Matrix of burned cells
- `weather_interp`: Optional WeatherInterpolator for wind-field ember advection
- `use_sardoy`: Use Sardoy model for distribution parameters
- `rng`: Random number generator

# Returns
- Vector of potential spot fire locations (may be empty)
"""
function generate_spot_fires(
    ix0::Int, iy0::Int,
    flin::T,
    ws20::T,
    wd20::T,
    crown_fire_type::Int,
    params::SpottingParameters{T},
    cellsize::T,
    ncols::Int, nrows::Int,
    time_now::T,
    burned::BitMatrix;
    weather_interp::Union{Nothing, WeatherInterpolator{T}} = nothing,
    use_sardoy::Bool = false,
    rng::AbstractRNG = Random.default_rng()
) where {T<:AbstractFloat}
    spot_fires = SpotFire{T}[]

    # Calculate number of embers
    dt = one(T)  # Assume 1 minute timestep for ember generation
    nembers = compute_num_embers(params, flin, crown_fire_type, dt, cellsize; rng=rng)

    if nembers == 0
        return spot_fires
    end

    # Calculate distribution parameters
    local mu_dist, sigma_dist
    if use_sardoy
        mu_dist, sigma_dist, _, _ = sardoy_parameters(ws20, flin)
    else
        # Scale mean distance by wind speed and fireline intensity
        msd = max(
            params.mean_distance * (flin^params.flin_exponent) * (ws20^params.ws_exponent),
            one(T)
        )
        mu_dist, sigma_dist = lognormal_params(msd, params.normalized_variance)
    end

    # Generate embers
    for _ in 1:nembers
        # Sample spotting distance
        spotting_dist = sample_lognormal(mu_dist, sigma_dist, rng)
        spotting_dist = clamp(spotting_dist, params.min_distance, params.max_distance)

        # Transport ember
        ix, iy, dist = transport_ember(
            T(ix0), T(iy0),
            ws20, wd20,
            spotting_dist,
            cellsize,
            ncols, nrows;
            weather_interp=weather_interp,
            time_now=time_now,
            rng=rng
        )

        # Check if landing location is valid for ignition
        # Must be unburned and far enough from source
        if !burned[ix, iy] && dist > T(0.5) * cellsize * ft_to_m(T)
            # Random ignition probability
            if rand(rng, T) * T(100) < params.pign
                push!(spot_fires, SpotFire{T}(ix, iy, time_now, dist))
            end
        end
    end

    return spot_fires
end


#-----------------------------------------------------------------------------#
#                     Spot Fire Management
#-----------------------------------------------------------------------------#

"""
    SpotFireTracker{T<:AbstractFloat}

Tracks pending and active spot fires during simulation.
"""
mutable struct SpotFireTracker{T<:AbstractFloat}
    pending::Vector{SpotFire{T}}   # Spot fires waiting to ignite
    ignition_delay::T              # Time delay before spot fire becomes active (min)
end

SpotFireTracker{T}(; ignition_delay::T = T(1)) where {T<:AbstractFloat} =
    SpotFireTracker{T}(SpotFire{T}[], ignition_delay)

SpotFireTracker() = SpotFireTracker{Float64}()


"""
    add_spot_fires!(tracker::SpotFireTracker{T}, fires::Vector{SpotFire{T}})

Add new spot fires to the pending queue.
"""
function add_spot_fires!(tracker::SpotFireTracker{T}, fires::Vector{SpotFire{T}}) where {T<:AbstractFloat}
    append!(tracker.pending, fires)
    return nothing
end


"""
    get_ready_ignitions!(tracker::SpotFireTracker{T}, time_now::T) -> Vector{Tuple{Int,Int}}

Get spot fires that are ready to ignite and remove them from pending queue.

Returns grid indices of cells to ignite.
"""
function get_ready_ignitions!(tracker::SpotFireTracker{T}, time_now::T) where {T<:AbstractFloat}
    ignitions = Tuple{Int,Int}[]

    # Partition in-place: move ready fires to front, keep pending at end
    write_idx = 0
    for i in eachindex(tracker.pending)
        fire = tracker.pending[i]
        if fire.time + tracker.ignition_delay <= time_now
            push!(ignitions, (fire.ix, fire.iy))
        else
            write_idx += 1
            tracker.pending[write_idx] = fire
        end
    end
    resize!(tracker.pending, write_idx)

    return ignitions
end
