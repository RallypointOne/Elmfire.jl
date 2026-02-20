#-----------------------------------------------------------------------------#
#                     Monte Carlo Ensemble Framework
#-----------------------------------------------------------------------------#
#
# Implements Monte Carlo ensemble simulations with parameter perturbations
# for probabilistic fire spread forecasting.
#-----------------------------------------------------------------------------#

using Distributions
using ProgressMeter

#-----------------------------------------------------------------------------#
#                     Configuration Types
#-----------------------------------------------------------------------------#

"""
    PerturbationConfig{T<:AbstractFloat}

Configuration for stochastic parameter perturbations.
"""
struct PerturbationConfig{T<:AbstractFloat}
    ignition_perturb_radius::T         # Radius for ignition point perturbation (grid cells)
    wind_speed_factor_range::Tuple{T,T}  # (min, max) multiplier for wind speed
    wind_direction_std::T              # Standard deviation for wind direction (degrees)
    moisture_factor_range::Tuple{T,T}  # (min, max) multiplier for fuel moisture
    spread_rate_factor_range::Tuple{T,T}  # (min, max) multiplier for spread rate
end

Base.eltype(::PerturbationConfig{T}) where {T} = T

function PerturbationConfig{T}(;
    ignition_perturb_radius::T = zero(T),
    wind_speed_factor_range::Tuple{T,T} = (one(T), one(T)),
    wind_direction_std::T = zero(T),
    moisture_factor_range::Tuple{T,T} = (one(T), one(T)),
    spread_rate_factor_range::Tuple{T,T} = (one(T), one(T))
) where {T<:AbstractFloat}
    PerturbationConfig{T}(
        ignition_perturb_radius,
        wind_speed_factor_range,
        wind_direction_std,
        moisture_factor_range,
        spread_rate_factor_range
    )
end

PerturbationConfig(; kwargs...) = PerturbationConfig{Float64}(; kwargs...)


"""
    EnsembleConfig{T<:AbstractFloat}

Configuration for ensemble simulation runs.
"""
struct EnsembleConfig{T<:AbstractFloat}
    n_simulations::Int
    base_seed::UInt64
    perturbation::PerturbationConfig{T}
    simulation_config::SimulationConfig{T}
    save_individual_results::Bool
end

Base.eltype(::EnsembleConfig{T}) where {T} = T

function EnsembleConfig{T}(;
    n_simulations::Int = 100,
    base_seed::UInt64 = UInt64(12345),
    perturbation::PerturbationConfig{T} = PerturbationConfig{T}(),
    simulation_config::SimulationConfig{T} = SimulationConfig{T}(),
    save_individual_results::Bool = false
) where {T<:AbstractFloat}
    EnsembleConfig{T}(n_simulations, base_seed, perturbation, simulation_config, save_individual_results)
end

EnsembleConfig(; kwargs...) = EnsembleConfig{Float64}(; kwargs...)


#-----------------------------------------------------------------------------#
#                     Ensemble Results
#-----------------------------------------------------------------------------#

"""
    EnsembleMember{T<:AbstractFloat}

Results from a single ensemble member simulation.
"""
struct EnsembleMember{T<:AbstractFloat}
    id::Int
    seed::UInt64
    burned::BitMatrix
    time_of_arrival::Matrix{T}
    burned_area_acres::T
    max_spread_rate::T
end

Base.eltype(::EnsembleMember{T}) where {T} = T


"""
    EnsembleResult{T<:AbstractFloat}

Aggregated results from an ensemble of simulations.
"""
mutable struct EnsembleResult{T<:AbstractFloat}
    config::EnsembleConfig{T}
    members::Vector{EnsembleMember{T}}
    burn_probability::Matrix{T}
    mean_arrival_time::Matrix{T}
    std_arrival_time::Matrix{T}
    mean_burned_area::T
    std_burned_area::T
    convergence_history::Vector{T}  # RMS change in burn probability
end

Base.eltype(::EnsembleResult{T}) where {T} = T

function EnsembleResult{T}(config::EnsembleConfig{T}, ncols::Int, nrows::Int) where {T<:AbstractFloat}
    EnsembleResult{T}(
        config,
        EnsembleMember{T}[],
        zeros(T, ncols, nrows),
        fill(-one(T), ncols, nrows),
        zeros(T, ncols, nrows),
        zero(T),
        zero(T),
        T[]
    )
end


#-----------------------------------------------------------------------------#
#                     Perturbation Functions
#-----------------------------------------------------------------------------#

"""
    perturb_weather(weather::ConstantWeather{T}, config::PerturbationConfig{T}, rng::AbstractRNG) -> ConstantWeather{T}

Apply stochastic perturbations to weather conditions.
"""
function perturb_weather(
    weather::ConstantWeather{T},
    config::PerturbationConfig{T},
    rng::AbstractRNG
) where {T<:AbstractFloat}
    # Wind speed perturbation (handle trivial range)
    ws_factor = if config.wind_speed_factor_range[1] < config.wind_speed_factor_range[2]
        rand(rng, Uniform(config.wind_speed_factor_range[1], config.wind_speed_factor_range[2]))
    else
        config.wind_speed_factor_range[1]
    end
    new_wind_speed = weather.wind_speed_20ft * ws_factor

    # Wind direction perturbation
    if config.wind_direction_std > zero(T)
        wd_perturb = rand(rng, Normal(zero(T), config.wind_direction_std))
        new_wind_dir = weather.wind_direction + wd_perturb
        # Normalize to 0-360
        while new_wind_dir < zero(T)
            new_wind_dir += T(360)
        end
        while new_wind_dir >= T(360)
            new_wind_dir -= T(360)
        end
    else
        new_wind_dir = weather.wind_direction
    end

    # Moisture perturbation (handle trivial range)
    m_factor = if config.moisture_factor_range[1] < config.moisture_factor_range[2]
        rand(rng, Uniform(config.moisture_factor_range[1], config.moisture_factor_range[2]))
    else
        config.moisture_factor_range[1]
    end
    new_m1 = clamp(weather.M1 * m_factor, T(0.01), T(0.5))
    new_m10 = clamp(weather.M10 * m_factor, T(0.01), T(0.5))
    new_m100 = clamp(weather.M100 * m_factor, T(0.01), T(0.5))

    return ConstantWeather{T}(
        new_wind_speed,
        new_wind_dir,
        new_m1,
        new_m10,
        new_m100,
        weather.MLH,
        weather.MLW
    )
end


"""
    perturb_ignition(ix::Int, iy::Int, radius::T, ncols::Int, nrows::Int, rng::AbstractRNG) -> Tuple{Int, Int}

Perturb ignition location within a given radius.
"""
function perturb_ignition(
    ix::Int, iy::Int,
    radius::T,
    ncols::Int, nrows::Int,
    rng::AbstractRNG
) where {T<:AbstractFloat}
    if radius <= zero(T)
        return (ix, iy)
    end

    # Sample random angle and distance
    angle = rand(rng, Uniform(zero(T), T(2) * pi_val(T)))
    dist = rand(rng, Uniform(zero(T), radius))

    # Compute new position
    new_ix = round(Int, ix + dist * cos(angle))
    new_iy = round(Int, iy + dist * sin(angle))

    # Clamp to grid bounds
    new_ix = clamp(new_ix, 1, ncols)
    new_iy = clamp(new_iy, 1, nrows)

    return (new_ix, new_iy)
end


#-----------------------------------------------------------------------------#
#                     Statistics Functions
#-----------------------------------------------------------------------------#

"""
    compute_burn_probability(members::Vector{EnsembleMember{T}}, ncols::Int, nrows::Int) -> Matrix{T}

Compute burn probability from ensemble members.
"""
function compute_burn_probability(
    members::Vector{EnsembleMember{T}},
    ncols::Int, nrows::Int
) where {T<:AbstractFloat}
    n = length(members)
    if n == 0
        return zeros(T, ncols, nrows)
    end

    burn_count = zeros(Int, ncols, nrows)
    for member in members
        for ix in 1:ncols
            for iy in 1:nrows
                if member.burned[ix, iy]
                    burn_count[ix, iy] += 1
                end
            end
        end
    end

    return T.(burn_count) ./ T(n)
end


"""
    compute_mean_arrival_time(members::Vector{EnsembleMember{T}}, ncols::Int, nrows::Int) -> Tuple{Matrix{T}, Matrix{T}}

Compute mean and standard deviation of time of arrival from ensemble members.
Returns (mean_toa, std_toa).
"""
function compute_mean_arrival_time(
    members::Vector{EnsembleMember{T}},
    ncols::Int, nrows::Int
) where {T<:AbstractFloat}
    n = length(members)
    if n == 0
        return (fill(-one(T), ncols, nrows), zeros(T, ncols, nrows))
    end

    # Sum and sum of squares for Welford's algorithm
    mean_toa = fill(-one(T), ncols, nrows)
    std_toa = zeros(T, ncols, nrows)
    count = zeros(Int, ncols, nrows)

    # First pass: compute mean
    sum_toa = zeros(T, ncols, nrows)
    for member in members
        for ix in 1:ncols
            for iy in 1:nrows
                toa = member.time_of_arrival[ix, iy]
                if toa >= zero(T)
                    sum_toa[ix, iy] += toa
                    count[ix, iy] += 1
                end
            end
        end
    end

    for ix in 1:ncols
        for iy in 1:nrows
            if count[ix, iy] > 0
                mean_toa[ix, iy] = sum_toa[ix, iy] / T(count[ix, iy])
            end
        end
    end

    # Second pass: compute variance
    sum_sq = zeros(T, ncols, nrows)
    for member in members
        for ix in 1:ncols
            for iy in 1:nrows
                toa = member.time_of_arrival[ix, iy]
                if toa >= zero(T) && mean_toa[ix, iy] >= zero(T)
                    diff = toa - mean_toa[ix, iy]
                    sum_sq[ix, iy] += diff^2
                end
            end
        end
    end

    for ix in 1:ncols
        for iy in 1:nrows
            if count[ix, iy] > 1
                std_toa[ix, iy] = sqrt(sum_sq[ix, iy] / T(count[ix, iy] - 1))
            end
        end
    end

    return (mean_toa, std_toa)
end


"""
    aggregate_ensemble_statistics!(result::EnsembleResult{T})

Compute aggregate statistics from ensemble members.
"""
function aggregate_ensemble_statistics!(result::EnsembleResult{T}) where {T<:AbstractFloat}
    members = result.members
    n = length(members)

    if n == 0
        return nothing
    end

    ncols, nrows = size(result.burn_probability)

    # Burn probability
    result.burn_probability = compute_burn_probability(members, ncols, nrows)

    # Mean and std of arrival time
    result.mean_arrival_time, result.std_arrival_time = compute_mean_arrival_time(members, ncols, nrows)

    # Mean and std of burned area
    areas = [m.burned_area_acres for m in members]
    result.mean_burned_area = sum(areas) / T(n)
    if n > 1
        result.std_burned_area = sqrt(sum((a - result.mean_burned_area)^2 for a in areas) / T(n - 1))
    end

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Main Ensemble Function
#-----------------------------------------------------------------------------#

"""
    run_ensemble!(
        config::EnsembleConfig{T},
        state_template::FireState{T},
        fuel_ids::AbstractMatrix{Int},
        fuel_table::FuelModelTable{T},
        weather::ConstantWeather{T},
        slope::AbstractMatrix{T},
        aspect::AbstractMatrix{T},
        ignition_ix::Int,
        ignition_iy::Int,
        t_start::T,
        t_stop::T;
        canopy::Union{Nothing, CanopyGrid{T}} = nothing,
        show_progress::Bool = true,
        callback::Union{Nothing, Function} = nothing
    ) -> EnsembleResult{T}

Run a Monte Carlo ensemble of fire simulations.

# Arguments
- `config`: Ensemble configuration
- `state_template`: Template FireState to clone for each simulation
- `fuel_ids`: Matrix of fuel model IDs
- `fuel_table`: Fuel model lookup table
- `weather`: Base weather conditions (will be perturbed)
- `slope`: Slope in degrees
- `aspect`: Aspect in degrees
- `ignition_ix, ignition_iy`: Grid coordinates of ignition point
- `t_start, t_stop`: Simulation time range (minutes)
- `canopy`: Optional canopy grid for crown fire
- `show_progress`: Show progress bar (default true)
- `callback`: Optional callback(member_id, state) called after each member completes
"""
function run_ensemble!(
    config::EnsembleConfig{T},
    state_template::FireState{T},
    fuel_ids::AbstractMatrix{Int},
    fuel_table::FuelModelTable{T},
    weather::ConstantWeather{T},
    slope::AbstractMatrix{T},
    aspect::AbstractMatrix{T},
    ignition_ix::Int,
    ignition_iy::Int,
    t_start::T,
    t_stop::T;
    canopy::Union{Nothing, CanopyGrid{T}} = nothing,
    show_progress::Bool = true,
    callback::CB = nothing
) where {T<:AbstractFloat, CB}
    ncols = state_template.ncols
    nrows = state_template.nrows

    # Initialize result
    result = EnsembleResult{T}(config, ncols, nrows)

    # Progress meter
    progress = if show_progress
        Progress(config.n_simulations, desc="Running ensemble: ")
    else
        nothing
    end

    prev_burn_prob = zeros(T, ncols, nrows)
    burn_count = zeros(Int, ncols, nrows)

    for i in 1:config.n_simulations
        # Create RNG for this member
        seed = config.base_seed + UInt64(i)
        rng = MersenneTwister(seed)

        # Clone the state
        state = copy(state_template)
        reset!(state)

        # Perturb weather
        perturbed_weather = perturb_weather(weather, config.perturbation, rng)

        # Perturb ignition location
        new_ix, new_iy = perturb_ignition(
            ignition_ix, ignition_iy,
            config.perturbation.ignition_perturb_radius,
            ncols, nrows,
            rng
        )

        # Ignite
        ignite!(state, new_ix, new_iy, t_start)

        # Create weather interpolator
        weather_interp = create_constant_interpolator(perturbed_weather, ncols, nrows, state.cellsize)

        # Run simulation
        simulate_full!(
            state,
            fuel_ids,
            fuel_table,
            weather_interp,
            slope,
            aspect,
            t_start,
            t_stop;
            canopy = canopy,
            config = config.simulation_config,
            rng = rng
        )

        # Apply spread rate factor if specified
        if config.perturbation.spread_rate_factor_range != (one(T), one(T))
            # Note: This is a simplification - in a more sophisticated implementation,
            # the spread rate factor would be applied within the simulation loop
        end

        # Compute member statistics
        burned_area = get_burned_area_acres(state)
        max_spread = maximum(state.spread_rate)

        # Create member record
        member = EnsembleMember{T}(
            i,
            seed,
            config.save_individual_results ? copy(state.burned) : state.burned,
            config.save_individual_results ? copy(state.time_of_arrival) : state.time_of_arrival,
            burned_area,
            max_spread
        )
        push!(result.members, member)

        # Incremental burn probability for convergence tracking
        for ix in 1:ncols
            for iy in 1:nrows
                if state.burned[ix, iy]
                    burn_count[ix, iy] += 1
                end
            end
        end
        inv_i = one(T) / T(i)
        rms_sum = zero(T)
        for ix in 1:ncols
            for iy in 1:nrows
                new_prob = T(burn_count[ix, iy]) * inv_i
                diff = new_prob - prev_burn_prob[ix, iy]
                rms_sum += diff * diff
                prev_burn_prob[ix, iy] = new_prob
            end
        end
        push!(result.convergence_history, sqrt(rms_sum / (ncols * nrows)))

        # Callback
        if callback !== nothing
            callback(i, state)
        end

        # Update progress
        if progress !== nothing
            next!(progress)
        end
    end

    # Compute final statistics
    aggregate_ensemble_statistics!(result)

    return result
end


"""
    check_convergence(result::EnsembleResult{T}; threshold::T = T(0.001)) -> Bool

Check if the ensemble has converged based on burn probability changes.
"""
function check_convergence(result::EnsembleResult{T}; threshold::T = T(0.001)) where {T<:AbstractFloat}
    history = result.convergence_history
    if length(history) < 10
        return false
    end

    # Check if last 10 iterations are below threshold
    return all(h < threshold for h in history[end-9:end])
end


"""
    get_exceedance_probability(result::EnsembleResult{T}, threshold_acres::T) -> T

Calculate the probability that burned area exceeds a threshold.
"""
function get_exceedance_probability(result::EnsembleResult{T}, threshold_acres::T) where {T<:AbstractFloat}
    members = result.members
    if isempty(members)
        return zero(T)
    end

    count_exceeded = count(m -> m.burned_area_acres > threshold_acres, members)
    return T(count_exceeded) / T(length(members))
end


"""
    get_percentile_fire(result::EnsembleResult{T}, percentile::T) -> Union{Nothing, EnsembleMember{T}}

Get the ensemble member closest to a given percentile of burned area.
"""
function get_percentile_fire(result::EnsembleResult{T}, percentile::T) where {T<:AbstractFloat}
    members = result.members
    if isempty(members)
        return nothing
    end

    # Sort by burned area
    sorted = sort(members, by = m -> m.burned_area_acres)

    # Find index at percentile
    idx = clamp(ceil(Int, percentile / T(100) * length(sorted)), 1, length(sorted))

    return sorted[idx]
end
