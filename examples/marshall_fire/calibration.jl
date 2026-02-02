#=
Calibration Framework for Marshall Fire
========================================

Parameter optimization to match simulated fire spread to observed data.
Uses black-box optimization (BBO) to search parameter space.
=#

using Optimization
using OptimizationBBO
using Elmfire

#-----------------------------------------------------------------------------#
#                           Calibration Parameters
#-----------------------------------------------------------------------------#

"""
    CalibrationParams{T}

Parameters to calibrate for fire spread simulation.

# Fields
- `wind_speed_factor`: Multiplier on wind speed [0.5, 2.0]
- `spread_rate_factor`: Multiplier on base spread rate [0.5, 2.0]
- `wind_dir_bias`: Degrees to add to wind direction [-30, 30]
- `moisture_factor`: Multiplier on fuel moisture [0.5, 1.5]
"""
struct CalibrationParams{T<:AbstractFloat}
    wind_speed_factor::T
    spread_rate_factor::T
    wind_dir_bias::T
    moisture_factor::T
end

function CalibrationParams(vec::AbstractVector{T}) where {T}
    CalibrationParams{T}(vec[1], vec[2], vec[3], vec[4])
end

function Base.Vector(p::CalibrationParams{T}) where {T}
    T[p.wind_speed_factor, p.spread_rate_factor, p.wind_dir_bias, p.moisture_factor]
end

# Default parameters (no adjustment)
default_params(::Type{T}=Float64) where {T} = CalibrationParams{T}(
    one(T), one(T), zero(T), one(T)
)

# Parameter bounds
const PARAM_LOWER_BOUNDS = [0.5, 0.5, -30.0, 0.5]
const PARAM_UPPER_BOUNDS = [2.0, 2.0, 30.0, 1.5]
const PARAM_NAMES = ["wind_speed_factor", "spread_rate_factor", "wind_dir_bias", "moisture_factor"]

#-----------------------------------------------------------------------------#
#                           Weather Adjustment
#-----------------------------------------------------------------------------#

"""
    apply_param_adjustments(weather_series, params::CalibrationParams)

Apply calibration parameters to weather series.
"""
function apply_param_adjustments(
    weather_series::Elmfire.WeatherTimeSeries{T},
    params::CalibrationParams
) where {T}
    adjusted_grids = Elmfire.WeatherGrid{T}[]

    for grid in weather_series.grids
        # Adjust wind speed
        new_ws = grid.ws .* T(params.wind_speed_factor)

        # Adjust wind direction
        new_wd = mod.(grid.wd .+ T(params.wind_dir_bias), T(360))

        # Adjust fuel moisture
        new_m1 = grid.m1 .* T(params.moisture_factor)
        new_m10 = grid.m10 .* T(params.moisture_factor)
        new_m100 = grid.m100 .* T(params.moisture_factor)
        new_mlh = grid.mlh .* T(params.moisture_factor)
        new_mlw = grid.mlw .* T(params.moisture_factor)

        adjusted_grid = Elmfire.WeatherGrid{T}(
            new_ws, new_wd,
            new_m1, new_m10, new_m100, new_mlh, new_mlw,
            grid.ncols, grid.nrows, grid.cellsize,
            grid.xllcorner, grid.yllcorner
        )

        push!(adjusted_grids, adjusted_grid)
    end

    return Elmfire.WeatherTimeSeries{T}(adjusted_grids, weather_series.times)
end

#-----------------------------------------------------------------------------#
#                           Simulation Runner
#-----------------------------------------------------------------------------#

"""
    run_simulation_with_params(params::CalibrationParams, data::MarshallFireData;
                                duration_minutes::Float64=360.0,
                                base_wind_speed::Float64=100.0,
                                base_wind_dir::Float64=270.0)

Run fire simulation with given calibration parameters.
"""
function run_simulation_with_params(
    params::CalibrationParams,
    data::MarshallFireData{T};
    duration_minutes::Float64 = 360.0,
    base_wind_speed::Float64 = 100.0,
    base_wind_dir::Float64 = 270.0
) where {T}
    ncols = size(data.fuel_ids, 1)
    nrows = size(data.fuel_ids, 2)

    # Apply parameter adjustments to create weather
    adjusted_wind_speed = base_wind_speed * params.wind_speed_factor
    adjusted_wind_dir = mod(base_wind_dir + params.wind_dir_bias, 360.0)
    adjusted_moisture = 0.03 * params.moisture_factor  # Base 3% moisture

    weather = Elmfire.ConstantWeather(
        wind_speed_mph = adjusted_wind_speed,
        wind_direction = adjusted_wind_dir,
        M1 = adjusted_moisture,
        M10 = adjusted_moisture * 1.33,
        M100 = adjusted_moisture * 2.0,
        MLH = 0.30 * params.moisture_factor,
        MLW = 0.60 * params.moisture_factor
    )

    # Create fire state (cellsize in feet)
    state = Elmfire.FireState{T}(ncols, nrows, T(CELLSIZE_FEET))

    # Ignite at known ignition point
    Elmfire.ignite!(state, data.ignition_ix, data.ignition_iy, zero(T))

    # Create fuel table
    fuel_table = Elmfire.create_standard_fuel_table(T)

    # Run simulation
    Elmfire.simulate!(
        state,
        data.fuel_ids,
        fuel_table,
        weather,
        data.slope,
        data.aspect,
        zero(T),
        T(duration_minutes);
        dt_initial = T(0.5),
        target_cfl = T(0.9),
        dt_max = T(5)
    )

    return state
end

#-----------------------------------------------------------------------------#
#                           Objective Function
#-----------------------------------------------------------------------------#

"""
    objective_function(param_vec, data, observed)

Objective function for optimization (to minimize).
Returns negative Sorensen coefficient.
"""
function objective_function(param_vec::AbstractVector, (data, observed))
    params = CalibrationParams(param_vec)

    try
        state = run_simulation_with_params(params, data)
        score = sorensen(state.burned, observed)
        return -score  # Negative for minimization
    catch e
        @warn "Simulation failed with params $param_vec: $e"
        return 0.0  # Return worst possible score
    end
end

"""
    combined_objective(param_vec, data, observed)

Combined objective that balances Sorensen similarity with area accuracy.
"""
function combined_objective(param_vec::AbstractVector, (data, observed))
    params = CalibrationParams(param_vec)

    try
        state = run_simulation_with_params(params, data)

        # Compute metrics
        s = sorensen(state.burned, observed)
        ae = abs(area_error(state.burned, observed))

        # Combined objective: weight Sorensen heavily, penalize area error
        # Higher is better, so negate for minimization
        combined = s - 0.2 * ae

        return -combined
    catch e
        @warn "Simulation failed with params $param_vec: $e"
        return 0.0
    end
end

#-----------------------------------------------------------------------------#
#                           Calibration Runner
#-----------------------------------------------------------------------------#

"""
    CalibrationResult

Results from calibration run.
"""
struct CalibrationResult
    best_params::CalibrationParams{Float64}
    best_score::Float64
    iterations::Int
    convergence_history::Vector{Float64}
end

"""
    run_calibration(data::MarshallFireData, observed::BitMatrix;
                    max_iters::Int=100, verbose::Bool=true, use_combined::Bool=true)

Run calibration optimization.

# Arguments
- `data`: Marshall fire data
- `observed`: Observed burned area
- `max_iters`: Maximum optimization iterations
- `verbose`: Print progress during optimization
- `use_combined`: Use combined objective (Sorensen + area penalty)
"""
function run_calibration(
    data::MarshallFireData,
    observed::BitMatrix;
    max_iters::Int = 100,
    verbose::Bool = true,
    use_combined::Bool = true
)
    # Select objective function
    obj_func = use_combined ? combined_objective : objective_function

    # Track convergence
    history = Float64[]

    # Callback for tracking
    function callback(state, best_fitness)
        push!(history, -best_fitness)  # Convert back to positive score
        if verbose && length(history) % 10 == 0
            println("  Iteration $(length(history)): Best score = $(round(-best_fitness, digits=4))")
        end
        return false  # Don't stop
    end

    # Initial guess (default parameters)
    x0 = Vector(default_params())

    # Set up optimization problem
    f = OptimizationFunction(obj_func)
    prob = OptimizationProblem(
        f, x0, (data, observed);
        lb = PARAM_LOWER_BOUNDS,
        ub = PARAM_UPPER_BOUNDS
    )

    # Solve using BBO (adaptive differential evolution)
    if verbose
        println("Starting calibration optimization...")
        println("  Max iterations: $max_iters")
        println("  Parameter bounds:")
        for (name, lb, ub) in zip(PARAM_NAMES, PARAM_LOWER_BOUNDS, PARAM_UPPER_BOUNDS)
            println("    $name: [$lb, $ub]")
        end
    end

    sol = solve(
        prob,
        BBO_adaptive_de_rand_1_bin_radiuslimited();
        maxiters = max_iters,
        callback = callback
    )

    best_params = CalibrationParams(sol.u)
    best_score = -sol.objective

    if verbose
        println("\nCalibration complete!")
        println("  Best score: $(round(best_score, digits=4))")
        println("  Best parameters:")
        println("    wind_speed_factor:  $(round(best_params.wind_speed_factor, digits=4))")
        println("    spread_rate_factor: $(round(best_params.spread_rate_factor, digits=4))")
        println("    wind_dir_bias:      $(round(best_params.wind_dir_bias, digits=2))°")
        println("    moisture_factor:    $(round(best_params.moisture_factor, digits=4))")
    end

    return CalibrationResult(best_params, best_score, length(history), history)
end

#-----------------------------------------------------------------------------#
#                           Grid Search Alternative
#-----------------------------------------------------------------------------#

"""
    run_grid_search(data::MarshallFireData, observed::BitMatrix;
                    wind_speeds=[0.7, 0.85, 1.0, 1.15, 1.3],
                    spread_factors=[0.7, 0.85, 1.0, 1.15, 1.3],
                    verbose::Bool=true)

Run grid search over parameter combinations.
Simpler but slower than BBO optimization.
"""
function run_grid_search(
    data::MarshallFireData,
    observed::BitMatrix;
    wind_speeds::Vector{Float64} = [0.7, 0.85, 1.0, 1.15, 1.3],
    spread_factors::Vector{Float64} = [0.7, 0.85, 1.0, 1.15, 1.3],
    verbose::Bool = true
)
    best_score = 0.0
    best_params = default_params()

    total = length(wind_speeds) * length(spread_factors)
    current = 0

    for ws_factor in wind_speeds
        for sf_factor in spread_factors
            current += 1
            params = CalibrationParams{Float64}(ws_factor, sf_factor, 0.0, 1.0)

            try
                state = run_simulation_with_params(params, data)
                score = sorensen(state.burned, observed)

                if score > best_score
                    best_score = score
                    best_params = params
                    if verbose
                        println("  [$current/$total] New best: score=$(round(score, digits=4)), " *
                                "ws_factor=$(ws_factor), sf_factor=$(sf_factor)")
                    end
                end
            catch e
                if verbose
                    println("  [$current/$total] Failed: ws_factor=$(ws_factor), sf_factor=$(sf_factor)")
                end
            end
        end
    end

    return CalibrationResult(best_params, best_score, current, Float64[])
end

#-----------------------------------------------------------------------------#
#                           Parameter Sensitivity
#-----------------------------------------------------------------------------#

"""
    parameter_sensitivity(data::MarshallFireData, observed::BitMatrix,
                          base_params::CalibrationParams;
                          perturbation::Float64=0.1)

Compute sensitivity of objective to each parameter.
"""
function parameter_sensitivity(
    data::MarshallFireData,
    observed::BitMatrix,
    base_params::CalibrationParams;
    perturbation::Float64 = 0.1
)
    base_vec = Vector(base_params)
    base_state = run_simulation_with_params(base_params, data)
    base_score = sorensen(base_state.burned, observed)

    sensitivities = Dict{String, Float64}()

    for (i, name) in enumerate(PARAM_NAMES)
        # Perturb parameter up
        perturbed_vec = copy(base_vec)
        perturbed_vec[i] *= (1.0 + perturbation)

        # Clamp to bounds
        perturbed_vec[i] = clamp(perturbed_vec[i], PARAM_LOWER_BOUNDS[i], PARAM_UPPER_BOUNDS[i])

        try
            perturbed_params = CalibrationParams(perturbed_vec)
            state = run_simulation_with_params(perturbed_params, data)
            perturbed_score = sorensen(state.burned, observed)

            # Sensitivity: change in score / change in parameter
            delta_param = (perturbed_vec[i] - base_vec[i]) / base_vec[i]
            delta_score = perturbed_score - base_score

            sensitivities[name] = delta_param > 0 ? delta_score / delta_param : 0.0
        catch
            sensitivities[name] = 0.0
        end
    end

    return sensitivities
end
