#-----------------------------------------------------------------------------#
#                     Fire Suppression Models
#-----------------------------------------------------------------------------#
#
# Implements fire suppression and containment models including:
# - Suppression resources (hand crews, engines, dozers, aircraft)
# - Containment line construction
# - Fire containment and spread reduction
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Suppression Resources
#-----------------------------------------------------------------------------#

"""
    SuppressionResource{T<:AbstractFloat}

A fire suppression resource (crew, engine, etc.).
"""
mutable struct SuppressionResource{T<:AbstractFloat}
    id::Int
    resource_type::Symbol      # :hand_crew, :engine, :dozer, :aircraft
    location_x::T              # Current x position (grid units)
    location_y::T              # Current y position (grid units)
    line_production_rate::T    # Fireline construction rate (ft/min)
    effective_width::T         # Width of constructed line (ft)
    status::Symbol             # :available, :deployed, :traveling, :resting
    assignment_ix::Int         # Current assignment x coordinate
    assignment_iy::Int         # Current assignment y coordinate
    fatigue::T                 # Fatigue level (0-1)
end

Base.eltype(::SuppressionResource{T}) where {T} = T

function SuppressionResource{T}(
    id::Int,
    resource_type::Symbol;
    location_x::T = zero(T),
    location_y::T = zero(T)
) where {T<:AbstractFloat}
    # Default parameters by resource type
    # Based on NWCG standards
    line_rate, eff_width = if resource_type == :hand_crew
        (T(2.5), T(6))     # 20-person crew: ~150 ft/hr = 2.5 ft/min
    elseif resource_type == :engine
        (T(5), T(10))      # Engine with hose lay
    elseif resource_type == :dozer
        (T(20), T(15))     # Bulldozer: ~1200 ft/hr
    elseif resource_type == :aircraft
        (T(100), T(100))   # Air tanker retardant drop
    else
        (one(T), T(3))
    end

    SuppressionResource{T}(
        id, resource_type,
        location_x, location_y,
        line_rate, eff_width,
        :available, 0, 0, zero(T)
    )
end


#-----------------------------------------------------------------------------#
#                     Containment Lines
#-----------------------------------------------------------------------------#

"""
    ContainmentLine{T<:AbstractFloat}

A fire containment line (fireline, retardant line, etc.).
"""
struct ContainmentLine{T<:AbstractFloat}
    cells::Vector{Tuple{Int,Int}}  # Grid cells in the line
    effectiveness::T               # Fire spread reduction (0-1)
    width::T                       # Line width (ft)
    construction_time::T           # Time when line was completed (min)
    resource_id::Int               # Resource that built this line
end

Base.eltype(::ContainmentLine{T}) where {T} = T


"""
    ContainmentLine{T}(cells; effectiveness=0.9, width=6.0, construction_time=0.0, resource_id=0)

Create a containment line from a vector of cells.
"""
function ContainmentLine{T}(
    cells::Vector{Tuple{Int,Int}};
    effectiveness::T = T(0.9),
    width::T = T(6),
    construction_time::T = zero(T),
    resource_id::Int = 0
) where {T<:AbstractFloat}
    ContainmentLine{T}(cells, effectiveness, width, construction_time, resource_id)
end


#-----------------------------------------------------------------------------#
#                     Suppression State
#-----------------------------------------------------------------------------#

"""
    SuppressionState{T<:AbstractFloat}

Complete state of fire suppression activities.
"""
mutable struct SuppressionState{T<:AbstractFloat}
    resources::Vector{SuppressionResource{T}}
    containment_lines::Vector{ContainmentLine{T}}
    contained_cells::BitMatrix     # Cells with containment lines
    containment_effectiveness::Matrix{T}  # Spread reduction factor per cell
    total_line_constructed::T      # Total line length constructed (ft)
    active_assignments::Dict{Int, Vector{Tuple{Int,Int}}}  # Resource ID -> target cells
end

Base.eltype(::SuppressionState{T}) where {T} = T


"""
    SuppressionState{T}(ncols::Int, nrows::Int)

Create an empty suppression state.
"""
function SuppressionState{T}(ncols::Int, nrows::Int) where {T<:AbstractFloat}
    SuppressionState{T}(
        SuppressionResource{T}[],
        ContainmentLine{T}[],
        falses(ncols, nrows),
        ones(T, ncols, nrows),  # Default effectiveness = 1 (no reduction)
        zero(T),
        Dict{Int, Vector{Tuple{Int,Int}}}()
    )
end


"""
    add_resource!(state::SuppressionState{T}, resource::SuppressionResource{T})

Add a suppression resource to the state.
"""
function add_resource!(state::SuppressionState{T}, resource::SuppressionResource{T}) where {T<:AbstractFloat}
    push!(state.resources, resource)
    return nothing
end


#-----------------------------------------------------------------------------#
#                     Line Construction
#-----------------------------------------------------------------------------#

"""
    construct_containment_line!(
        state::SuppressionState{T},
        resource::SuppressionResource{T},
        start_ix::Int, start_iy::Int,
        target_ix::Int, target_iy::Int,
        dt::T,
        cellsize::T,
        t::T
    ) -> Tuple{Vector{Tuple{Int,Int}}, T}

Construct a containment line from start to target (or as far as possible in dt).

# Arguments
- `state`: Suppression state (modified in place)
- `resource`: Resource constructing the line
- `start_ix, start_iy`: Starting grid coordinates
- `target_ix, target_iy`: Target grid coordinates
- `dt`: Time step (minutes)
- `cellsize`: Grid cell size (ft)
- `t`: Current simulation time (minutes)

# Returns
- Tuple of (cells_constructed, length_constructed)
"""
function construct_containment_line!(
    state::SuppressionState{T},
    resource::SuppressionResource{T},
    start_ix::Int, start_iy::Int,
    target_ix::Int, target_iy::Int,
    dt::T,
    cellsize::T,
    t::T
) where {T<:AbstractFloat}
    if resource.status != :deployed
        return (Tuple{Int,Int}[], zero(T))
    end

    # Calculate maximum line length constructable in dt
    # Account for fatigue
    fatigue_factor = one(T) - T(0.3) * resource.fatigue
    max_length = resource.line_production_rate * dt * max(fatigue_factor, T(0.5))

    # Direction and distance to target
    dx = T(target_ix - start_ix)
    dy = T(target_iy - start_iy)
    total_dist = sqrt(dx^2 + dy^2) * cellsize

    if total_dist < T(0.1)
        return (Tuple{Int,Int}[], zero(T))
    end

    # Normalize direction
    nx = dx * cellsize / total_dist
    ny = dy * cellsize / total_dist

    # Construct line cells
    cells_constructed = Tuple{Int,Int}[]
    length_constructed = zero(T)
    current_x = T(start_ix) * cellsize
    current_y = T(start_iy) * cellsize

    while length_constructed < max_length && length_constructed < total_dist
        # Current grid cell
        ix = round(Int, current_x / cellsize)
        iy = round(Int, current_y / cellsize)

        # Add to line if valid
        ncols, nrows = size(state.contained_cells)
        if 1 <= ix <= ncols && 1 <= iy <= nrows
            if !state.contained_cells[ix, iy]
                push!(cells_constructed, (ix, iy))
                state.contained_cells[ix, iy] = true

                # Effectiveness based on resource type and terrain
                eff = if resource.resource_type == :dozer
                    T(0.95)  # Very effective
                elseif resource.resource_type == :hand_crew
                    T(0.85)  # Good effectiveness
                elseif resource.resource_type == :engine
                    T(0.80)  # Moderate
                elseif resource.resource_type == :aircraft
                    T(0.60)  # Retardant less reliable
                else
                    T(0.70)
                end

                state.containment_effectiveness[ix, iy] = min(
                    state.containment_effectiveness[ix, iy],
                    one(T) - eff
                )
            end
        end

        # Move to next position
        step_size = cellsize
        current_x += nx * step_size
        current_y += ny * step_size
        length_constructed += step_size
    end

    # Update resource position
    resource.location_x = current_x / cellsize
    resource.location_y = current_y / cellsize

    # Update fatigue
    resource.fatigue = min(one(T), resource.fatigue + dt / T(480))  # 8-hour shift

    # Create containment line record
    if !isempty(cells_constructed)
        line = ContainmentLine{T}(
            cells_constructed;
            effectiveness = T(0.85),
            width = resource.effective_width,
            construction_time = t,
            resource_id = resource.id
        )
        push!(state.containment_lines, line)
    end

    # Update total line constructed
    state.total_line_constructed += length_constructed

    return (cells_constructed, length_constructed)
end


#-----------------------------------------------------------------------------#
#                     Containment Application
#-----------------------------------------------------------------------------#

"""
    apply_containment!(fire_state::FireState{T}, suppression_state::SuppressionState{T})

Apply containment effects to the fire state by reducing spread velocities
at contained cells.
"""
function apply_containment!(
    fire_state::FireState{T},
    suppression_state::SuppressionState{T}
) where {T<:AbstractFloat}
    ncols = fire_state.ncols
    nrows = fire_state.nrows

    for ix in 1:ncols
        for iy in 1:nrows
            if suppression_state.contained_cells[ix, iy]
                px, py = grid_to_padded(fire_state, ix, iy)

                # Reduce velocity by containment effectiveness
                eff = suppression_state.containment_effectiveness[ix, iy]
                fire_state.ux[px, py] *= eff
                fire_state.uy[px, py] *= eff
            end
        end
    end

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Suppression State Update
#-----------------------------------------------------------------------------#

"""
    update_suppression_state!(
        suppression::SuppressionState{T},
        fire_state::FireState{T},
        dt::T,
        t::T
    )

Update suppression state: move resources, construct lines, etc.
"""
function update_suppression_state!(
    suppression::SuppressionState{T},
    fire_state::FireState{T},
    dt::T,
    t::T
) where {T<:AbstractFloat}
    for resource in suppression.resources
        if resource.status == :deployed
            # Get assignment
            if haskey(suppression.active_assignments, resource.id)
                targets = suppression.active_assignments[resource.id]
                if !isempty(targets)
                    target_ix, target_iy = targets[1]

                    # Construct line toward target
                    start_ix = round(Int, resource.location_x)
                    start_iy = round(Int, resource.location_y)

                    cells, length = construct_containment_line!(
                        suppression, resource,
                        start_ix, start_iy,
                        target_ix, target_iy,
                        dt, fire_state.cellsize, t
                    )

                    # Check if target reached
                    dist_to_target = sqrt(
                        (resource.location_x - target_ix)^2 +
                        (resource.location_y - target_iy)^2
                    )

                    if dist_to_target < T(1.5)  # Within 1.5 cells
                        popfirst!(targets)
                        if isempty(targets)
                            resource.status = :available
                        end
                    end
                end
            end
        elseif resource.status == :resting
            # Recover from fatigue
            resource.fatigue = max(zero(T), resource.fatigue - dt / T(240))  # 4-hour recovery
            if resource.fatigue < T(0.3)
                resource.status = :available
            end
        end

        # Check for excessive fatigue
        if resource.fatigue > T(0.8)
            resource.status = :resting
        end
    end

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Assignment and Tactics
#-----------------------------------------------------------------------------#

"""
    assign_resource!(
        suppression::SuppressionState{T},
        resource_id::Int,
        targets::Vector{Tuple{Int,Int}}
    )

Assign a resource to construct lines at target locations.
"""
function assign_resource!(
    suppression::SuppressionState{T},
    resource_id::Int,
    targets::Vector{Tuple{Int,Int}}
) where {T<:AbstractFloat}
    # Find resource
    for resource in suppression.resources
        if resource.id == resource_id
            resource.status = :deployed
            resource.assignment_ix = targets[1][1]
            resource.assignment_iy = targets[1][2]
            suppression.active_assignments[resource_id] = copy(targets)
            break
        end
    end

    return nothing
end


"""
    plan_indirect_attack(
        fire_state::FireState{T},
        weather::ConstantWeather{T},
        buffer_distance::Int
    ) -> Vector{Tuple{Int,Int}}

Plan an indirect attack line ahead of the fire front.

# Arguments
- `fire_state`: Current fire state
- `weather`: Weather conditions for wind direction
- `buffer_distance`: Distance ahead of fire front (cells)

# Returns
- Vector of grid cells for the planned line
"""
function plan_indirect_attack(
    fire_state::FireState{T},
    weather::ConstantWeather{T},
    buffer_distance::Int
) where {T<:AbstractFloat}
    # Find fire perimeter
    perimeter = get_fire_perimeter(fire_state)

    if isempty(perimeter)
        return Tuple{Int,Int}[]
    end

    # Wind direction (FROM, so fire spreads in opposite direction)
    wind_rad = (weather.wind_direction + T(180)) * pio180(T)
    wind_dx = round(Int, buffer_distance * cos(wind_rad))
    wind_dy = round(Int, buffer_distance * sin(wind_rad))

    # Create line ahead of perimeter in downwind direction
    line_cells = Tuple{Int,Int}[]

    for (px, py) in perimeter
        # Check if this is a downwind perimeter cell
        # (simplified: all perimeter cells)
        line_x = px + wind_dx
        line_y = py + wind_dy

        # Check bounds
        if 1 <= line_x <= fire_state.ncols && 1 <= line_y <= fire_state.nrows
            if !fire_state.burned[line_x, line_y]
                push!(line_cells, (line_x, line_y))
            end
        end
    end

    return unique(line_cells)
end


"""
    plan_direct_attack(fire_state::FireState{T}) -> Vector{Tuple{Int,Int}}

Plan a direct attack on the current fire perimeter.
"""
function plan_direct_attack(fire_state::FireState{T}) where {T<:AbstractFloat}
    return get_fire_perimeter(fire_state)
end


#-----------------------------------------------------------------------------#
#                     Simulation with Suppression
#-----------------------------------------------------------------------------#

"""
    simulate_with_suppression!(
        state::FireState{T},
        suppression::SuppressionState{T},
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
        callback::Union{Nothing, Function} = nothing
    )

Run fire simulation with active suppression.
"""
function simulate_with_suppression!(
    state::FireState{T},
    suppression::SuppressionState{T},
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
    callback::Union{Nothing, Function} = nothing
) where {T<:AbstractFloat}
    t = t_start
    dt = dt_initial
    iteration = 0

    # Create weather interpolator
    weather_interp = create_constant_interpolator(weather, state.ncols, state.nrows, state.cellsize)

    while t < t_stop
        iteration += 1

        # Update suppression state
        update_suppression_state!(suppression, state, dt, t)

        # Apply containment effects
        apply_containment!(state, suppression)

        # Get active cells
        active_cells = get_active_cells(state.narrow_band)

        if isempty(active_cells)
            break
        end

        # Compute spread rates (similar to simulate_full!)
        for idx in active_cells
            px, py = idx[1], idx[2]
            ix, iy = padded_to_grid(state, px, py)

            if ix < 1 || ix > state.ncols || iy < 1 || iy > state.nrows
                continue
            end

            if state.burned[ix, iy]
                state.ux[px, py] = zero(T)
                state.uy[px, py] = zero(T)
                continue
            end

            w = get_weather_at(weather_interp, ix, iy, t)
            live_moisture_class = clamp(round(Int, T(100) * w.mlh), 30, 120)
            fuel_id = fuel_ids[ix, iy]
            fm = get_fuel_model(fuel_table, fuel_id, live_moisture_class)

            if isnonburnable(fm)
                state.ux[px, py] = zero(T)
                state.uy[px, py] = zero(T)
                continue
            end

            waf = wind_adjustment_factor(fm.delta)
            ws20_ftpmin = w.ws * T(88)
            wsmf = ws20_ftpmin * waf
            tanslp2 = calculate_tanslp2(slope[ix, iy])

            result = surface_spread_rate(
                fm,
                w.m1, w.m10, w.m100,
                w.mlh, w.mlw,
                wsmf, tanslp2
            )

            normal_x, normal_y = compute_normal(state.phi, px, py, state.cellsize)
            effective_ws_mph = w.ws * waf / T(1.47)
            es = elliptical_spread(result.velocity, effective_ws_mph)
            wind_dir_rad = w.wd * pio180(T)

            ux, uy = velocity_components(
                es.head, es.back,
                wind_dir_rad,
                normal_x, normal_y
            )

            # Apply containment reduction
            eff = suppression.containment_effectiveness[ix, iy]
            state.ux[px, py] = ux * eff
            state.uy[px, py] = uy * eff
        end

        # CFL timestep
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

        if t + dt > t_stop
            dt = t_stop - t
        end

        # RK2 integration
        for idx in active_cells
            state.phi_old[idx] = state.phi[idx]
        end

        rk2_step!(state.phi, state.phi_old, state.ux, state.uy, active_cells, dt, state.cellsize, 1)
        rk2_step!(state.phi, state.phi_old, state.ux, state.uy, active_cells, dt, state.cellsize, 2)

        # Update burned cells
        cells_to_tag = CartesianIndex{2}[]
        for idx in active_cells
            px, py = idx[1], idx[2]
            ix, iy = padded_to_grid(state, px, py)

            if ix < 1 || ix > state.ncols || iy < 1 || iy > state.nrows
                continue
            end

            if state.phi[px, py] <= zero(T) && !state.burned[ix, iy]
                state.burned[ix, iy] = true
                state.time_of_arrival[ix, iy] = t + dt
                push!(cells_to_tag, idx)
            end
        end

        # Update narrow band
        nx_pad = state.ncols + 2*state.padding
        ny_pad = state.nrows + 2*state.padding
        for idx in cells_to_tag
            tag_band!(state.narrow_band, idx, nx_pad, ny_pad, state.padding)
        end
        untag_isolated!(state.narrow_band, state.phi, state.burned, state.padding)

        t += dt

        if callback !== nothing
            callback(state, t, dt, iteration)
        end
    end

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Suppression Statistics
#-----------------------------------------------------------------------------#

"""
    get_suppression_statistics(suppression::SuppressionState{T}) -> NamedTuple

Get summary statistics for suppression activities.
"""
function get_suppression_statistics(suppression::SuppressionState{T}) where {T<:AbstractFloat}
    n_resources = length(suppression.resources)
    n_available = count(r -> r.status == :available, suppression.resources)
    n_deployed = count(r -> r.status == :deployed, suppression.resources)
    n_resting = count(r -> r.status == :resting, suppression.resources)

    n_lines = length(suppression.containment_lines)
    n_contained_cells = count(suppression.contained_cells)

    mean_fatigue = if n_resources > 0
        sum(r.fatigue for r in suppression.resources) / T(n_resources)
    else
        zero(T)
    end

    return (
        total_resources = n_resources,
        available = n_available,
        deployed = n_deployed,
        resting = n_resting,
        containment_lines = n_lines,
        contained_cells = n_contained_cells,
        total_line_feet = suppression.total_line_constructed,
        total_line_miles = suppression.total_line_constructed / T(5280),
        mean_fatigue = mean_fatigue
    )
end
