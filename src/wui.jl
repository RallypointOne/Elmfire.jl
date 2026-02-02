#-----------------------------------------------------------------------------#
#                     Wildland-Urban Interface (WUI) Models
#-----------------------------------------------------------------------------#
#
# Implements building ignition and urban fire spread models including:
# - Radiative heat flux calculations
# - Building ignition probability models
# - Hamada urban fire spread model
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Building Structures
#-----------------------------------------------------------------------------#

"""
    WUIBuilding{T<:AbstractFloat}

A building in the WUI that can be affected by wildfire.
"""
struct WUIBuilding{T<:AbstractFloat}
    id::Int
    ix::Int                    # Grid x coordinate
    iy::Int                    # Grid y coordinate
    construction_type::Symbol  # :wood, :masonry, :mixed
    combustible_fraction::T    # Fraction of exterior that is combustible
    ignition_temperature::T    # Critical ignition temperature (°C)
end

Base.eltype(::WUIBuilding{T}) where {T} = T

function WUIBuilding{T}(
    id::Int, ix::Int, iy::Int;
    construction_type::Symbol = :wood,
    combustible_fraction::T = T(0.7),
    ignition_temperature::T = T(300)
) where {T<:AbstractFloat}
    WUIBuilding{T}(id, ix, iy, construction_type, combustible_fraction, ignition_temperature)
end


"""
    WUIGrid{T<:AbstractFloat}

Grid of buildings for WUI fire simulation.
"""
struct WUIGrid{T<:AbstractFloat}
    buildings::Vector{WUIBuilding{T}}
    building_map::BitMatrix        # true where buildings exist
    building_ids::Matrix{Int}       # Building ID at each cell (0 if none)
    building_density::Matrix{T}    # Buildings per hectare
    ignited::BitVector             # Whether each building has ignited
    ignition_time::Vector{T}       # Time of ignition for each building
end

Base.eltype(::WUIGrid{T}) where {T} = T


"""
    WUIGrid{T}(buildings::Vector{WUIBuilding{T}}, ncols::Int, nrows::Int)

Create a WUI grid from a vector of buildings.
"""
function WUIGrid{T}(buildings::Vector{WUIBuilding{T}}, ncols::Int, nrows::Int) where {T<:AbstractFloat}
    building_map = falses(ncols, nrows)
    building_ids = zeros(Int, ncols, nrows)
    building_density = zeros(T, ncols, nrows)

    for b in buildings
        if 1 <= b.ix <= ncols && 1 <= b.iy <= nrows
            building_map[b.ix, b.iy] = true
            building_ids[b.ix, b.iy] = b.id
        end
    end

    ignited = falses(length(buildings))
    ignition_time = fill(-one(T), length(buildings))

    WUIGrid{T}(buildings, building_map, building_ids, building_density, ignited, ignition_time)
end

"""
    WUIGrid{T}(ncols::Int, nrows::Int)

Create an empty WUI grid.
"""
function WUIGrid{T}(ncols::Int, nrows::Int) where {T<:AbstractFloat}
    WUIGrid{T}(
        WUIBuilding{T}[],
        falses(ncols, nrows),
        zeros(Int, ncols, nrows),
        zeros(T, ncols, nrows),
        BitVector(),
        T[]
    )
end


#-----------------------------------------------------------------------------#
#                     Hamada Urban Fire Spread Parameters
#-----------------------------------------------------------------------------#

"""
    HamadaParameters{T<:AbstractFloat}

Parameters for the Hamada urban fire spread model.

The Hamada model describes fire spread between buildings based on
separation distance, wind conditions, and building characteristics.
"""
struct HamadaParameters{T<:AbstractFloat}
    critical_separation::T     # Distance beyond which no ignition (m)
    wind_spread_factor::T      # Wind enhancement factor
    ember_generation_rate::T   # Embers per unit fireline intensity
    base_spread_rate::T        # Base inter-building spread rate (m/min)
end

Base.eltype(::HamadaParameters{T}) where {T} = T

function HamadaParameters{T}(;
    critical_separation::T = T(10),
    wind_spread_factor::T = T(1.5),
    ember_generation_rate::T = T(0.1),
    base_spread_rate::T = T(1.0)
) where {T<:AbstractFloat}
    HamadaParameters{T}(critical_separation, wind_spread_factor, ember_generation_rate, base_spread_rate)
end

HamadaParameters() = HamadaParameters{Float64}()


#-----------------------------------------------------------------------------#
#                     Building Ignition Results
#-----------------------------------------------------------------------------#

"""
    BuildingIgnitionResult{T<:AbstractFloat}

Result of a building ignition calculation.
"""
struct BuildingIgnitionResult{T<:AbstractFloat}
    building_id::Int
    ignition_time::T
    ignition_source::Symbol    # :radiation, :ember, :flame_contact, :building_spread
    ignition_probability::T
end

Base.eltype(::BuildingIgnitionResult{T}) where {T} = T


#-----------------------------------------------------------------------------#
#                     Radiative Heat Flux Calculations
#-----------------------------------------------------------------------------#

"""
    compute_radiative_heat_flux(flin::T, distance::T, flame_length::T) -> T

Compute radiative heat flux (kW/m²) at a given distance from the fire.

Based on the solid flame model with view factor calculations.

# Arguments
- `flin`: Fireline intensity (kW/m)
- `distance`: Distance from fire front (m)
- `flame_length`: Flame length (m)

# Returns
- Heat flux in kW/m²
"""
function compute_radiative_heat_flux(flin::T, distance::T, flame_length::T) where {T<:AbstractFloat}
    if flin <= zero(T) || distance <= zero(T) || flame_length <= zero(T)
        return zero(T)
    end

    # Stefan-Boltzmann constant (kW/m²/K⁴)
    SIGMA = T(5.67e-11)

    # Effective flame temperature (K)
    # Empirical relationship: T_flame ≈ 1200 - 1500 K
    T_flame = T(1200)

    # Flame emissivity (typically 0.7-0.9 for vegetation fires)
    emissivity = T(0.8)

    # Compute view factor using rectangular flame approximation
    view_factor = compute_view_factor(flame_length, distance)

    # Radiative heat flux (Stefan-Boltzmann)
    q_rad = emissivity * SIGMA * T_flame^4 * view_factor

    # Alternative empirical formula based on fireline intensity
    # q_rad = flin / (4 * pi * distance^2)
    # This is a simplified inverse-square law

    # Use empirical correlation for practical applications
    # Sullivan et al. (2003): q = 0.3 * I / d  for d in meters, I in kW/m
    q_empirical = T(0.3) * flin / max(distance, T(0.1))

    # Return minimum of theoretical and empirical (conservative)
    return min(q_rad, q_empirical)
end


"""
    compute_view_factor(flame_height::T, distance::T) -> T

Compute the view factor between a vertical rectangular flame and a point.

Uses the approximation for a vertical radiating surface.

# Arguments
- `flame_height`: Height of the flame (m)
- `distance`: Horizontal distance to the target point (m)
"""
function compute_view_factor(flame_height::T, distance::T) where {T<:AbstractFloat}
    if flame_height <= zero(T) || distance <= zero(T)
        return zero(T)
    end

    # Aspect ratio
    H = flame_height
    D = distance

    # View factor for vertical rectangle to point at distance D
    # Simplified approximation
    X = H / D

    # Approximation for perpendicular view factor
    vf = X / (T(2) * pi_val(T) * sqrt(one(T) + X^2))

    # Clamp to valid range
    return clamp(vf, zero(T), one(T))
end


#-----------------------------------------------------------------------------#
#                     Building Ignition Probability
#-----------------------------------------------------------------------------#

"""
    building_ignition_probability(
        building::WUIBuilding{T},
        heat_flux::T,
        exposure_time::T
    ) -> T

Calculate the probability of building ignition given heat flux and exposure time.

Uses a dose-response model based on heat flux accumulation.

# Arguments
- `building`: Building properties
- `heat_flux`: Incident heat flux (kW/m²)
- `exposure_time`: Duration of exposure (minutes)

# Returns
- Ignition probability (0-1)
"""
function building_ignition_probability(
    building::WUIBuilding{T},
    heat_flux::T,
    exposure_time::T
) where {T<:AbstractFloat}
    if heat_flux <= zero(T) || exposure_time <= zero(T)
        return zero(T)
    end

    # Critical heat flux thresholds (kW/m²) by construction type
    # Based on NIST and fire research literature
    q_crit = if building.construction_type == :wood
        T(12.5)   # Wood ignition ~12.5 kW/m²
    elseif building.construction_type == :masonry
        T(25.0)   # Masonry is more resistant
    else  # :mixed
        T(15.0)
    end

    # Adjust for combustible fraction
    q_effective = heat_flux * building.combustible_fraction

    # If below critical threshold, no ignition possible
    if q_effective < q_crit * T(0.5)
        return zero(T)
    end

    # Dose-response model
    # Time to ignition decreases exponentially with heat flux above critical
    # t_ig = A * (q - q_crit)^(-n)  where n ≈ 1.6

    # Probability based on accumulated dose
    dose = q_effective * exposure_time

    # Critical dose for ignition (empirical)
    dose_crit = if building.construction_type == :wood
        T(50)     # kW·min/m²
    elseif building.construction_type == :masonry
        T(150)
    else
        T(80)
    end

    # Probability using logistic function
    # P(ignition) = 1 / (1 + exp(-k*(dose - dose_crit)))
    k = T(0.05)  # Slope parameter
    x = (dose - dose_crit) / dose_crit
    prob = one(T) / (one(T) + exp(-T(4) * x))

    return clamp(prob, zero(T), one(T))
end


#-----------------------------------------------------------------------------#
#                     Hamada Urban Fire Spread
#-----------------------------------------------------------------------------#

"""
    hamada_spread_probability(
        source::WUIBuilding{T},
        target::WUIBuilding{T},
        params::HamadaParameters{T},
        wind_speed::T,
        wind_direction::T,
        cellsize::T
    ) -> T

Calculate the probability of fire spreading between buildings using the Hamada model.

# Arguments
- `source`: Building that is on fire
- `target`: Potential target building
- `params`: Hamada model parameters
- `wind_speed`: Wind speed (mph)
- `wind_direction`: Wind direction (degrees, FROM)
- `cellsize`: Grid cell size (ft)
"""
function hamada_spread_probability(
    source::WUIBuilding{T},
    target::WUIBuilding{T},
    params::HamadaParameters{T},
    wind_speed::T,
    wind_direction::T,
    cellsize::T
) where {T<:AbstractFloat}
    # Calculate distance between buildings (meters)
    dx = T(target.ix - source.ix) * cellsize * ft_to_m(T)
    dy = T(target.iy - source.iy) * cellsize * ft_to_m(T)
    distance = sqrt(dx^2 + dy^2)

    # If too far, no spread possible
    if distance > params.critical_separation
        return zero(T)
    end

    # Direction from source to target (mathematical angle)
    if distance < T(0.1)
        return zero(T)  # Same building
    end

    spread_angle = atan(dy, dx) / pio180(T)  # degrees

    # Convert wind direction to spread enhancement
    # Wind FROM wind_direction means wind vector points in opposite direction
    wind_vector_angle = wind_direction + T(180)
    if wind_vector_angle >= T(360)
        wind_vector_angle -= T(360)
    end

    # Angle between wind vector and spread direction
    angle_diff = abs(spread_angle - wind_vector_angle)
    if angle_diff > T(180)
        angle_diff = T(360) - angle_diff
    end

    # Wind enhancement: maximum when spreading downwind
    wind_factor = one(T) + (params.wind_spread_factor - one(T)) * cos(angle_diff * pio180(T))
    wind_factor = max(wind_factor, T(0.5))  # Minimum factor

    # Base probability decreases with distance
    # P_base = exp(-distance / critical_separation)
    p_base = exp(-distance / params.critical_separation)

    # Adjust for construction types
    combustibility_factor = source.combustible_fraction * target.combustible_fraction

    # Final probability
    prob = p_base * wind_factor * combustibility_factor

    return clamp(prob, zero(T), one(T))
end


#-----------------------------------------------------------------------------#
#                     WUI State Update
#-----------------------------------------------------------------------------#

"""
    update_wui_state!(
        wui_grid::WUIGrid{T},
        fire_state::FireState{T},
        weather::ConstantWeather{T},
        t::T,
        dt::T,
        rng::AbstractRNG
    ) -> Vector{BuildingIgnitionResult{T}}

Update the WUI grid based on fire state and check for building ignitions.

# Arguments
- `wui_grid`: WUI grid (modified in place)
- `fire_state`: Current fire simulation state
- `weather`: Weather conditions
- `t`: Current time (minutes)
- `dt`: Time step (minutes)
- `rng`: Random number generator

# Returns
- Vector of new building ignitions during this time step
"""
function update_wui_state!(
    wui_grid::WUIGrid{T},
    fire_state::FireState{T},
    weather::ConstantWeather{T},
    t::T,
    dt::T,
    rng::AbstractRNG
) where {T<:AbstractFloat}
    results = BuildingIgnitionResult{T}[]
    cellsize_m = fire_state.cellsize * ft_to_m(T)

    for (i, building) in enumerate(wui_grid.buildings)
        # Skip if already ignited
        if wui_grid.ignited[i]
            continue
        end

        ix, iy = building.ix, building.iy

        # Check bounds
        if ix < 1 || ix > fire_state.ncols || iy < 1 || iy > fire_state.nrows
            continue
        end

        # Check for direct flame contact (cell is burning)
        if fire_state.burned[ix, iy]
            wui_grid.ignited[i] = true
            wui_grid.ignition_time[i] = t
            push!(results, BuildingIgnitionResult{T}(
                building.id, t, :flame_contact, one(T)
            ))
            continue
        end

        # Check for radiative ignition from nearby burning cells
        max_heat_flux = zero(T)
        for dx in -5:5
            for dy in -5:5
                if dx == 0 && dy == 0
                    continue
                end

                nx, ny = ix + dx, iy + dy
                if nx < 1 || nx > fire_state.ncols || ny < 1 || ny > fire_state.nrows
                    continue
                end

                if fire_state.burned[nx, ny] && fire_state.fireline_intensity[nx, ny] > zero(T)
                    # Calculate distance
                    distance = sqrt(T(dx^2 + dy^2)) * cellsize_m

                    # Get heat flux from this burning cell
                    flin = fire_state.fireline_intensity[nx, ny]
                    flame_len = fire_state.flame_length[nx, ny] * ft_to_m(T)

                    heat_flux = compute_radiative_heat_flux(flin, distance, flame_len)
                    max_heat_flux = max(max_heat_flux, heat_flux)
                end
            end
        end

        # Calculate ignition probability from radiation
        if max_heat_flux > zero(T)
            prob = building_ignition_probability(building, max_heat_flux, dt)

            if rand(rng) < prob
                wui_grid.ignited[i] = true
                wui_grid.ignition_time[i] = t
                push!(results, BuildingIgnitionResult{T}(
                    building.id, t, :radiation, prob
                ))
                continue
            end
        end

        # Check for building-to-building spread (Hamada model)
        params = HamadaParameters{T}()
        for (j, other_building) in enumerate(wui_grid.buildings)
            if !wui_grid.ignited[j]
                continue
            end

            # Only consider buildings that ignited recently (within 10 minutes)
            if t - wui_grid.ignition_time[j] > T(10)
                continue
            end

            prob = hamada_spread_probability(
                other_building, building, params,
                weather.wind_speed_20ft, weather.wind_direction,
                fire_state.cellsize
            )

            if rand(rng) < prob * dt
                wui_grid.ignited[i] = true
                wui_grid.ignition_time[i] = t
                push!(results, BuildingIgnitionResult{T}(
                    building.id, t, :building_spread, prob
                ))
                break
            end
        end
    end

    return results
end


#-----------------------------------------------------------------------------#
#                     WUI Statistics
#-----------------------------------------------------------------------------#

"""
    get_wui_statistics(wui_grid::WUIGrid{T}) -> NamedTuple

Get summary statistics for the WUI simulation.
"""
function get_wui_statistics(wui_grid::WUIGrid{T}) where {T<:AbstractFloat}
    n_buildings = length(wui_grid.buildings)
    n_ignited = count(wui_grid.ignited)

    # Ignition times for ignited buildings
    ignition_times = [wui_grid.ignition_time[i] for i in 1:n_buildings if wui_grid.ignited[i]]

    mean_ignition_time = isempty(ignition_times) ? zero(T) : sum(ignition_times) / T(length(ignition_times))
    first_ignition_time = isempty(ignition_times) ? -one(T) : minimum(ignition_times)
    last_ignition_time = isempty(ignition_times) ? -one(T) : maximum(ignition_times)

    # Count by construction type
    n_wood_ignited = count(i -> wui_grid.ignited[i] && wui_grid.buildings[i].construction_type == :wood, 1:n_buildings)
    n_masonry_ignited = count(i -> wui_grid.ignited[i] && wui_grid.buildings[i].construction_type == :masonry, 1:n_buildings)

    return (
        total_buildings = n_buildings,
        ignited_buildings = n_ignited,
        ignition_fraction = n_buildings > 0 ? T(n_ignited) / T(n_buildings) : zero(T),
        mean_ignition_time = mean_ignition_time,
        first_ignition_time = first_ignition_time,
        last_ignition_time = last_ignition_time,
        wood_ignited = n_wood_ignited,
        masonry_ignited = n_masonry_ignited
    )
end


#-----------------------------------------------------------------------------#
#                     Building Placement Utilities
#-----------------------------------------------------------------------------#

"""
    create_building_grid(
        ncols::Int, nrows::Int,
        building_spacing::Int,
        building_footprint::Int,
        origin_x::Int, origin_y::Int;
        construction_type::Symbol = :wood
    ) -> Vector{WUIBuilding{T}}

Create a regular grid of buildings for WUI simulation.

# Arguments
- `ncols, nrows`: Grid dimensions
- `building_spacing`: Spacing between buildings (cells)
- `building_footprint`: Building footprint size (cells)
- `origin_x, origin_y`: Starting position for building grid
- `construction_type`: Default construction type for buildings
"""
function create_building_grid(
    ::Type{T},
    ncols::Int, nrows::Int,
    building_spacing::Int,
    building_footprint::Int,
    origin_x::Int, origin_y::Int;
    construction_type::Symbol = :wood
) where {T<:AbstractFloat}
    buildings = WUIBuilding{T}[]
    id = 0

    for ix in origin_x:building_spacing:ncols-building_footprint
        for iy in origin_y:building_spacing:nrows-building_footprint
            # Place building at center of footprint
            center_x = ix + div(building_footprint, 2)
            center_y = iy + div(building_footprint, 2)

            id += 1
            push!(buildings, WUIBuilding{T}(
                id, center_x, center_y;
                construction_type = construction_type
            ))
        end
    end

    return buildings
end

create_building_grid(ncols::Int, nrows::Int, args...; kwargs...) =
    create_building_grid(Float64, ncols, nrows, args...; kwargs...)
