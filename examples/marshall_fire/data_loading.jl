#=
Data Loading Module for Marshall Fire Calibration
==================================================

Loads and prepares data for Marshall Fire simulation:
- LANDFIRE fuel, elevation, slope, aspect data (or synthetic fallback)
- HRRR wind data (or constant wind fallback)
- Observed fire perimeter from Boulder County
=#

using Dates
using GeoJSON
using Elmfire

#-----------------------------------------------------------------------------#
#                           Constants
#-----------------------------------------------------------------------------#

# Geographic extent for Marshall Fire area (WGS84)
# Expanded to give fire room to spread from ignition point
const EXTENT_X = (-105.28, -105.10)  # Wider extent
const EXTENT_Y = (39.90, 40.00)      # Taller extent

# Ignition point (lon, lat) - where fire started
# Placing more centered to allow full spread
const IGNITION_POINT = (-105.231, 39.955)

# Time range (UTC) - Marshall Fire spread rapidly on Dec 30, 2021
const START_TIME = DateTime(2021, 12, 30, 17, 0, 0)  # 10:00 AM MT = 17:00 UTC
const STOP_TIME = DateTime(2022, 1, 1, 13, 0, 0)     # 6:00 AM MT = 13:00 UTC

# Simulation duration in minutes (~6 hours of rapid spread)
const SIMULATION_DURATION_MINUTES = 360.0

# Cell size (LANDFIRE native resolution is 30m, convert to feet for Elmfire)
const CELLSIZE_METERS = 30.0
const CELLSIZE_FEET = CELLSIZE_METERS * 3.28084  # ~98.4 ft

# Grid dimensions based on extent
const GRID_NCOLS = ceil(Int, (EXTENT_X[2] - EXTENT_X[1]) * 111000 / CELLSIZE_METERS)  # ~deg to m
const GRID_NROWS = ceil(Int, (EXTENT_Y[2] - EXTENT_Y[1]) * 111000 / CELLSIZE_METERS)

#-----------------------------------------------------------------------------#
#                           Data Structures
#-----------------------------------------------------------------------------#

"""
    MarshallFireData{T}

Container for all Marshall Fire simulation data.
"""
struct MarshallFireData{T<:AbstractFloat}
    fuel_ids::Matrix{Int}
    slope::Matrix{T}
    aspect::Matrix{T}
    elevation::Matrix{T}
    weather_series::Elmfire.WeatherTimeSeries{T}
    observed_burned::BitMatrix
    metadata::Elmfire.GeoMetadata{T}
    ignition_ix::Int
    ignition_iy::Int
end

#-----------------------------------------------------------------------------#
#                           Synthetic Data Generation
#-----------------------------------------------------------------------------#

"""
    generate_synthetic_terrain(ncols, nrows)

Generate synthetic terrain data for testing when real data is unavailable.
The Marshall Fire area has relatively flat terrain with grass fuels.
"""
function generate_synthetic_terrain(ncols::Int, nrows::Int)
    # Elevation: gentle slope from west (higher) to east (lower)
    # Marshall Fire area elevation ~5,400-5,800 ft
    elevation = zeros(Float64, ncols, nrows)
    for ix in 1:ncols
        for iy in 1:nrows
            # Base elevation around 1650m (5413 ft)
            # Slight increase toward west
            elevation[ix, iy] = 1650.0 + (ncols - ix) * 0.5 + randn() * 2
        end
    end

    # Compute slope and aspect from synthetic elevation
    slope = zeros(Float64, ncols, nrows)
    aspect = zeros(Float64, ncols, nrows)

    for ix in 2:ncols-1
        for iy in 2:nrows-1
            dzdx = (elevation[ix+1, iy] - elevation[ix-1, iy]) / (2 * CELLSIZE_METERS)
            dzdy = (elevation[ix, iy+1] - elevation[ix, iy-1]) / (2 * CELLSIZE_METERS)

            slope[ix, iy] = rad2deg(atan(sqrt(dzdx^2 + dzdy^2)))

            if abs(dzdx) > 1e-10 || abs(dzdy) > 1e-10
                aspect[ix, iy] = mod(180.0 - rad2deg(atan(dzdx, dzdy)), 360.0)
            end
        end
    end

    # Edge handling
    slope[1, :] .= slope[2, :]
    slope[end, :] .= slope[end-1, :]
    slope[:, 1] .= slope[:, 2]
    slope[:, end] .= slope[:, end-1]

    aspect[1, :] .= aspect[2, :]
    aspect[end, :] .= aspect[end-1, :]
    aspect[:, 1] .= aspect[:, 2]
    aspect[:, end] .= aspect[:, end-1]

    return (elevation=elevation, slope=slope, aspect=aspect)
end

"""
    generate_synthetic_fuel(ncols, nrows)

Generate synthetic fuel data.
Marshall Fire burned primarily through grass (fuel models 1-2).
"""
function generate_synthetic_fuel(ncols::Int, nrows::Int)
    fuel_ids = ones(Int, ncols, nrows)

    # Mostly fuel model 1 (short grass) with some model 2 (timber grass)
    for ix in 1:ncols
        for iy in 1:nrows
            r = rand()
            if r < 0.7
                fuel_ids[ix, iy] = 1  # Short grass
            elseif r < 0.9
                fuel_ids[ix, iy] = 2  # Timber grass
            else
                fuel_ids[ix, iy] = 3  # Tall grass
            end
        end
    end

    return fuel_ids
end

#-----------------------------------------------------------------------------#
#                           Weather Data
#-----------------------------------------------------------------------------#

"""
    create_constant_weather_series(wind_speed_mph, wind_direction,
                                   ncols, nrows, cellsize_ft)

Create a constant weather series for simplified simulations.
"""
function create_constant_weather_series(
    wind_speed_mph::Float64, wind_direction::Float64,
    ncols::Int, nrows::Int, cellsize_ft::Float64;
    m1::Float64 = 0.03,    # Very dry conditions (3% 1-hr moisture)
    m10::Float64 = 0.04,
    m100::Float64 = 0.06,
    mlh::Float64 = 0.30,   # Cured grass
    mlw::Float64 = 0.60
)
    weather = Elmfire.ConstantWeather{Float64}(
        wind_speed_mph, wind_direction,
        m1, m10, m100, mlh, mlw
    )

    return Elmfire.WeatherTimeSeries{Float64}(
        weather, ncols, nrows, cellsize_ft, SIMULATION_DURATION_MINUTES
    )
end

#-----------------------------------------------------------------------------#
#                           Observed Perimeter
#-----------------------------------------------------------------------------#

"""
    load_observed_perimeter(; perimeter_path::String="data/perimeter.geojson")

Load the observed fire perimeter from GeoJSON file.
"""
function load_observed_perimeter(; perimeter_path::String=joinpath(@__DIR__, "data", "perimeter.geojson"))
    if !isfile(perimeter_path)
        @warn "Perimeter file not found: $perimeter_path"
        return nothing
    end

    return GeoJSON.read(read(perimeter_path, String))
end

"""
    rasterize_perimeter(geom, metadata::Elmfire.GeoMetadata)

Convert polygon perimeter to BitMatrix matching simulation grid.
"""
function rasterize_perimeter(geom, metadata::Elmfire.GeoMetadata{T}) where {T}
    ncols = metadata.ncols
    nrows = metadata.nrows
    burned = falses(ncols, nrows)

    if geom === nothing
        return burned
    end

    # Get the geometry (handle FeatureCollection or single geometry)
    polygon = _extract_polygon(geom)

    if polygon === nothing
        @warn "Could not extract polygon from perimeter geometry"
        return burned
    end

    # Simple point-in-polygon rasterization
    for ix in 1:ncols
        for iy in 1:nrows
            x, y = Elmfire.grid_to_geo(ix, iy, metadata)
            if _point_in_polygon(x, y, polygon)
                burned[ix, iy] = true
            end
        end
    end

    return burned
end

function _extract_polygon(geom)
    # Handle GeoJSON.jl structures (FeatureCollection, Feature, Geometry)
    # GeoJSON.jl returns typed objects, not plain dicts

    # Try to get geometry from FeatureCollection
    try
        if geom isa GeoJSON.FeatureCollection
            for feature in geom
                g = GeoJSON.geometry(feature)
                if g isa GeoJSON.Polygon || g isa GeoJSON.MultiPolygon
                    return g
                end
            end
        elseif geom isa GeoJSON.Feature
            return GeoJSON.geometry(geom)
        elseif geom isa GeoJSON.Polygon || geom isa GeoJSON.MultiPolygon
            return geom
        end
    catch e
        @warn "Error extracting polygon: $e"
    end
    return nothing
end

function _point_in_polygon(x::T, y::T, polygon) where {T}
    # Handle GeoJSON.jl geometry types
    try
        if polygon isa GeoJSON.Polygon
            coords = GeoJSON.coordinates(polygon)
            return _point_in_ring(x, y, coords[1])  # First ring is outer
        elseif polygon isa GeoJSON.MultiPolygon
            coords = GeoJSON.coordinates(polygon)
            for poly_coords in coords
                if _point_in_ring(x, y, poly_coords[1])
                    return true
                end
            end
            return false
        end
    catch e
        @warn "Error in point_in_polygon: $e"
    end
    return false
end

function _point_in_ring(x::T, y::T, ring) where {T}
    n = length(ring)
    inside = false

    j = n
    for i in 1:n
        xi, yi = ring[i][1], ring[i][2]
        xj, yj = ring[j][1], ring[j][2]

        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
            inside = !inside
        end
        j = i
    end

    return inside
end

"""
    generate_synthetic_perimeter(ncols, nrows, ignition_ix, ignition_iy)

Generate a synthetic observed perimeter for testing.
Creates an elliptical shape elongated in the wind direction.
"""
function generate_synthetic_perimeter(ncols::Int, nrows::Int, ignition_ix::Int, ignition_iy::Int)
    burned = falses(ncols, nrows)

    # Create elliptical burned area elongated to the east (wind from west)
    # Marshall Fire burned ~6000 acres in ~6 hours
    # At 30m resolution, that's roughly 80,000 cells (sqrt ~ 280)
    semi_major = min(200, ncols ÷ 3)  # East-west (larger, downwind)
    semi_minor = min(80, nrows ÷ 3)   # North-south

    # Center offset slightly east of ignition
    cx = ignition_ix + semi_major ÷ 2
    cy = ignition_iy

    for ix in 1:ncols
        for iy in 1:nrows
            dx = (ix - cx) / semi_major
            dy = (iy - cy) / semi_minor
            if dx^2 + dy^2 <= 1.0
                burned[ix, iy] = true
            end
        end
    end

    return burned
end

#-----------------------------------------------------------------------------#
#                           Complete Data Loading
#-----------------------------------------------------------------------------#

"""
    load_all_data(; use_synthetic::Bool=true,
                   default_wind_speed::Float64=70.0,
                   default_wind_dir::Float64=270.0)

Load all data required for Marshall Fire simulation.

# Arguments
- `use_synthetic`: If true, generate synthetic terrain/fuel data (faster for testing)
- `default_wind_speed`: Wind speed in mph (Marshall Fire: 70-100+ mph)
- `default_wind_dir`: Wind direction FROM (270 = from west)
"""
function load_all_data(;
    use_synthetic::Bool = true,
    default_wind_speed::Float64 = 70.0,
    default_wind_dir::Float64 = 270.0
)
    println("Loading Marshall Fire simulation data...")

    ncols = GRID_NCOLS
    nrows = GRID_NROWS
    println("  Grid dimensions: $ncols x $nrows cells")

    # Build metadata
    xllcorner = EXTENT_X[1]
    yllcorner = EXTENT_Y[1]
    cellsize_deg = (EXTENT_X[2] - EXTENT_X[1]) / ncols

    # Transform for GeoMetadata
    xmin = xllcorner
    ymax = EXTENT_Y[2]
    transform = (xmin, cellsize_deg, 0.0, ymax, 0.0, -cellsize_deg)

    metadata = Elmfire.GeoMetadata{Float64}(
        ncols = ncols,
        nrows = nrows,
        cellsize = cellsize_deg,
        xllcorner = xllcorner,
        yllcorner = yllcorner,
        nodata_value = -9999.0,
        crs = "EPSG:4326",
        transform = transform
    )

    # Load or generate terrain
    println("  Generating terrain data...")
    terrain = generate_synthetic_terrain(ncols, nrows)

    # Load or generate fuel
    println("  Generating fuel data...")
    fuel_ids = generate_synthetic_fuel(ncols, nrows)

    # Create weather
    println("  Creating weather data...")
    println("    Wind: $(default_wind_speed) mph from $(default_wind_dir)°")
    weather_series = create_constant_weather_series(
        default_wind_speed, default_wind_dir,
        ncols, nrows, Float64(CELLSIZE_FEET)
    )

    # Load observed perimeter
    println("  Loading observed perimeter...")
    perimeter_geom = load_observed_perimeter()

    # Calculate ignition point in grid coordinates
    ignition_ix = floor(Int, (IGNITION_POINT[1] - xllcorner) / cellsize_deg) + 1
    ignition_iy = floor(Int, (IGNITION_POINT[2] - yllcorner) / cellsize_deg) + 1

    # Clamp to valid range
    ignition_ix = clamp(ignition_ix, 5, ncols - 5)
    ignition_iy = clamp(ignition_iy, 5, nrows - 5)

    println("  Ignition point: ($ignition_ix, $ignition_iy)")

    # Rasterize perimeter or generate synthetic
    observed_burned = if perimeter_geom !== nothing
        rasterize_perimeter(perimeter_geom, metadata)
    else
        println("  Generating synthetic observed perimeter...")
        generate_synthetic_perimeter(ncols, nrows, ignition_ix, ignition_iy)
    end

    println("  Observed burned cells: $(count(observed_burned))")

    data = MarshallFireData{Float64}(
        fuel_ids,
        terrain.slope,
        terrain.aspect,
        terrain.elevation,
        weather_series,
        observed_burned,
        metadata,
        ignition_ix,
        ignition_iy
    )

    println("Data loading complete.")
    return data
end
