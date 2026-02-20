#-----------------------------------------------------------------------------#
#                     Weather Interpolation
#-----------------------------------------------------------------------------#
#
# Implements spatially and temporally varying weather conditions.
# Supports:
# - Temporal interpolation between weather time steps
# - Spatial interpolation from coarse weather grid to fine simulation grid
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Constant Weather (Uniform Conditions)
#-----------------------------------------------------------------------------#

"""
    ConstantWeather{T<:AbstractFloat}

Constant (spatially and temporally uniform) weather conditions.
"""
struct ConstantWeather{T<:AbstractFloat}
    wind_speed_20ft::T      # 20-ft wind speed (mph)
    wind_direction::T       # Wind direction (degrees, meteorological: FROM)
    M1::T                   # 1-hr dead fuel moisture (fraction)
    M10::T                  # 10-hr dead fuel moisture (fraction)
    M100::T                 # 100-hr dead fuel moisture (fraction)
    MLH::T                  # Live herbaceous moisture (fraction)
    MLW::T                  # Live woody moisture (fraction)
end

Base.eltype(::ConstantWeather{T}) where {T} = T

function ConstantWeather{T}(;
    wind_speed_mph::T = T(10),
    wind_direction::T = zero(T),
    M1::T = T(0.06),
    M10::T = T(0.08),
    M100::T = T(0.10),
    MLH::T = T(0.60),
    MLW::T = T(0.90)
) where {T<:AbstractFloat}
    ConstantWeather{T}(wind_speed_mph, wind_direction, M1, M10, M100, MLH, MLW)
end

# Default to Float64 for backwards compatibility
function ConstantWeather(;
    wind_speed_mph::Float64 = 10.0,
    wind_direction::Float64 = 0.0,
    M1::Float64 = 0.06,
    M10::Float64 = 0.08,
    M100::Float64 = 0.10,
    MLH::Float64 = 0.60,
    MLW::Float64 = 0.90
)
    ConstantWeather{Float64}(wind_speed_mph, wind_direction, M1, M10, M100, MLH, MLW)
end


#-----------------------------------------------------------------------------#
#                     Weather Grid
#-----------------------------------------------------------------------------#

"""
    WeatherGrid{T<:AbstractFloat}

A grid of weather values at a single time.
"""
struct WeatherGrid{T<:AbstractFloat}
    ws::Matrix{T}      # Wind speed (mph)
    wd::Matrix{T}      # Wind direction (degrees, FROM)
    m1::Matrix{T}      # 1-hour dead fuel moisture (fraction)
    m10::Matrix{T}     # 10-hour dead fuel moisture (fraction)
    m100::Matrix{T}    # 100-hour dead fuel moisture (fraction)
    mlh::Matrix{T}     # Live herbaceous moisture (fraction)
    mlw::Matrix{T}     # Live woody moisture (fraction)
    ncols::Int         # Number of columns
    nrows::Int         # Number of rows
    cellsize::T        # Cell size (m)
    xllcorner::T       # X coordinate of lower-left corner
    yllcorner::T       # Y coordinate of lower-left corner
end

Base.eltype(::WeatherGrid{T}) where {T} = T


"""
    WeatherGrid{T}(ncols, nrows, cellsize; xllcorner=zero(T), yllcorner=zero(T))

Create an empty weather grid.
"""
function WeatherGrid{T}(
    ncols::Int, nrows::Int, cellsize::T;
    xllcorner::T = zero(T),
    yllcorner::T = zero(T)
) where {T<:AbstractFloat}
    WeatherGrid{T}(
        zeros(T, ncols, nrows),  # ws
        zeros(T, ncols, nrows),  # wd
        fill(T(0.06), ncols, nrows),  # m1
        fill(T(0.08), ncols, nrows),  # m10
        fill(T(0.10), ncols, nrows),  # m100
        fill(T(0.60), ncols, nrows),  # mlh
        fill(T(0.90), ncols, nrows),  # mlw
        ncols, nrows, cellsize,
        xllcorner, yllcorner
    )
end


"""
    WeatherGrid{T}(weather::ConstantWeather{T}, ncols, nrows, cellsize)

Create a uniform weather grid from constant weather conditions.
"""
function WeatherGrid{T}(
    weather::ConstantWeather{T},
    ncols::Int, nrows::Int, cellsize::T;
    xllcorner::T = zero(T),
    yllcorner::T = zero(T)
) where {T<:AbstractFloat}
    WeatherGrid{T}(
        fill(weather.wind_speed_20ft, ncols, nrows),
        fill(weather.wind_direction, ncols, nrows),
        fill(weather.M1, ncols, nrows),
        fill(weather.M10, ncols, nrows),
        fill(weather.M100, ncols, nrows),
        fill(weather.MLH, ncols, nrows),
        fill(weather.MLW, ncols, nrows),
        ncols, nrows, cellsize,
        xllcorner, yllcorner
    )
end


#-----------------------------------------------------------------------------#
#                     Weather Time Series
#-----------------------------------------------------------------------------#

"""
    WeatherTimeSeries{T<:AbstractFloat}

A time series of weather grids.
"""
struct WeatherTimeSeries{T<:AbstractFloat}
    grids::Vector{WeatherGrid{T}}  # Weather grids at each time
    times::Vector{T}               # Times (minutes from start)
    dt::T                          # Time step between grids (minutes)
end

Base.eltype(::WeatherTimeSeries{T}) where {T} = T


"""
    WeatherTimeSeries{T}(grids, times)

Create a weather time series from a vector of grids and times.
"""
function WeatherTimeSeries{T}(grids::Vector{WeatherGrid{T}}, times::Vector{T}) where {T<:AbstractFloat}
    @assert length(grids) == length(times) "Number of grids must match number of times"
    @assert length(times) >= 1 "Must have at least one time point"

    dt = if length(times) > 1
        times[2] - times[1]
    else
        one(T)
    end

    WeatherTimeSeries{T}(grids, times, dt)
end


"""
    WeatherTimeSeries{T}(weather::ConstantWeather{T}, ncols, nrows, cellsize, duration)

Create a constant weather time series.
"""
function WeatherTimeSeries{T}(
    weather::ConstantWeather{T},
    ncols::Int, nrows::Int, cellsize::T,
    duration::T;
    xllcorner::T = zero(T),
    yllcorner::T = zero(T)
) where {T<:AbstractFloat}
    grid = WeatherGrid{T}(weather, ncols, nrows, cellsize;
        xllcorner=xllcorner, yllcorner=yllcorner)
    WeatherTimeSeries{T}([grid], [zero(T)])
end


#-----------------------------------------------------------------------------#
#                     Temporal Interpolation
#-----------------------------------------------------------------------------#

"""
    find_time_indices(wts::WeatherTimeSeries{T}, t::T) -> Tuple{Int, Int, T}

Find the indices and interpolation weight for time t.

Returns (i_lo, i_hi, f) where the interpolated value is:
value = (1-f) * grids[i_lo] + f * grids[i_hi]
"""
function find_time_indices(wts::WeatherTimeSeries{T}, t::T) where {T<:AbstractFloat}
    n = length(wts.times)

    if n == 1
        return (1, 1, zero(T))
    end

    # Find bracketing indices
    i_lo = 1
    for i in 1:n-1
        if wts.times[i] <= t < wts.times[i+1]
            i_lo = i
            break
        elseif i == n-1 && t >= wts.times[n]
            i_lo = n
        end
    end

    i_hi = min(i_lo + 1, n)

    # Interpolation weight
    if i_lo == i_hi
        f = zero(T)
    else
        f = (t - wts.times[i_lo]) / (wts.times[i_hi] - wts.times[i_lo])
        f = clamp(f, zero(T), one(T))
    end

    return (i_lo, i_hi, f)
end


"""
    interpolate_wind_direction(wd1::T, wd2::T, f::T) -> T

Interpolate wind direction, handling the 0°/360° wrap-around.
"""
function interpolate_wind_direction(wd1::T, wd2::T, f::T) where {T<:AbstractFloat}
    # Convert to radians
    r1 = wd1 * pio180(T)
    r2 = wd2 * pio180(T)

    # Compute unit vectors
    x1, y1 = sin(r1), cos(r1)
    x2, y2 = sin(r2), cos(r2)

    # Interpolate unit vectors
    x = (one(T) - f) * x1 + f * x2
    y = (one(T) - f) * y1 + f * y2

    # Convert back to degrees
    wd = atan(x, y) / pio180(T)
    if wd < zero(T)
        wd += T(360)
    end

    return wd
end


#-----------------------------------------------------------------------------#
#                     Spatial Interpolation
#-----------------------------------------------------------------------------#

"""
    create_grid_mapping(
        weather_grid::WeatherGrid{T},
        sim_ncols::Int, sim_nrows::Int,
        sim_cellsize::T,
        sim_xllcorner::T,
        sim_yllcorner::T
    ) -> Tuple{Vector{Int}, Vector{Int}}

Create mapping from simulation grid to weather grid indices.

Returns (icol_weather, irow_weather) vectors such that
simulation cell (ix, iy) maps to weather cell (icol_weather[ix], irow_weather[iy]).
"""
function create_grid_mapping(
    weather_grid::WeatherGrid{T},
    sim_ncols::Int, sim_nrows::Int,
    sim_cellsize::T,
    sim_xllcorner::T,
    sim_yllcorner::T
) where {T<:AbstractFloat}
    # X coordinates of simulation grid cell centers
    x_sim = [sim_xllcorner + (T(ix) - T(0.5)) * sim_cellsize * ft_to_m(T) for ix in 1:sim_ncols]

    # Y coordinates of simulation grid cell centers
    y_sim = [sim_yllcorner + (T(iy) - T(0.5)) * sim_cellsize * ft_to_m(T) for iy in 1:sim_nrows]

    # Map to weather grid indices
    icol_weather = [
        clamp(ceil(Int, (x - weather_grid.xllcorner) / weather_grid.cellsize), 1, weather_grid.ncols)
        for x in x_sim
    ]

    irow_weather = [
        clamp(ceil(Int, (y - weather_grid.yllcorner) / weather_grid.cellsize), 1, weather_grid.nrows)
        for y in y_sim
    ]

    return (icol_weather, irow_weather)
end


#-----------------------------------------------------------------------------#
#                     Weather Interpolator
#-----------------------------------------------------------------------------#

"""
    WeatherInterpolator{T<:AbstractFloat}

Handles interpolation of weather data to simulation grid and time.
"""
struct WeatherInterpolator{T<:AbstractFloat}
    weather_series::WeatherTimeSeries{T}
    icol_map::Vector{Int}    # Mapping from sim column to weather column
    irow_map::Vector{Int}    # Mapping from sim row to weather row
    sim_ncols::Int
    sim_nrows::Int
end

Base.eltype(::WeatherInterpolator{T}) where {T} = T


"""
    WeatherInterpolator(
        weather_series::WeatherTimeSeries{T},
        sim_ncols::Int, sim_nrows::Int,
        sim_cellsize::T,
        sim_xllcorner::T = zero(T),
        sim_yllcorner::T = zero(T)
    )

Create a weather interpolator for the given simulation grid.
"""
function WeatherInterpolator(
    weather_series::WeatherTimeSeries{T},
    sim_ncols::Int, sim_nrows::Int,
    sim_cellsize::T,
    sim_xllcorner::T = zero(T),
    sim_yllcorner::T = zero(T)
) where {T<:AbstractFloat}
    # Create mapping using first weather grid
    first_grid = weather_series.grids[1]
    icol_map, irow_map = create_grid_mapping(
        first_grid,
        sim_ncols, sim_nrows,
        sim_cellsize,
        sim_xllcorner, sim_yllcorner
    )

    WeatherInterpolator{T}(weather_series, icol_map, irow_map, sim_ncols, sim_nrows)
end


"""
    get_weather_at(interp::WeatherInterpolator{T}, ix::Int, iy::Int, t::T) -> NamedTuple

Get interpolated weather values at simulation grid cell (ix, iy) and time t.

Returns named tuple with fields: ws, wd, m1, m10, m100, mlh, mlw
"""
function get_weather_at(interp::WeatherInterpolator{T}, ix::Int, iy::Int, t::T) where {T<:AbstractFloat}
    wts = interp.weather_series

    # Get weather grid indices
    wix = interp.icol_map[ix]
    wiy = interp.irow_map[iy]

    # Short-circuit for single time step (constant weather)
    if length(wts.times) == 1
        g = wts.grids[1]
        return (ws=g.ws[wix, wiy], wd=g.wd[wix, wiy], m1=g.m1[wix, wiy],
                m10=g.m10[wix, wiy], m100=g.m100[wix, wiy], mlh=g.mlh[wix, wiy], mlw=g.mlw[wix, wiy])
    end

    # Get time indices and interpolation weight
    i_lo, i_hi, f = find_time_indices(wts, t)

    g_lo = wts.grids[i_lo]
    g_hi = wts.grids[i_hi]

    # Interpolate values
    ws = (one(T) - f) * g_lo.ws[wix, wiy] + f * g_hi.ws[wix, wiy]
    wd = interpolate_wind_direction(g_lo.wd[wix, wiy], g_hi.wd[wix, wiy], f)
    m1 = (one(T) - f) * g_lo.m1[wix, wiy] + f * g_hi.m1[wix, wiy]
    m10 = (one(T) - f) * g_lo.m10[wix, wiy] + f * g_hi.m10[wix, wiy]
    m100 = (one(T) - f) * g_lo.m100[wix, wiy] + f * g_hi.m100[wix, wiy]
    mlh = (one(T) - f) * g_lo.mlh[wix, wiy] + f * g_hi.mlh[wix, wiy]
    mlw = (one(T) - f) * g_lo.mlw[wix, wiy] + f * g_hi.mlw[wix, wiy]

    return (ws=ws, wd=wd, m1=m1, m10=m10, m100=m100, mlh=mlh, mlw=mlw)
end


#-----------------------------------------------------------------------------#
#                     Convenience Functions
#-----------------------------------------------------------------------------#

"""
    create_constant_interpolator(
        weather::ConstantWeather{T},
        sim_ncols::Int, sim_nrows::Int,
        sim_cellsize::T
    ) -> WeatherInterpolator{T}

Create a weather interpolator for constant weather conditions.
"""
function create_constant_interpolator(
    weather::ConstantWeather{T},
    sim_ncols::Int, sim_nrows::Int,
    sim_cellsize::T
) where {T<:AbstractFloat}
    # Create a 1x1 weather grid
    wgrid = WeatherGrid{T}(weather, 1, 1, T(1e6))  # Large cell to cover everything
    wts = WeatherTimeSeries{T}([wgrid], [zero(T)])

    WeatherInterpolator(wts, sim_ncols, sim_nrows, sim_cellsize)
end
