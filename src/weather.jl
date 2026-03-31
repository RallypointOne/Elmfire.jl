#-----------------------------------------------------------------------------#
#                     Weather Interpolation
#-----------------------------------------------------------------------------#
#
# Implements spatially and temporally varying weather conditions.
# Supports:
# - Temporal interpolation between weather time steps
# - Spatial interpolation from coarse weather grid to fine simulation grid
#-----------------------------------------------------------------------------#

using Interpolations


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
    ) -> Tuple{Vector{T}, Vector{T}}

Create mapping from simulation grid to weather grid fractional coordinates.

Returns (xcol_weather, yrow_weather) vectors of fractional weather grid positions
for interpolation. Values are clamped to [1, ncols] and [1, nrows].
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

    # Map to fractional weather grid coordinates (cell-centered: 1.0 = center of first cell)
    xcol_weather = [
        clamp((x - weather_grid.xllcorner) / weather_grid.cellsize + T(0.5), one(T), T(weather_grid.ncols))
        for x in x_sim
    ]

    yrow_weather = [
        clamp((y - weather_grid.yllcorner) / weather_grid.cellsize + T(0.5), one(T), T(weather_grid.nrows))
        for y in y_sim
    ]

    return (xcol_weather, yrow_weather)
end


# Build an Interpolations.jl bilinear extrapolant (flat boundary) for a single field matrix.
# Singleton dimensions are padded to 2 so BSpline(Linear()) can be used uniformly.
function _make_field_itp(field::Matrix{T}) where {T<:AbstractFloat}
    nc, nr = size(field)
    padded = if nc == 1 && nr == 1
        fill(field[1,1], 2, 2)
    elseif nc == 1
        vcat(field, field)
    elseif nr == 1
        hcat(field, field)
    else
        field
    end
    extrapolate(interpolate(padded, BSpline(Linear())), Flat())
end


#-----------------------------------------------------------------------------#
#                     Weather Interpolator
#-----------------------------------------------------------------------------#

"""
    WeatherInterpolator{T<:AbstractFloat, ITP}

Handles spatial and temporal interpolation of weather data to the simulation
grid and time. Spatial interpolation uses precomputed Interpolations.jl
bilinear interpolants (one per field per time step). Wind direction is
decomposed into sin/cos components to handle 0°/360° wrap-around.
"""
struct WeatherInterpolator{T<:AbstractFloat, ITP}
    weather_series::WeatherTimeSeries{T}
    xcol_map::Vector{T}        # Fractional weather column for each sim column
    yrow_map::Vector{T}        # Fractional weather row for each sim row
    sim_ncols::Int
    sim_nrows::Int
    ws_itps::Vector{ITP}       # Wind speed interpolants (one per time step)
    wd_sin_itps::Vector{ITP}   # sin(wind direction) interpolants
    wd_cos_itps::Vector{ITP}   # cos(wind direction) interpolants
    m1_itps::Vector{ITP}
    m10_itps::Vector{ITP}
    m100_itps::Vector{ITP}
    mlh_itps::Vector{ITP}
    mlw_itps::Vector{ITP}
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
    first_grid = weather_series.grids[1]
    xcol_map, yrow_map = create_grid_mapping(
        first_grid, sim_ncols, sim_nrows,
        sim_cellsize, sim_xllcorner, sim_yllcorner
    )

    ws_itps     = [_make_field_itp(g.ws)  for g in weather_series.grids]
    wd_sin_itps = [_make_field_itp(sin.(g.wd .* pio180(T))) for g in weather_series.grids]
    wd_cos_itps = [_make_field_itp(cos.(g.wd .* pio180(T))) for g in weather_series.grids]
    m1_itps     = [_make_field_itp(g.m1)  for g in weather_series.grids]
    m10_itps    = [_make_field_itp(g.m10) for g in weather_series.grids]
    m100_itps   = [_make_field_itp(g.m100) for g in weather_series.grids]
    mlh_itps    = [_make_field_itp(g.mlh) for g in weather_series.grids]
    mlw_itps    = [_make_field_itp(g.mlw) for g in weather_series.grids]

    ITP = typeof(ws_itps[1])
    WeatherInterpolator{T, ITP}(
        weather_series, xcol_map, yrow_map, sim_ncols, sim_nrows,
        ws_itps, wd_sin_itps, wd_cos_itps,
        m1_itps, m10_itps, m100_itps, mlh_itps, mlw_itps
    )
end


"""
    get_weather_at(interp::WeatherInterpolator{T}, ix::Int, iy::Int, t::T) -> NamedTuple

Get interpolated weather values at simulation grid cell (ix, iy) and time t.

Returns named tuple with fields: ws, wd, m1, m10, m100, mlh, mlw
"""
function get_weather_at(interp::WeatherInterpolator{T}, ix::Int, iy::Int, t::T) where {T<:AbstractFloat}
    wts = interp.weather_series
    fx  = interp.xcol_map[ix]
    fy  = interp.yrow_map[iy]

    function _wd_from_sincos(s, c)
        wd = atan(s, c) / pio180(T)
        wd < zero(T) && (wd += T(360))
        wd
    end

    if length(wts.times) == 1
        ws  = interp.ws_itps[1](fx, fy)
        wd  = _wd_from_sincos(interp.wd_sin_itps[1](fx, fy), interp.wd_cos_itps[1](fx, fy))
        m1  = interp.m1_itps[1](fx, fy)
        m10 = interp.m10_itps[1](fx, fy)
        m100 = interp.m100_itps[1](fx, fy)
        mlh = interp.mlh_itps[1](fx, fy)
        mlw = interp.mlw_itps[1](fx, fy)
        return (ws=ws, wd=wd, m1=m1, m10=m10, m100=m100, mlh=mlh, mlw=mlw)
    end

    i_lo, i_hi, f = find_time_indices(wts, t)
    lo, hi = one(T) - f, f

    ws  = lo * interp.ws_itps[i_lo](fx, fy)  + hi * interp.ws_itps[i_hi](fx, fy)

    wd_lo = _wd_from_sincos(interp.wd_sin_itps[i_lo](fx, fy), interp.wd_cos_itps[i_lo](fx, fy))
    wd_hi = _wd_from_sincos(interp.wd_sin_itps[i_hi](fx, fy), interp.wd_cos_itps[i_hi](fx, fy))
    wd = interpolate_wind_direction(wd_lo, wd_hi, f)

    m1   = lo * interp.m1_itps[i_lo](fx, fy)   + hi * interp.m1_itps[i_hi](fx, fy)
    m10  = lo * interp.m10_itps[i_lo](fx, fy)  + hi * interp.m10_itps[i_hi](fx, fy)
    m100 = lo * interp.m100_itps[i_lo](fx, fy) + hi * interp.m100_itps[i_hi](fx, fy)
    mlh  = lo * interp.mlh_itps[i_lo](fx, fy)  + hi * interp.mlh_itps[i_hi](fx, fy)
    mlw  = lo * interp.mlw_itps[i_lo](fx, fy)  + hi * interp.mlw_itps[i_hi](fx, fy)

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
