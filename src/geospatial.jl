#-----------------------------------------------------------------------------#
#                     Geospatial I/O
#-----------------------------------------------------------------------------#
#
# Provides GeoTIFF reading/writing, raster processing, and landscape data
# management for ELMFIRE simulations.
#
# Uses ArchGDAL for low-level raster operations and Rasters.jl for
# high-level geospatial data structures.
#-----------------------------------------------------------------------------#

using ArchGDAL
using Rasters
using GeoInterface
using GeoJSON

#-----------------------------------------------------------------------------#
#                     Geospatial Metadata
#-----------------------------------------------------------------------------#

"""
    GeoMetadata{T<:AbstractFloat}

Geospatial metadata for a raster dataset.
"""
struct GeoMetadata{T<:AbstractFloat}
    ncols::Int
    nrows::Int
    cellsize::T
    xllcorner::T
    yllcorner::T
    nodata_value::T
    crs::String
    transform::NTuple{6,T}  # GDAL GeoTransform: (xmin, xres, 0, ymax, 0, -yres)
end

Base.eltype(::GeoMetadata{T}) where {T} = T

function GeoMetadata{T}(;
    ncols::Int,
    nrows::Int,
    cellsize::T,
    xllcorner::T = zero(T),
    yllcorner::T = zero(T),
    nodata_value::T = T(-9999),
    crs::String = "EPSG:4326",
    transform::Union{Nothing, NTuple{6,T}} = nothing
) where {T<:AbstractFloat}
    # Default transform from corner coordinates and cell size
    xform = if transform === nothing
        (xllcorner, cellsize, zero(T), yllcorner + cellsize * nrows, zero(T), -cellsize)
    else
        transform
    end

    GeoMetadata{T}(ncols, nrows, cellsize, xllcorner, yllcorner, nodata_value, crs, xform)
end


#-----------------------------------------------------------------------------#
#                     GeoRaster
#-----------------------------------------------------------------------------#

"""
    GeoRaster{T<:AbstractFloat}

A raster dataset with geospatial metadata.
"""
struct GeoRaster{T<:AbstractFloat}
    data::Matrix{T}
    metadata::GeoMetadata{T}
end

Base.eltype(::GeoRaster{T}) where {T} = T
Base.size(gr::GeoRaster) = size(gr.data)


#-----------------------------------------------------------------------------#
#                     LandscapeData
#-----------------------------------------------------------------------------#

"""
    LandscapeData{T<:AbstractFloat}

Complete landscape data for fire simulation including fuel, topography, and canopy.
"""
struct LandscapeData{T<:AbstractFloat}
    fuel_ids::Matrix{Int}
    slope::Matrix{T}
    aspect::Matrix{T}
    elevation::Matrix{T}
    canopy::Union{Nothing, CanopyGrid{T}}
    metadata::GeoMetadata{T}
end

Base.eltype(::LandscapeData{T}) where {T} = T


#-----------------------------------------------------------------------------#
#                     Reading Functions
#-----------------------------------------------------------------------------#

"""
    read_geotiff(::Type{T}, path::String) -> GeoRaster{T}

Read a GeoTIFF file and return a GeoRaster with the specified precision.

# Arguments
- `T`: Element type for the raster data (e.g., Float64, Float32)
- `path`: Path to the GeoTIFF file
"""
function read_geotiff(::Type{T}, path::String) where {T<:AbstractFloat}
    ArchGDAL.read(path) do dataset
        band = ArchGDAL.getband(dataset, 1)

        # Read data and convert to desired precision
        data_raw = ArchGDAL.read(band)
        data = T.(data_raw)

        # Get dimensions
        ncols = ArchGDAL.width(dataset)
        nrows = ArchGDAL.height(dataset)

        # Get geotransform
        gt = ArchGDAL.getgeotransform(dataset)
        xmin = T(gt[1])
        xres = T(gt[2])
        ymax = T(gt[4])
        yres = T(gt[6])  # Typically negative

        # Calculate lower-left corner
        xllcorner = xmin
        yllcorner = ymax + yres * nrows  # yres is negative

        cellsize = xres

        # Get nodata value
        nodata_val = try
            T(ArchGDAL.getnodatavalue(band))
        catch
            T(-9999)
        end

        # Get CRS
        crs = try
            ArchGDAL.toPROJ4(ArchGDAL.getspatialref(dataset))
        catch
            "EPSG:4326"
        end

        # Transform tuple
        transform = (T(gt[1]), T(gt[2]), T(gt[3]), T(gt[4]), T(gt[5]), T(gt[6]))

        metadata = GeoMetadata{T}(
            ncols, nrows, cellsize, xllcorner, yllcorner, nodata_val, crs, transform
        )

        # Transpose to match Julia's column-major order
        # GDAL returns row-major data, so we need to permute
        data_transposed = permutedims(data, (2, 1))

        return GeoRaster{T}(data_transposed, metadata)
    end
end

# Default to Float64
read_geotiff(path::String) = read_geotiff(Float64, path)


"""
    read_fuel_raster(::Type{T}, path::String) -> Tuple{Matrix{Int}, GeoMetadata{T}}

Read a fuel model raster (integer fuel IDs) and return the matrix with metadata.
"""
function read_fuel_raster(::Type{T}, path::String) where {T<:AbstractFloat}
    ArchGDAL.read(path) do dataset
        band = ArchGDAL.getband(dataset, 1)

        # Read data as integers
        data_raw = ArchGDAL.read(band)
        data = Int.(data_raw)

        # Get dimensions
        ncols = ArchGDAL.width(dataset)
        nrows = ArchGDAL.height(dataset)

        # Get geotransform
        gt = ArchGDAL.getgeotransform(dataset)
        xmin = T(gt[1])
        xres = T(gt[2])
        ymax = T(gt[4])
        yres = T(gt[6])

        xllcorner = xmin
        yllcorner = ymax + yres * nrows
        cellsize = xres

        nodata_val = try
            T(ArchGDAL.getnodatavalue(band))
        catch
            T(-9999)
        end

        crs = try
            ArchGDAL.toPROJ4(ArchGDAL.getspatialref(dataset))
        catch
            "EPSG:4326"
        end

        transform = (T(gt[1]), T(gt[2]), T(gt[3]), T(gt[4]), T(gt[5]), T(gt[6]))

        metadata = GeoMetadata{T}(
            ncols, nrows, cellsize, xllcorner, yllcorner, nodata_val, crs, transform
        )

        # Transpose for Julia column-major order
        data_transposed = permutedims(data, (2, 1))

        return (data_transposed, metadata)
    end
end

read_fuel_raster(path::String) = read_fuel_raster(Float64, path)


"""
    read_dem(::Type{T}, path::String) -> GeoRaster{T}

Read a Digital Elevation Model raster.
"""
read_dem(::Type{T}, path::String) where {T<:AbstractFloat} = read_geotiff(T, path)
read_dem(path::String) = read_dem(Float64, path)


#-----------------------------------------------------------------------------#
#                     Terrain Processing
#-----------------------------------------------------------------------------#

"""
    compute_slope_aspect(dem::GeoRaster{T}) -> Tuple{Matrix{T}, Matrix{T}}

Compute slope (degrees) and aspect (degrees) from a DEM using a 3x3 moving window.

Aspect is in degrees clockwise from north (0-360), with flat areas set to 0.
"""
function compute_slope_aspect(dem::GeoRaster{T}) where {T<:AbstractFloat}
    data = dem.data
    cellsize = dem.metadata.cellsize
    ncols, nrows = size(data)

    slope_deg = zeros(T, ncols, nrows)
    aspect_deg = zeros(T, ncols, nrows)

    # Use 2*cellsize as the window width for central differences
    dx_factor = one(T) / (T(2) * cellsize)

    for ix in 2:ncols-1
        for iy in 2:nrows-1
            # 3x3 window
            z_w = data[ix-1, iy]      # West
            z_e = data[ix+1, iy]      # East
            z_s = data[ix, iy-1]      # South
            z_n = data[ix, iy+1]      # North

            # Gradients
            dzdx = (z_e - z_w) * dx_factor
            dzdy = (z_n - z_s) * dx_factor

            # Slope
            slope_rad = atan(sqrt(dzdx^2 + dzdy^2))
            slope_deg[ix, iy] = slope_rad / pio180(T)

            # Aspect (degrees clockwise from north)
            if abs(dzdx) < T(1e-10) && abs(dzdy) < T(1e-10)
                aspect_deg[ix, iy] = zero(T)  # Flat
            else
                # atan2 gives angle from east, counter-clockwise
                # Convert to angle from north, clockwise
                aspect_rad = atan(dzdx, dzdy)
                aspect_degrees = aspect_rad / pio180(T)
                # Convert to 0-360 range
                aspect_degrees = T(180) - aspect_degrees
                if aspect_degrees < zero(T)
                    aspect_degrees += T(360)
                end
                if aspect_degrees >= T(360)
                    aspect_degrees -= T(360)
                end
                aspect_deg[ix, iy] = aspect_degrees
            end
        end
    end

    # Handle edges by copying from adjacent cells
    slope_deg[1, :] .= slope_deg[2, :]
    slope_deg[end, :] .= slope_deg[end-1, :]
    slope_deg[:, 1] .= slope_deg[:, 2]
    slope_deg[:, end] .= slope_deg[:, end-1]

    aspect_deg[1, :] .= aspect_deg[2, :]
    aspect_deg[end, :] .= aspect_deg[end-1, :]
    aspect_deg[:, 1] .= aspect_deg[:, 2]
    aspect_deg[:, end] .= aspect_deg[:, end-1]

    return (slope_deg, aspect_deg)
end


"""
    compute_slope_aspect(elevation::Matrix{T}, cellsize::T) -> Tuple{Matrix{T}, Matrix{T}}

Compute slope and aspect from an elevation matrix.
"""
function compute_slope_aspect(elevation::Matrix{T}, cellsize::T) where {T<:AbstractFloat}
    ncols, nrows = size(elevation)
    metadata = GeoMetadata{T}(ncols=ncols, nrows=nrows, cellsize=cellsize)
    dem = GeoRaster{T}(elevation, metadata)
    return compute_slope_aspect(dem)
end


#-----------------------------------------------------------------------------#
#                     Landscape Reading
#-----------------------------------------------------------------------------#

"""
    read_landscape(::Type{T}, fuel_path, dem_path;
                   cbd_path=nothing, cbh_path=nothing, cc_path=nothing, ch_path=nothing
    ) -> LandscapeData{T}

Read a complete landscape dataset from GeoTIFF files.

# Arguments
- `T`: Element type for the data
- `fuel_path`: Path to fuel model ID raster
- `dem_path`: Path to DEM raster
- `cbd_path`: Optional path to canopy bulk density raster (kg/m³)
- `cbh_path`: Optional path to canopy base height raster (m)
- `cc_path`: Optional path to canopy cover raster (fraction 0-1)
- `ch_path`: Optional path to canopy height raster (m)
"""
function read_landscape(::Type{T}, fuel_path::String, dem_path::String;
    cbd_path::Union{Nothing,String} = nothing,
    cbh_path::Union{Nothing,String} = nothing,
    cc_path::Union{Nothing,String} = nothing,
    ch_path::Union{Nothing,String} = nothing
) where {T<:AbstractFloat}
    # Read fuel raster
    fuel_ids, metadata = read_fuel_raster(T, fuel_path)

    # Read DEM and compute slope/aspect
    dem = read_dem(T, dem_path)
    slope, aspect = compute_slope_aspect(dem)
    elevation = dem.data

    # Handle nodata in elevation
    nodata = dem.metadata.nodata_value
    for ix in eachindex(elevation)
        if elevation[ix] == nodata
            elevation[ix] = zero(T)
        end
    end

    # Read canopy data if provided
    canopy = if cbd_path !== nothing && cbh_path !== nothing &&
                cc_path !== nothing && ch_path !== nothing
        cbd_raster = read_geotiff(T, cbd_path)
        cbh_raster = read_geotiff(T, cbh_path)
        cc_raster = read_geotiff(T, cc_path)
        ch_raster = read_geotiff(T, ch_path)

        # Handle nodata values
        cbd_data = replace(cbd_raster.data, nodata => zero(T))
        cbh_data = replace(cbh_raster.data, nodata => zero(T))
        cc_data = replace(cc_raster.data, nodata => zero(T))
        ch_data = replace(ch_raster.data, nodata => zero(T))

        CanopyGrid{T}(cbd_data, cbh_data, cc_data, ch_data)
    else
        nothing
    end

    return LandscapeData{T}(fuel_ids, slope, aspect, elevation, canopy, metadata)
end

read_landscape(fuel_path::String, dem_path::String; kwargs...) =
    read_landscape(Float64, fuel_path, dem_path; kwargs...)


#-----------------------------------------------------------------------------#
#                     Writing Functions
#-----------------------------------------------------------------------------#

"""
    write_geotiff(path::String, data::AbstractMatrix{T}, metadata::GeoMetadata{T})

Write a matrix to a GeoTIFF file with the given metadata.
"""
function write_geotiff(path::String, data::AbstractMatrix{T}, metadata::GeoMetadata{T}) where {T<:AbstractFloat}
    ncols, nrows = size(data)

    # Transpose back to GDAL's row-major order
    data_transposed = permutedims(data, (2, 1))

    # Create the GeoTIFF
    ArchGDAL.create(
        path,
        driver = ArchGDAL.getdriver("GTiff"),
        width = ncols,
        height = nrows,
        nbands = 1,
        dtype = T == Float64 ? Float64 : Float32
    ) do dataset
        # Set the geotransform
        ArchGDAL.setgeotransform!(dataset, collect(metadata.transform))

        # Set the projection
        try
            ArchGDAL.setproj!(dataset, metadata.crs)
        catch
            # Projection might fail for some CRS strings
        end

        # Get the band and write data
        band = ArchGDAL.getband(dataset, 1)
        ArchGDAL.setnodatavalue!(band, metadata.nodata_value)
        ArchGDAL.write!(band, data_transposed)
    end

    return nothing
end


"""
    write_geotiff(path::String, raster::GeoRaster{T})

Write a GeoRaster to a GeoTIFF file.
"""
write_geotiff(path::String, raster::GeoRaster{T}) where {T<:AbstractFloat} =
    write_geotiff(path, raster.data, raster.metadata)


#-----------------------------------------------------------------------------#
#                     Fire Perimeter Export
#-----------------------------------------------------------------------------#

"""
    write_fire_perimeter(path::String, state::FireState, metadata::GeoMetadata;
                         format::Symbol=:geojson)

Write the fire perimeter to a GeoJSON or other format.

# Arguments
- `path`: Output file path
- `state`: Fire simulation state
- `metadata`: Geospatial metadata for coordinate conversion
- `format`: Output format (:geojson supported)
"""
function write_fire_perimeter(
    path::String,
    state::FireState{T},
    metadata::GeoMetadata{T};
    format::Symbol = :geojson
) where {T<:AbstractFloat}
    if format != :geojson
        error("Only :geojson format is currently supported")
    end

    # Get perimeter cells
    perimeter = get_fire_perimeter(state)

    if isempty(perimeter)
        # Write empty feature collection
        open(path, "w") do io
            write(io, """{"type":"FeatureCollection","features":[]}""")
        end
        return nothing
    end

    # Convert grid coordinates to geographic coordinates
    xform = metadata.transform
    xmin = xform[1]
    xres = xform[2]
    ymax = xform[4]
    yres = xform[6]  # Negative

    # Create coordinate array for the perimeter
    coords = Vector{Tuple{T, T}}()
    for (ix, iy) in perimeter
        # Convert to geographic coordinates (cell center)
        x = xmin + (T(ix) - T(0.5)) * xres
        y = ymax + (T(iy) - T(0.5)) * yres  # yres is negative
        push!(coords, (x, y))
    end

    # Create GeoJSON structure
    features = [
        Dict(
            "type" => "Feature",
            "geometry" => Dict(
                "type" => "MultiPoint",
                "coordinates" => [[c[1], c[2]] for c in coords]
            ),
            "properties" => Dict(
                "type" => "fire_perimeter",
                "cell_count" => length(perimeter)
            )
        )
    ]

    geojson = Dict(
        "type" => "FeatureCollection",
        "features" => features
    )

    # Write to file
    open(path, "w") do io
        GeoJSON.write(io, geojson)
    end

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Utility Functions
#-----------------------------------------------------------------------------#

"""
    grid_to_geo(ix::Int, iy::Int, metadata::GeoMetadata{T}) -> Tuple{T, T}

Convert grid indices to geographic coordinates (cell center).
"""
function grid_to_geo(ix::Int, iy::Int, metadata::GeoMetadata{T}) where {T<:AbstractFloat}
    xform = metadata.transform
    x = xform[1] + (T(ix) - T(0.5)) * xform[2]
    y = xform[4] + (T(iy) - T(0.5)) * xform[6]
    return (x, y)
end


"""
    geo_to_grid(x::T, y::T, metadata::GeoMetadata{T}) -> Tuple{Int, Int}

Convert geographic coordinates to grid indices.
"""
function geo_to_grid(x::T, y::T, metadata::GeoMetadata{T}) where {T<:AbstractFloat}
    xform = metadata.transform
    ix = floor(Int, (x - xform[1]) / xform[2]) + 1
    iy = floor(Int, (y - xform[4]) / xform[6]) + 1
    return (ix, iy)
end


"""
    resample_to_match(source::GeoRaster{T}, target_metadata::GeoMetadata{T}) -> GeoRaster{T}

Resample a raster to match the resolution and extent of another.
Uses nearest-neighbor interpolation.
"""
function resample_to_match(source::GeoRaster{T}, target_metadata::GeoMetadata{T}) where {T<:AbstractFloat}
    ncols = target_metadata.ncols
    nrows = target_metadata.nrows

    result = fill(target_metadata.nodata_value, ncols, nrows)

    for ix in 1:ncols
        for iy in 1:nrows
            # Get geographic coordinate of target cell center
            x, y = grid_to_geo(ix, iy, target_metadata)

            # Convert to source grid indices
            src_ix, src_iy = geo_to_grid(x, y, source.metadata)

            # Check bounds and copy value
            if 1 <= src_ix <= source.metadata.ncols && 1 <= src_iy <= source.metadata.nrows
                result[ix, iy] = source.data[src_ix, src_iy]
            end
        end
    end

    return GeoRaster{T}(result, target_metadata)
end
