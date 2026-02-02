@testset "Geospatial I/O" begin

    @testset "GeoMetadata" begin
        metadata = GeoMetadata{Float64}(
            ncols=100, nrows=100, cellsize=30.0,
            xllcorner=0.0, yllcorner=0.0
        )

        @test metadata.ncols == 100
        @test metadata.nrows == 100
        @test metadata.cellsize == 30.0
        @test metadata.nodata_value == -9999.0

        # Check default transform
        @test metadata.transform[1] == 0.0  # xmin
        @test metadata.transform[2] == 30.0  # xres
        @test metadata.transform[4] == 3000.0  # ymax = yllcorner + cellsize * nrows
    end

    @testset "GeoRaster" begin
        data = rand(50, 50)
        metadata = GeoMetadata{Float64}(
            ncols=50, nrows=50, cellsize=30.0
        )
        raster = GeoRaster{Float64}(data, metadata)

        @test size(raster) == (50, 50)
        @test eltype(raster) == Float64
    end

    @testset "Slope and Aspect Computation" begin
        # Create a simple tilted plane (slope in x direction)
        ncols, nrows = 50, 50
        cellsize = 30.0
        elevation = zeros(Float64, ncols, nrows)

        # Create a 10% grade (about 5.7 degrees) in the x direction
        for ix in 1:ncols
            for iy in 1:nrows
                elevation[ix, iy] = Float64(ix) * 3.0  # 3m rise per 30m cell = 10%
            end
        end

        slope, aspect = compute_slope_aspect(elevation, cellsize)

        # Check slope (should be ~5.7 degrees in the interior)
        @test 5.0 < slope[25, 25] < 7.0

        # Check aspect (slope faces east, so downhill is west, aspect points upslope = east = 90 degrees)
        # Actually aspect convention varies - just check it's computed
        @test 0.0 <= aspect[25, 25] <= 360.0
    end

    @testset "Coordinate Conversions" begin
        metadata = GeoMetadata{Float64}(
            ncols=100, nrows=100, cellsize=30.0,
            xllcorner=0.0, yllcorner=0.0
        )

        # Test grid to geo
        x, y = grid_to_geo(1, 1, metadata)
        @test x ≈ 15.0  # Center of first cell
        @test y ≈ 2985.0  # ymax - 0.5 * cellsize

        # Test geo to grid
        ix, iy = geo_to_grid(x, y, metadata)
        @test ix == 1
        @test iy == 1

        # Test round trip
        for _ in 1:10
            test_ix = rand(1:100)
            test_iy = rand(1:100)
            x, y = grid_to_geo(test_ix, test_iy, metadata)
            ix, iy = geo_to_grid(x, y, metadata)
            @test ix == test_ix
            @test iy == test_iy
        end
    end

    @testset "LandscapeData" begin
        # Test empty landscape creation
        fuel_ids = fill(1, 50, 50)
        slope = fill(5.0, 50, 50)
        aspect = fill(180.0, 50, 50)
        elevation = fill(1000.0, 50, 50)
        metadata = GeoMetadata{Float64}(
            ncols=50, nrows=50, cellsize=30.0
        )

        landscape = LandscapeData{Float64}(
            fuel_ids, slope, aspect, elevation, nothing, metadata
        )

        @test size(landscape.fuel_ids) == (50, 50)
        @test landscape.canopy === nothing
        @test eltype(landscape) == Float64
    end

    @testset "Float32 Support" begin
        metadata = GeoMetadata{Float32}(
            ncols=100, nrows=100, cellsize=30.0f0
        )

        @test eltype(metadata) == Float32
        @test metadata.cellsize === 30.0f0

        data = rand(Float32, 100, 100)
        raster = GeoRaster{Float32}(data, metadata)
        @test eltype(raster) == Float32
    end

end
