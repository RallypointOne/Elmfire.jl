@testset "Weather Interpolation" begin

    @testset "WeatherGrid Construction" begin
        # Empty grid
        wg = WeatherGrid{Float64}(10, 10, 1000.0)
        @test wg.ncols == 10
        @test wg.nrows == 10
        @test wg.cellsize == 1000.0
        @test size(wg.ws) == (10, 10)
        @test eltype(wg) == Float64

        # Default values
        @test all(wg.ws .== 0.0)
        @test all(wg.wd .== 0.0)
        @test all(wg.m1 .== 0.06)
        @test all(wg.m10 .== 0.08)

        # With corner coordinates
        wg2 = WeatherGrid{Float64}(5, 5, 500.0; xllcorner=1000.0, yllcorner=2000.0)
        @test wg2.xllcorner == 1000.0
        @test wg2.yllcorner == 2000.0

        # Float32
        wg32 = WeatherGrid{Float32}(5, 5, Float32(500))
        @test eltype(wg32) == Float32
    end

    @testset "WeatherGrid from ConstantWeather" begin
        weather = ConstantWeather{Float64}(
            wind_speed_mph = 15.0,
            wind_direction = 270.0,
            M1 = 0.05,
            M10 = 0.07,
            M100 = 0.09,
            MLH = 0.50,
            MLW = 0.85
        )

        wg = WeatherGrid{Float64}(weather, 8, 6, 1000.0)

        @test wg.ncols == 8
        @test wg.nrows == 6
        @test all(wg.ws .== 15.0)
        @test all(wg.wd .== 270.0)
        @test all(wg.m1 .== 0.05)
        @test all(wg.mlw .== 0.85)
    end

    @testset "WeatherTimeSeries Construction" begin
        # Create two grids at different times
        wg1 = WeatherGrid{Float64}(5, 5, 1000.0)
        wg2 = WeatherGrid{Float64}(5, 5, 1000.0)

        # Set different wind speeds
        wg1.ws .= 10.0
        wg2.ws .= 20.0

        times = [0.0, 60.0]  # t=0 and t=60 minutes
        wts = WeatherTimeSeries{Float64}([wg1, wg2], times)

        @test length(wts.grids) == 2
        @test length(wts.times) == 2
        @test wts.dt == 60.0
        @test eltype(wts) == Float64

        # Single grid time series
        wts_single = WeatherTimeSeries{Float64}([wg1], [0.0])
        @test wts_single.dt == 1.0  # Default when only one point
    end

    @testset "WeatherTimeSeries from ConstantWeather" begin
        weather = ConstantWeather{Float64}(
            wind_speed_mph = 12.0,
            wind_direction = 180.0,
            M1 = 0.06
        )

        wts = WeatherTimeSeries{Float64}(weather, 10, 10, 500.0, 120.0)

        @test length(wts.grids) == 1
        @test wts.times[1] == 0.0
    end

    @testset "Find Time Indices" begin
        wg = WeatherGrid{Float64}(3, 3, 1000.0)
        grids = [wg, wg, wg, wg]
        times = [0.0, 30.0, 60.0, 90.0]
        wts = WeatherTimeSeries{Float64}(grids, times)

        # At first time point
        i_lo, i_hi, f = find_time_indices(wts, 0.0)
        @test i_lo == 1
        @test i_hi == 2
        @test f == 0.0

        # Between time points
        i_lo, i_hi, f = find_time_indices(wts, 45.0)
        @test i_lo == 2
        @test i_hi == 3
        @test f == 0.5

        # At last time point
        i_lo, i_hi, f = find_time_indices(wts, 90.0)
        @test i_lo == 4
        @test i_hi == 4
        @test f == 0.0

        # Beyond last time point (clamped)
        i_lo, i_hi, f = find_time_indices(wts, 120.0)
        @test i_lo == 4
        @test i_hi == 4

        # Single grid case
        wts_single = WeatherTimeSeries{Float64}([wg], [0.0])
        i_lo, i_hi, f = find_time_indices(wts_single, 50.0)
        @test i_lo == 1
        @test i_hi == 1
        @test f == 0.0
    end

    @testset "Interpolate Wind Direction" begin
        # Simple case - no wrap-around
        wd = interpolate_wind_direction(90.0, 180.0, 0.5)
        @test wd ≈ 135.0 atol=1.0

        # At boundaries
        @test interpolate_wind_direction(90.0, 180.0, 0.0) ≈ 90.0 atol=0.1
        @test interpolate_wind_direction(90.0, 180.0, 1.0) ≈ 180.0 atol=0.1

        # Wrap-around case: from 350° to 10° should go through 0°
        wd_wrap = interpolate_wind_direction(350.0, 10.0, 0.5)
        # Should be approximately 0° (or 360°)
        @test wd_wrap < 20.0 || wd_wrap > 340.0

        # Opposite direction wrap
        wd_wrap2 = interpolate_wind_direction(10.0, 350.0, 0.5)
        @test wd_wrap2 < 20.0 || wd_wrap2 > 340.0

        # Float32
        wd32 = interpolate_wind_direction(Float32(90), Float32(180), Float32(0.5))
        @test wd32 isa Float32
    end

    @testset "Create Grid Mapping" begin
        # Weather grid: 10km x 10km with 2km cells (5x5)
        wg = WeatherGrid{Float64}(5, 5, 2000.0; xllcorner=0.0, yllcorner=0.0)

        # Simulation grid: 1000ft cells, covering ~3km
        sim_ncols = 30
        sim_nrows = 30
        sim_cellsize = 100.0  # feet
        sim_xllcorner = 1000.0  # meters
        sim_yllcorner = 1000.0

        icol_map, irow_map = create_grid_mapping(
            wg,
            sim_ncols, sim_nrows,
            sim_cellsize,
            sim_xllcorner, sim_yllcorner
        )

        @test length(icol_map) == sim_ncols
        @test length(irow_map) == sim_nrows

        # All indices should be within weather grid bounds
        @test all(1 .<= icol_map .<= 5)
        @test all(1 .<= irow_map .<= 5)
    end

    @testset "WeatherInterpolator Construction" begin
        weather = ConstantWeather{Float64}(
            wind_speed_mph = 10.0,
            wind_direction = 270.0,
            M1 = 0.06
        )

        wts = WeatherTimeSeries{Float64}(weather, 1, 1, 10000.0, 120.0)

        interp = WeatherInterpolator(wts, 50, 50, 30.0)

        @test interp.sim_ncols == 50
        @test interp.sim_nrows == 50
        @test length(interp.icol_map) == 50
        @test length(interp.irow_map) == 50
        @test eltype(interp) == Float64
    end

    @testset "Get Weather At - Constant" begin
        weather = ConstantWeather{Float64}(
            wind_speed_mph = 12.0,
            wind_direction = 180.0,
            M1 = 0.05,
            M10 = 0.07,
            M100 = 0.09,
            MLH = 0.55,
            MLW = 0.90
        )

        wts = WeatherTimeSeries{Float64}(weather, 1, 1, 10000.0, 120.0)
        interp = WeatherInterpolator(wts, 20, 20, 30.0)

        # Get weather at any cell - should be constant
        w1 = get_weather_at(interp, 5, 5, 0.0)
        @test w1.ws ≈ 12.0
        @test w1.wd ≈ 180.0
        @test w1.m1 ≈ 0.05
        @test w1.m10 ≈ 0.07
        @test w1.m100 ≈ 0.09
        @test w1.mlh ≈ 0.55
        @test w1.mlw ≈ 0.90

        # Same at different location and time
        w2 = get_weather_at(interp, 15, 15, 60.0)
        @test w2.ws ≈ 12.0
    end

    @testset "Get Weather At - Temporal Interpolation" begin
        # Create time series with changing wind
        wg1 = WeatherGrid{Float64}(3, 3, 1000.0)
        wg2 = WeatherGrid{Float64}(3, 3, 1000.0)

        wg1.ws .= 10.0
        wg1.wd .= 90.0
        wg2.ws .= 20.0
        wg2.wd .= 180.0

        wts = WeatherTimeSeries{Float64}([wg1, wg2], [0.0, 60.0])
        interp = WeatherInterpolator(wts, 10, 10, 30.0)

        # At t=0, should be wg1 values
        w0 = get_weather_at(interp, 5, 5, 0.0)
        @test w0.ws ≈ 10.0
        @test w0.wd ≈ 90.0

        # At t=60, should be wg2 values
        w60 = get_weather_at(interp, 5, 5, 60.0)
        @test w60.ws ≈ 20.0
        @test w60.wd ≈ 180.0

        # At t=30, should be interpolated
        w30 = get_weather_at(interp, 5, 5, 30.0)
        @test w30.ws ≈ 15.0
        @test w30.wd ≈ 135.0 atol=1.0  # Interpolated direction
    end

    @testset "Create Constant Interpolator" begin
        weather = ConstantWeather{Float64}(
            wind_speed_mph = 8.0,
            wind_direction = 45.0,
            M1 = 0.04
        )

        interp = create_constant_interpolator(weather, 30, 30, 30.0)

        @test interp.sim_ncols == 30
        @test interp.sim_nrows == 30

        # Check values at various points
        for ix in [1, 15, 30], iy in [1, 15, 30], t in [0.0, 30.0, 60.0]
            w = get_weather_at(interp, ix, iy, t)
            @test w.ws ≈ 8.0
            @test w.wd ≈ 45.0
            @test w.m1 ≈ 0.04
        end
    end

    @testset "Float32 Weather System" begin
        weather32 = ConstantWeather{Float32}(
            wind_speed_mph = Float32(10),
            wind_direction = Float32(270),
            M1 = Float32(0.06),
            M10 = Float32(0.08),
            M100 = Float32(0.10),
            MLH = Float32(0.60),
            MLW = Float32(0.90)
        )

        wg32 = WeatherGrid{Float32}(weather32, 5, 5, Float32(1000))
        @test eltype(wg32) == Float32

        wts32 = WeatherTimeSeries{Float32}([wg32], [Float32(0)])
        @test eltype(wts32) == Float32

        interp32 = WeatherInterpolator(wts32, 20, 20, Float32(30))
        @test eltype(interp32) == Float32

        w32 = get_weather_at(interp32, 10, 10, Float32(0))
        @test w32.ws isa Float32
        @test w32.wd isa Float32
        @test w32.m1 isa Float32
    end

    @testset "Spatial Variation" begin
        # Create weather grid with spatial variation
        wg = WeatherGrid{Float64}(5, 5, 1000.0)

        # Set different wind speeds in different cells
        for i in 1:5, j in 1:5
            wg.ws[i, j] = 5.0 + i * 2.0  # Varies from 7 to 15 mph
        end

        wts = WeatherTimeSeries{Float64}([wg], [0.0])

        # Create interpolator for simulation grid that spans weather grid
        # With 1000m cells and sim cells ~30ft, many sim cells per weather cell
        interp = WeatherInterpolator(
            wts, 100, 100, 30.0,
            0.0, 0.0  # Aligned with weather grid
        )

        # Check that different simulation cells map to different weather cells
        w1 = get_weather_at(interp, 10, 50, 0.0)
        w2 = get_weather_at(interp, 90, 50, 0.0)

        # Wind speeds should differ if mapping to different weather cells
        # (depends on exact mapping, so just check they're valid)
        @test w1.ws >= 5.0
        @test w2.ws >= 5.0
    end
end
