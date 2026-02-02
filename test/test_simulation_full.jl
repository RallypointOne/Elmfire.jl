using Random

@testset "Full Simulation (Phase 2)" begin

    @testset "SimulationConfig Construction" begin
        # Default config
        config = SimulationConfig()
        @test config.enable_crown_fire == false
        @test config.enable_spotting == false
        @test config.crown_fire_adj == 1.0
        @test eltype(config) == Float64

        # With crown fire enabled
        config_crown = SimulationConfig{Float64}(
            enable_crown_fire = true,
            crown_fire_adj = 1.2,
            foliar_moisture = 90.0
        )
        @test config_crown.enable_crown_fire == true
        @test config_crown.crown_fire_adj == 1.2

        # With spotting enabled
        spot_params = SpottingParameters{Float64}(pign = 50.0)
        config_spot = SimulationConfig{Float64}(
            enable_spotting = true,
            spotting_params = spot_params
        )
        @test config_spot.enable_spotting == true
        @test config_spot.spotting_params !== nothing
    end

    @testset "CanopyGrid Construction" begin
        # Empty grid
        cg = CanopyGrid{Float64}(30, 30)
        @test size(cg.cbd) == (30, 30)
        @test all(cg.cbd .== 0.0)
        @test eltype(cg) == Float64

        # Uniform grid
        cg_uniform = CanopyGrid{Float64}(20, 20, 0.15, 5.0, 0.7, 18.0)
        @test all(cg_uniform.cbd .== 0.15)
        @test all(cg_uniform.cbh .== 5.0)
        @test all(cg_uniform.cc .== 0.7)
        @test all(cg_uniform.ch .== 18.0)

        # Get properties
        props = get_canopy_properties(cg_uniform, 10, 10)
        @test props.cbd == 0.15
        @test props.cbh == 5.0
    end

    @testset "simulate_full! Basic (No Phase 2 Features)" begin
        # Run full simulation with Phase 2 features disabled
        state = FireState(30, 30, 10.0)
        ignite!(state, 15, 15, 0.0)

        fuel_table = create_standard_fuel_table()
        weather = ConstantWeather(
            wind_speed_mph = 10.0,
            wind_direction = 270.0,
            M1 = 0.05
        )

        config = SimulationConfig()  # All features disabled

        simulate_full_uniform!(
            state, 1, fuel_table, weather,
            0.0, 0.0,  # Flat terrain
            0.0, 3.0;  # 3 minutes
            config = config,
            dt_initial = 0.2
        )

        # Fire should have spread
        @test count(state.burned) > 1
        @test get_burned_area(state) > state.cellsize^2
    end

    @testset "simulate_full! with Crown Fire" begin
        state = FireState(40, 40, 10.0)
        ignite!(state, 20, 20, 0.0)

        fuel_table = create_standard_fuel_table()
        weather = ConstantWeather(
            wind_speed_mph = 15.0,
            wind_direction = 270.0,
            M1 = 0.04,
            M10 = 0.06,
            M100 = 0.08,
            MLH = 0.50,
            MLW = 0.80
        )

        config = SimulationConfig{Float64}(
            enable_crown_fire = true,
            crown_fire_adj = 1.0,
            foliar_moisture = 100.0
        )

        # With dense canopy that can support crown fire
        simulate_full_uniform!(
            state, 1, fuel_table, weather,
            0.0, 0.0,
            0.0, 3.0;
            canopy_cbd = 0.15,
            canopy_cbh = 3.0,
            canopy_cc = 0.7,
            canopy_ch = 18.0,
            config = config,
            dt_initial = 0.2
        )

        # Fire should spread
        @test count(state.burned) > 1

        # Should have higher fireline intensity due to crown contribution
        max_flin = maximum(state.fireline_intensity)
        @test max_flin > 0
    end

    @testset "simulate_full! with Spotting" begin
        rng = MersenneTwister(12345)

        state = FireState(50, 50, 10.0)
        ignite!(state, 25, 25, 0.0)

        fuel_table = create_standard_fuel_table()
        weather = ConstantWeather(
            wind_speed_mph = 20.0,  # Strong wind for spotting
            wind_direction = 270.0,
            M1 = 0.03,  # Very dry
            M10 = 0.05,
            M100 = 0.07,
            MLH = 0.40,
            MLW = 0.70
        )

        spot_params = SpottingParameters{Float64}(
            mean_distance = 30.0,  # Short distance for test grid
            surface_spotting_percent = 50.0,
            crown_spotting_percent = 80.0,
            nembers_max = 5,
            pign = 80.0,  # High ignition probability
            min_distance = 5.0,
            max_distance = 200.0
        )

        config = SimulationConfig{Float64}(
            enable_spotting = true,
            spotting_params = spot_params
        )

        tracker = simulate_full_uniform!(
            state, 1, fuel_table, weather,
            0.0, 0.0,
            0.0, 5.0;  # 5 minutes
            config = config,
            dt_initial = 0.2,
            rng = rng
        )

        # Fire should spread
        @test count(state.burned) > 1

        # Tracker should be returned
        @test tracker !== nothing
    end

    @testset "simulate_full! with Weather Interpolation" begin
        state = FireState(30, 30, 10.0)
        ignite!(state, 15, 15, 0.0)

        fuel_table = create_standard_fuel_table()

        # Create time-varying weather
        wg1 = WeatherGrid{Float64}(3, 3, 1000.0)
        wg2 = WeatherGrid{Float64}(3, 3, 1000.0)

        # Different wind at t=0 vs t=60
        wg1.ws .= 10.0
        wg1.wd .= 270.0
        wg1.m1 .= 0.06
        wg1.m10 .= 0.08
        wg1.m100 .= 0.10
        wg1.mlh .= 0.60
        wg1.mlw .= 0.90

        wg2.ws .= 15.0
        wg2.wd .= 180.0  # Wind shifts to from south
        wg2.m1 .= 0.05
        wg2.m10 .= 0.07
        wg2.m100 .= 0.09
        wg2.mlh .= 0.55
        wg2.mlw .= 0.85

        wts = WeatherTimeSeries{Float64}([wg1, wg2], [0.0, 60.0])
        weather_interp = WeatherInterpolator(wts, 30, 30, 10.0)

        fuel_ids = fill(1, 30, 30)
        slope = zeros(Float64, 30, 30)
        aspect = zeros(Float64, 30, 30)

        config = SimulationConfig{Float64}()

        simulate_full!(
            state, fuel_ids, fuel_table, weather_interp,
            slope, aspect,
            0.0, 3.0;
            config = config,
            dt_initial = 0.2
        )

        # Fire should spread
        @test count(state.burned) > 1
    end

    @testset "simulate_full! All Features Combined" begin
        rng = MersenneTwister(54321)

        state = FireState(50, 50, 10.0)
        ignite!(state, 25, 25, 0.0)

        fuel_table = create_standard_fuel_table()
        weather = ConstantWeather(
            wind_speed_mph = 18.0,
            wind_direction = 270.0,
            M1 = 0.04,
            M10 = 0.06,
            M100 = 0.08,
            MLH = 0.45,
            MLW = 0.75
        )

        spot_params = SpottingParameters{Float64}(
            mean_distance = 25.0,
            surface_spotting_percent = 30.0,
            crown_spotting_percent = 60.0,
            pign = 50.0
        )

        config = SimulationConfig{Float64}(
            enable_crown_fire = true,
            enable_spotting = true,
            crown_fire_adj = 1.0,
            spotting_params = spot_params
        )

        tracker = simulate_full_uniform!(
            state, 1, fuel_table, weather,
            0.0, 0.0,
            0.0, 5.0;
            canopy_cbd = 0.12,
            canopy_cbh = 4.0,
            canopy_cc = 0.65,
            canopy_ch = 16.0,
            config = config,
            dt_initial = 0.2,
            rng = rng
        )

        # Verify simulation ran
        @test count(state.burned) > 1
        @test get_burned_area(state) > 0

        # Check output fields populated
        @test maximum(state.spread_rate) > 0
        @test maximum(state.fireline_intensity) > 0
        @test maximum(state.flame_length) > 0
    end

    @testset "Float32 Full Simulation" begin
        state32 = FireState{Float32}(30, 30, Float32(10))
        ignite!(state32, 15, 15, Float32(0))

        fuel_table32 = create_standard_fuel_table(Float32)
        weather32 = ConstantWeather{Float32}(
            wind_speed_mph = Float32(12),
            wind_direction = Float32(270),
            M1 = Float32(0.05)
        )

        spot_params32 = SpottingParameters{Float32}(
            mean_distance = Float32(30),
            pign = Float32(60)
        )

        config32 = SimulationConfig{Float32}(
            enable_crown_fire = true,
            enable_spotting = true,
            spotting_params = spot_params32
        )

        rng = MersenneTwister(11111)

        simulate_full_uniform!(
            state32, 1, fuel_table32, weather32,
            Float32(0), Float32(0),
            Float32(0), Float32(3);
            canopy_cbd = Float32(0.10),
            canopy_cbh = Float32(4),
            canopy_cc = Float32(0.6),
            canopy_ch = Float32(15),
            config = config32,
            dt_initial = Float32(0.2),
            rng = rng
        )

        @test count(state32.burned) > 1
        @test get_burned_area(state32) isa Float32
    end

    @testset "Error Handling" begin
        state = FireState(20, 20, 10.0)
        ignite!(state, 10, 10, 0.0)

        fuel_table = create_standard_fuel_table()
        weather = ConstantWeather()

        # Create grids for simulate_full!
        fuel_ids = fill(1, 20, 20)
        slope = zeros(Float64, 20, 20)
        aspect = zeros(Float64, 20, 20)
        weather_interp = create_constant_interpolator(weather, 20, 20, 10.0)

        # Crown fire enabled but no canopy passed to simulate_full! - should error
        config_no_canopy = SimulationConfig{Float64}(enable_crown_fire = true)
        @test_throws ErrorException simulate_full!(
            state, fuel_ids, fuel_table, weather_interp,
            slope, aspect, 0.0, 1.0;
            canopy = nothing,  # Explicitly no canopy
            config = config_no_canopy
        )

        # Spotting enabled but no params - should error
        config_no_params = SimulationConfig{Float64}(enable_spotting = true)
        @test_throws ErrorException simulate_full!(
            state, fuel_ids, fuel_table, weather_interp,
            slope, aspect, 0.0, 1.0;
            config = config_no_params
        )
    end

    @testset "Callback Function" begin
        state = FireState(20, 20, 10.0)
        ignite!(state, 10, 10, 0.0)

        fuel_table = create_standard_fuel_table()
        weather = ConstantWeather(wind_speed_mph = 8.0)

        config = SimulationConfig()

        callback_count = Ref(0)
        function my_callback(s, t, dt, iter)
            callback_count[] += 1
        end

        simulate_full_uniform!(
            state, 1, fuel_table, weather,
            0.0, 0.0, 0.0, 2.0;
            config = config,
            callback = my_callback,
            dt_initial = 0.5
        )

        # Callback should have been called
        @test callback_count[] > 0
    end
end
