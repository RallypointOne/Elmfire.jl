@testset "Generic Precision (Float32)" begin

    @testset "Constants" begin
        # Test that constant functions work with Float32
        @test Elmfire.pi_val(Float32) isa Float32
        @test Elmfire.pio180(Float32) isa Float32
        @test Elmfire.btupft2min_to_kwpm2(Float32) isa Float32
        @test Elmfire.ft_to_m(Float32) isa Float32
        @test Elmfire.rhop_default(Float32) isa Float32
        @test Elmfire.etas_default(Float32) isa Float32

        # Check approximate values match Float64
        @test Elmfire.pi_val(Float32) ≈ Float32(π)
        @test Elmfire.ft_to_m(Float32) ≈ Float32(0.3048)
    end

    @testset "Fuel Models Float32" begin
        # Create Float32 fuel model table
        table32 = create_standard_fuel_table(Float32)

        @test eltype(table32) == Float32

        # Get a fuel model
        fm32 = get_fuel_model(table32, 1, 60)

        @test eltype(fm32) == Float32
        @test fm32.delta isa Float32
        @test fm32.rhob isa Float32
        @test all(x -> x isa Float32, fm32.W0)
        @test all(x -> x isa Float32, fm32.SIG)
    end

    @testset "Rothermel Float32" begin
        table32 = create_standard_fuel_table(Float32)
        fm32 = get_fuel_model(table32, 1, 60)

        # Test spread rate calculation with Float32
        result = surface_spread_rate(
            fm32,
            Float32(0.06), Float32(0.08), Float32(0.10),
            Float32(0.60), Float32(0.90),
            Float32(100), Float32(0.0)
        )

        @test eltype(result) == Float32
        @test result.velocity isa Float32
        @test result.ir isa Float32
        @test result.hpua isa Float32
        @test result.flin isa Float32

        # Should produce positive spread rate
        @test result.velocity > 0
    end

    @testset "Elliptical Spread Float32" begin
        es = elliptical_spread(Float32(50), Float32(10))

        @test eltype(es) == Float32
        @test es.head isa Float32
        @test es.back isa Float32
        @test es.flank isa Float32
        @test es.eccentricity isa Float32
    end

    @testset "Level Set Float32" begin
        # Test half_superbee
        @test Elmfire.half_superbee(Float32(0.5)) isa Float32
        @test Elmfire.half_superbee(Float32(1.0)) ≈ Float32(0.5)

        # Test limit_gradients with Float32
        phi32 = zeros(Float32, 10, 10)
        for i in 1:10
            phi32[i, :] .= Float32(i)
        end

        dphidx, dphidy = Elmfire.limit_gradients(phi32, Float32(1.0), Float32(0.0), 5, 5, Float32(1.0))
        @test dphidx isa Float32
        @test dphidy isa Float32

        # Test compute_normal
        nx, ny = Elmfire.compute_normal(phi32, 5, 5, Float32(1.0))
        @test nx isa Float32
        @test ny isa Float32
    end

    @testset "Weather Float32" begin
        weather32 = ConstantWeather{Float32}(
            wind_speed_mph = Float32(10),
            wind_direction = Float32(270),
            M1 = Float32(0.06),
            M10 = Float32(0.08),
            M100 = Float32(0.10),
            MLH = Float32(0.60),
            MLW = Float32(0.90)
        )

        @test eltype(weather32) == Float32
        @test weather32.wind_speed_20ft isa Float32
        @test weather32.M1 isa Float32
    end

    @testset "FireState Float32" begin
        state32 = FireState{Float32}(50, 50, Float32(30))

        @test eltype(state32) == Float32
        @test state32.cellsize isa Float32
        @test eltype(state32.phi) == Float32
        @test eltype(state32.ux) == Float32
        @test eltype(state32.time_of_arrival) == Float32
    end

    @testset "Full Simulation Float32" begin
        # Create a complete Float32 simulation
        state32 = FireState{Float32}(30, 30, Float32(10))

        # Ignite center
        ignite!(state32, 15, 15, Float32(0))

        @test state32.burned[15, 15]
        @test state32.time_of_arrival[15, 15] == Float32(0)

        # Create Float32 fuel table
        fuel_table32 = create_standard_fuel_table(Float32)

        # Float32 weather
        weather32 = ConstantWeather{Float32}(
            wind_speed_mph = Float32(10),
            wind_direction = Float32(0),
            M1 = Float32(0.04),
            M10 = Float32(0.06),
            M100 = Float32(0.08),
            MLH = Float32(0.50),
            MLW = Float32(0.80)
        )

        # Run simulation
        simulate_uniform!(
            state32,
            1,  # FBFM01
            fuel_table32,
            weather32,
            Float32(0),  # Flat
            Float32(0),  # No aspect
            Float32(0),  # Start time
            Float32(3);  # 3 minutes
            dt_initial = Float32(0.2)
        )

        # Fire should have spread
        burned_count = count(state32.burned)
        @test burned_count > 1

        # Check output types
        @test get_burned_area(state32) isa Float32
        @test get_burned_area_acres(state32) isa Float32
    end

    @testset "Precision Comparison" begin
        # Compare Float32 vs Float64 results - should be similar but not identical
        table32 = create_standard_fuel_table(Float32)
        table64 = create_standard_fuel_table(Float64)

        fm32 = get_fuel_model(table32, 1, 60)
        fm64 = get_fuel_model(table64, 1, 60)

        result32 = surface_spread_rate(
            fm32,
            Float32(0.06), Float32(0.08), Float32(0.10),
            Float32(0.60), Float32(0.90),
            Float32(100), Float32(0.0)
        )

        result64 = surface_spread_rate(
            fm64,
            0.06, 0.08, 0.10,
            0.60, 0.90,
            100.0, 0.0
        )

        # Results should be close but may differ slightly due to precision
        @test Float64(result32.velocity) ≈ result64.velocity rtol=1e-4
        @test Float64(result32.ir) ≈ result64.ir rtol=1e-4
    end
end
