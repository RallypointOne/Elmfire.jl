@testset "Rothermel Model" begin

    @testset "Moisture Damping" begin
        # At zero moisture ratio, damping should be 1.0
        @test Elmfire.moisture_damping(0.0) ≈ 1.0

        # At extinction (ratio = 1.0), damping should be ≈ 0
        @test Elmfire.moisture_damping(1.0) ≈ 0.0 atol=0.01

        # Intermediate values should be between 0 and 1
        @test 0.0 < Elmfire.moisture_damping(0.5) < 1.0

        # Should handle values outside [0, 1]
        @test Elmfire.moisture_damping(-0.1) ≈ 1.0
        @test Elmfire.moisture_damping(1.5) ≈ 0.0 atol=0.01
    end

    @testset "Surface Spread Rate - Basic" begin
        table = Elmfire.create_standard_fuel_table()
        fm = Elmfire.get_fuel_model(table, 1, 60)  # FBFM01

        # Low moisture, moderate wind, flat ground
        result = Elmfire.surface_spread_rate(
            fm,
            0.06, 0.08, 0.10,  # M1, M10, M100
            0.60, 0.90,        # MLH, MLW
            440.0,             # 5 mph mid-flame wind (5 * 88 ft/min)
            0.0                # Flat (tan²(0) = 0)
        )

        @test result.velocity > 0
        @test result.vs0 > 0
        @test result.ir > 0
        @test result.hpua > 0
        @test result.flin > 0
        @test result.phiw >= 0
        @test result.phis == 0  # Flat ground
    end

    @testset "Surface Spread Rate - Wind Effect" begin
        table = Elmfire.create_standard_fuel_table()
        fm = Elmfire.get_fuel_model(table, 1, 60)

        # No wind
        result_nowind = Elmfire.surface_spread_rate(
            fm, 0.06, 0.08, 0.10, 0.60, 0.90,
            0.0, 0.0
        )

        # With wind
        result_wind = Elmfire.surface_spread_rate(
            fm, 0.06, 0.08, 0.10, 0.60, 0.90,
            880.0, 0.0  # 10 mph
        )

        # Wind should increase spread rate
        @test result_wind.velocity > result_nowind.velocity
        @test result_wind.phiw > result_nowind.phiw
    end

    @testset "Surface Spread Rate - Slope Effect" begin
        table = Elmfire.create_standard_fuel_table()
        fm = Elmfire.get_fuel_model(table, 1, 60)

        # Flat ground
        result_flat = Elmfire.surface_spread_rate(
            fm, 0.06, 0.08, 0.10, 0.60, 0.90,
            0.0, 0.0
        )

        # 30-degree slope (tan²(30°) ≈ 0.333)
        tanslp2_30 = tan(30 * Elmfire.PI / 180)^2
        result_slope = Elmfire.surface_spread_rate(
            fm, 0.06, 0.08, 0.10, 0.60, 0.90,
            0.0, tanslp2_30
        )

        # Slope should increase spread rate
        @test result_slope.velocity > result_flat.velocity
        @test result_slope.phis > result_flat.phis
    end

    @testset "Surface Spread Rate - Moisture Effect" begin
        table = Elmfire.create_standard_fuel_table()
        fm = Elmfire.get_fuel_model(table, 1, 60)

        # Dry fuels
        result_dry = Elmfire.surface_spread_rate(
            fm, 0.03, 0.05, 0.07, 0.40, 0.60,
            440.0, 0.0
        )

        # Wet fuels
        result_wet = Elmfire.surface_spread_rate(
            fm, 0.10, 0.12, 0.15, 0.80, 1.00,
            440.0, 0.0
        )

        # Dry fuels should spread faster
        @test result_dry.velocity > result_wet.velocity
        @test result_dry.ir > result_wet.ir
    end

    @testset "Surface Spread Rate - Non-burnable" begin
        table = Elmfire.create_standard_fuel_table()
        fm = Elmfire.get_fuel_model(table, 256, 60)  # Non-burnable

        result = Elmfire.surface_spread_rate(
            fm, 0.06, 0.08, 0.10, 0.60, 0.90,
            440.0, 0.0
        )

        @test result.velocity == 0.0
        @test result.ir == 0.0
    end

    @testset "Surface Spread Rate - Tuple Input" begin
        table = Elmfire.create_standard_fuel_table()
        fm = Elmfire.get_fuel_model(table, 1, 60)

        moisture = (0.06, 0.08, 0.10, 0.60, 0.90)
        result = Elmfire.surface_spread_rate(fm, moisture, 440.0, 0.0)

        @test result.velocity > 0
    end

    @testset "Elliptical Spread" begin
        # No wind - should be nearly circular
        es_nowind = Elmfire.elliptical_spread(10.0, 0.0)
        @test es_nowind.length_to_breadth ≈ 1.0 atol=0.1
        @test es_nowind.head ≈ es_nowind.back atol=0.5

        # With wind - should be elongated
        es_wind = Elmfire.elliptical_spread(10.0, 5.0)
        @test es_wind.length_to_breadth > 1.0
        @test es_wind.head > es_wind.back
        @test es_wind.head > es_wind.flank

        # Higher wind - more elongated (use values below L/B cap of 8.0)
        es_highwind = Elmfire.elliptical_spread(10.0, 8.0)
        @test es_highwind.length_to_breadth > es_wind.length_to_breadth
    end

    @testset "Velocity at Angle" begin
        es = Elmfire.elliptical_spread(100.0, 10.0)

        # Head fire (θ = 0)
        v_head = Elmfire.velocity_at_angle(es, 0.0)
        @test v_head ≈ es.head atol=5.0

        # Backing fire (θ = π)
        v_back = Elmfire.velocity_at_angle(es, Elmfire.PI)
        @test v_back < v_head

        # Flanking (θ = π/2)
        v_flank = Elmfire.velocity_at_angle(es, Elmfire.PI/2)
        @test v_back < v_flank < v_head
    end
end
