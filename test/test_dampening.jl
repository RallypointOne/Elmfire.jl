@testset "Spread Rate Dampening" begin
    @testset "SpreadRateDampeningConfig Construction" begin
        # Default: NO_DAMPENING
        cfg = SpreadRateDampeningConfig{Float64}()
        @test cfg.mode == NO_DAMPENING
        @test cfg.max_spread_rate == 250.0
        @test cfg.linear_threshold == 400.0

        # Convenience constructor
        cfg2 = SpreadRateDampeningConfig(WIND_SPEED_CAP)
        @test cfg2.mode == WIND_SPEED_CAP
        @test eltype(cfg2) == Float64

        # Custom parameters
        cfg3 = SpreadRateDampeningConfig{Float64}(ABSOLUTE_CAP, max_spread_rate=200.0)
        @test cfg3.max_spread_rate == 200.0

        cfg4 = SpreadRateDampeningConfig{Float64}(LINEAR_DAMPENING, linear_threshold=300.0)
        @test cfg4.linear_threshold == 300.0

        # Float32
        cfg5 = SpreadRateDampeningConfig{Float32}()
        @test eltype(cfg5) == Float32
    end

    @testset "apply_spread_rate_dampening — NO_DAMPENING" begin
        cfg = SpreadRateDampeningConfig{Float64}()
        # Should return velocity unchanged
        vel = apply_spread_rate_dampening(100.0, 500.0, 10.0, 0.5, 8.0, 0.01, 1.5, cfg)
        @test vel == 100.0

        vel2 = apply_spread_rate_dampening(999.0, 200.0, 5.0, 0.0, 50.0, 0.01, 1.5, cfg)
        @test vel2 == 999.0
    end

    @testset "apply_spread_rate_dampening — WIND_SPEED_CAP" begin
        cfg = SpreadRateDampeningConfig{Float64}(WIND_SPEED_CAP)

        # velocity < wsmf: no change
        vel = apply_spread_rate_dampening(100.0, 500.0, 10.0, 0.5, 8.0, 0.01, 1.5, cfg)
        @test vel == 100.0

        # velocity > wsmf: capped at wsmf
        vel2 = apply_spread_rate_dampening(600.0, 400.0, 10.0, 0.5, 8.0, 0.01, 1.5, cfg)
        @test vel2 == 400.0

        # velocity == wsmf: unchanged
        vel3 = apply_spread_rate_dampening(300.0, 300.0, 10.0, 0.5, 8.0, 0.01, 1.5, cfg)
        @test vel3 == 300.0
    end

    @testset "apply_spread_rate_dampening — ABSOLUTE_CAP" begin
        cfg = SpreadRateDampeningConfig{Float64}(ABSOLUTE_CAP, max_spread_rate=200.0)

        # velocity < cap: no change
        vel = apply_spread_rate_dampening(150.0, 500.0, 10.0, 0.5, 8.0, 0.01, 1.5, cfg)
        @test vel == 150.0

        # velocity > cap: capped
        vel2 = apply_spread_rate_dampening(300.0, 500.0, 10.0, 0.5, 8.0, 0.01, 1.5, cfg)
        @test vel2 == 200.0
    end

    @testset "apply_spread_rate_dampening — LINEAR_DAMPENING" begin
        cfg = SpreadRateDampeningConfig{Float64}(LINEAR_DAMPENING, linear_threshold=400.0)

        # Below threshold: velocity unchanged
        vs0 = 10.0
        phis = 0.5
        phiwterm = 0.01
        B = 1.5
        wsmf_low = 300.0
        phiw_low = phiwterm * wsmf_low^B
        velocity_low = vs0 * (1.0 + phis + phiw_low)
        vel = apply_spread_rate_dampening(velocity_low, wsmf_low, vs0, phis, phiw_low, phiwterm, B, cfg)
        @test vel == velocity_low

        # Above threshold: linear extrapolation gives lower velocity than power law
        wsmf_high = 600.0
        phiw_high = phiwterm * wsmf_high^B  # What power law would give
        velocity_high = vs0 * (1.0 + phis + phiw_high)
        vel2 = apply_spread_rate_dampening(velocity_high, wsmf_high, vs0, phis, phiw_high, phiwterm, B, cfg)
        @test vel2 < velocity_high  # Linear dampening should reduce velocity

        # At threshold: should equal original velocity
        wsmf_thresh = 400.0
        phiw_thresh = phiwterm * wsmf_thresh^B
        velocity_thresh = vs0 * (1.0 + phis + phiw_thresh)
        vel3 = apply_spread_rate_dampening(velocity_thresh, wsmf_thresh, vs0, phis, phiw_thresh, phiwterm, B, cfg)
        @test vel3 ≈ velocity_thresh
    end

    @testset "Integration: WIND_SPEED_CAP burns less area" begin
        fuel_table = create_standard_fuel_table()

        weather = ConstantWeather(
            wind_speed_mph = 40.0,
            wind_direction = 270.0,
            M1 = 0.04,
            M10 = 0.06,
            M100 = 0.08,
            MLH = 0.50,
            MLW = 0.80
        )

        # Run without dampening
        state_no = FireState(30, 30, 10.0)
        ignite!(state_no, 15, 15, 0.0)
        simulate_uniform!(state_no, 1, fuel_table, weather,
            0.0, 0.0, 0.0, 3.0;
            dt_initial = 0.2,
            dampening = SpreadRateDampeningConfig{Float64}(NO_DAMPENING))
        area_no = get_burned_area(state_no)

        # Run with WIND_SPEED_CAP dampening
        state_cap = FireState(30, 30, 10.0)
        ignite!(state_cap, 15, 15, 0.0)
        simulate_uniform!(state_cap, 1, fuel_table, weather,
            0.0, 0.0, 0.0, 3.0;
            dt_initial = 0.2,
            dampening = SpreadRateDampeningConfig{Float64}(WIND_SPEED_CAP))
        area_cap = get_burned_area(state_cap)

        @test area_cap <= area_no
    end
end
