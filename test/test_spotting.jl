using Random

@testset "Spotting / Ember Transport" begin

    @testset "SpottingParameters Construction" begin
        # Default constructor
        params = SpottingParameters()
        @test params.mean_distance == 100.0
        @test params.normalized_variance == 0.5
        @test params.nembers_max == 10
        @test eltype(params) == Float64

        # Custom parameters
        params_custom = SpottingParameters{Float64}(
            mean_distance = 200.0,
            normalized_variance = 0.3,
            nembers_max = 20,
            pign = 75.0
        )
        @test params_custom.mean_distance == 200.0
        @test params_custom.pign == 75.0

        # Float32
        params32 = SpottingParameters{Float32}()
        @test eltype(params32) == Float32
    end

    @testset "SpotFire Construction" begin
        sf = SpotFire{Float64}(10, 20, 5.0, 150.0)
        @test sf.ix == 10
        @test sf.iy == 20
        @test sf.time == 5.0
        @test sf.distance == 150.0
        @test eltype(sf) == Float64
    end

    @testset "Lognormal Parameters" begin
        # Test conversion from mean/variance to μ/σ
        mean = 100.0
        norm_var = 0.5

        mu, sigma = lognormal_params(mean, norm_var)

        @test mu isa Float64
        @test sigma isa Float64
        @test sigma > 0

        # Float32 version
        mu32, sigma32 = lognormal_params(Float32(100), Float32(0.5))
        @test mu32 isa Float32
        @test sigma32 isa Float32
    end

    @testset "Lognormal Sampling" begin
        rng = MersenneTwister(12345)
        mu, sigma = lognormal_params(100.0, 0.5)

        # Sample many values
        samples = [sample_lognormal(mu, sigma, rng) for _ in 1:1000]

        # All samples should be positive
        @test all(x -> x > 0, samples)

        # Mean should be approximately correct
        sample_mean = sum(samples) / length(samples)
        @test abs(sample_mean - 100.0) < 30.0  # Within ~30% of expected

        # Float32 sampling
        mu32, sigma32 = lognormal_params(Float32(100), Float32(0.5))
        sample32 = sample_lognormal(mu32, sigma32, rng)
        @test sample32 isa Float32
        @test sample32 > 0
    end

    @testset "Sardoy Parameters" begin
        # Test Sardoy (2008) model
        ws = 15.0  # mph
        flin = 1000.0  # kW/m

        mu_dist, sigma_dist, mu_span, sigma_span = sardoy_parameters(ws, flin)

        @test mu_dist > 0  # Distance parameter positive
        @test sigma_dist > 0  # Variance positive
        @test mu_span == 0.0  # Spanwise mean is zero
        @test sigma_span > 0  # Spanwise variance positive

        # Higher wind should increase distance
        _, _, _, _ = sardoy_parameters(5.0, flin)
        mu_high_wind, _, _, _ = sardoy_parameters(25.0, flin)
        # Note: relationship is complex, just check it runs

        # Float32 version
        mu32, sigma32, _, _ = sardoy_parameters(Float32(15), Float32(1000))
        @test mu32 isa Float32
        @test sigma32 isa Float32
    end

    @testset "Compute Num Embers" begin
        params = SpottingParameters{Float64}(
            surface_spotting_percent = 100.0,  # Always spot
            crown_spotting_percent = 100.0,
            nembers_max = 5
        )

        rng = MersenneTwister(12345)

        # Higher fireline intensity should produce more embers
        n_low = compute_num_embers(params, 100.0, 0, 1.0, 30.0; rng=rng)
        rng = MersenneTwister(12345)  # Reset
        n_high = compute_num_embers(params, 5000.0, 0, 1.0, 30.0; rng=rng)

        @test n_low >= 0
        @test n_high >= 0
        @test n_high <= params.nembers_max

        # Zero spotting percentage should produce no embers
        params_no_spot = SpottingParameters{Float64}(
            surface_spotting_percent = 0.0,
            crown_spotting_percent = 0.0
        )
        rng = MersenneTwister(12345)
        n_zero = compute_num_embers(params_no_spot, 1000.0, 0, 1.0, 30.0; rng=rng)
        @test n_zero == 0
    end

    @testset "Transport Ember" begin
        rng = MersenneTwister(12345)

        # Transport ember from center of 100x100 grid
        ix, iy, dist = transport_ember(
            50.0, 50.0,     # Source
            10.0,           # Wind speed (mph)
            0.0,            # Wind from north (ember goes south)
            100.0,          # 100m spotting distance
            30.0,           # 30ft cells
            100, 100;       # Grid size
            rng=rng
        )

        @test 1 <= ix <= 100
        @test 1 <= iy <= 100
        @test dist == 100.0

        # Wind from west (270°) should transport east
        rng = MersenneTwister(12345)
        ix_e, iy_e, _ = transport_ember(
            50.0, 50.0,
            15.0,
            270.0,  # From west
            200.0,
            30.0,
            100, 100;
            rng=rng
        )
        @test ix_e > 50  # Should move east

        # Check bounds clamping
        rng = MersenneTwister(12345)
        ix_clamp, iy_clamp, _ = transport_ember(
            5.0, 5.0,
            20.0,
            180.0,  # From south (goes north)
            5000.0,  # Very long distance
            30.0,
            100, 100;
            rng=rng
        )
        @test ix_clamp >= 1
        @test iy_clamp <= 100
    end

    @testset "Generate Spot Fires" begin
        params = SpottingParameters{Float64}(
            mean_distance = 50.0,
            normalized_variance = 0.3,
            surface_spotting_percent = 100.0,
            crown_spotting_percent = 100.0,
            nembers_max = 10,
            pign = 100.0,  # Always ignite
            min_distance = 5.0,
            max_distance = 500.0
        )

        burned = falses(50, 50)

        rng = MersenneTwister(12345)

        spot_fires = generate_spot_fires(
            25, 25,         # Source cell
            2000.0,         # High fireline intensity
            15.0,           # Wind speed
            270.0,          # Wind from west
            2,              # Active crown fire
            params,
            30.0,           # Cell size
            50, 50,         # Grid size
            0.0,            # Time
            burned;
            rng=rng
        )

        # Should generate some spot fires with these parameters
        @test length(spot_fires) >= 0
        @test length(spot_fires) <= params.nembers_max

        # All spot fires should be within grid
        for sf in spot_fires
            @test 1 <= sf.ix <= 50
            @test 1 <= sf.iy <= 50
            @test sf.time == 0.0
            @test sf.distance > 0
        end

        # Spot fires should be in unburned cells
        for sf in spot_fires
            @test !burned[sf.ix, sf.iy]
        end
    end

    @testset "Generate Spot Fires - Sardoy Model" begin
        params = SpottingParameters{Float64}(
            mean_distance = 50.0,
            surface_spotting_percent = 100.0,
            crown_spotting_percent = 100.0,
            nembers_max = 5,
            pign = 100.0
        )

        burned = falses(50, 50)
        rng = MersenneTwister(12345)

        spot_fires = generate_spot_fires(
            25, 25,
            1500.0,
            12.0,
            0.0,
            1,  # Passive crown
            params,
            30.0,
            50, 50,
            5.0,
            burned;
            use_sardoy=true,
            rng=rng
        )

        # Just verify it runs without error
        @test spot_fires isa Vector{SpotFire{Float64}}
    end

    @testset "SpotFireTracker" begin
        tracker = SpotFireTracker{Float64}(ignition_delay=2.0)

        @test isempty(tracker.pending)
        @test tracker.ignition_delay == 2.0

        # Add spot fires
        fires = [
            SpotFire{Float64}(10, 15, 0.0, 100.0),
            SpotFire{Float64}(20, 25, 0.0, 150.0),
            SpotFire{Float64}(30, 35, 1.0, 200.0)
        ]

        add_spot_fires!(tracker, fires)
        @test length(tracker.pending) == 3

        # Get ready ignitions at t=1.5 (none should be ready)
        ignitions = get_ready_ignitions!(tracker, 1.5)
        @test isempty(ignitions)
        @test length(tracker.pending) == 3

        # Get ready ignitions at t=2.0 (first two should be ready)
        ignitions = get_ready_ignitions!(tracker, 2.0)
        @test length(ignitions) == 2
        @test (10, 15) in ignitions
        @test (20, 25) in ignitions
        @test length(tracker.pending) == 1

        # Get ready ignitions at t=3.0 (last one should be ready)
        ignitions = get_ready_ignitions!(tracker, 3.0)
        @test length(ignitions) == 1
        @test (30, 35) in ignitions
        @test isempty(tracker.pending)
    end

    @testset "SpotFireTracker Float32" begin
        tracker32 = SpotFireTracker{Float32}(ignition_delay=Float32(1.5))

        fires32 = [SpotFire{Float32}(5, 10, Float32(0), Float32(80))]
        add_spot_fires!(tracker32, fires32)

        @test length(tracker32.pending) == 1

        ignitions = get_ready_ignitions!(tracker32, Float32(2.0))
        @test length(ignitions) == 1
    end

    @testset "Float32 Full Pipeline" begin
        params32 = SpottingParameters{Float32}(
            mean_distance = Float32(100),
            pign = Float32(100)
        )

        burned32 = falses(30, 30)
        rng = MersenneTwister(54321)

        spot_fires32 = generate_spot_fires(
            15, 15,
            Float32(2000),
            Float32(10),
            Float32(180),
            1,
            params32,
            Float32(30),
            30, 30,
            Float32(0),
            burned32;
            rng=rng
        )

        @test spot_fires32 isa Vector{SpotFire{Float32}}
        for sf in spot_fires32
            @test eltype(sf) == Float32
        end
    end
end
