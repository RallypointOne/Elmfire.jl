@testset "Monte Carlo Ensemble" begin

    @testset "PerturbationConfig" begin
        config = PerturbationConfig{Float64}(
            ignition_perturb_radius = 5.0,
            wind_speed_factor_range = (0.8, 1.2),
            wind_direction_std = 10.0,
            moisture_factor_range = (0.9, 1.1),
            spread_rate_factor_range = (0.95, 1.05)
        )

        @test config.ignition_perturb_radius == 5.0
        @test config.wind_speed_factor_range == (0.8, 1.2)
        @test config.wind_direction_std == 10.0

        # Default constructor
        default_config = PerturbationConfig{Float64}()
        @test default_config.ignition_perturb_radius == 0.0
    end

    @testset "EnsembleConfig" begin
        pert = PerturbationConfig{Float64}()
        sim_config = SimulationConfig{Float64}()

        config = EnsembleConfig{Float64}(
            n_simulations = 50,
            base_seed = UInt64(12345),
            perturbation = pert,
            simulation_config = sim_config,
            save_individual_results = true
        )

        @test config.n_simulations == 50
        @test config.base_seed == UInt64(12345)
        @test config.save_individual_results == true
    end

    @testset "Weather Perturbation" begin
        using Random

        weather = ConstantWeather{Float64}(
            wind_speed_mph = 10.0,
            wind_direction = 180.0,
            M1 = 0.06,
            M10 = 0.08,
            M100 = 0.10,
            MLH = 0.60,
            MLW = 0.90
        )

        config = PerturbationConfig{Float64}(
            wind_speed_factor_range = (0.5, 1.5),
            wind_direction_std = 20.0,
            moisture_factor_range = (0.8, 1.2)
        )

        rng = MersenneTwister(42)
        perturbed = perturb_weather(weather, config, rng)

        # Wind speed should be within factor range
        @test 5.0 <= perturbed.wind_speed_20ft <= 15.0

        # Wind direction should be perturbed (might wrap)
        @test 0.0 <= perturbed.wind_direction < 360.0

        # Moisture should be within factor range and clamped
        @test 0.01 <= perturbed.M1 <= 0.5

        # Multiple perturbations should vary
        rng2 = MersenneTwister(999)
        perturbed2 = perturb_weather(weather, config, rng2)
        @test perturbed2.wind_speed_20ft != perturbed.wind_speed_20ft
    end

    @testset "Ignition Perturbation" begin
        using Random

        rng = MersenneTwister(42)
        ix, iy = 50, 50
        radius = 10.0
        ncols, nrows = 100, 100

        # With zero radius, should return same point
        new_ix, new_iy = perturb_ignition(ix, iy, 0.0, ncols, nrows, rng)
        @test new_ix == ix
        @test new_iy == iy

        # With radius, should be within bounds and within radius
        rng = MersenneTwister(42)
        new_ix, new_iy = perturb_ignition(ix, iy, radius, ncols, nrows, rng)
        @test 1 <= new_ix <= ncols
        @test 1 <= new_iy <= nrows
        dist = sqrt((new_ix - ix)^2 + (new_iy - iy)^2)
        @test dist <= radius + 1  # +1 for rounding

        # Edge case: near boundary should clamp
        rng = MersenneTwister(42)
        new_ix, new_iy = perturb_ignition(5, 5, 20.0, ncols, nrows, rng)
        @test 1 <= new_ix <= ncols
        @test 1 <= new_iy <= nrows
    end

    @testset "Burn Probability Computation" begin
        ncols, nrows = 20, 20

        # Create mock ensemble members
        members = EnsembleMember{Float64}[]

        for i in 1:10
            burned = falses(ncols, nrows)
            toa = fill(-1.0, ncols, nrows)

            # All members burn center region
            burned[8:12, 8:12] .= true
            toa[8:12, 8:12] .= Float64(i)

            # Half burn additional region
            if i <= 5
                burned[13:15, 8:12] .= true
                toa[13:15, 8:12] .= Float64(i) + 5.0
            end

            push!(members, EnsembleMember{Float64}(
                i, UInt64(i), burned, toa, 100.0 * i, 10.0
            ))
        end

        burn_prob = compute_burn_probability(members, ncols, nrows)

        # Center should have 100% probability
        @test burn_prob[10, 10] == 1.0

        # Extended region should have 50% probability
        @test burn_prob[14, 10] ≈ 0.5

        # Outside region should have 0% probability
        @test burn_prob[1, 1] == 0.0
    end

    @testset "Mean Arrival Time Computation" begin
        ncols, nrows = 10, 10

        members = EnsembleMember{Float64}[]
        for i in 1:5
            burned = trues(ncols, nrows)
            toa = fill(Float64(i * 10), ncols, nrows)

            push!(members, EnsembleMember{Float64}(
                i, UInt64(i), burned, toa, 100.0, 10.0
            ))
        end

        mean_toa, std_toa = compute_mean_arrival_time(members, ncols, nrows)

        # Mean should be (10+20+30+40+50)/5 = 30
        @test mean_toa[5, 5] ≈ 30.0

        # Standard deviation of [10,20,30,40,50] = sqrt(250) ≈ 15.81
        @test 15.0 < std_toa[5, 5] < 16.5
    end

    @testset "EnsembleResult" begin
        config = EnsembleConfig{Float64}(n_simulations = 10)
        result = EnsembleResult{Float64}(config, 50, 50)

        @test size(result.burn_probability) == (50, 50)
        @test isempty(result.members)
        @test result.mean_burned_area == 0.0
    end

    @testset "Small Ensemble Run" begin
        # Set up a small test case
        ncols, nrows = 30, 30
        cellsize = 30.0

        state = FireState{Float64}(ncols, nrows, cellsize)
        fuel_table = create_standard_fuel_table(Float64)
        fuel_ids = fill(1, ncols, nrows)  # Short grass
        slope = zeros(Float64, ncols, nrows)
        aspect = zeros(Float64, ncols, nrows)

        weather = ConstantWeather{Float64}(
            wind_speed_mph = 15.0,
            wind_direction = 0.0,
            M1 = 0.06,
            M10 = 0.08,
            M100 = 0.10,
            MLH = 0.60,
            MLW = 0.90
        )

        perturbation = PerturbationConfig{Float64}(
            ignition_perturb_radius = 2.0,
            wind_speed_factor_range = (0.9, 1.1),
            wind_direction_std = 5.0
        )

        config = EnsembleConfig{Float64}(
            n_simulations = 3,
            base_seed = UInt64(42),
            perturbation = perturbation,
            save_individual_results = true
        )

        result = run_ensemble!(
            config, state, fuel_ids, fuel_table, weather,
            slope, aspect,
            15, 15,  # ignition point
            0.0, 30.0;  # 30 minutes
            show_progress = false
        )

        @test length(result.members) == 3
        @test all(m -> m.burned_area_acres > 0, result.members)
        @test all(0.0 .<= result.burn_probability .<= 1.0)
    end

    @testset "Convergence Check" begin
        config = EnsembleConfig{Float64}(n_simulations = 20)
        result = EnsembleResult{Float64}(config, 50, 50)

        # Add small convergence values
        for _ in 1:15
            push!(result.convergence_history, 0.0001)
        end

        @test check_convergence(result, threshold = 0.001)

        # Add large value
        push!(result.convergence_history, 0.1)
        @test !check_convergence(result, threshold = 0.001)
    end

    @testset "Exceedance Probability" begin
        config = EnsembleConfig{Float64}(n_simulations = 10)
        result = EnsembleResult{Float64}(config, 50, 50)

        # Add mock members with varying burned areas
        for i in 1:10
            push!(result.members, EnsembleMember{Float64}(
                i, UInt64(i),
                falses(50, 50), fill(-1.0, 50, 50),
                Float64(i * 10),  # 10, 20, ..., 100 acres
                10.0
            ))
        end

        # 50% should exceed 50 acres
        @test get_exceedance_probability(result, 50.0) ≈ 0.5

        # 90% should exceed 10 acres
        @test get_exceedance_probability(result, 10.0) ≈ 0.9

        # None should exceed 200 acres
        @test get_exceedance_probability(result, 200.0) == 0.0
    end

    @testset "Float32 Support" begin
        config = PerturbationConfig{Float32}(
            ignition_perturb_radius = 5.0f0,
            wind_speed_factor_range = (0.8f0, 1.2f0)
        )

        @test eltype(config) == Float32
        @test config.ignition_perturb_radius === 5.0f0
    end

end
