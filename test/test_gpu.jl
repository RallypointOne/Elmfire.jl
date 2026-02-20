@testset "GPU Extension (CPU Backend)" begin
    using KernelAbstractions
    using Adapt

    @testset "simulate_gpu! produces fire spread" begin
        # The GPU path uses true parallel RK2 updates while the CPU path
        # has serial order-dependent updates that produce different (but both
        # valid) results. Test that the GPU path produces reasonable fire spread.
        fuel_table = create_standard_fuel_table()
        fuel_array = FuelModelArray(fuel_table)

        weather = ConstantWeather(
            wind_speed_mph = 10.0,
            wind_direction = 270.0,
            M1 = 0.04,
            M10 = 0.06,
            M100 = 0.08,
            MLH = 0.50,
            MLW = 0.80
        )

        # --- GPU simulation (CPU backend) ---
        state = FireState(30, 30, 10.0)
        ignite!(state, 15, 15, 0.0)
        simulate_gpu_uniform!(
            state, 1, fuel_array, weather,
            0.0, 0.0, 0.0, 3.0;
            dt_initial = 0.2,
            backend = KernelAbstractions.CPU()
        )

        # Should have meaningful fire spread
        @test count(state.burned) > 10
        @test get_burned_area(state) > 10 * state.cellsize^2

        # Should have valid output fields for fire-spread cells (not the ignition cell)
        burned_cells = findall(state.burned)
        for idx in burned_cells
            @test state.time_of_arrival[idx] >= 0.0
            if state.time_of_arrival[idx] > 0.0  # skip ignition cell
                @test state.spread_rate[idx] > 0.0
            end
        end
    end

    @testset "simulate_gpu! with wind shows directionality" begin
        fuel_table = create_standard_fuel_table()
        fuel_array = FuelModelArray(fuel_table)

        weather = ConstantWeather(
            wind_speed_mph = 15.0,
            wind_direction = 270.0,  # from west, fire spreads east
            M1 = 0.04,
            M10 = 0.06,
            M100 = 0.08,
            MLH = 0.50,
            MLW = 0.80
        )

        state = FireState(50, 50, 10.0)
        ignite!(state, 25, 25, 0.0)

        simulate_gpu_uniform!(
            state, 1, fuel_array, weather,
            0.0, 0.0, 0.0, 5.0;
            dt_initial = 0.2,
            backend = KernelAbstractions.CPU()
        )

        # Fire should spread more to the east (downwind)
        east_burned = sum(state.burned[26:end, :])
        west_burned = sum(state.burned[1:24, :])
        @test east_burned >= west_burned
    end

    @testset "simulate_gpu! with slope" begin
        fuel_table = create_standard_fuel_table()
        fuel_array = FuelModelArray(fuel_table)

        weather = ConstantWeather(
            wind_speed_mph = 0.0,
            wind_direction = 0.0,
            M1 = 0.04,
            M10 = 0.06,
            M100 = 0.08,
            MLH = 0.50,
            MLW = 0.80
        )

        state = FireState(40, 40, 10.0)
        ignite!(state, 20, 20, 0.0)

        fuel_ids = fill(1, 40, 40)
        slope = fill(20.0, 40, 40)
        aspect = fill(0.0, 40, 40)

        simulate_gpu!(
            state, fuel_ids, fuel_array, weather, slope, aspect,
            0.0, 5.0;
            dt_initial = 0.2,
            backend = KernelAbstractions.CPU()
        )

        @test count(state.burned) > 1
        @test get_burned_area(state) > state.cellsize^2
    end

    @testset "simulate_gpu! multi-fuel" begin
        fuel_table = create_standard_fuel_table()
        fuel_array = FuelModelArray(fuel_table)

        weather = ConstantWeather(
            wind_speed_mph = 10.0,
            wind_direction = 0.0,
            M1 = 0.06,
            M10 = 0.08,
            M100 = 0.10,
            MLH = 0.60,
            MLW = 0.90
        )

        state = FireState(40, 40, 10.0)
        ignite!(state, 20, 20, 0.0)

        # Mix of fuel types
        fuel_ids = fill(1, 40, 40)
        fuel_ids[1:20, :] .= 5   # FBFM05 (brush)
        fuel_ids[21:40, :] .= 1  # FBFM01 (short grass)

        slope = zeros(40, 40)
        aspect = zeros(40, 40)

        simulate_gpu!(
            state, fuel_ids, fuel_array, weather, slope, aspect,
            0.0, 5.0;
            dt_initial = 0.2,
            backend = KernelAbstractions.CPU()
        )

        @test count(state.burned) > 1
    end

    @testset "simulate_gpu! callback" begin
        fuel_table = create_standard_fuel_table()
        fuel_array = FuelModelArray(fuel_table)

        weather = ConstantWeather(
            wind_speed_mph = 10.0,
            wind_direction = 0.0,
            M1 = 0.06,
            M10 = 0.08,
            M100 = 0.10,
            MLH = 0.60,
            MLW = 0.90
        )

        state = FireState(20, 20, 10.0)
        ignite!(state, 10, 10, 0.0)

        callback_count = Ref(0)
        simulate_gpu_uniform!(
            state, 1, fuel_array, weather,
            0.0, 0.0, 0.0, 2.0;
            dt_initial = 0.5,
            backend = KernelAbstractions.CPU(),
            callback = (s, t, dt, iter) -> callback_count[] += 1
        )

        @test callback_count[] > 0
    end

    @testset "simulate_gpu! timestep behavior" begin
        # GPU uses true parallel RK2 updates which produce different burn
        # patterns than the CPU's serial updates. Both are valid. Verify
        # the GPU timestep machinery works correctly on its own.
        fuel_table = create_standard_fuel_table()
        fuel_array = FuelModelArray(fuel_table)

        weather = ConstantWeather(
            wind_speed_mph = 10.0,
            wind_direction = 270.0,
            M1 = 0.04, M10 = 0.06, M100 = 0.08,
            MLH = 0.50, MLW = 0.80
        )

        state = FireState(30, 30, 10.0)
        ignite!(state, 15, 15, 0.0)
        gpu_dts = Float64[]
        gpu_times = Float64[]
        simulate_gpu_uniform!(
            state, 1, fuel_array, weather, 0.0, 0.0, 0.0, 3.0;
            dt_initial = 0.2,
            backend = KernelAbstractions.CPU(),
            callback = (s, t, dt, iter) -> begin
                push!(gpu_dts, dt)
                push!(gpu_times, t)
            end
        )

        # Should produce reasonable timesteps
        @test length(gpu_dts) > 5
        @test all(dt -> dt > 0, gpu_dts)
        @test all(dt -> dt <= 10.0, gpu_dts)  # bounded by dt_max

        # Should end at or very close to t_stop
        @test gpu_times[end] ≈ 3.0 atol=0.01
    end
end
