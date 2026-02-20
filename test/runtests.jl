using Elmfire
using Test

using Random  # Required for ensemble tests

@testset "Elmfire.jl" begin
    include("test_fuel_models.jl")
    include("test_rothermel.jl")
    include("test_level_set.jl")
    include("test_precision.jl")
    include("test_crown_fire.jl")
    include("test_spotting.jl")
    include("test_weather.jl")
    include("test_simulation_full.jl")

    # Phase 3: Operational features
    include("test_geospatial.jl")
    include("test_ensemble.jl")
    include("test_wui.jl")
    include("test_suppression.jl")

    # GPU extension (tested with CPU backend)
    include("test_gpu.jl")

    @testset "Simulation" begin
        @testset "FireState Construction" begin
            state = FireState(100, 100, 30.0)

            @test state.ncols == 100
            @test state.nrows == 100
            @test state.cellsize == 30.0
            @test state.padding == 2

            # Check dimensions with padding
            @test size(state.phi) == (104, 104)  # 100 + 2*2
            @test size(state.burned) == (100, 100)
        end

        @testset "Ignition" begin
            state = FireState(50, 50, 30.0)

            # Ignite a cell
            ignite!(state, 25, 25, 0.0)

            @test state.burned[25, 25] == true
            @test state.time_of_arrival[25, 25] == 0.0

            # Check phi is negative at ignition
            px, py = grid_to_padded(state, 25, 25)
            @test state.phi[px, py] < 0

            # Check narrow band is populated
            @test !isempty(state.narrow_band.active)
        end

        @testset "Ignite Circle" begin
            state = FireState(50, 50, 30.0)

            ignite_circle!(state, 25, 25, 3.0, 0.0)

            @test state.burned[25, 25] == true
            @test state.burned[26, 25] == true
            @test state.burned[25, 26] == true
            @test state.burned[28, 28] == false  # Outside circle
        end

        @testset "Weather" begin
            weather = ConstantWeather(
                wind_speed_mph = 10.0,
                wind_direction = 270.0,
                M1 = 0.06
            )

            @test weather.wind_speed_20ft == 10.0
            @test weather.wind_direction == 270.0
            @test weather.M1 == 0.06
        end

        @testset "Wind Adjustment Factor" begin
            # Typical fuel bed depth
            waf = wind_adjustment_factor(1.0)
            @test 0.1 < waf < 1.0

            # Shallow fuel bed - lower WAF
            waf_shallow = wind_adjustment_factor(0.2)
            @test waf_shallow < waf

            # Deep fuel bed - higher WAF
            waf_deep = wind_adjustment_factor(3.0)
            @test waf_deep > waf
        end

        @testset "Basic Simulation" begin
            # Create a small simulation with smaller cells for faster spread
            state = FireState(30, 30, 10.0)  # 30x30 grid, 10ft cells

            # Ignite center
            ignite!(state, 15, 15, 0.0)

            # Create fuel table
            fuel_table = create_standard_fuel_table()

            # Set up uniform conditions
            weather = ConstantWeather(
                wind_speed_mph = 10.0,  # Moderate wind for faster spread
                wind_direction = 0.0,   # From north
                M1 = 0.04,              # Dry fuels
                M10 = 0.06,
                M100 = 0.08,
                MLH = 0.50,
                MLW = 0.80
            )

            # Run simulation - need enough time to spread across cells
            # At ~30-50 ft/min spread rate with 10ft cells, ~1min per cell
            simulate_uniform!(
                state,
                1,  # FBFM01
                fuel_table,
                weather,
                0.0,  # Flat
                0.0,  # No aspect
                0.0,  # Start time
                3.0;  # 3 minutes
                dt_initial = 0.2
            )

            # Fire should have spread
            @test get_burned_area(state) > state.cellsize^2  # More than 1 cell

            # Should have multiple burned cells
            burned_count = count(state.burned)
            @test burned_count > 1
        end

        @testset "Fire Perimeter" begin
            state = FireState(20, 20, 30.0)

            # Create a small burned area
            ignite!(state, 10, 10, 0.0)
            ignite!(state, 10, 11, 0.0)
            ignite!(state, 11, 10, 0.0)
            ignite!(state, 11, 11, 0.0)

            perimeter = get_fire_perimeter(state)

            # All 4 cells should be on perimeter (small fire)
            @test length(perimeter) == 4
        end

        @testset "Burned Area Calculation" begin
            state = FireState(20, 20, 30.0)  # 30 ft cells

            ignite!(state, 10, 10, 0.0)
            ignite!(state, 10, 11, 0.0)

            area = get_burned_area(state)
            @test area ≈ 2 * 30.0^2  # 2 cells at 30 ft each

            acres = get_burned_area_acres(state)
            @test acres ≈ area / 43560.0
        end
    end

    @testset "Integration" begin
        @testset "Circular Fire No Wind" begin
            # A fire with no wind on flat ground should spread roughly circularly
            # Use small cells for faster spread
            state = FireState(50, 50, 10.0)  # 10ft cells
            ignite!(state, 25, 25, 0.0)

            fuel_table = create_standard_fuel_table()
            weather = ConstantWeather(
                wind_speed_mph = 0.0,  # No wind
                wind_direction = 0.0,
                M1 = 0.04,   # Dry fuels
                M10 = 0.06,
                M100 = 0.08,
                MLH = 0.50,
                MLW = 0.80
            )

            # With no wind, spread rate is ~5 ft/min (base rate)
            # Need ~2 min per 10ft cell
            simulate_uniform!(
                state, 1, fuel_table, weather,
                0.0, 0.0,  # Flat
                0.0, 5.0;  # 5 minutes
                dt_initial = 0.2
            )

            # Check that fire spread in multiple directions
            @test state.burned[26, 25]  # East
            @test state.burned[24, 25]  # West
            @test state.burned[25, 26]  # North
            @test state.burned[25, 24]  # South
        end

        @testset "Wind-Driven Fire" begin
            # Fire with wind should spread faster downwind
            state = FireState(50, 50, 10.0)  # 10ft cells
            ignite!(state, 25, 25, 0.0)

            fuel_table = create_standard_fuel_table()
            weather = ConstantWeather(
                wind_speed_mph = 15.0,  # Strong wind
                wind_direction = 270.0, # From west (blowing east)
                M1 = 0.04,
                M10 = 0.06,
                M100 = 0.08,
                MLH = 0.50,
                MLW = 0.80
            )

            simulate_uniform!(
                state, 1, fuel_table, weather,
                0.0, 0.0,
                0.0, 5.0;  # 5 minutes
                dt_initial = 0.2
            )

            # Fire should spread more to the east (downwind)
            # Count burned cells east vs west of center
            east_burned = sum(state.burned[26:end, :])
            west_burned = sum(state.burned[1:24, :])

            @test east_burned >= west_burned  # More spread downwind
        end
    end
end
