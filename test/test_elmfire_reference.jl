#-----------------------------------------------------------------------------#
#                     ELMFIRE Reference Value Regression Tests
#-----------------------------------------------------------------------------#
# Values here are computed independently from the ELMFIRE Fortran source
# (elmfire/build/source/*.f90) and its technical reference, not from Elmfire.jl.
# They pin the behaviours that a prior audit found had drifted.

using Test
using Elmfire

@testset "ELMFIRE Reference Values" begin

    @testset "Length-to-breadth uses effective wind in ELMFIRE units" begin
        # elmfire_level_set.f90:1810 with WSMFEFF_LOW_MULT = 5.07955E-3.
        # 15 mph mid-flame = 1320 ft/min -> U = 6.7050 (m/s)
        # L/B = 0.936*exp(0.2566*U) + 0.461*exp(-0.1548*U) - 0.397
        @test length_to_breadth(1320.0) ≈ 4.9963 atol = 1e-3
        @test length_to_breadth(440.0) ≈ 1.5900 atol = 1e-3
        @test length_to_breadth(880.0) ≈ 2.7810 atol = 1e-3

        # Zero wind gives a circle
        @test length_to_breadth(0.0) ≈ 1.0 atol = 1e-6

        # MAX_LOW default of 8
        @test length_to_breadth(1e6) == 8.0
        @test length_to_breadth(1e6; lb_cap = 4.0) == 4.0
    end

    @testset "Backing rate matches ELMFIRE BOH" begin
        # elmfire_level_set.f90:1813-1817
        boh(LB) = (LB - sqrt(LB^2 - 1)) / (LB + sqrt(LB^2 - 1))

        for wsmfeff in (300.0, 600.0, 900.0, 1500.0)
            es = elliptical_spread(100.0, wsmfeff)
            @test es.back / es.head ≈ boh(es.length_to_breadth) rtol = 1e-10
        end

        # Spot value: L/B = 2 -> (2-sqrt3)/(2+sqrt3)
        es2 = EllipticalSpread{Float64}(100.0, 0.0, 0.0, sqrt(3.0) / 2, 2.0)
        @test boh(2.0) ≈ 0.0717968 atol = 1e-6
        @test (1 - es2.eccentricity) / (1 + es2.eccentricity) ≈ boh(2.0) atol = 1e-10
    end

    @testset "Wind and slope combine as vectors" begin
        # Wind from the west (270) pushes east; aspect 90 (east-facing) pushes west.
        # Opposed factors cancel: |phi| = |phis - phiw|
        phimag, dx, dy = spread_direction(2.0, 3.0, 90.0, 270.0)
        @test phimag ≈ 1.0 atol = 1e-10

        # Aligned: aspect 270 (west-facing slope) pushes east with the wind
        phimag, dx, dy = spread_direction(2.0, 3.0, 270.0, 270.0)
        @test phimag ≈ 5.0 atol = 1e-10
        @test dx ≈ 1.0 atol = 1e-9    # heading east
        @test dy ≈ 0.0 atol = 1e-9

        # Perpendicular
        phimag, _, _ = spread_direction(2.0, 3.0, 180.0, 270.0)
        @test phimag ≈ sqrt(13.0) atol = 1e-10

        # Wind only, from the north: fire heads south
        _, dx, dy = spread_direction(0.0, 1.0, 0.0, 0.0)
        @test dx ≈ 0.0 atol = 1e-9
        @test dy ≈ -1.0 atol = 1e-9
    end

    @testset "Effective wind speed inverts the wind factor" begin
        table = create_standard_fuel_table()
        fm = get_fuel_model(table, 1, 60)

        # With no slope, |phi| = phiw, so the inversion must return wsmf exactly
        for wsmf in (100.0, 500.0, 1200.0)
            phiw = fm.phiwterm * wsmf^fm.B
            @test effective_wind_speed(phiw, fm.wsmfeff_coeff, fm.B_inverse) ≈ wsmf rtol = 1e-9
        end

        # Slope raises the effective wind above the actual wind
        wsmf = 500.0
        phiw = fm.phiwterm * wsmf^fm.B
        @test effective_wind_speed(phiw + 1.0, fm.wsmfeff_coeff, fm.B_inverse) > wsmf
    end

    @testset "Slope projection factors" begin
        # Flat ground is a no-op
        @test all(slope_projection_factors(0.0, 137.0) .≈ (1.0, 1.0))

        # East-facing 30-degree slope: x shrinks by cos(gamma), y is untouched
        ux, uy = slope_projection_factors(30.0, 90.0)
        @test ux ≈ cosd(30.0) atol = 1e-10
        @test uy ≈ 1.0 atol = 1e-10

        # North-facing: the roles swap
        ux, uy = slope_projection_factors(30.0, 0.0)
        @test ux ≈ 1.0 atol = 1e-10
        @test uy ≈ cosd(30.0) atol = 1e-10
    end

    @testset "Wind adjustment factor" begin
        # Unsheltered, elmfire_init.f90:632-641 with H_f/H = 1
        waf_expected(H) = 1.36 * (log(1.36 / 0.13) - 1) / log((20 + 0.36H) / (0.13H))
        for H in (0.2, 1.0, 3.0, 6.0)
            @test wind_adjustment_factor(H) ≈ waf_expected(H) rtol = 1e-12
        end
        @test wind_adjustment_factor(1.0) ≈ 0.362674 atol = 1e-5
        @test wind_adjustment_factor(0.0) == 0.0

        # Sheltered, elmfire_init.f90:624-631 with CROWN_RATIO = 1
        function sheltered(cc, ch_m, crown_ratio = 1.0)
            hft = ch_m / 0.3048
            (1 / log((20 + 0.36hft) / (0.13hft))) * 0.555 / sqrt(0.3333 * cc * crown_ratio * hft)
        end
        @test wind_adjustment_factor(1.0, 0.5, 15.0) ≈ sheltered(0.5, 15.0) rtol = 1e-12
        @test wind_adjustment_factor(1.0, 0.5, 15.0) ≈ 0.10924 atol = 1e-4
        @test wind_adjustment_factor(1.0, 0.7, 18.0) ≈ sheltered(0.7, 18.0) rtol = 1e-12

        # crown_ratio is honoured
        @test wind_adjustment_factor(1.0, 0.5, 15.0; crown_ratio = 1/3) ≈
              sheltered(0.5, 15.0, 1/3) rtol = 1e-12

        # No canopy falls back to the unsheltered profile
        @test wind_adjustment_factor(1.0, 0.0, 15.0) == wind_adjustment_factor(1.0)
    end

    @testset "Crown fire limits match ELMFIRE namelist defaults" begin
        canopy = CanopyProperties{Float64}(cbd = 0.2, cbh = 2.0, cc = 0.7, ch = 20.0)
        # Extreme wind: CROSA must be clipped at CROWN_FIRE_SPREAD_RATE_LIMIT = 250
        r = crown_spread_rate(canopy, 5000.0, 60.0, 0.03, 20.0)
        @test r.spread_rate <= 250.0

        # CRITICAL_CANOPY_COVER default is 0.39, not 0.40
        canopy_low = CanopyProperties{Float64}(cbd = 0.2, cbh = 2.0, cc = 0.395, ch = 20.0)
        r_low = crown_spread_rate(canopy_low, 5000.0, 30.0, 0.03, 20.0)
        @test r_low.spread_rate > 0.0
    end

    @testset "Non-burnable fuel codes" begin
        # elmfire_init.f90:156-166 treats 90-100, <=0 and 256 as non-burnable
        for id in (0, -9999, 91, 98, 100, 256)
            @test Elmfire.isnonburnable(id)
        end
        for id in (1, 13, 89, 101, 165, 204)
            @test !Elmfire.isnonburnable(id)
        end

        # LANDFIRE urban/water codes resolve to the non-burnable model
        table = create_standard_fuel_table()
        fm = Elmfire.get_fuel_model_or_nonburnable(table, 91, 60)
        @test Elmfire.isnonburnable(fm)
        @test_throws KeyError Elmfire.get_fuel_model_or_nonburnable(table, 165, 60)
    end

    @testset "Level set is isotropic" begin
        # The RK2 update reads phi through a 5-point stencil, so gradients must be
        # computed for every active cell before any of them is written. Fusing the
        # two passes makes the result depend on iteration order and biases spread
        # toward -x and -y. Opposite winds must mirror.
        table = create_standard_fuel_table()
        function burn(wd)
            state = FireState(80, 80, 30.0)
            weather = ConstantWeather(
                wind_speed_mph = 15.0, wind_direction = wd,
                M1 = 0.06, M10 = 0.08, M100 = 0.10, MLH = 0.60, MLW = 0.90
            )
            ignite!(state, 40, 40, 0.0)
            simulate_uniform!(state, 1, table, weather, 0.0, 0.0, 0.0, 25.0)
            count(state.burned)
        end

        cardinal = [burn(wd) for wd in (0.0, 90.0, 180.0, 270.0)]
        diagonal = [burn(wd) for wd in (45.0, 135.0, 225.0, 315.0)]

        # Every cardinal run is the same fire in a different orientation
        @test maximum(cardinal) / minimum(cardinal) < 1.05
        @test maximum(diagonal) / minimum(diagonal) < 1.05
        @test minimum(cardinal) > 100   # guard against all four collapsing to nothing
    end

    @testset "Burn stays connected" begin
        # With uniform fuel and no spotting the burn must be simply connected.
        # A band wider than ELMFIRE's BANDTHICKNESS=2 tags cells while their phi is
        # still the far-field value, and advection across that jump flips isolated
        # cells negative — detached islands ahead of a diagonally-driven fire.
        table = create_standard_fuel_table()

        function n_detached(wd)
            state = FireState(80, 80, 30.0)
            weather = ConstantWeather(
                wind_speed_mph = 15.0, wind_direction = wd,
                M1 = 0.06, M10 = 0.08, M100 = 0.10, MLH = 0.60, MLW = 0.90
            )
            ignite!(state, 40, 40, 0.0)
            simulate_uniform!(state, 1, table, weather, 0.0, 0.0, 0.0, 25.0)

            B = state.burned
            seen = falses(size(B))
            sizes = Int[]
            for I in CartesianIndices(B)
                (B[I] && !seen[I]) || continue
                c = 0
                stack = [I]
                seen[I] = true
                while !isempty(stack)
                    J = pop!(stack)
                    c += 1
                    for d in (CartesianIndex(1, 0), CartesianIndex(-1, 0),
                              CartesianIndex(0, 1), CartesianIndex(0, -1))
                        K = J + d
                        if checkbounds(Bool, B, K) && B[K] && !seen[K]
                            seen[K] = true
                            push!(stack, K)
                        end
                    end
                end
                push!(sizes, c)
            end
            sort!(sizes, rev = true)
            sum(sizes[2:end]; init = 0)
        end

        # Diagonals are where the artifact showed up
        for wd in (0.0, 45.0, 135.0, 225.0, 315.0)
            @test n_detached(wd) == 0
        end
    end

    @testset "Fireline intensity varies around the perimeter" begin
        # Head fire intensity must exceed backing intensity for a wind-driven fire
        state = FireState(60, 60, 30.0)
        table = create_standard_fuel_table()
        weather = ConstantWeather(
            wind_speed_mph = 12.0, wind_direction = 270.0,
            M1 = 0.05, M10 = 0.06, M100 = 0.08, MLH = 0.60, MLW = 0.90
        )
        ignite!(state, 30, 30, 0.0)
        simulate_uniform!(state, 1, table, weather, 0.0, 0.0, 0.0, 20.0)

        # Along the row through the ignition point, wind from the west drives the
        # fire further east than west, and the head burns hotter than the back.
        row = [x for x in 1:state.ncols if state.burned[x, 30]]
        @test !isempty(row)
        east = maximum(row)
        west = minimum(row)
        @test east - 30 > 30 - west
        @test state.fireline_intensity[east, 30] > state.fireline_intensity[west, 30]
        # A head/back intensity ratio this large is only possible once fireline
        # intensity is built from the local spread rate rather than the head rate
        @test state.fireline_intensity[east, 30] > 2 * state.fireline_intensity[west, 30]
    end

    @testset "Slope steers the fire with no wind" begin
        # Aspect 180 (south-facing) means the fire runs upslope, toward -y... the
        # push direction is aspect - 180 = 0 degrees, i.e. north (+y).
        state = FireState(60, 60, 30.0)
        table = create_standard_fuel_table()
        weather = ConstantWeather(
            wind_speed_mph = 0.0, wind_direction = 0.0,
            M1 = 0.05, M10 = 0.06, M100 = 0.08, MLH = 0.60, MLW = 0.90
        )
        ignite!(state, 30, 30, 0.0)
        simulate_uniform!(state, 1, table, weather, 25.0, 180.0, 0.0, 25.0)

        col = [y for y in 1:state.nrows if state.burned[30, y]]
        @test !isempty(col)
        north = maximum(col)
        south = minimum(col)
        @test north - 30 > 30 - south
    end
end
