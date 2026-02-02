@testset "WUI Models" begin

    @testset "WUIBuilding" begin
        building = WUIBuilding{Float64}(
            1, 10, 20;
            construction_type = :wood,
            combustible_fraction = 0.7,
            ignition_temperature = 300.0
        )

        @test building.id == 1
        @test building.ix == 10
        @test building.iy == 20
        @test building.construction_type == :wood
        @test building.combustible_fraction == 0.7
        @test eltype(building) == Float64
    end

    @testset "WUIGrid" begin
        buildings = [
            WUIBuilding{Float64}(1, 10, 10),
            WUIBuilding{Float64}(2, 20, 20),
            WUIBuilding{Float64}(3, 30, 30)
        ]

        grid = WUIGrid{Float64}(buildings, 50, 50)

        @test length(grid.buildings) == 3
        @test grid.building_map[10, 10] == true
        @test grid.building_map[15, 15] == false
        @test grid.building_ids[20, 20] == 2
        @test all(.!grid.ignited)

        # Test empty grid
        empty_grid = WUIGrid{Float64}(50, 50)
        @test isempty(empty_grid.buildings)
    end

    @testset "Radiative Heat Flux" begin
        # Test basic calculation
        flin = 1000.0  # kW/m
        distance = 10.0  # m
        flame_length = 5.0  # m

        q = compute_radiative_heat_flux(flin, distance, flame_length)

        @test q > 0.0
        @test q < flin  # Should be less than source intensity

        # Heat flux should decrease with distance
        q_close = compute_radiative_heat_flux(flin, 5.0, flame_length)
        q_far = compute_radiative_heat_flux(flin, 20.0, flame_length)
        @test q_close > q_far

        # Zero inputs should give zero output
        @test compute_radiative_heat_flux(0.0, distance, flame_length) == 0.0
        @test compute_radiative_heat_flux(flin, 0.0, flame_length) == 0.0
        @test compute_radiative_heat_flux(flin, distance, 0.0) == 0.0
    end

    @testset "View Factor" begin
        # Basic view factor test
        vf = compute_view_factor(5.0, 10.0)
        @test 0.0 <= vf <= 1.0

        # Larger flames have larger view factor at same distance
        vf_small = compute_view_factor(2.0, 10.0)
        vf_large = compute_view_factor(10.0, 10.0)
        @test vf_large > vf_small

        # Closer distance has larger view factor
        vf_close = compute_view_factor(5.0, 5.0)
        vf_far = compute_view_factor(5.0, 20.0)
        @test vf_close > vf_far

        # Edge cases
        @test compute_view_factor(0.0, 10.0) == 0.0
        @test compute_view_factor(10.0, 0.0) == 0.0
    end

    @testset "Building Ignition Probability" begin
        wood_building = WUIBuilding{Float64}(1, 10, 10, :wood, 0.7, 300.0)
        masonry_building = WUIBuilding{Float64}(2, 20, 20, :masonry, 0.3, 400.0)

        # Low heat flux, short exposure - low probability
        p_low = building_ignition_probability(wood_building, 5.0, 1.0)
        @test p_low < 0.5

        # High heat flux, long exposure - high probability
        p_high = building_ignition_probability(wood_building, 50.0, 30.0)
        @test p_high > 0.5

        # Masonry should be more resistant
        p_wood = building_ignition_probability(wood_building, 30.0, 10.0)
        p_masonry = building_ignition_probability(masonry_building, 30.0, 10.0)
        @test p_masonry < p_wood

        # Zero inputs
        @test building_ignition_probability(wood_building, 0.0, 10.0) == 0.0
        @test building_ignition_probability(wood_building, 30.0, 0.0) == 0.0
    end

    @testset "Hamada Spread Probability" begin
        source = WUIBuilding{Float64}(1, 10, 10, :wood, 0.8, 300.0)
        # Use small cell size (1 ft) so buildings are close in meters
        target = WUIBuilding{Float64}(2, 12, 10, :wood, 0.7, 300.0)
        params = HamadaParameters{Float64}(critical_separation = 20.0)  # Larger separation for test

        # Test spread probability with small cells (close buildings)
        prob = hamada_spread_probability(
            source, target, params,
            10.0,  # wind speed
            0.0,   # wind direction
            1.0    # small cellsize = 1 ft ≈ 0.3m per cell
        )

        @test 0.0 <= prob <= 1.0

        # Far building should have lower probability (but still > 0 due to larger critical separation)
        far_target = WUIBuilding{Float64}(3, 25, 10, :wood, 0.7, 300.0)
        prob_far = hamada_spread_probability(
            source, far_target, params,
            10.0, 0.0, 1.0
        )
        @test prob_far <= prob

        # Beyond critical separation should have zero probability
        very_far_target = WUIBuilding{Float64}(4, 200, 10, :wood, 0.7, 300.0)
        prob_very_far = hamada_spread_probability(
            source, very_far_target, params,
            10.0, 0.0, 1.0
        )
        @test prob_very_far == 0.0
    end

    @testset "HamadaParameters" begin
        params = HamadaParameters{Float64}(
            critical_separation = 15.0,
            wind_spread_factor = 2.0
        )

        @test params.critical_separation == 15.0
        @test params.wind_spread_factor == 2.0

        # Default parameters
        default_params = HamadaParameters{Float64}()
        @test default_params.critical_separation == 10.0
    end

    @testset "Building Grid Creation" begin
        buildings = create_building_grid(
            Float64,
            100, 100,
            20,   # spacing
            5,    # footprint
            10, 10;  # origin
            construction_type = :masonry
        )

        @test !isempty(buildings)
        @test all(b -> b.construction_type == :masonry, buildings)

        # Check spacing
        if length(buildings) >= 2
            b1, b2 = buildings[1], buildings[2]
            # Buildings should be spaced apart
            dist = sqrt((b2.ix - b1.ix)^2 + (b2.iy - b1.iy)^2)
            @test dist >= 15  # At least spacing - footprint
        end
    end

    @testset "WUI Statistics" begin
        buildings = [
            WUIBuilding{Float64}(1, 10, 10, :wood, 0.7, 300.0),
            WUIBuilding{Float64}(2, 20, 20, :wood, 0.7, 300.0),
            WUIBuilding{Float64}(3, 30, 30, :masonry, 0.3, 400.0)
        ]

        grid = WUIGrid{Float64}(buildings, 50, 50)

        # Ignite first two buildings
        grid.ignited[1] = true
        grid.ignition_time[1] = 10.0
        grid.ignited[2] = true
        grid.ignition_time[2] = 20.0

        stats = get_wui_statistics(grid)

        @test stats.total_buildings == 3
        @test stats.ignited_buildings == 2
        @test stats.ignition_fraction ≈ 2/3
        @test stats.mean_ignition_time ≈ 15.0
        @test stats.first_ignition_time ≈ 10.0
        @test stats.last_ignition_time ≈ 20.0
        @test stats.wood_ignited == 2
        @test stats.masonry_ignited == 0
    end

    @testset "Float32 Support" begin
        building = WUIBuilding{Float32}(1, 10, 10)
        @test eltype(building) == Float32

        grid = WUIGrid{Float32}(50, 50)
        @test eltype(grid) == Float32

        params = HamadaParameters{Float32}()
        @test eltype(params) == Float32
    end

end
