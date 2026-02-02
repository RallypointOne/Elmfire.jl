@testset "Suppression Models" begin

    @testset "SuppressionResource" begin
        # Test hand crew
        crew = SuppressionResource{Float64}(1, :hand_crew; location_x = 10.0, location_y = 20.0)

        @test crew.id == 1
        @test crew.resource_type == :hand_crew
        @test crew.location_x == 10.0
        @test crew.line_production_rate == 2.5  # Default for hand crew
        @test crew.status == :available
        @test crew.fatigue == 0.0

        # Test dozer
        dozer = SuppressionResource{Float64}(2, :dozer)
        @test dozer.line_production_rate == 20.0  # Default for dozer
        @test dozer.effective_width == 15.0

        # Test engine
        engine = SuppressionResource{Float64}(3, :engine)
        @test engine.line_production_rate == 5.0

        # Test aircraft
        aircraft = SuppressionResource{Float64}(4, :aircraft)
        @test aircraft.line_production_rate == 100.0
    end

    @testset "ContainmentLine" begin
        cells = [(10, 10), (11, 10), (12, 10)]
        line = ContainmentLine{Float64}(
            cells;
            effectiveness = 0.95,
            width = 10.0,
            construction_time = 30.0,
            resource_id = 1
        )

        @test length(line.cells) == 3
        @test line.effectiveness == 0.95
        @test line.width == 10.0
        @test line.construction_time == 30.0
    end

    @testset "SuppressionState" begin
        state = SuppressionState{Float64}(50, 50)

        @test size(state.contained_cells) == (50, 50)
        @test all(.!state.contained_cells)
        @test isempty(state.resources)
        @test state.total_line_constructed == 0.0

        # Add a resource
        crew = SuppressionResource{Float64}(1, :hand_crew)
        add_resource!(state, crew)
        @test length(state.resources) == 1
    end

    @testset "Resource Assignment" begin
        state = SuppressionState{Float64}(50, 50)
        crew = SuppressionResource{Float64}(1, :hand_crew; location_x = 10.0, location_y = 10.0)
        add_resource!(state, crew)

        # Assign resource
        targets = [(20, 20), (30, 30)]
        assign_resource!(state, 1, targets)

        @test state.resources[1].status == :deployed
        @test haskey(state.active_assignments, 1)
        @test length(state.active_assignments[1]) == 2
    end

    @testset "Line Construction" begin
        ncols, nrows = 50, 50
        state = SuppressionState{Float64}(ncols, nrows)
        cellsize = 30.0  # feet

        crew = SuppressionResource{Float64}(1, :hand_crew; location_x = 10.0, location_y = 10.0)
        crew.status = :deployed
        add_resource!(state, crew)

        # Construct line
        cells, length_constructed = construct_containment_line!(
            state, crew,
            10, 10,  # start
            15, 10,  # target
            10.0,    # dt (minutes)
            cellsize,
            0.0      # current time
        )

        # Should have constructed some line
        @test length_constructed > 0.0
        @test !isempty(cells)
        @test state.total_line_constructed > 0.0

        # Cells should be marked as contained
        @test any(state.contained_cells)

        # Containment effectiveness should be reduced
        @test any(state.containment_effectiveness .< 1.0)
    end

    @testset "Containment Application" begin
        ncols, nrows = 20, 20
        cellsize = 30.0

        fire_state = FireState{Float64}(ncols, nrows, cellsize)
        supp_state = SuppressionState{Float64}(ncols, nrows)

        # Set some velocities
        for ix in 1:ncols
            for iy in 1:nrows
                px, py = grid_to_padded(fire_state, ix, iy)
                fire_state.ux[px, py] = 10.0
                fire_state.uy[px, py] = 5.0
            end
        end

        # Mark some cells as contained
        supp_state.contained_cells[10, 10] = true
        supp_state.containment_effectiveness[10, 10] = 0.1  # 90% reduction

        # Apply containment
        apply_containment!(fire_state, supp_state)

        # Check that velocities were reduced
        px, py = grid_to_padded(fire_state, 10, 10)
        @test fire_state.ux[px, py] ≈ 1.0  # 10 * 0.1
        @test fire_state.uy[px, py] ≈ 0.5  # 5 * 0.1

        # Uncontained cells should be unchanged
        px2, py2 = grid_to_padded(fire_state, 5, 5)
        @test fire_state.ux[px2, py2] == 10.0
    end

    @testset "Indirect Attack Planning" begin
        ncols, nrows = 50, 50
        cellsize = 30.0

        fire_state = FireState{Float64}(ncols, nrows, cellsize)

        # Create a small fire
        ignite!(fire_state, 25, 25, 0.0)
        for ix in 23:27, iy in 23:27
            fire_state.burned[ix, iy] = true
        end

        weather = ConstantWeather{Float64}(
            wind_speed_mph = 10.0,
            wind_direction = 180.0  # Wind from south
        )

        # Plan indirect attack
        line_cells = plan_indirect_attack(fire_state, weather, 5)

        # Should have some cells (unless fire is at boundary)
        # The exact cells depend on perimeter and wind direction
        @test line_cells isa Vector{Tuple{Int,Int}}
    end

    @testset "Direct Attack Planning" begin
        ncols, nrows = 50, 50
        cellsize = 30.0

        fire_state = FireState{Float64}(ncols, nrows, cellsize)

        # Create a small fire
        for ix in 20:30, iy in 20:30
            fire_state.burned[ix, iy] = true
        end

        # Plan direct attack
        perimeter = plan_direct_attack(fire_state)

        # Should return perimeter cells
        @test !isempty(perimeter)
        @test all(cell -> fire_state.burned[cell[1], cell[2]], perimeter)
    end

    @testset "Suppression Statistics" begin
        ncols, nrows = 50, 50
        state = SuppressionState{Float64}(ncols, nrows)

        # Add some resources
        crew1 = SuppressionResource{Float64}(1, :hand_crew)
        crew1.status = :deployed
        crew1.fatigue = 0.3

        crew2 = SuppressionResource{Float64}(2, :hand_crew)
        crew2.status = :available
        crew2.fatigue = 0.1

        dozer = SuppressionResource{Float64}(3, :dozer)
        dozer.status = :resting
        dozer.fatigue = 0.9

        add_resource!(state, crew1)
        add_resource!(state, crew2)
        add_resource!(state, dozer)

        # Add some containment
        state.total_line_constructed = 2640.0  # 0.5 miles
        state.contained_cells[1:10, 1:10] .= true

        stats = get_suppression_statistics(state)

        @test stats.total_resources == 3
        @test stats.available == 1
        @test stats.deployed == 1
        @test stats.resting == 1
        @test stats.contained_cells == 100
        @test stats.total_line_feet == 2640.0
        @test stats.total_line_miles ≈ 0.5
        @test 0.3 < stats.mean_fatigue < 0.5  # (0.3 + 0.1 + 0.9) / 3 ≈ 0.43
    end

    @testset "Fatigue and Resting" begin
        ncols, nrows = 20, 20
        fire_state = FireState{Float64}(ncols, nrows, 30.0)
        supp_state = SuppressionState{Float64}(ncols, nrows)

        crew = SuppressionResource{Float64}(1, :hand_crew)
        crew.status = :resting
        crew.fatigue = 0.5
        add_resource!(supp_state, crew)

        # Update state should reduce fatigue
        update_suppression_state!(supp_state, fire_state, 60.0, 0.0)

        @test supp_state.resources[1].fatigue < 0.5

        # After enough rest, should become available
        supp_state.resources[1].fatigue = 0.1
        update_suppression_state!(supp_state, fire_state, 60.0, 60.0)
        @test supp_state.resources[1].status == :available
    end

    @testset "Float32 Support" begin
        resource = SuppressionResource{Float32}(1, :hand_crew)
        @test eltype(resource) == Float32

        state = SuppressionState{Float32}(50, 50)
        @test eltype(state) == Float32

        line = ContainmentLine{Float32}([(1,1)]; effectiveness = 0.9f0)
        @test eltype(line) == Float32
    end

end
