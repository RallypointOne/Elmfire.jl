@testset "Level Set Solver" begin

    @testset "Half Superbee Limiter" begin
        # Test key properties of the limiter

        # At r = 0, limiter should be 0
        @test Elmfire.half_superbee(0.0) == 0.0

        # At r = 1, limiter should be 0.5
        @test Elmfire.half_superbee(1.0) == 0.5

        # Negative r should give 0
        @test Elmfire.half_superbee(-1.0) == 0.0

        # Large positive r should be capped
        @test Elmfire.half_superbee(10.0) == 1.0

        # Intermediate values
        @test 0.0 < Elmfire.half_superbee(0.5) <= 0.5
        @test 0.0 < Elmfire.half_superbee(2.0) <= 1.0
    end

    @testset "Limit Gradients" begin
        # Create a simple phi field with linear gradient
        phi = zeros(10, 10)
        for i in 1:10
            phi[i, :] .= Float64(i)
        end

        # Positive ux (flow from left to right)
        dphidx, dphidy = Elmfire.limit_gradients(phi, 1.0, 0.0, 5, 5, 1.0)
        @test dphidx ≈ 1.0 atol=0.1  # Gradient is 1 per cell
        @test abs(dphidy) < 0.1  # No gradient in y

        # Negative ux (flow from right to left)
        dphidx2, dphidy2 = Elmfire.limit_gradients(phi, -1.0, 0.0, 5, 5, 1.0)
        @test dphidx2 ≈ 1.0 atol=0.1
    end

    @testset "CFL Timestep" begin
        # Create velocity fields
        ux = zeros(10, 10)
        uy = zeros(10, 10)
        ux[5, 5] = 10.0  # 10 ft/min

        active = [CartesianIndex(5, 5)]
        dx = 30.0  # 30 ft cells

        # Initial dt should be adjusted based on CFL
        dt = Elmfire.compute_cfl_timestep(ux, uy, active, dx, 1.0; target_cfl=0.9)

        # CFL = umax * dt / dx = 10 * 1 / 30 = 0.33 (already < 0.9)
        # So dt can be increased
        @test dt > 0

        # With higher velocity, dt should be smaller
        ux[5, 5] = 100.0
        dt_fast = Elmfire.compute_cfl_timestep(ux, uy, active, dx, 1.0; target_cfl=0.9)
        @test dt_fast < dt
    end

    @testset "RK2 Step" begin
        # Simple advection test with a gradient
        phi = zeros(10, 10)
        phi_old = zeros(10, 10)
        ux = ones(10, 10) * 10.0  # Constant velocity in +x direction
        uy = zeros(10, 10)

        # Initialize with linear gradient (phi increases with x)
        for i in 1:10
            phi[i, :] .= Float64(i - 5)  # -4, -3, -2, -1, 0, 1, 2, 3, 4, 5
        end

        # Focus on cell at the interface (cell 5 or 6)
        # At cell 6, phi = 1, neighbors: phi[5]=0, phi[7]=2
        # Gradient dφ/dx ≈ 1
        active = vec([CartesianIndex(i, j) for i in 4:7, j in 4:7])  # Small region
        dt = 0.1
        dx = 1.0

        # Save old state
        for idx in active
            phi_old[idx] = phi[idx]
        end

        # Stage 1: φ_new = φ_old - dt * (ux * dφ/dx)
        # At cell (6,5): φ_new ≈ 1 - 0.1 * (10 * 1) = 0
        Elmfire.rk2_step!(phi, phi_old, ux, uy, active, dt, dx, 1)

        # Phi should have changed at the interface
        # Since we have upwind advection with ux > 0, phi should decrease
        @test phi[6, 5] < phi_old[6, 5]
    end

    @testset "Normal Vector" begin
        phi = zeros(10, 10)

        # Create a circular pattern (negative inside, positive outside)
        cx, cy = 5, 5
        for i in 1:10, j in 1:10
            phi[i, j] = sqrt((i-cx)^2 + (j-cy)^2) - 2.0
        end

        # Normal at (7, 5) should point right (+x direction)
        nx, ny = Elmfire.compute_normal(phi, 7, 5, 1.0)
        @test nx > 0.9  # Should be mostly in +x
        @test abs(ny) < 0.2  # Little y component

        # Normal at (5, 7) should point up (+y direction)
        nx, ny = Elmfire.compute_normal(phi, 5, 7, 1.0)
        @test abs(nx) < 0.2
        @test ny > 0.9
    end

    @testset "Narrow Band" begin
        nb = Elmfire.NarrowBand(3)  # Band thickness of 3

        @test isempty(nb.active)
        @test isempty(nb.ever_tagged)

        # Tag around a center point
        Elmfire.tag_band!(nb, CartesianIndex(10, 10), 20, 20, 2)

        # Should have tagged cells in a square around (10, 10)
        @test !isempty(nb.active)
        @test CartesianIndex(10, 10) in nb.active
        @test CartesianIndex(11, 10) in nb.active
        @test CartesianIndex(10, 11) in nb.active

        # ever_tagged should track these
        @test !isempty(nb.ever_tagged)
    end

    @testset "Initialize Phi" begin
        phi = zeros(20, 20)
        ignition_points = [(10, 10), (10, 11)]

        Elmfire.initialize_phi!(phi, ignition_points, 1.0)

        # Ignition points should be negative
        @test phi[10, 10] < 0
        @test phi[10, 11] < 0

        # Far points should be positive
        @test phi[1, 1] > 0
        @test phi[20, 20] > 0

        # Values should increase with distance
        @test phi[1, 1] > phi[5, 5]
    end

    @testset "Initialize Circular Fire" begin
        phi = zeros(21, 21)

        Elmfire.initialize_circular_fire!(phi, 11, 11, 3.0, 1.0)

        # Center should be negative
        @test phi[11, 11] < 0

        # Points within radius should be negative
        @test phi[11, 12] < 0
        @test phi[12, 11] < 0

        # Points outside radius should be positive
        @test phi[11, 16] > 0
        @test phi[1, 1] > 0
    end
end
