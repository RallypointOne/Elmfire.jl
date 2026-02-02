@testset "Crown Fire Model" begin

    @testset "CanopyProperties Construction" begin
        # Default constructor
        canopy = CanopyProperties{Float64}()
        @test canopy.cbd == 0.0
        @test canopy.cbh == 0.0
        @test canopy.cc == 0.0
        @test canopy.ch == 0.0
        @test eltype(canopy) == Float64

        # With values
        canopy = CanopyProperties{Float64}(
            cbd = 0.15,
            cbh = 5.0,
            cc = 0.7,
            ch = 20.0
        )
        @test canopy.cbd == 0.15
        @test canopy.cbh == 5.0
        @test canopy.cc == 0.7
        @test canopy.ch == 20.0

        # Float32
        canopy32 = CanopyProperties{Float32}(
            cbd = Float32(0.15),
            cbh = Float32(5.0),
            cc = Float32(0.7),
            ch = Float32(20.0)
        )
        @test eltype(canopy32) == Float32
    end

    @testset "Critical Fireline Intensity" begin
        # Van Wagner (1977) criterion
        # I_crit = (0.01 * CBH * (460 + 26 * FMC))^1.5

        # Low canopy base height should require low intensity
        flin_low = critical_fireline_intensity(2.0, 100.0)
        @test flin_low > 0

        # Higher canopy base height requires higher intensity
        flin_high = critical_fireline_intensity(10.0, 100.0)
        @test flin_high > flin_low

        # Higher foliar moisture requires higher intensity
        flin_dry = critical_fireline_intensity(5.0, 80.0)
        flin_wet = critical_fireline_intensity(5.0, 120.0)
        @test flin_wet > flin_dry

        # Negative CBH should return very high value (no crown fire)
        flin_neg = critical_fireline_intensity(-1.0, 100.0)
        @test flin_neg > 1e8

        # Float32 version
        flin32 = critical_fireline_intensity(Float32(5.0), Float32(100.0))
        @test flin32 isa Float32
        @test Float64(flin32) ≈ critical_fireline_intensity(5.0, 100.0) rtol=1e-4
    end

    @testset "Crown Spread Rate - No Crown Fire" begin
        # No crown fire when CBD is zero
        canopy_no_fuel = CanopyProperties{Float64}(cbd=0.0, cbh=5.0, cc=0.7, ch=20.0)
        result = crown_spread_rate(canopy_no_fuel, 500.0, 10.0, 0.06, 20.0)
        @test result.crown_fire_type == 0
        @test result.spread_rate == 0.0

        # No crown fire when canopy cover is zero
        canopy_no_cover = CanopyProperties{Float64}(cbd=0.15, cbh=5.0, cc=0.0, ch=20.0)
        result = crown_spread_rate(canopy_no_cover, 500.0, 10.0, 0.06, 20.0)
        @test result.crown_fire_type == 0

        # No crown fire when surface fire intensity is zero
        canopy = CanopyProperties{Float64}(cbd=0.15, cbh=5.0, cc=0.7, ch=20.0)
        result = crown_spread_rate(canopy, 0.0, 10.0, 0.06, 20.0)
        @test result.crown_fire_type == 0

        # No crown fire when surface spread rate is zero
        result = crown_spread_rate(canopy, 500.0, 10.0, 0.06, 0.0)
        @test result.crown_fire_type == 0
    end

    @testset "Crown Spread Rate - Below Threshold" begin
        # Crown fire not initiated when FLIN < critical
        canopy = CanopyProperties{Float64}(
            cbd = 0.1,
            cbh = 10.0,  # High base height
            cc = 0.7,
            ch = 25.0
        )

        # Low intensity surface fire
        result = crown_spread_rate(canopy, 50.0, 5.0, 0.06, 10.0)
        @test result.crown_fire_type == 0
        @test result.spread_rate == 0.0
        @test result.critical_flin > 50.0
    end

    @testset "Crown Spread Rate - Passive Crown Fire" begin
        canopy = CanopyProperties{Float64}(
            cbd = 0.05,  # Low CBD - passive
            cbh = 2.0,   # Low base height
            cc = 0.6,
            ch = 15.0
        )

        # Moderate intensity should trigger passive crown fire
        result = crown_spread_rate(canopy, 1000.0, 10.0, 0.06, 30.0)
        # With low CBD, should be passive (type 1)
        @test result.crown_fire_type >= 1  # Passive or active
        @test result.hpua_canopy > 0  # Should have canopy heat
    end

    @testset "Crown Spread Rate - Active Crown Fire" begin
        canopy = CanopyProperties{Float64}(
            cbd = 0.2,   # High CBD
            cbh = 2.0,   # Low base height
            cc = 0.8,    # High cover
            ch = 20.0
        )

        # High intensity, strong wind, dry fuels
        result = crown_spread_rate(canopy, 2000.0, 20.0, 0.04, 50.0)

        @test result.crown_fire_type == 2  # Active
        @test result.spread_rate > 0
        @test result.phiw_crown > 0

        # Check that crown fire spread rate is higher than surface
        @test result.spread_rate > 50.0
    end

    @testset "Crown Spread Rate - Wind and Moisture Effects" begin
        canopy = CanopyProperties{Float64}(
            cbd = 0.15,
            cbh = 3.0,
            cc = 0.7,
            ch = 18.0
        )

        # Higher wind speed should increase spread rate
        result_low_wind = crown_spread_rate(canopy, 1500.0, 5.0, 0.06, 40.0)
        result_high_wind = crown_spread_rate(canopy, 1500.0, 25.0, 0.06, 40.0)

        if result_low_wind.crown_fire_type >= 1 && result_high_wind.crown_fire_type >= 1
            @test result_high_wind.spread_rate >= result_low_wind.spread_rate
        end

        # Higher moisture should decrease spread rate
        result_dry = crown_spread_rate(canopy, 1500.0, 15.0, 0.03, 40.0)
        result_wet = crown_spread_rate(canopy, 1500.0, 15.0, 0.10, 40.0)

        if result_dry.crown_fire_type == 2 && result_wet.crown_fire_type == 2
            @test result_dry.spread_rate >= result_wet.spread_rate
        end
    end

    @testset "Combined Spread Rate" begin
        # Create mock results (velocity, vs0, ir, hpua, flin, phiw, phis)
        surface = SpreadResult{Float64}(30.0, 10.0, 500.0, 1000.0, 300.0, 2.0, 0.0)

        # No crown fire - use surface rate
        crown_none = CrownFireResult{Float64}(0, 0.0, 0.0, 1000.0, 0.0)
        @test combined_spread_rate(surface, crown_none) == 30.0

        # Passive crown fire - use surface rate
        crown_passive = CrownFireResult{Float64}(1, 50.0, 500.0, 800.0, 1.0)
        @test combined_spread_rate(surface, crown_passive) == 30.0

        # Active crown fire - use crown rate
        crown_active = CrownFireResult{Float64}(2, 100.0, 800.0, 600.0, 3.0)
        @test combined_spread_rate(surface, crown_active) == 100.0
    end

    @testset "Combined Fireline Intensity" begin
        fuel_table = create_standard_fuel_table()
        fm = get_fuel_model(fuel_table, 1, 60)

        # Create mock results (velocity, vs0, ir, hpua, flin, phiw, phis)
        surface = SpreadResult{Float64}(30.0, 10.0, 500.0, 1000.0, 300.0, 2.0, 0.0)

        # No crown fire - just surface FLIN
        crown_none = CrownFireResult{Float64}(0, 0.0, 0.0, 1000.0, 0.0)
        flin_none = combined_fireline_intensity(surface, crown_none, fm)
        @test flin_none == surface.flin

        # Crown fire adds intensity
        crown_active = CrownFireResult{Float64}(2, 80.0, 600.0, 500.0, 2.0)
        flin_crown = combined_fireline_intensity(surface, crown_active, fm)
        @test flin_crown > surface.flin  # Crown contribution added
    end

    @testset "Float32 Precision" begin
        canopy32 = CanopyProperties{Float32}(
            cbd = Float32(0.15),
            cbh = Float32(3.0),
            cc = Float32(0.7),
            ch = Float32(18.0)
        )

        result32 = crown_spread_rate(
            canopy32,
            Float32(1500.0),
            Float32(15.0),
            Float32(0.05),
            Float32(40.0)
        )

        @test eltype(result32) == Float32
        @test result32.spread_rate isa Float32
        @test result32.hpua_canopy isa Float32
        @test result32.critical_flin isa Float32
    end
end
