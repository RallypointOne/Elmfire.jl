@testset "Fuel Models" begin

    @testset "RawFuelModel Construction" begin
        raw = Elmfire.RawFuelModel(
            1, "FBFM01", false,
            0.034, 0.0, 0.0, 0.0, 0.0,
            3500.0, 9999.0, 9999.0,
            1.0, 0.12, 8000.0
        )

        @test raw.id == 1
        @test raw.name == "FBFM01"
        @test raw.dynamic == false
        @test raw.W0_1hr ≈ 0.034
        @test raw.mex_dead ≈ 0.12
    end

    @testset "FuelModel Coefficient Computation" begin
        # Test with FBFM01 (simple grass model)
        raw = Elmfire.RawFuelModel(
            1, "FBFM01", false,
            0.034, 0.0, 0.0, 0.0, 0.0,
            3500.0, 9999.0, 9999.0,
            1.0, 0.12, 8000.0
        )

        fm = Elmfire.compute_fuel_model(raw, 30)

        @test fm.id == 1
        @test fm.name == "FBFM01"
        @test fm.dynamic == false

        # Check fuel loadings
        @test fm.W0[1] ≈ 0.034
        @test fm.W0[2] ≈ 0.0
        @test fm.W0[3] ≈ 0.0

        # Check SAV ratios
        @test fm.SIG[1] ≈ 3500.0
        @test fm.SIG[2] ≈ 109.0  # Fixed 10-hr SAV

        # Check derived properties
        @test fm.rhop ≈ 32.0
        @test fm.delta ≈ 1.0
        @test fm.mex_dead ≈ 0.12

        # Check Rothermel coefficients are computed
        @test fm.A > 0
        @test fm.B > 0
        @test fm.C > 0
        @test fm.E > 0
        @test fm.gammaprime > 0
        @test fm.tr > 0
        @test fm.xi > 0
        @test fm.beta > 0
        @test fm.betaop > 0
    end

    @testset "Dynamic Fuel Model" begin
        # GR1 - dynamic grass model
        raw = Elmfire.RawFuelModel(
            101, "GR1", true,
            0.00459, 0.0, 0.0, 0.01377, 0.0,
            2200.0, 2000.0, 9999.0,
            0.4, 0.15, 8000.0
        )

        # At low moisture (30%), most herb should be dead
        fm_low = Elmfire.compute_fuel_model(raw, 30)
        # At high moisture (120%), all herb should be live
        fm_high = Elmfire.compute_fuel_model(raw, 120)

        # Live fuel should increase with moisture class
        @test fm_high.W0[5] > fm_low.W0[5]
    end

    @testset "FuelModelTable" begin
        table = Elmfire.FuelModelTable()

        raw = Elmfire.RawFuelModel(
            1, "FBFM01", false,
            0.034, 0.0, 0.0, 0.0, 0.0,
            3500.0, 9999.0, 9999.0,
            1.0, 0.12, 8000.0
        )

        Elmfire.add_raw_model!(table, raw)

        # Should have entries for all moisture classes 30-120
        for ilh in 30:120
            fm = Elmfire.get_fuel_model(table, 1, ilh)
            @test fm.id == 1
        end

        # Test with float moisture
        fm = Elmfire.get_fuel_model(table, 1, 0.65)  # 65% moisture
        @test fm.id == 1
    end

    @testset "Standard Fuel Table" begin
        table = Elmfire.create_standard_fuel_table()

        # Check that standard models are loaded
        for id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 256]
            fm = Elmfire.get_fuel_model(table, id, 60)
            @test fm.id == id
        end

        # Non-burnable should have very low fuel loading
        nb = Elmfire.get_fuel_model(table, 256, 60)
        @test Elmfire.isnonburnable(nb)
    end

    @testset "isnonburnable" begin
        table = Elmfire.create_standard_fuel_table()

        # FBFM01 should be burnable
        fm1 = Elmfire.get_fuel_model(table, 1, 60)
        @test !Elmfire.isnonburnable(fm1)

        # NB should be non-burnable
        nb = Elmfire.get_fuel_model(table, 256, 60)
        @test Elmfire.isnonburnable(nb)
    end
end
