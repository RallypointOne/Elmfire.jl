#=
Quick Grid Search Calibration for Marshall Fire
================================================

Faster alternative to BBO that tests a grid of parameter values.
=#

using Elmfire
using JSON3
using Plots

include("data_loading.jl")
include("validation.jl")
include("calibration.jl")

function run_quick_calibration()
    println("="^60)
    println("Marshall Fire Quick Grid Search Calibration")
    println("="^60)

    println("\n[1] Loading data...")
    data = load_all_data(
        use_synthetic = true,
        default_wind_speed = 100.0,
        default_wind_dir = 270.0
    )

    ncols, nrows = size(data.fuel_ids)
    println("   Grid: $ncols x $nrows cells")

    println("\n[2] Running quick grid search...")
    println("   Testing 4x4 = 16 parameter combinations")

    # Quick grid search over most impactful parameters
    best_score = 0.0
    best_params = default_params()

    wind_factors = [0.8, 1.0, 1.2, 1.5]
    spread_factors = [0.8, 1.0, 1.2, 1.5]

    for wf in wind_factors
        for sf in spread_factors
            params = CalibrationParams{Float64}(wf, sf, 0.0, 1.0)

            try
                state = run_simulation_with_params(params, data)
                score = sorensen(state.burned, data.observed_burned)
                ae = abs(area_error(state.burned, data.observed_burned))

                println("   wf=$wf, sf=$sf => Sorensen=$(round(score, digits=4)), Area Error=$(round(100*ae, digits=1))%")

                if score > best_score
                    best_score = score
                    best_params = params
                end
            catch e
                println("   wf=$wf, sf=$sf => FAILED: $e")
            end
        end
    end

    println("\n[3] Best parameters found:")
    println("   wind_speed_factor:  $(best_params.wind_speed_factor)")
    println("   spread_rate_factor: $(best_params.spread_rate_factor)")
    println("   Best Sorensen:      $(round(best_score, digits=4))")

    println("\n[4] Running final simulation with best params...")
    final_state = run_simulation_with_params(best_params, data)
    final_result = compute_validation_metrics(final_state.burned, data.observed_burned)

    print_validation_summary(final_result; cellsize_ft=CELLSIZE_FEET)

    println("\n[5] Saving results...")

    output_dir = joinpath(@__DIR__, "results")
    mkpath(output_dir)

    params_dict = Dict(
        "wind_speed_factor" => best_params.wind_speed_factor,
        "spread_rate_factor" => best_params.spread_rate_factor,
        "wind_dir_bias" => best_params.wind_dir_bias,
        "moisture_factor" => best_params.moisture_factor,
        "best_score" => best_score,
        "method" => "grid_search"
    )

    open(joinpath(output_dir, "best_params.json"), "w") do f
        JSON3.write(f, params_dict)
    end

    metrics_dict = Dict(
        "sorensen" => final_result.sorensen,
        "jaccard" => final_result.jaccard,
        "kappa" => final_result.kappa,
        "commission_error" => final_result.commission_error,
        "omission_error" => final_result.omission_error,
        "area_error" => final_result.area_error
    )

    open(joinpath(output_dir, "validation_metrics.json"), "w") do f
        JSON3.write(f, metrics_dict)
    end

    println("   Saved best_params.json and validation_metrics.json")

    # Generate comparison plots
    println("\n[6] Creating visualizations...")

    p1 = heatmap(
        Float64.(final_state.burned)',
        aspect_ratio = 1,
        title = "Calibrated Simulation\n(Sorensen=$(round(best_score, digits=3)))",
        color = :YlOrRd,
        xlabel = "X (cells)",
        ylabel = "Y (cells)"
    )

    p2 = heatmap(
        Float64.(data.observed_burned)',
        aspect_ratio = 1,
        title = "Observed",
        color = :YlOrRd,
        xlabel = "X (cells)",
        ylabel = "Y (cells)"
    )

    error_map = create_error_map(final_state.burned, data.observed_burned)
    p3 = heatmap(
        error_map',
        aspect_ratio = 1,
        title = "Error Map (G=TP, R=FP, B=FN)",
        color = [:white, :green, :red, :blue],
        clims = (0, 3),
        xlabel = "X (cells)",
        ylabel = "Y (cells)",
        colorbar = false
    )

    toa = copy(final_state.time_of_arrival)
    toa[toa .< 0] .= NaN
    p4 = heatmap(
        toa',
        aspect_ratio = 1,
        title = "Time of Arrival (min)",
        color = :viridis,
        xlabel = "X (cells)",
        ylabel = "Y (cells)"
    )

    plot_summary = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 900))
    savefig(plot_summary, joinpath(output_dir, "calibration_results.png"))
    println("   Saved: calibration_results.png")

    println("\n" * "="^60)
    println("Calibration Complete")
    println("="^60)
    println("\nBest parameters: wf=$(best_params.wind_speed_factor), sf=$(best_params.spread_rate_factor)")
    println("Best Sorensen: $(round(best_score, digits=4))")
    println("Results saved to: $output_dir")
    println("="^60)

    return (best_params=best_params, best_score=best_score, final_result=final_result)
end

# Run it
run_quick_calibration()
