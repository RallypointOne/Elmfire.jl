#=
Calibration Script for Marshall Fire
=====================================

Run parameter optimization to fit simulation to observed fire perimeter.

Usage:
    julia --project=. 02_run_calibration.jl [--max-iters=100] [--grid-search]
=#

using Elmfire
using JSON3

# Include local modules
include("data_loading.jl")
include("validation.jl")
include("calibration.jl")

# Parse command line arguments
function parse_args()
    max_iters = 100
    use_grid_search = false

    for arg in ARGS
        if startswith(arg, "--max-iters=")
            max_iters = parse(Int, split(arg, "=")[2])
        elseif arg == "--grid-search"
            use_grid_search = true
        end
    end

    return (max_iters=max_iters, use_grid_search=use_grid_search)
end

args = parse_args()

println("="^60)
println("Marshall Fire Calibration")
println("="^60)

#-----------------------------------------------------------------------------#
#                           Load Data
#-----------------------------------------------------------------------------#

println("\n[1] Loading data...")

data = load_all_data(
    use_synthetic = true,
    default_wind_speed = 100.0,
    default_wind_dir = 270.0
)

ncols, nrows = size(data.fuel_ids)
println("   Grid: $ncols x $nrows cells")

observed_cells = count(data.observed_burned)
observed_acres = observed_cells * CELLSIZE_FEET^2 / 43560
println("   Observed burned area: $(round(observed_acres, digits=1)) acres")

#-----------------------------------------------------------------------------#
#                           Baseline Simulation
#-----------------------------------------------------------------------------#

println("\n[2] Running baseline simulation...")

baseline_params = default_params()
println("   Parameters: wind_factor=1.0, spread_factor=1.0, wind_bias=0.0, moisture_factor=1.0")

baseline_state = run_simulation_with_params(baseline_params, data)

baseline_result = compute_validation_metrics(baseline_state.burned, data.observed_burned)
println("   Baseline Sorensen: $(round(baseline_result.sorensen, digits=4))")
println("   Baseline Area Error: $(round(100*baseline_result.area_error, digits=2))%")

#-----------------------------------------------------------------------------#
#                           Run Calibration
#-----------------------------------------------------------------------------#

println("\n[3] Running calibration...")

output_dir = joinpath(@__DIR__, "results")
mkpath(output_dir)

start_time = time()

if args.use_grid_search
    println("   Mode: Grid Search")
    cal_result = run_grid_search(
        data,
        data.observed_burned;
        wind_speeds = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
        spread_factors = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6],
        verbose = true
    )
else
    println("   Mode: Black-Box Optimization (BBO)")
    println("   Max iterations: $(args.max_iters)")
    cal_result = run_calibration(
        data,
        data.observed_burned;
        max_iters = args.max_iters,
        verbose = true,
        use_combined = true
    )
end

elapsed = time() - start_time
println("\n   Calibration completed in $(round(elapsed, digits=1)) seconds")

#-----------------------------------------------------------------------------#
#                           Final Simulation with Best Params
#-----------------------------------------------------------------------------#

println("\n[4] Running final simulation with calibrated parameters...")

final_state = run_simulation_with_params(cal_result.best_params, data)
final_result = compute_validation_metrics(final_state.burned, data.observed_burned)

print_validation_summary(final_result; cellsize_ft=CELLSIZE_FEET)

#-----------------------------------------------------------------------------#
#                           Save Results
#-----------------------------------------------------------------------------#

println("\n[5] Saving results...")

# Save best parameters to JSON
params_dict = Dict(
    "wind_speed_factor" => cal_result.best_params.wind_speed_factor,
    "spread_rate_factor" => cal_result.best_params.spread_rate_factor,
    "wind_dir_bias" => cal_result.best_params.wind_dir_bias,
    "moisture_factor" => cal_result.best_params.moisture_factor,
    "best_score" => cal_result.best_score,
    "iterations" => cal_result.iterations,
    "baseline_sorensen" => baseline_result.sorensen,
    "final_sorensen" => final_result.sorensen,
    "final_jaccard" => final_result.jaccard,
    "final_area_error" => final_result.area_error
)

params_path = joinpath(output_dir, "best_params.json")
open(params_path, "w") do f
    JSON3.write(f, params_dict)
end
println("   Saved parameters: $params_path")

# Save convergence history
if !isempty(cal_result.convergence_history)
    history_path = joinpath(output_dir, "convergence_history.json")
    open(history_path, "w") do f
        JSON3.write(f, cal_result.convergence_history)
    end
    println("   Saved convergence history: $history_path")
end

# Save validation metrics
metrics_dict = Dict(
    "sorensen" => final_result.sorensen,
    "jaccard" => final_result.jaccard,
    "kappa" => final_result.kappa,
    "commission_error" => final_result.commission_error,
    "omission_error" => final_result.omission_error,
    "area_error" => final_result.area_error,
    "true_positives" => final_result.true_positives,
    "false_positives" => final_result.false_positives,
    "false_negatives" => final_result.false_negatives,
    "true_negatives" => final_result.true_negatives,
    "simulated_cells" => final_result.simulated_cells,
    "observed_cells" => final_result.observed_cells,
    "simulated_acres" => final_result.simulated_cells * CELLSIZE_FEET^2 / 43560,
    "observed_acres" => final_result.observed_cells * CELLSIZE_FEET^2 / 43560
)

metrics_path = joinpath(output_dir, "validation_metrics.json")
open(metrics_path, "w") do f
    JSON3.write(f, metrics_dict)
end
println("   Saved validation metrics: $metrics_path")

#-----------------------------------------------------------------------------#
#                           Quick Visualization
#-----------------------------------------------------------------------------#

println("\n[6] Creating visualizations...")

using Plots

# Comparison plot
p1 = heatmap(
    Float64.(final_state.burned)',
    aspect_ratio = 1,
    title = "Calibrated Simulation",
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

# Convergence plot
p4 = if !isempty(cal_result.convergence_history)
    plot(
        cal_result.convergence_history,
        xlabel = "Iteration",
        ylabel = "Sorensen Score",
        title = "Calibration Convergence",
        legend = false,
        linewidth = 2,
        color = :blue
    )
else
    plot(title = "No convergence data", framestyle = :none)
end

# Combined plot
plot_summary = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 900))
savefig(plot_summary, joinpath(output_dir, "calibration_results.png"))
println("   Saved: calibration_results.png")

#-----------------------------------------------------------------------------#
#                           Parameter Sensitivity
#-----------------------------------------------------------------------------#

println("\n[7] Computing parameter sensitivity...")

sensitivities = parameter_sensitivity(data, data.observed_burned, cal_result.best_params)

println("   Parameter sensitivities (d_score / d_param):")
for (param, sens) in sensitivities
    println("     $param: $(round(sens, digits=4))")
end

# Save sensitivities
sens_path = joinpath(output_dir, "parameter_sensitivity.json")
open(sens_path, "w") do f
    JSON3.write(f, sensitivities)
end
println("   Saved: parameter_sensitivity.json")

#-----------------------------------------------------------------------------#
#                           Summary
#-----------------------------------------------------------------------------#

println("\n" * "="^60)
println("Calibration Complete")
println("="^60)

println("\nImprovement:")
println("  Baseline Sorensen: $(round(baseline_result.sorensen, digits=4))")
println("  Final Sorensen:    $(round(final_result.sorensen, digits=4))")
println("  Improvement:       $(round(final_result.sorensen - baseline_result.sorensen, digits=4))")

println("\nBest Parameters:")
println("  wind_speed_factor:  $(round(cal_result.best_params.wind_speed_factor, digits=4))")
println("  spread_rate_factor: $(round(cal_result.best_params.spread_rate_factor, digits=4))")
println("  wind_dir_bias:      $(round(cal_result.best_params.wind_dir_bias, digits=2))°")
println("  moisture_factor:    $(round(cal_result.best_params.moisture_factor, digits=4))")

println("\nTarget Check:")
if final_result.sorensen > 0.5
    println("  ✓ Sorensen > 0.5 target met!")
else
    println("  ✗ Sorensen below 0.5 target")
end

if abs(final_result.area_error) < 0.3
    println("  ✓ Area error < 30% target met!")
else
    println("  ✗ Area error above 30% target")
end

println("\nResults saved to: $output_dir")
println("\nNext steps:")
println("  1. Run 03_generate_report.jl for detailed analysis")
println("  2. Try increasing --max-iters for better fit")
println("  3. Consider using HRRR wind data for time-varying conditions")
println("="^60)
