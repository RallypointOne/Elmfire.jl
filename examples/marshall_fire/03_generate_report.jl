#=
Report Generation for Marshall Fire Calibration
================================================

Generate detailed visualizations and analysis of calibration results.

Usage:
    julia --project=. 03_generate_report.jl
=#

using Elmfire
using Plots
using JSON3
using Dates

# Include local modules
include("data_loading.jl")
include("validation.jl")
include("calibration.jl")

println("="^60)
println("Marshall Fire Calibration Report")
println("="^60)

#-----------------------------------------------------------------------------#
#                           Load Results
#-----------------------------------------------------------------------------#

println("\n[1] Loading calibration results...")

output_dir = joinpath(@__DIR__, "results")

# Load best parameters
params_path = joinpath(output_dir, "best_params.json")
if !isfile(params_path)
    error("Calibration results not found. Run 02_run_calibration.jl first.")
end

params_dict = JSON3.read(read(params_path, String))
best_params = CalibrationParams{Float64}(
    params_dict.wind_speed_factor,
    params_dict.spread_rate_factor,
    params_dict.wind_dir_bias,
    params_dict.moisture_factor
)

println("   Loaded parameters:")
println("     wind_speed_factor:  $(round(best_params.wind_speed_factor, digits=4))")
println("     spread_rate_factor: $(round(best_params.spread_rate_factor, digits=4))")
println("     wind_dir_bias:      $(round(best_params.wind_dir_bias, digits=2))°")
println("     moisture_factor:    $(round(best_params.moisture_factor, digits=4))")

# Load convergence history if available
history_path = joinpath(output_dir, "convergence_history.json")
convergence_history = if isfile(history_path)
    Float64.(JSON3.read(read(history_path, String)))
else
    Float64[]
end

#-----------------------------------------------------------------------------#
#                           Load Data and Run Simulation
#-----------------------------------------------------------------------------#

println("\n[2] Loading data and running calibrated simulation...")

data = load_all_data(
    use_synthetic = true,
    default_wind_speed = 100.0,
    default_wind_dir = 270.0
)

ncols, nrows = size(data.fuel_ids)

# Run with calibrated parameters
state = run_simulation_with_params(best_params, data)

# Compute metrics
result = compute_validation_metrics(state.burned, data.observed_burned)
print_validation_summary(result; cellsize_ft=CELLSIZE_FEET)

#-----------------------------------------------------------------------------#
#                           Generate Visualizations
#-----------------------------------------------------------------------------#

println("\n[3] Generating visualizations...")

# Set plot defaults
default(fontfamily="Computer Modern", titlefont=(12, "bold"))

# Helper function
function fire_heatmap(data, title; kwargs...)
    heatmap(
        data',
        aspect_ratio = 1,
        title = title,
        xlabel = "X (cells)",
        ylabel = "Y (cells)",
        framestyle = :box;
        kwargs...
    )
end

#--- Plot 1: Main Comparison ---
println("   Creating comparison plots...")

p_sim = fire_heatmap(Float64.(state.burned), "Simulated (Calibrated)", color=:YlOrRd)
p_obs = fire_heatmap(Float64.(data.observed_burned), "Observed", color=:YlOrRd)

# Overlay plot
overlay = zeros(ncols, nrows)
for ix in 1:ncols
    for iy in 1:nrows
        if state.burned[ix, iy] && data.observed_burned[ix, iy]
            overlay[ix, iy] = 3  # Both (green)
        elseif state.burned[ix, iy]
            overlay[ix, iy] = 2  # Simulated only (red)
        elseif data.observed_burned[ix, iy]
            overlay[ix, iy] = 1  # Observed only (blue)
        end
    end
end

p_overlay = heatmap(
    overlay',
    aspect_ratio = 1,
    title = "Overlay (G=Both, R=Sim Only, B=Obs Only)",
    color = [:white, :blue, :red, :green],
    clims = (0, 3),
    colorbar = false,
    xlabel = "X (cells)",
    ylabel = "Y (cells)"
)

plot_comparison = plot(p_sim, p_obs, p_overlay, layout=(1,3), size=(1400, 400))
savefig(plot_comparison, joinpath(output_dir, "report_comparison.png"))
println("   Saved: report_comparison.png")

#--- Plot 2: Time of Arrival ---
println("   Creating time of arrival plot...")

toa = copy(state.time_of_arrival)
toa[toa .< 0] .= NaN

p_toa = fire_heatmap(
    toa,
    "Time of Arrival (minutes)",
    color = :viridis,
    colorbar_title = "Minutes"
)
savefig(p_toa, joinpath(output_dir, "report_time_of_arrival.png"))
println("   Saved: report_time_of_arrival.png")

#--- Plot 3: Spread Rate ---
println("   Creating spread rate plot...")

ros = copy(state.spread_rate)
ros[ros .== 0] .= NaN

p_ros = fire_heatmap(
    ros,
    "Spread Rate (ft/min)",
    color = :plasma,
    colorbar_title = "ft/min"
)
savefig(p_ros, joinpath(output_dir, "report_spread_rate.png"))
println("   Saved: report_spread_rate.png")

#--- Plot 4: Fireline Intensity ---
println("   Creating fireline intensity plot...")

flin = copy(state.fireline_intensity)
flin[flin .== 0] .= NaN

p_flin = fire_heatmap(
    flin,
    "Fireline Intensity (kW/m)",
    color = :hot,
    colorbar_title = "kW/m"
)
savefig(p_flin, joinpath(output_dir, "report_fireline_intensity.png"))
println("   Saved: report_fireline_intensity.png")

#--- Plot 5: Error Analysis ---
println("   Creating error analysis...")

error_map = create_error_map(state.burned, data.observed_burned)

p_error = heatmap(
    error_map',
    aspect_ratio = 1,
    title = "Error Classification",
    color = [:white, :green, :red, :blue],
    clims = (0, 3),
    xlabel = "X (cells)",
    ylabel = "Y (cells)"
)

# Add legend
annotate!(p_error, [
    (ncols*0.05, nrows*0.95, text("Legend:", 8, :left)),
    (ncols*0.05, nrows*0.90, text("Green = True Positive", 8, :left)),
    (ncols*0.05, nrows*0.85, text("Red = False Positive (Commission)", 8, :left)),
    (ncols*0.05, nrows*0.80, text("Blue = False Negative (Omission)", 8, :left))
])

savefig(p_error, joinpath(output_dir, "report_error_map.png"))
println("   Saved: report_error_map.png")

#--- Plot 6: Convergence History ---
if !isempty(convergence_history)
    println("   Creating convergence plot...")

    p_conv = plot(
        convergence_history,
        xlabel = "Iteration",
        ylabel = "Sorensen Score",
        title = "Calibration Convergence",
        legend = false,
        linewidth = 2,
        color = :blue,
        grid = true,
        minorgrid = true
    )

    # Add horizontal line at final value
    hline!(p_conv, [convergence_history[end]], linestyle=:dash, color=:red,
           label="Final: $(round(convergence_history[end], digits=4))")

    savefig(p_conv, joinpath(output_dir, "report_convergence.png"))
    println("   Saved: report_convergence.png")
end

#--- Plot 7: Fuel and Terrain ---
println("   Creating fuel and terrain plots...")

# Fuel model map
p_fuel = fire_heatmap(
    Float64.(data.fuel_ids),
    "Fuel Models (FBFM40)",
    color = :Set3
)

# Slope map
p_slope = fire_heatmap(
    data.slope,
    "Slope (degrees)",
    color = :terrain,
    colorbar_title = "degrees"
)

# Elevation map
p_elev = fire_heatmap(
    data.elevation,
    "Elevation (m)",
    color = :terrain,
    colorbar_title = "m"
)

# Aspect map
p_aspect = fire_heatmap(
    data.aspect,
    "Aspect (degrees from N)",
    color = :hsv,
    colorbar_title = "degrees"
)

plot_terrain = plot(p_fuel, p_slope, p_elev, p_aspect, layout=(2,2), size=(1000, 900))
savefig(plot_terrain, joinpath(output_dir, "report_terrain.png"))
println("   Saved: report_terrain.png")

#--- Plot 8: Summary Dashboard ---
println("   Creating summary dashboard...")

# Metrics bar chart
metrics_names = ["Sorensen", "Jaccard", "Kappa", "1-CommErr", "1-OmitErr"]
metrics_values = [
    result.sorensen,
    result.jaccard,
    result.kappa,
    1 - result.commission_error,
    1 - result.omission_error
]

p_metrics = bar(
    metrics_names,
    metrics_values,
    title = "Validation Metrics",
    ylabel = "Score (higher is better)",
    legend = false,
    ylims = (0, 1),
    color = [:blue, :blue, :blue, :orange, :orange]
)
hline!(p_metrics, [0.5], linestyle=:dash, color=:red, label="Target")

# Confusion matrix
conf_labels = ["Sim Yes", "Sim No"]
conf_values = [
    result.true_positives result.false_negatives;
    result.false_positives result.true_negatives
]

p_conf = heatmap(
    conf_values,
    xticks = (1:2, ["Obs Yes", "Obs No"]),
    yticks = (1:2, conf_labels),
    title = "Confusion Matrix",
    color = :Blues,
    aspect_ratio = 1
)

# Add text annotations to confusion matrix
annotate!(p_conf, [
    (1, 1, text("TP\n$(result.true_positives)", 10, :center)),
    (2, 1, text("FN\n$(result.false_negatives)", 10, :center)),
    (1, 2, text("FP\n$(result.false_positives)", 10, :center)),
    (2, 2, text("TN\n$(result.true_negatives)", 10, :center))
])

# Area comparison
area_labels = ["Simulated", "Observed"]
area_values = [
    result.simulated_cells * CELLSIZE_FEET^2 / 43560,
    result.observed_cells * CELLSIZE_FEET^2 / 43560
]

p_area = bar(
    area_labels,
    area_values,
    title = "Burned Area Comparison",
    ylabel = "Acres",
    legend = false,
    color = [:red, :blue]
)

# Parameter values
param_names = ["Wind\nFactor", "Spread\nFactor", "Wind Bias\n(deg/30)", "Moisture\nFactor"]
param_values = [
    best_params.wind_speed_factor,
    best_params.spread_rate_factor,
    best_params.wind_dir_bias / 30,  # Normalize to [0,1] range
    best_params.moisture_factor
]

p_params = bar(
    param_names,
    param_values,
    title = "Calibrated Parameters",
    ylabel = "Value (normalized)",
    legend = false,
    color = :green,
    ylims = (0, 2)
)
hline!(p_params, [1.0], linestyle=:dash, color=:black, label="Default")

# Combine
plot_dashboard = plot(
    p_sim, p_obs, p_overlay,
    p_metrics, p_area, p_params,
    layout = @layout([a b c; d e f]),
    size = (1400, 800)
)
savefig(plot_dashboard, joinpath(output_dir, "report_dashboard.png"))
println("   Saved: report_dashboard.png")

#--- Plot 9: Fire Progression ---
println("   Creating fire progression plot...")

# Create time slices
time_slices = [60.0, 120.0, 180.0, 240.0, 300.0, 360.0]  # minutes

progression_plots = []
for t in time_slices
    burned_at_t = state.time_of_arrival .<= t
    burned_at_t[state.time_of_arrival .< 0] .= false  # Handle unburned

    p = fire_heatmap(
        Float64.(burned_at_t),
        "t = $(Int(t)) min",
        color = :YlOrRd
    )
    push!(progression_plots, p)
end

plot_progression = plot(progression_plots..., layout=(2,3), size=(1200, 700))
savefig(plot_progression, joinpath(output_dir, "report_progression.png"))
println("   Saved: report_progression.png")

#-----------------------------------------------------------------------------#
#                           Generate Text Report
#-----------------------------------------------------------------------------#

println("\n[4] Generating text report...")

report_path = joinpath(output_dir, "calibration_report.txt")
open(report_path, "w") do f
    write(f, "="^60 * "\n")
    write(f, "Marshall Fire Calibration Report\n")
    write(f, "Generated: $(Dates.now())\n")
    write(f, "="^60 * "\n\n")

    write(f, "MARSHALL FIRE OVERVIEW\n")
    write(f, "-"^40 * "\n")
    write(f, "Date: December 30, 2021\n")
    write(f, "Location: Boulder County, Colorado\n")
    write(f, "Ignition: ($IGNITION_POINT)\n")
    write(f, "Duration: ~6 hours of rapid spread\n")
    write(f, "Conditions: Extreme winds (70-100+ mph)\n\n")

    write(f, "SIMULATION PARAMETERS\n")
    write(f, "-"^40 * "\n")
    write(f, "Grid: $ncols x $nrows cells\n")
    write(f, "Cell size: $(round(CELLSIZE_FEET, digits=1)) ft ($(round(CELLSIZE_METERS, digits=1)) m)\n")
    write(f, "Duration: $(SIMULATION_DURATION_MINUTES) minutes\n")
    write(f, "Base wind: 70 mph from 270° (west)\n\n")

    write(f, "CALIBRATED PARAMETERS\n")
    write(f, "-"^40 * "\n")
    write(f, "wind_speed_factor:  $(round(best_params.wind_speed_factor, digits=4))\n")
    write(f, "spread_rate_factor: $(round(best_params.spread_rate_factor, digits=4))\n")
    write(f, "wind_dir_bias:      $(round(best_params.wind_dir_bias, digits=2))°\n")
    write(f, "moisture_factor:    $(round(best_params.moisture_factor, digits=4))\n\n")

    write(f, "VALIDATION METRICS\n")
    write(f, "-"^40 * "\n")
    write(f, "Sorensen-Dice:    $(round(result.sorensen, digits=4))\n")
    write(f, "Jaccard Index:    $(round(result.jaccard, digits=4))\n")
    write(f, "Cohen's Kappa:    $(round(result.kappa, digits=4))\n")
    write(f, "Commission Error: $(round(100*result.commission_error, digits=2))%\n")
    write(f, "Omission Error:   $(round(100*result.omission_error, digits=2))%\n")
    write(f, "Area Error:       $(round(100*result.area_error, digits=2))%\n\n")

    write(f, "CONFUSION MATRIX\n")
    write(f, "-"^40 * "\n")
    write(f, "True Positives:  $(result.true_positives)\n")
    write(f, "False Positives: $(result.false_positives)\n")
    write(f, "False Negatives: $(result.false_negatives)\n")
    write(f, "True Negatives:  $(result.true_negatives)\n\n")

    write(f, "BURNED AREA\n")
    write(f, "-"^40 * "\n")
    sim_acres = result.simulated_cells * CELLSIZE_FEET^2 / 43560
    obs_acres = result.observed_cells * CELLSIZE_FEET^2 / 43560
    write(f, "Simulated: $(result.simulated_cells) cells ($(round(sim_acres, digits=1)) acres)\n")
    write(f, "Observed:  $(result.observed_cells) cells ($(round(obs_acres, digits=1)) acres)\n\n")

    write(f, "TARGET METRICS\n")
    write(f, "-"^40 * "\n")
    write(f, "Sorensen > 0.5:    $(result.sorensen > 0.5 ? "PASS" : "FAIL")\n")
    write(f, "Area Error < 30%:  $(abs(result.area_error) < 0.3 ? "PASS" : "FAIL")\n\n")

    write(f, "="^60 * "\n")
end

println("   Saved: calibration_report.txt")

#-----------------------------------------------------------------------------#
#                           Summary
#-----------------------------------------------------------------------------#

println("\n" * "="^60)
println("Report Generation Complete")
println("="^60)

println("\nGenerated files:")
println("  - report_comparison.png       : Simulated vs Observed comparison")
println("  - report_time_of_arrival.png  : Fire arrival time map")
println("  - report_spread_rate.png      : Spread rate distribution")
println("  - report_fireline_intensity.png: Fireline intensity map")
println("  - report_error_map.png        : Error classification map")
if !isempty(convergence_history)
    println("  - report_convergence.png      : Optimization convergence")
end
println("  - report_terrain.png          : Fuel and terrain inputs")
println("  - report_dashboard.png        : Summary dashboard")
println("  - report_progression.png      : Fire progression over time")
println("  - calibration_report.txt      : Text summary")

println("\nAll outputs saved to: $output_dir")
println("="^60)
