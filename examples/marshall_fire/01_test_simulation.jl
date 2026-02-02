#=
Test Simulation for Marshall Fire
=================================

Run a single simulation to verify setup and data loading.
Uses default parameters with extreme wind conditions characteristic
of the Marshall Fire.

Usage:
    julia --project=. 01_test_simulation.jl
=#

using Elmfire
using Plots

# Include local modules
include("data_loading.jl")
include("validation.jl")

println("="^60)
println("Marshall Fire Test Simulation")
println("="^60)

#-----------------------------------------------------------------------------#
#                           Load Data
#-----------------------------------------------------------------------------#

println("\n[1] Loading data...")

# Load all data (use synthetic terrain for initial testing)
# Marshall Fire had 70-100+ mph winds from the west
data = load_all_data(
    use_synthetic = true,        # Use synthetic data for faster testing
    default_wind_speed = 100.0,  # mph - extreme gusts
    default_wind_dir = 270.0     # FROM west
)

ncols, nrows = size(data.fuel_ids)
println("   Grid dimensions: $ncols x $nrows")
println("   Total cells: $(ncols * nrows)")
println("   Cell size: $(round(CELLSIZE_FEET, digits=1)) ft")
println("   Domain: $(round(ncols * CELLSIZE_FEET / 5280, digits=2)) x $(round(nrows * CELLSIZE_FEET / 5280, digits=2)) miles")

# Print observed fire stats
observed_cells = count(data.observed_burned)
observed_acres = observed_cells * CELLSIZE_FEET^2 / 43560
println("\n   Observed fire:")
println("     Burned cells: $observed_cells")
println("     Burned area: $(round(observed_acres, digits=1)) acres")

#-----------------------------------------------------------------------------#
#                           Run Simulation
#-----------------------------------------------------------------------------#

println("\n[2] Running simulation...")

# Create fire state
state = Elmfire.FireState{Float64}(ncols, nrows, CELLSIZE_FEET)

# Ignite at known ignition point
println("   Ignition point: ($(data.ignition_ix), $(data.ignition_iy))")
Elmfire.ignite!(state, data.ignition_ix, data.ignition_iy, 0.0)

# Create fuel table
fuel_table = Elmfire.create_standard_fuel_table(Float64)

# Create constant weather (simpler than interpolator for testing)
weather = Elmfire.ConstantWeather(
    wind_speed_mph = 100.0,  # Extreme wind
    wind_direction = 270.0,  # FROM west
    M1 = 0.03,               # Very dry
    M10 = 0.04,
    M100 = 0.06,
    MLH = 0.30,              # Cured grass
    MLW = 0.60
)

# Run simulation
duration = SIMULATION_DURATION_MINUTES
println("   Duration: $duration minutes ($(round(duration/60, digits=1)) hours)")

start_time = time()

# Track progress
iteration_count = Ref(0)
callback = (state, t, dt, iter) -> begin
    iteration_count[] = iter
    if iter % 100 == 0
        println("     t=$(round(t, digits=1)) min, burned=$(count(state.burned)) cells")
    end
end

Elmfire.simulate!(
    state,
    data.fuel_ids,
    fuel_table,
    weather,
    data.slope,
    data.aspect,
    0.0,
    duration;
    dt_initial = 0.5,
    target_cfl = 0.9,
    dt_max = 5.0,
    callback = callback
)

println("   Total iterations: $(iteration_count[])")

elapsed = time() - start_time
println("   Simulation completed in $(round(elapsed, digits=2)) seconds")

#-----------------------------------------------------------------------------#
#                           Validate Results
#-----------------------------------------------------------------------------#

println("\n[3] Validating results...")

# Get simulation results
sim_burned_cells = count(state.burned)
sim_burned_acres = Elmfire.get_burned_area_acres(state)
println("   Simulated burned cells: $sim_burned_cells")
println("   Simulated burned area: $(round(sim_burned_acres, digits=1)) acres")

# Compute validation metrics
result = validate_simulation(state, data.observed_burned; cellsize_ft=CELLSIZE_FEET)

# Target metrics
println("\n[4] Target Metrics:")
println("   Target Sorensen: > 0.5 (reasonable fit)")
println("   Target Area Error: < 30%")

if result.sorensen > 0.5
    println("   ✓ Sorensen target met!")
else
    println("   ✗ Sorensen below target - consider adjusting parameters")
end

if abs(result.area_error) < 0.3
    println("   ✓ Area error target met!")
else
    println("   ✗ Area error above target - fire spread may be too fast/slow")
end

#-----------------------------------------------------------------------------#
#                           Visualization
#-----------------------------------------------------------------------------#

println("\n[5] Creating visualizations...")

# Create output directory
output_dir = joinpath(@__DIR__, "results")
mkpath(output_dir)

# Helper for heatmaps
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

# Plot 1: Simulated vs Observed burned area
p1 = fire_heatmap(Float64.(state.burned), "Simulated Burned Area", color=:YlOrRd)
p2 = fire_heatmap(Float64.(data.observed_burned), "Observed Burned Area", color=:YlOrRd)

plot_comparison = plot(p1, p2, layout=(1,2), size=(1000, 450))
savefig(plot_comparison, joinpath(output_dir, "burned_comparison.png"))
println("   Saved: burned_comparison.png")

# Plot 2: Error map
error_map = create_error_map(state.burned, data.observed_burned)
p3 = heatmap(
    error_map',
    aspect_ratio = 1,
    title = "Error Map",
    xlabel = "X (cells)",
    ylabel = "Y (cells)",
    color = [:white, :green, :red, :blue],  # TN, TP, FP (commission), FN (omission)
    clims = (0, 3),
    colorbar = false
)

# Add legend annotation
annotate!(p3, [(10, nrows-10, text("Green=TP, Red=FP, Blue=FN", 8, :left))])

savefig(p3, joinpath(output_dir, "error_map.png"))
println("   Saved: error_map.png")

# Plot 3: Time of arrival
toa = copy(state.time_of_arrival)
toa[toa .< 0] .= NaN  # Mask unburned cells

p4 = fire_heatmap(
    toa,
    "Time of Arrival",
    color = :viridis,
    colorbar_title = "Minutes"
)
savefig(p4, joinpath(output_dir, "time_of_arrival.png"))
println("   Saved: time_of_arrival.png")

# Plot 4: Summary (4 panels)
p_summary = plot(
    fire_heatmap(Float64.(state.burned), "Simulated", color=:YlOrRd),
    fire_heatmap(Float64.(data.observed_burned), "Observed", color=:YlOrRd),
    fire_heatmap(Float64.(error_map), "Error Map", color=[:white, :green, :red, :blue], clims=(0,3)),
    fire_heatmap(toa, "Time of Arrival", color=:viridis),
    layout = (2, 2),
    size = (900, 800)
)
savefig(p_summary, joinpath(output_dir, "test_simulation_summary.png"))
println("   Saved: test_simulation_summary.png")

#-----------------------------------------------------------------------------#
#                           Summary
#-----------------------------------------------------------------------------#

println("\n" * "="^60)
println("Test Simulation Complete")
println("="^60)
println("\nResults saved to: $output_dir")
println("\nNext steps:")
println("  1. If results look reasonable, run 02_run_calibration.jl")
println("  2. Adjust wind speed/direction if fire shape doesn't match")
println("  3. Check fuel model assignments if spread rate is wrong")
println("="^60)
