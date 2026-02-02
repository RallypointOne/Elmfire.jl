#=
Spotting (Ember Transport) Example
===================================

This example demonstrates:
1. Fire simulation with ember spotting enabled
2. How spot fires can cause fire to jump ahead of the main front
3. Comparison with and without spotting

Requirements: Plots.jl, Random
=#

using Elmfire
using Plots
using Random

# Create output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("=" ^ 60)
println("Elmfire.jl - Spotting Example")
println("=" ^ 60)

#-----------------------------------------------------------------------------#
#                     Setup Common Parameters
#-----------------------------------------------------------------------------#

# Use a fixed seed for reproducibility
rng = MersenneTwister(42)

# Larger grid to see spotting effects
ncols, nrows = 150, 100
cellsize = 10.0  # 10 ft cells

# Weather: strong wind, dry conditions (favorable for spotting)
weather = ConstantWeather(
    wind_speed_mph = 25.0,
    wind_direction = 270.0,  # Wind from west
    M1 = 0.03,               # Very dry
    M10 = 0.05,
    M100 = 0.07,
    MLH = 0.35,
    MLW = 0.60
)

fuel_table = create_standard_fuel_table()

#-----------------------------------------------------------------------------#
#                     Simulation WITHOUT Spotting
#-----------------------------------------------------------------------------#

println("\n[1] Running simulation without spotting...")

state_no_spot = FireState(ncols, nrows, cellsize)
ignite!(state_no_spot, 30, 50, 0.0)  # Ignite on left side

config_no_spot = SimulationConfig{Float64}(
    enable_crown_fire = true,
    enable_spotting = false
)

simulate_full_uniform!(
    state_no_spot,
    1,
    fuel_table,
    weather,
    0.0, 0.0,
    0.0, 15.0;  # 15 minutes
    canopy_cbd = 0.12,
    canopy_cbh = 4.0,
    canopy_cc = 0.65,
    canopy_ch = 16.0,
    config = config_no_spot,
    dt_initial = 0.2
)

println("   Burned: $(round(get_burned_area_acres(state_no_spot), digits=3)) acres")

#-----------------------------------------------------------------------------#
#                     Simulation WITH Spotting
#-----------------------------------------------------------------------------#

println("\n[2] Running simulation with spotting enabled...")

state_spot = FireState(ncols, nrows, cellsize)
ignite!(state_spot, 30, 50, 0.0)

# Configure spotting parameters
spot_params = SpottingParameters{Float64}(
    mean_distance = 80.0,           # Mean spotting distance (m)
    normalized_variance = 0.6,
    ws_exponent = 1.2,              # Wind speed effect on distance
    flin_exponent = 0.4,            # Intensity effect on distance
    nembers_max = 8,
    surface_spotting_percent = 5.0, # 5% chance from surface fire
    crown_spotting_percent = 40.0,  # 40% chance from crown fire
    pign = 60.0,                    # 60% ignition probability
    min_distance = 10.0,
    max_distance = 500.0
)

config_spot = SimulationConfig{Float64}(
    enable_crown_fire = true,
    enable_spotting = true,
    spotting_params = spot_params
)

tracker = simulate_full_uniform!(
    state_spot,
    1,
    fuel_table,
    weather,
    0.0, 0.0,
    0.0, 15.0;
    canopy_cbd = 0.12,
    canopy_cbh = 4.0,
    canopy_cc = 0.65,
    canopy_ch = 16.0,
    config = config_spot,
    dt_initial = 0.2,
    rng = rng
)

println("   Burned: $(round(get_burned_area_acres(state_spot), digits=3)) acres")
println("   Pending spot fires: $(length(tracker.pending))")

#-----------------------------------------------------------------------------#
#                     Visualization
#-----------------------------------------------------------------------------#

println("\n[3] Creating visualizations...")

# Compare burned areas
p1 = heatmap(
    Float64.(state_no_spot.burned)',
    aspect_ratio = 1,
    title = "Without Spotting",
    xlabel = "X (cells)",
    ylabel = "Y (cells)",
    color = :OrRd,
    colorbar = false
)

p2 = heatmap(
    Float64.(state_spot.burned)',
    aspect_ratio = 1,
    title = "With Spotting",
    xlabel = "X (cells)",
    ylabel = "Y (cells)",
    color = :OrRd,
    colorbar = false
)

plot_compare = plot(p1, p2, layout=(1,2), size=(1000, 400))
savefig(plot_compare, joinpath(output_dir, "spotting_comparison.png"))
println("   Saved: spotting_comparison.png")

# Time of arrival comparison
toa_no_spot = copy(state_no_spot.time_of_arrival)
toa_no_spot[toa_no_spot .< 0] .= NaN

toa_spot = copy(state_spot.time_of_arrival)
toa_spot[toa_spot .< 0] .= NaN

p3 = heatmap(
    toa_no_spot',
    aspect_ratio = 1,
    title = "Time of Arrival (No Spotting)",
    xlabel = "X", ylabel = "Y",
    color = :viridis,
    clims = (0, 15)
)

p4 = heatmap(
    toa_spot',
    aspect_ratio = 1,
    title = "Time of Arrival (With Spotting)",
    xlabel = "X", ylabel = "Y",
    color = :viridis,
    clims = (0, 15)
)

plot_toa = plot(p3, p4, layout=(1,2), size=(1000, 400))
savefig(plot_toa, joinpath(output_dir, "spotting_toa.png"))
println("   Saved: spotting_toa.png")

# Show fire perimeter progression (using contours of TOA)
p5 = contour(
    1:ncols, 1:nrows, toa_no_spot',
    levels = 0:2:14,
    fill = false,
    linewidth = 2,
    aspect_ratio = 1,
    title = "Fire Progression (No Spotting)",
    xlabel = "X", ylabel = "Y",
    legend = :topright,
    color = :thermal
)

p6 = contour(
    1:ncols, 1:nrows, toa_spot',
    levels = 0:2:14,
    fill = false,
    linewidth = 2,
    aspect_ratio = 1,
    title = "Fire Progression (With Spotting)",
    xlabel = "X", ylabel = "Y",
    legend = :topright,
    color = :thermal
)

plot_contour = plot(p5, p6, layout=(1,2), size=(1000, 400))
savefig(plot_contour, joinpath(output_dir, "spotting_contours.png"))
println("   Saved: spotting_contours.png")

#-----------------------------------------------------------------------------#
#                     Summary
#-----------------------------------------------------------------------------#

println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)

area_no_spot = get_burned_area_acres(state_no_spot)
area_spot = get_burned_area_acres(state_spot)
increase = (area_spot - area_no_spot) / area_no_spot * 100

println("\nSpotting Effect on Burned Area:")
println("  Without spotting: $(round(area_no_spot, digits=3)) acres")
println("  With spotting:    $(round(area_spot, digits=3)) acres")
println("  Increase:         $(round(increase, digits=1))%")

# Find the easternmost burned cell (how far fire spread)
max_x_no_spot = maximum(ix for ix in 1:ncols for iy in 1:nrows if state_no_spot.burned[ix, iy])
max_x_spot = maximum(ix for ix in 1:ncols for iy in 1:nrows if state_spot.burned[ix, iy])

println("\nFire Spread Distance (eastward):")
println("  Without spotting: $((max_x_no_spot - 30) * cellsize) ft")
println("  With spotting:    $((max_x_spot - 30) * cellsize) ft")

println("\nPlots saved to: $output_dir")
println("=" ^ 60)
