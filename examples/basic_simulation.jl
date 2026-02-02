#=
Basic Fire Simulation Example
==============================

This example demonstrates:
1. Setting up a fire simulation grid
2. Running simulations with different wind conditions
3. Visualizing fire spread patterns

Requirements: Plots.jl (run `using Pkg; Pkg.add("Plots")` if not installed)
=#

using Elmfire
using Plots

# Create output directory for plots
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("=" ^ 60)
println("Elmfire.jl - Basic Simulation Example")
println("=" ^ 60)

#-----------------------------------------------------------------------------#
#                     Example 1: No Wind (Circular Spread)
#-----------------------------------------------------------------------------#

println("\n[1] Running simulation with no wind...")

# Create simulation state: 100x100 grid with 10ft cells
state_no_wind = FireState(100, 100, 10.0)

# Ignite center of the grid
ignite!(state_no_wind, 50, 50, 0.0)

# Create fuel table (standard 40 fuel models)
fuel_table = create_standard_fuel_table()

# Set up weather conditions - no wind, moderate moisture
weather_no_wind = ConstantWeather(
    wind_speed_mph = 0.0,
    wind_direction = 0.0,
    M1 = 0.06,      # 6% dead fuel moisture
    M10 = 0.08,
    M100 = 0.10,
    MLH = 0.60,     # 60% live herbaceous
    MLW = 0.90      # 90% live woody
)

# Run simulation for 10 minutes
simulate_uniform!(
    state_no_wind,
    1,              # Fuel model 1 (short grass)
    fuel_table,
    weather_no_wind,
    0.0,            # Flat terrain (0 degrees slope)
    0.0,            # No aspect
    0.0,            # Start time
    10.0;           # End time (10 minutes)
    dt_initial = 0.2
)

burned_acres_no_wind = get_burned_area_acres(state_no_wind)
println("   Burned area: $(round(burned_acres_no_wind, digits=3)) acres")

#-----------------------------------------------------------------------------#
#                     Example 2: With Wind (Elliptical Spread)
#-----------------------------------------------------------------------------#

println("\n[2] Running simulation with wind from west...")

state_wind = FireState(100, 100, 10.0)
ignite!(state_wind, 50, 50, 0.0)

# Wind from west (270°) - fire spreads east
weather_wind = ConstantWeather(
    wind_speed_mph = 15.0,
    wind_direction = 270.0,  # FROM west
    M1 = 0.05,
    M10 = 0.07,
    M100 = 0.09,
    MLH = 0.50,
    MLW = 0.80
)

simulate_uniform!(
    state_wind,
    1,
    fuel_table,
    weather_wind,
    0.0,
    0.0,
    0.0,
    10.0;
    dt_initial = 0.2
)

burned_acres_wind = get_burned_area_acres(state_wind)
println("   Burned area: $(round(burned_acres_wind, digits=3)) acres")

#-----------------------------------------------------------------------------#
#                     Example 3: Crown Fire Simulation
#-----------------------------------------------------------------------------#

println("\n[3] Running simulation with crown fire enabled...")

state_crown = FireState(100, 100, 10.0)
ignite!(state_crown, 50, 50, 0.0)

# Dry, windy conditions favorable for crown fire
weather_crown = ConstantWeather(
    wind_speed_mph = 20.0,
    wind_direction = 270.0,
    M1 = 0.03,      # Very dry
    M10 = 0.05,
    M100 = 0.07,
    MLH = 0.40,
    MLW = 0.70
)

# Enable crown fire
config_crown = SimulationConfig{Float64}(
    enable_crown_fire = true,
    crown_fire_adj = 1.0,
    foliar_moisture = 100.0
)

simulate_full_uniform!(
    state_crown,
    1,
    fuel_table,
    weather_crown,
    0.0,
    0.0,
    0.0,
    10.0;
    canopy_cbd = 0.15,   # Canopy bulk density (kg/m³)
    canopy_cbh = 3.0,    # Canopy base height (m)
    canopy_cc = 0.7,     # Canopy cover (70%)
    canopy_ch = 18.0,    # Canopy height (m)
    config = config_crown,
    dt_initial = 0.2
)

burned_acres_crown = get_burned_area_acres(state_crown)
println("   Burned area: $(round(burned_acres_crown, digits=3)) acres")

#-----------------------------------------------------------------------------#
#                     Visualization
#-----------------------------------------------------------------------------#

println("\n[4] Creating visualizations...")

# Helper function to create a clean heatmap
function fire_heatmap(data, title; kwargs...)
    heatmap(
        data',  # Transpose for correct orientation
        aspect_ratio = 1,
        title = title,
        xlabel = "X (cells)",
        ylabel = "Y (cells)",
        color = :YlOrRd,
        framestyle = :box;
        kwargs...
    )
end

# Plot 1: Compare burned areas (no wind vs wind)
p1 = fire_heatmap(
    Float64.(state_no_wind.burned),
    "No Wind - Burned Area"
)

p2 = fire_heatmap(
    Float64.(state_wind.burned),
    "15 mph Wind from West - Burned Area"
)

plot_burned = plot(p1, p2, layout=(1,2), size=(1000, 400))
savefig(plot_burned, joinpath(output_dir, "burned_comparison.png"))
println("   Saved: burned_comparison.png")

# Plot 2: Time of Arrival
toa_no_wind = copy(state_no_wind.time_of_arrival)
toa_no_wind[toa_no_wind .< 0] .= NaN  # Mask unburned cells

toa_wind = copy(state_wind.time_of_arrival)
toa_wind[toa_wind .< 0] .= NaN

p3 = fire_heatmap(
    toa_no_wind,
    "Time of Arrival (No Wind)",
    color = :viridis,
    colorbar_title = "Minutes"
)

p4 = fire_heatmap(
    toa_wind,
    "Time of Arrival (With Wind)",
    color = :viridis,
    colorbar_title = "Minutes"
)

plot_toa = plot(p3, p4, layout=(1,2), size=(1000, 400))
savefig(plot_toa, joinpath(output_dir, "time_of_arrival.png"))
println("   Saved: time_of_arrival.png")

# Plot 3: Fireline Intensity
flin_wind = copy(state_wind.fireline_intensity)
flin_wind[flin_wind .== 0] .= NaN

flin_crown = copy(state_crown.fireline_intensity)
flin_crown[flin_crown .== 0] .= NaN

p5 = fire_heatmap(
    flin_wind,
    "Fireline Intensity (Surface Fire)",
    color = :hot,
    colorbar_title = "kW/m"
)

p6 = fire_heatmap(
    flin_crown,
    "Fireline Intensity (Crown Fire)",
    color = :hot,
    colorbar_title = "kW/m"
)

plot_flin = plot(p5, p6, layout=(1,2), size=(1000, 400))
savefig(plot_flin, joinpath(output_dir, "fireline_intensity.png"))
println("   Saved: fireline_intensity.png")

# Plot 4: Spread Rate
ros_wind = copy(state_wind.spread_rate)
ros_wind[ros_wind .== 0] .= NaN

ros_crown = copy(state_crown.spread_rate)
ros_crown[ros_crown .== 0] .= NaN

p7 = fire_heatmap(
    ros_wind,
    "Spread Rate (Surface Fire)",
    color = :plasma,
    colorbar_title = "ft/min"
)

p8 = fire_heatmap(
    ros_crown,
    "Spread Rate (Crown Fire)",
    color = :plasma,
    colorbar_title = "ft/min"
)

plot_ros = plot(p7, p8, layout=(1,2), size=(1000, 400))
savefig(plot_ros, joinpath(output_dir, "spread_rate.png"))
println("   Saved: spread_rate.png")

# Plot 5: Summary comparison (4 panels)
p_summary = plot(
    fire_heatmap(Float64.(state_no_wind.burned), "No Wind"),
    fire_heatmap(Float64.(state_wind.burned), "15 mph Wind"),
    fire_heatmap(Float64.(state_crown.burned), "Crown Fire"),
    fire_heatmap(flin_crown, "Crown Fire Intensity", color=:hot),
    layout = (2, 2),
    size = (900, 800)
)
savefig(p_summary, joinpath(output_dir, "summary.png"))
println("   Saved: summary.png")

#-----------------------------------------------------------------------------#
#                     Summary Statistics
#-----------------------------------------------------------------------------#

println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
println("\nSimulation Parameters:")
println("  Grid size: 100 x 100 cells (10 ft each)")
println("  Domain: $(100 * 10) x $(100 * 10) ft = $(round(100*10*100*10/43560, digits=2)) acres")
println("  Duration: 10 minutes")
println("  Fuel model: 1 (short grass)")

println("\nResults:")
println("  No wind:     $(round(burned_acres_no_wind, digits=3)) acres burned")
println("  With wind:   $(round(burned_acres_wind, digits=3)) acres burned")
println("  Crown fire:  $(round(burned_acres_crown, digits=3)) acres burned")

println("\nMax Spread Rates:")
println("  Surface fire: $(round(maximum(state_wind.spread_rate), digits=1)) ft/min")
println("  Crown fire:   $(round(maximum(state_crown.spread_rate), digits=1)) ft/min")

println("\nMax Fireline Intensity:")
println("  Surface fire: $(round(maximum(state_wind.fireline_intensity), digits=1)) kW/m")
println("  Crown fire:   $(round(maximum(state_crown.fireline_intensity), digits=1)) kW/m")

println("\nPlots saved to: $output_dir")
println("=" ^ 60)
