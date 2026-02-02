#=
Fire Spread Animation
=====================

This example creates an animated GIF showing fire spread over time.

Requirements: Plots.jl
=#

using Elmfire
using Plots

# Create output directory
output_dir = joinpath(@__DIR__, "output")
mkpath(output_dir)

println("=" ^ 60)
println("Elmfire.jl - Animation Example")
println("=" ^ 60)

#-----------------------------------------------------------------------------#
#                     Setup Simulation
#-----------------------------------------------------------------------------#

println("\n[1] Setting up simulation...")

ncols, nrows = 80, 80
cellsize = 10.0

# Weather: moderate wind from southwest
weather = ConstantWeather(
    wind_speed_mph = 12.0,
    wind_direction = 225.0,  # From southwest
    M1 = 0.05,
    M10 = 0.07,
    M100 = 0.09,
    MLH = 0.50,
    MLW = 0.80
)

fuel_table = create_standard_fuel_table()

# We'll run the simulation in chunks and save frames
duration = 20.0  # Total simulation time (minutes)
frame_interval = 0.5  # Time between frames

#-----------------------------------------------------------------------------#
#                     Run Simulation with Snapshots
#-----------------------------------------------------------------------------#

println("\n[2] Running simulation and capturing frames...")

# Store snapshots
frames_burned = []
frames_toa = []
frame_times = Float64[]

state = FireState(ncols, nrows, cellsize)
ignite!(state, 20, 40, 0.0)  # Ignite in southwest quadrant

# Capture initial state
push!(frames_burned, copy(state.burned))
push!(frames_toa, copy(state.time_of_arrival))
push!(frame_times, 0.0)

t_current = 0.0
while t_current < duration
    t_next = min(t_current + frame_interval, duration)

    simulate_uniform!(
        state,
        1,
        fuel_table,
        weather,
        5.0,    # 5 degree slope
        45.0,   # Uphill to NE
        t_current,
        t_next;
        dt_initial = 0.1
    )

    t_current = t_next

    # Capture frame
    push!(frames_burned, copy(state.burned))
    push!(frames_toa, copy(state.time_of_arrival))
    push!(frame_times, t_current)

    print("\r   Time: $(round(t_current, digits=1)) / $duration min")
end
println()

println("   Captured $(length(frames_burned)) frames")
println("   Final burned area: $(round(get_burned_area_acres(state), digits=3)) acres")

#-----------------------------------------------------------------------------#
#                     Create Animation
#-----------------------------------------------------------------------------#

println("\n[3] Creating animation...")

# Animation of burned area
anim_burned = @animate for (i, burned) in enumerate(frames_burned)
    t = frame_times[i]

    heatmap(
        Float64.(burned)',
        aspect_ratio = 1,
        title = "Fire Spread - t = $(round(t, digits=1)) min",
        xlabel = "X (cells)",
        ylabel = "Y (cells)",
        color = :YlOrRd,
        clims = (0, 1),
        colorbar = false,
        size = (500, 500)
    )
end

gif(anim_burned, joinpath(output_dir, "fire_spread.gif"), fps=4)
println("   Saved: fire_spread.gif")

# Animation with time of arrival coloring
anim_toa = @animate for (i, toa) in enumerate(frames_toa)
    t = frame_times[i]

    # Mask unburned cells
    toa_masked = copy(toa)
    toa_masked[toa_masked .< 0] .= NaN

    heatmap(
        toa_masked',
        aspect_ratio = 1,
        title = "Time of Arrival - t = $(round(t, digits=1)) min",
        xlabel = "X (cells)",
        ylabel = "Y (cells)",
        color = :viridis,
        clims = (0, duration),
        colorbar_title = "Minutes",
        size = (550, 500)
    )
end

gif(anim_toa, joinpath(output_dir, "fire_toa.gif"), fps=4)
println("   Saved: fire_toa.gif")

#-----------------------------------------------------------------------------#
#                     Fire Perimeter Evolution
#-----------------------------------------------------------------------------#

println("\n[4] Creating perimeter evolution plot...")

# Create a single plot showing fire perimeter at different times
perimeter_times = [2.0, 5.0, 10.0, 15.0, 20.0]
colors = [:yellow, :orange, :red, :darkred, :black]

p_perimeter = plot(
    aspect_ratio = 1,
    title = "Fire Perimeter Evolution",
    xlabel = "X (cells)",
    ylabel = "Y (cells)",
    xlims = (0, ncols),
    ylims = (0, nrows),
    legend = :topright,
    size = (600, 600)
)

for (target_t, c) in zip(perimeter_times, colors)
    # Find the frame closest to target time
    idx = argmin(abs.(frame_times .- target_t))
    toa = frames_toa[idx]

    # Create contour at this time
    toa_masked = copy(toa)
    toa_masked[toa_masked .< 0] .= NaN
    toa_masked[toa_masked .> frame_times[idx]] .= NaN

    # Find perimeter (cells that are burned)
    burned = frames_burned[idx]
    perimeter_x = Int[]
    perimeter_y = Int[]

    for ix in 1:ncols, iy in 1:nrows
        if burned[ix, iy]
            # Check if on perimeter (has unburned neighbor)
            is_perimeter = false
            for (dx, dy) in [(1,0), (-1,0), (0,1), (0,-1)]
                nx, ny = ix + dx, iy + dy
                if 1 <= nx <= ncols && 1 <= ny <= nrows
                    if !burned[nx, ny]
                        is_perimeter = true
                        break
                    end
                end
            end
            if is_perimeter
                push!(perimeter_x, ix)
                push!(perimeter_y, iy)
            end
        end
    end

    scatter!(p_perimeter, perimeter_x, perimeter_y,
        label = "t = $(Int(target_t)) min",
        color = c,
        markersize = 2,
        markerstrokewidth = 0
    )
end

# Add ignition point
scatter!(p_perimeter, [20], [40],
    label = "Ignition",
    color = :blue,
    markersize = 8,
    marker = :star5
)

savefig(p_perimeter, joinpath(output_dir, "perimeter_evolution.png"))
println("   Saved: perimeter_evolution.png")

#-----------------------------------------------------------------------------#
#                     Summary
#-----------------------------------------------------------------------------#

println("\n" * "=" ^ 60)
println("Summary")
println("=" ^ 60)
println("\nSimulation Parameters:")
println("  Grid: $ncols x $nrows cells ($cellsize ft each)")
println("  Duration: $duration minutes")
println("  Wind: 12 mph from southwest (225°)")
println("  Slope: 5° uphill to northeast")

println("\nOutput Files:")
println("  fire_spread.gif      - Burned area animation")
println("  fire_toa.gif         - Time of arrival animation")
println("  perimeter_evolution.png - Perimeter at different times")

println("\nPlots saved to: $output_dir")
println("=" ^ 60)
