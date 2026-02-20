using Elmfire
using KernelAbstractions
using Adapt
using Statistics: median
using Printf
using Plots
using Dates

# Try to load Metal
const HAS_METAL = try
    @eval using Metal
    Metal.functional()
catch
    false
end

#--------------------------------------------------------------------------------# Config
const GRID_SIZES = [100, 256, 512, 1024]
const N_RUNS = 5
const FUEL_ID = 1
const WIND_MPH = 15.0
const SIM_DURATION = 10.0  # minutes
const OUTDIR = joinpath(@__DIR__, "results")

#--------------------------------------------------------------------------------# Setup
function make_state(::Type{T}, n::Int) where {T}
    FireState{T}(n, n, T(10))
end

function make_weather(::Type{T}) where {T}
    ConstantWeather{T}(
        wind_speed_mph = T(WIND_MPH),
        wind_direction = T(270),
        M1 = T(0.04),
        M10 = T(0.06),
        M100 = T(0.08),
        MLH = T(0.50),
        MLW = T(0.80),
    )
end

#--------------------------------------------------------------------------------# CPU benchmark
function bench_cpu(::Type{T}, n::Int) where {T}
    state = make_state(T, n)
    fuel_table = create_standard_fuel_table(T)
    weather = make_weather(T)
    center = div(n, 2)

    # Warmup
    ignite!(state, center, center, T(0))
    simulate_uniform!(state, FUEL_ID, fuel_table, weather,
        T(0), T(0), T(0), T(SIM_DURATION); dt_initial = T(0.2))
    burned = count(state.burned)

    # Timed runs
    times = Float64[]
    for _ in 1:N_RUNS
        reset!(state)
        ignite!(state, center, center, T(0))
        t = @elapsed simulate_uniform!(state, FUEL_ID, fuel_table, weather,
            T(0), T(0), T(0), T(SIM_DURATION); dt_initial = T(0.2))
        push!(times, t)
    end

    return times, burned
end

#--------------------------------------------------------------------------------# GPU benchmark (parameterized backend)
function bench_gpu(::Type{T}, n::Int; backend = KernelAbstractions.CPU()) where {T}
    state = make_state(T, n)
    fuel_table = create_standard_fuel_table(T)
    fuel_array = FuelModelArray(fuel_table)
    weather = make_weather(T)
    center = div(n, 2)

    # Warmup
    ignite!(state, center, center, T(0))
    simulate_gpu_uniform!(state, FUEL_ID, fuel_array, weather,
        T(0), T(0), T(0), T(SIM_DURATION);
        dt_initial = T(0.2), backend = backend)
    burned = count(state.burned)

    # Timed runs
    times = Float64[]
    for _ in 1:N_RUNS
        reset!(state)
        ignite!(state, center, center, T(0))
        t = @elapsed simulate_gpu_uniform!(state, FUEL_ID, fuel_array, weather,
            T(0), T(0), T(0), T(SIM_DURATION);
            dt_initial = T(0.2), backend = backend)
        push!(times, t)
    end

    return times, burned
end

#--------------------------------------------------------------------------------# Run benchmarks and generate report
function main()
    mkpath(OUTDIR)

    # Collect results: (grid, precision, backend) → (times, burned)
    results = Dict{Tuple{Int, String, String}, Tuple{Vector{Float64}, Int}}()

    # Determine which backends to run
    backends = [
        ("CPU",      (T, n) -> bench_cpu(T, n)),
        ("KA.CPU",   (T, n) -> bench_gpu(T, n; backend = KernelAbstractions.CPU())),
    ]
    if HAS_METAL
        push!(backends, ("Metal", (T, n) -> bench_gpu(T, n; backend = Metal.MetalBackend())))
    end

    # Determine which precisions to test per backend
    precisions_for = Dict(
        "CPU"    => [(Float64, "Float64"), (Float32, "Float32")],
        "KA.CPU" => [(Float64, "Float64"), (Float32, "Float32")],
        "Metal"  => [(Float32, "Float32")],  # Metal does not support Float64
    )

    println("Running benchmarks...")
    if HAS_METAL
        println("  Metal GPU detected: $(Metal.current_device())")
    else
        println("  Metal GPU not available — skipping Metal benchmarks")
    end
    println()

    for n in GRID_SIZES
        for (label, bench_fn) in backends
            for (T, prec) in precisions_for[label]
                print("  $(n)x$(n) $prec $label ... ")
                times, burned = bench_fn(T, n)
                results[(n, prec, label)] = (times, burned)
                @printf("%.3fs (median)\n", median(times))
            end
        end
    end

    # --- Collect all backend labels that were actually run ---
    all_backends = [b[1] for b in backends]

    # --- Generate plots ---
    println("\nGenerating plots...")

    # Plot 1: Median time by grid size
    p1 = let
        configs = Tuple{String, String}[]
        for b in all_backends
            for (_, prec) in precisions_for[b]
                push!(configs, (prec, b))
            end
        end
        p = plot(
            title = "Median Simulation Time by Grid Size",
            xlabel = "Grid Size", ylabel = "Time (s)",
            legend = :topleft, size = (700, 450),
            xticks = (1:length(GRID_SIZES), string.(GRID_SIZES)),
            margin = 5Plots.mm,
        )
        styles = Dict("CPU" => :solid, "KA.CPU" => :dash, "Metal" => :dot)
        markers = Dict("CPU" => :circle, "KA.CPU" => :diamond, "Metal" => :star5)
        for (prec, backend) in configs
            meds = [median(results[(n, prec, backend)][1]) for n in GRID_SIZES]
            plot!(p, 1:length(GRID_SIZES), meds,
                label = "$prec $backend", lw = 2,
                ls = get(styles, backend, :solid),
                marker = get(markers, backend, :circle), ms = 6)
        end
        p
    end
    savefig(p1, joinpath(OUTDIR, "time_by_grid.png"))

    # Plot 2: Burned cells by grid size
    p2 = let
        configs = Tuple{String, String}[]
        for b in all_backends
            for (_, prec) in precisions_for[b]
                push!(configs, (prec, b))
            end
        end
        p = plot(
            title = "Burned Cells by Grid Size",
            xlabel = "Grid Size", ylabel = "Burned Cells",
            legend = :topleft, size = (700, 450),
            xticks = (1:length(GRID_SIZES), string.(GRID_SIZES)),
            margin = 5Plots.mm,
        )
        styles = Dict("CPU" => :solid, "KA.CPU" => :dash, "Metal" => :dot)
        markers = Dict("CPU" => :circle, "KA.CPU" => :diamond, "Metal" => :star5)
        for (prec, backend) in configs
            cells = [results[(n, prec, backend)][2] for n in GRID_SIZES]
            plot!(p, 1:length(GRID_SIZES), cells,
                label = "$prec $backend", lw = 2,
                ls = get(styles, backend, :solid),
                marker = get(markers, backend, :circle), ms = 6)
        end
        p
    end
    savefig(p2, joinpath(OUTDIR, "burned_cells.png"))

    # Plot 3: Float32 speedup over Float64 (only for backends with both)
    p3 = let
        p = plot(
            title = "Float32 Speedup over Float64",
            xlabel = "Grid Size", ylabel = "Speedup (×)",
            legend = :topleft, size = (700, 450),
            xticks = (1:length(GRID_SIZES), string.(GRID_SIZES)),
            margin = 5Plots.mm,
        )
        styles = Dict("CPU" => :solid, "KA.CPU" => :dash)
        markers = Dict("CPU" => :circle, "KA.CPU" => :diamond)
        for b in all_backends
            has_both = any(p == "Float64" for (_, p) in precisions_for[b]) &&
                       any(p == "Float32" for (_, p) in precisions_for[b])
            if has_both
                speedup = [median(results[(n, "Float64", b)][1]) / median(results[(n, "Float32", b)][1]) for n in GRID_SIZES]
                plot!(p, 1:length(GRID_SIZES), speedup,
                    label = b, lw = 2,
                    ls = get(styles, b, :solid),
                    marker = get(markers, b, :circle), ms = 6)
            end
        end
        hline!(p, [1.0], ls = :dot, color = :gray, label = "1× (no speedup)")
        p
    end
    savefig(p3, joinpath(OUTDIR, "float32_speedup.png"))

    # --- Generate markdown report ---
    println("Writing report...")
    report = joinpath(OUTDIR, "report.md")
    open(report, "w") do io
        println(io, "# Elmfire.jl Benchmark Report")
        println(io)
        println(io, "**Date:** $(Dates.format(now(), "yyyy-mm-dd HH:MM"))")
        println(io)

        # System info
        println(io, "## System")
        println(io)
        println(io, "| Property | Value |")
        println(io, "|----------|-------|")
        println(io, "| Julia | $(VERSION) |")
        println(io, "| OS | $(Sys.isapple() ? "macOS" : Sys.islinux() ? "Linux" : Sys.iswindows() ? "Windows" : string(Sys.KERNEL)) $(Sys.KERNEL == :Darwin ? strip(read(`sw_vers -productVersion`, String)) : "") |")
        println(io, "| CPU | $(Sys.cpu_info()[1].model) |")
        println(io, "| CPU Cores | $(Sys.CPU_THREADS) |")
        println(io, "| RAM | $(round(Sys.total_memory() / 2^30, digits=1)) GB |")
        if HAS_METAL
            println(io, "| GPU | $(Metal.current_device()) |")
        end
        println(io, "| Elmfire | $(pkgversion(Elmfire)) |")
        println(io, "| KernelAbstractions | $(pkgversion(KernelAbstractions)) |")
        if HAS_METAL
            println(io, "| Metal | $(pkgversion(Metal)) |")
        end
        println(io)

        println(io, "## Configuration")
        println(io)
        println(io, "| Parameter | Value |")
        println(io, "|-----------|-------|")
        println(io, "| Fuel Model | FBFM0$FUEL_ID |")
        println(io, "| Wind Speed | $(Int(WIND_MPH)) mph (from west) |")
        println(io, "| Sim Duration | $(Int(SIM_DURATION)) min |")
        println(io, "| Cell Size | 10 ft |")
        println(io, "| Runs per config | $N_RUNS (median reported) |")
        println(io)

        # Results table
        println(io, "## Results")
        println(io)
        println(io, "| Grid | Precision | Backend | Median (s) | Min (s) | Max (s) | Burned Cells |")
        println(io, "|------|-----------|---------|------------|---------|---------|--------------|")
        for n in GRID_SIZES
            for b in all_backends
                for (_, prec) in precisions_for[b]
                    times, burned = results[(n, prec, b)]
                    med = median(times)
                    mn = minimum(times)
                    mx = maximum(times)
                    @printf(io, "| %d | %s | %s | %.3f | %.3f | %.3f | %d |\n",
                        n, prec, b, med, mn, mx, burned)
                end
            end
        end

        # Speedup summary (only for backends with Float64 + Float32)
        backends_with_both = [b for b in all_backends
            if any(p == "Float64" for (_, p) in precisions_for[b]) &&
               any(p == "Float32" for (_, p) in precisions_for[b])]
        if !isempty(backends_with_both)
            println(io)
            println(io, "## Float32 vs Float64 Speedup")
            println(io)
            println(io, "| Grid | Backend | Speedup (×) |")
            println(io, "|------|---------|-------------|")
            for n in GRID_SIZES
                for b in backends_with_both
                    t64 = median(results[(n, "Float64", b)][1])
                    t32 = median(results[(n, "Float32", b)][1])
                    @printf(io, "| %d | %s | %.2f |\n", n, b, t64 / t32)
                end
            end
        end

        # Metal vs CPU speedup
        if HAS_METAL
            println(io)
            println(io, "## Metal GPU vs CPU Speedup (Float32)")
            println(io)
            println(io, "| Grid | CPU (s) | Metal (s) | Speedup (×) |")
            println(io, "|------|---------|-----------|-------------|")
            for n in GRID_SIZES
                t_cpu = median(results[(n, "Float32", "CPU")][1])
                t_metal = median(results[(n, "Float32", "Metal")][1])
                @printf(io, "| %d | %.3f | %.3f | %.2f |\n", n, t_cpu, t_metal, t_cpu / t_metal)
            end
        end

        # Plots
        println(io)
        println(io, "## Plots")
        println(io)
        println(io, "### Simulation Time by Grid Size")
        println(io, "![Time by Grid Size](time_by_grid.png)")
        println(io)
        println(io, "### Burned Cells by Grid Size")
        println(io, "![Burned Cells](burned_cells.png)")
        println(io)
        println(io, "### Float32 Speedup")
        println(io, "![Float32 Speedup](float32_speedup.png)")
        println(io)

        # Notes
        println(io, "## Notes")
        println(io)
        println(io, "- **KA.CPU** benchmarks the GPU code path running on CPU threads via")
        println(io, "  `KernelAbstractions.CPU()`, not an actual GPU.")
        if HAS_METAL
            println(io, "- **Metal** runs on the Apple Silicon GPU. Only Float32 is supported.")
        end
        println(io, "- CPU and GPU paths use different update strategies (serial vs parallel RK2),")
        println(io, "  so burned cell counts may differ. Both are valid.")
    end

    println("Done! Report written to: $report")
end

main()
