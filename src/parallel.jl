#-----------------------------------------------------------------------------#
#                     Parallel Execution Utilities
#-----------------------------------------------------------------------------#
#
# Provides multi-threaded execution for ensemble simulations.
# Uses Julia's built-in threading via Threads.@threads.
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Configuration
#-----------------------------------------------------------------------------#

"""
    ParallelConfig

Configuration for parallel execution.
"""
struct ParallelConfig
    n_workers::Int   # Number of worker threads (0 = auto-detect nthreads())
    chunk_size::Int  # Number of simulations per chunk for load balancing
end

function ParallelConfig(; n_workers::Int = 0, chunk_size::Int = 1)
    ParallelConfig(n_workers, chunk_size)
end


#-----------------------------------------------------------------------------#
#                     Thread-Local State
#-----------------------------------------------------------------------------#

"""
    ThreadLocalState{T<:AbstractFloat, R<:AbstractRNG}

Thread-local simulation state to avoid data races.
"""
mutable struct ThreadLocalState{T<:AbstractFloat, R<:AbstractRNG}
    fire_state::FireState{T}
    rng::R
end

Base.eltype(::ThreadLocalState{T}) where {T} = T


"""
    create_thread_local_states(template::FireState{T}, n::Int) -> Vector{ThreadLocalState{T, MersenneTwister}}

Create n thread-local states from a template.
"""
function create_thread_local_states(template::FireState{T}, n::Int) where {T<:AbstractFloat}
    states = ThreadLocalState{T, MersenneTwister}[]
    for i in 1:n
        state = copy(template)
        rng = MersenneTwister(i * 12345)
        push!(states, ThreadLocalState{T, MersenneTwister}(state, rng))
    end
    return states
end


#-----------------------------------------------------------------------------#
#                     Threaded Ensemble Execution
#-----------------------------------------------------------------------------#

"""
    run_ensemble_threaded!(
        config::EnsembleConfig{T},
        state_template::FireState{T},
        fuel_ids::AbstractMatrix{Int},
        fuel_table::FuelModelTable{T},
        weather::ConstantWeather{T},
        slope::AbstractMatrix{T},
        aspect::AbstractMatrix{T},
        ignition_ix::Int,
        ignition_iy::Int,
        t_start::T,
        t_stop::T;
        canopy::Union{Nothing, CanopyGrid{T}} = nothing,
        parallel_config::ParallelConfig = ParallelConfig(),
        show_progress::Bool = true,
        callback::Union{Nothing, Function} = nothing
    ) -> EnsembleResult{T}

Run a Monte Carlo ensemble using multiple threads.

# Arguments
Same as `run_ensemble!` with additional:
- `parallel_config`: Configuration for parallel execution
"""
function run_ensemble_threaded!(
    config::EnsembleConfig{T},
    state_template::FireState{T},
    fuel_ids::AbstractMatrix{Int},
    fuel_table::FuelModelTable{T},
    weather::ConstantWeather{T},
    slope::AbstractMatrix{T},
    aspect::AbstractMatrix{T},
    ignition_ix::Int,
    ignition_iy::Int,
    t_start::T,
    t_stop::T;
    canopy::Union{Nothing, CanopyGrid{T}} = nothing,
    parallel_config::ParallelConfig = ParallelConfig(),
    show_progress::Bool = true,
    callback::CB = nothing
) where {T<:AbstractFloat, CB}
    ncols = state_template.ncols
    nrows = state_template.nrows

    # Determine number of threads to use for work
    n_threads = if parallel_config.n_workers == 0
        Threads.nthreads()
    else
        min(parallel_config.n_workers, Threads.nthreads())
    end

    # Create thread-local states
    # Use maxthreadid() to handle interactive threads which have higher IDs
    max_tid = Threads.maxthreadid()
    thread_states = create_thread_local_states(state_template, max_tid)

    # Initialize result containers (thread-safe)
    members_lock = ReentrantLock()
    members = Vector{EnsembleMember{T}}(undef, config.n_simulations)

    # Progress tracking
    progress_counter = Threads.Atomic{Int}(0)
    progress = if show_progress
        Progress(config.n_simulations, desc="Running threaded ensemble: ")
    else
        nothing
    end

    # Run simulations in parallel
    Threads.@threads for i in 1:config.n_simulations
        tid = Threads.threadid()
        local_state = thread_states[tid]

        # Create deterministic RNG for this member
        seed = config.base_seed + UInt64(i)
        rng = MersenneTwister(seed)

        # Reset and configure the thread-local state
        reset!(local_state.fire_state)

        # Perturb weather
        perturbed_weather = perturb_weather(weather, config.perturbation, rng)

        # Perturb ignition location
        new_ix, new_iy = perturb_ignition(
            ignition_ix, ignition_iy,
            config.perturbation.ignition_perturb_radius,
            ncols, nrows,
            rng
        )

        # Ignite
        ignite!(local_state.fire_state, new_ix, new_iy, t_start)

        # Create weather interpolator
        weather_interp = create_constant_interpolator(perturbed_weather, ncols, nrows, local_state.fire_state.cellsize)

        # Run simulation
        simulate_full!(
            local_state.fire_state,
            fuel_ids,
            fuel_table,
            weather_interp,
            slope,
            aspect,
            t_start,
            t_stop;
            canopy = canopy,
            config = config.simulation_config,
            rng = rng
        )

        # Compute member statistics
        burned_area = get_burned_area_acres(local_state.fire_state)
        max_spread = maximum(local_state.fire_state.spread_rate)

        # Create member record (copy data to avoid race conditions)
        member = EnsembleMember{T}(
            i,
            seed,
            config.save_individual_results ? copy(local_state.fire_state.burned) : local_state.fire_state.burned,
            config.save_individual_results ? copy(local_state.fire_state.time_of_arrival) : local_state.fire_state.time_of_arrival,
            burned_area,
            max_spread
        )

        # Store result (atomic)
        members[i] = member

        # Update progress (thread-safe)
        Threads.atomic_add!(progress_counter, 1)
        if progress !== nothing && tid == 1
            # Only thread 1 updates progress bar to avoid flickering
            update!(progress, progress_counter[])
        end

        # Callback
        if callback !== nothing
            lock(members_lock) do
                callback(i, local_state.fire_state)
            end
        end
    end

    # Finish progress bar
    if progress !== nothing
        finish!(progress)
    end

    # Create result object
    result = EnsembleResult{T}(config, ncols, nrows)
    result.members = members

    # Compute convergence history incrementally
    prev_burn_prob = zeros(T, ncols, nrows)
    burn_count = zeros(Int, ncols, nrows)
    for i in 1:config.n_simulations
        member = result.members[i]
        for ix in 1:ncols
            for iy in 1:nrows
                if member.burned[ix, iy]
                    burn_count[ix, iy] += 1
                end
            end
        end
        inv_i = one(T) / T(i)
        rms_sum = zero(T)
        for ix in 1:ncols
            for iy in 1:nrows
                new_prob = T(burn_count[ix, iy]) * inv_i
                diff = new_prob - prev_burn_prob[ix, iy]
                rms_sum += diff * diff
                prev_burn_prob[ix, iy] = new_prob
            end
        end
        push!(result.convergence_history, sqrt(rms_sum / (ncols * nrows)))
    end

    # Compute final statistics
    aggregate_ensemble_statistics!(result)

    return result
end


#-----------------------------------------------------------------------------#
#                     Parallel Batch Processing
#-----------------------------------------------------------------------------#

"""
    BatchSimulationJob{T<:AbstractFloat}

A single simulation job for batch processing.
"""
struct BatchSimulationJob{T<:AbstractFloat}
    id::Int
    ignition_ix::Int
    ignition_iy::Int
    weather::ConstantWeather{T}
    t_start::T
    t_stop::T
end


"""
    BatchResult{T<:AbstractFloat}

Result from a single batch job.
"""
struct BatchResult{T<:AbstractFloat}
    job_id::Int
    burned::BitMatrix
    time_of_arrival::Matrix{T}
    burned_area_acres::T
    elapsed_time::Float64
end


"""
    run_batch_simulations(
        jobs::Vector{BatchSimulationJob{T}},
        state_template::FireState{T},
        fuel_ids::AbstractMatrix{Int},
        fuel_table::FuelModelTable{T},
        slope::AbstractMatrix{T},
        aspect::AbstractMatrix{T};
        canopy::Union{Nothing, CanopyGrid{T}} = nothing,
        config::SimulationConfig{T} = SimulationConfig{T}(),
        show_progress::Bool = true
    ) -> Vector{BatchResult{T}}

Run a batch of independent simulations in parallel.

This is useful for running multiple scenarios with different ignition points
or weather conditions.
"""
function run_batch_simulations(
    jobs::Vector{BatchSimulationJob{T}},
    state_template::FireState{T},
    fuel_ids::AbstractMatrix{Int},
    fuel_table::FuelModelTable{T},
    slope::AbstractMatrix{T},
    aspect::AbstractMatrix{T};
    canopy::Union{Nothing, CanopyGrid{T}} = nothing,
    config::SimulationConfig{T} = SimulationConfig{T}(),
    show_progress::Bool = true
) where {T<:AbstractFloat}
    n_jobs = length(jobs)
    n_threads = Threads.nthreads()

    # Create thread-local states
    thread_states = create_thread_local_states(state_template, n_threads)

    # Results array
    results = Vector{BatchResult{T}}(undef, n_jobs)

    # Progress
    progress = if show_progress
        Progress(n_jobs, desc="Running batch simulations: ")
    else
        nothing
    end
    progress_counter = Threads.Atomic{Int}(0)

    Threads.@threads for i in 1:n_jobs
        tid = Threads.threadid()
        job = jobs[i]
        local_state = thread_states[tid]

        # Record start time
        t0 = time()

        # Reset state
        reset!(local_state.fire_state)

        # Ignite
        ignite!(local_state.fire_state, job.ignition_ix, job.ignition_iy, job.t_start)

        # Create weather interpolator
        ncols = local_state.fire_state.ncols
        nrows = local_state.fire_state.nrows
        weather_interp = create_constant_interpolator(job.weather, ncols, nrows, local_state.fire_state.cellsize)

        # Create thread-local RNG
        rng = MersenneTwister(job.id * 12345)

        # Run simulation
        simulate_full!(
            local_state.fire_state,
            fuel_ids,
            fuel_table,
            weather_interp,
            slope,
            aspect,
            job.t_start,
            job.t_stop;
            canopy = canopy,
            config = config,
            rng = rng
        )

        # Compute elapsed time
        elapsed = time() - t0

        # Store result
        results[i] = BatchResult{T}(
            job.id,
            copy(local_state.fire_state.burned),
            copy(local_state.fire_state.time_of_arrival),
            get_burned_area_acres(local_state.fire_state),
            elapsed
        )

        # Update progress
        Threads.atomic_add!(progress_counter, 1)
        if progress !== nothing && tid == 1
            update!(progress, progress_counter[])
        end
    end

    if progress !== nothing
        finish!(progress)
    end

    return results
end


#-----------------------------------------------------------------------------#
#                     Parallel Map/Reduce Utilities
#-----------------------------------------------------------------------------#

"""
    parallel_map(f, items::AbstractVector; n_threads::Int = 0)

Apply function f to each item in parallel.
"""
function parallel_map(f::F, items::AbstractVector; n_threads::Int = 0) where {F}
    n = length(items)
    RT = Core.Compiler.return_type(f, Tuple{eltype(items)})
    results = Vector{RT}(undef, n)

    Threads.@threads for i in 1:n
        results[i] = f(items[i])
    end

    return results
end


"""
    parallel_reduce(f, g, items::AbstractVector; init, n_threads::Int = 0)

Apply function f to each item and reduce with g.

# Arguments
- `f`: Map function to apply to each item
- `g`: Reduce function to combine results (must be associative)
- `items`: Items to process
- `init`: Initial value for reduction
"""
function parallel_reduce(f::F, g::G, items::AbstractVector; init, n_threads::Int = 0) where {F, G}
    n = length(items)
    if n == 0
        return init
    end

    n_workers = n_threads == 0 ? Threads.nthreads() : n_threads

    # Split work among threads
    chunk_size = max(1, div(n, n_workers))
    IT = typeof(init)
    partial_results = Vector{IT}(undef, n_workers)

    Threads.@threads for tid in 1:n_workers
        start_idx = (tid - 1) * chunk_size + 1
        end_idx = tid == n_workers ? n : min(tid * chunk_size, n)

        if start_idx <= n
            local_result = init
            for i in start_idx:end_idx
                local_result = g(local_result, f(items[i]))
            end
            partial_results[tid] = local_result
        else
            partial_results[tid] = init
        end
    end

    # Final reduction
    result = init
    for pr in partial_results
        result = g(result, pr)
    end

    return result
end


#-----------------------------------------------------------------------------#
#                     Thread Safety Utilities
#-----------------------------------------------------------------------------#

"""
    atomic_max!(arr::Matrix{T}, ix::Int, iy::Int, val::T)

Atomically update arr[ix,iy] = max(arr[ix,iy], val).
Note: This is a simplified version that uses locks.
"""
function atomic_max!(arr::Matrix{T}, ix::Int, iy::Int, val::T, lock::ReentrantLock) where {T}
    Base.@lock lock begin
        if val > arr[ix, iy]
            arr[ix, iy] = val
        end
    end
end


"""
    atomic_add!(arr::Matrix{T}, ix::Int, iy::Int, val::T)

Atomically add val to arr[ix,iy].
"""
function atomic_add!(arr::Matrix{T}, ix::Int, iy::Int, val::T, lock::ReentrantLock) where {T}
    Base.@lock lock begin
        arr[ix, iy] += val
    end
end
