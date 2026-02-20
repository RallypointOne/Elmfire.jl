module ElmfireKAExt

using Elmfire
using KernelAbstractions
using Adapt

#-----------------------------------------------------------------------------#
#                     Adapt Rules for FireState
#-----------------------------------------------------------------------------#

function Adapt.adapt_structure(to, state::Elmfire.FireState{T}) where {T}
    Elmfire.FireState{T}(
        Adapt.adapt(to, state.phi),
        Adapt.adapt(to, state.phi_old),
        Adapt.adapt(to, state.time_of_arrival),
        state.burned,                            # stays CPU
        Adapt.adapt(to, state.spread_rate),
        Adapt.adapt(to, state.fireline_intensity),
        Adapt.adapt(to, state.flame_length),
        Adapt.adapt(to, state.ux),
        Adapt.adapt(to, state.uy),
        state.narrow_band,                       # stays CPU
        state.ncols, state.nrows, state.cellsize,
        state.xllcorner, state.yllcorner, state.padding
    )
end


#-----------------------------------------------------------------------------#
#                     Helper: half_superbee for kernels
#-----------------------------------------------------------------------------#

@inline function _half_superbee(r::T) where {T}
    max(zero(T), max(min(T(0.5) * r, one(T)), min(r, T(0.5))))
end


#-----------------------------------------------------------------------------#
#                     GPU Gather Kernel
#-----------------------------------------------------------------------------#

@kernel function gather_phi_kernel!(
    phi_out, @Const(phi), @Const(active_px), @Const(active_py)
)
    i = @index(Global, Linear)
    phi_out[i] = phi[active_px[i], active_py[i]]
end


#-----------------------------------------------------------------------------#
#                     GPU Copy-Active Kernel
#-----------------------------------------------------------------------------#

@kernel function copy_active_kernel!(
    dst, @Const(src), @Const(active_px), @Const(active_py)
)
    i = @index(Global, Linear)
    px = active_px[i]
    py = active_py[i]
    dst[px, py] = src[px, py]
end


#-----------------------------------------------------------------------------#
#                     GPU Spread Rate Kernel (index-list)
#-----------------------------------------------------------------------------#
# Split into two kernels for Metal shader compiler compatibility.
# Fuel model data is packed into two 3D arrays to reduce argument count:
#   fuel_scalars[fi, mi, 1:14] = delta, rhob, xi, B, GP_WND, GP_WNL,
#       phisterm, phiwterm, R_val, mex_dead, mex_live, F_dead, F_live, tr
#   fuel_components[fi, mi, 1:16] = F[1:6], FEPS[1:6], WPRIMENUMER[1:4]

@kernel function spread_rate_kernel!(
    vel_out, ecc_out, @Const(active_px), @Const(active_py),
    @Const(burned),
    @Const(fuel_ids), @Const(fuel_id_to_index),
    @Const(fuel_scalars), @Const(fa_nonburnable), @Const(fuel_components),
    @Const(slope),
    ws20_ftpmin, M1, M10, M100, MLH, MLW,
    live_moisture_class, spread_rate_adj,
    padding, ncols, nrows
)
    T = eltype(vel_out)
    i = @index(Global, Linear)
    px = active_px[i]
    py = active_py[i]

    ix = px - padding
    iy = py - padding

    if ix >= 1 && ix <= ncols && iy >= 1 && iy <= nrows
        if burned[ix, iy] != zero(eltype(burned))
            vel_out[i] = zero(T)
            ecc_out[i] = zero(T)
        else
            fuel_id = fuel_ids[ix, iy]
            fi = fuel_id_to_index[fuel_id]
            mi = live_moisture_class - 29

            if fa_nonburnable[fi, mi] != zero(eltype(fa_nonburnable))
                vel_out[i] = zero(T)
                ecc_out[i] = zero(T)
            else
                # Unpack fuel scalars
                delta           = fuel_scalars[fi, mi, 1]
                rhob            = fuel_scalars[fi, mi, 2]
                xi              = fuel_scalars[fi, mi, 3]
                B               = fuel_scalars[fi, mi, 4]
                GP_WND_ETAS_HOC = fuel_scalars[fi, mi, 5]
                GP_WNL_ETAS_HOC = fuel_scalars[fi, mi, 6]
                phisterm        = fuel_scalars[fi, mi, 7]
                phiwterm        = fuel_scalars[fi, mi, 8]
                R_val           = fuel_scalars[fi, mi, 9]
                mex_dead        = fuel_scalars[fi, mi, 10]
                mex_live_base   = fuel_scalars[fi, mi, 11]
                F_dead          = fuel_scalars[fi, mi, 12]
                F_live          = fuel_scalars[fi, mi, 13]

                # --- Wind adjustment factor ---
                waf = if delta < T(0.1)
                    T(0.1)
                else
                    clamp(T(1.83) / log((T(20) + T(0.36) * delta) / (T(0.13) * delta)), T(0.1), one(T))
                end
                wsmf = ws20_ftpmin * waf

                # --- Slope factor ---
                slope_rad = slope[ix, iy] * (Elmfire.pi_val(T) / T(180))
                tanslp2 = tan(slope_rad)^2

                # --- Rothermel spread rate ---
                M = (M1, M10, M100, M1, MLH, MLW)

                # WPRIMENUMER is at components 13:16
                SUM_MPRIMENUMER = zero(T)
                for k in 1:4
                    SUM_MPRIMENUMER += fuel_components[fi, mi, 12 + k] * M[k]
                end
                mex_live = mex_live_base * (one(T) - R_val * SUM_MPRIMENUMER) - T(0.226)
                mex_live = max(mex_live, mex_dead)

                # FEPS is at components 7:12
                RHOBEPSQIG_DEAD = zero(T)
                RHOBEPSQIG_LIVE = zero(T)
                for k in 1:4
                    RHOBEPSQIG_DEAD += fuel_components[fi, mi, 6 + k] * (T(250) + T(1116) * M[k])
                end
                for k in 5:6
                    RHOBEPSQIG_LIVE += fuel_components[fi, mi, 6 + k] * (T(250) + T(1116) * M[k])
                end
                RHOBEPSQIG_DEAD *= rhob
                RHOBEPSQIG_LIVE *= rhob
                RHOBEPSQIG = F_dead * RHOBEPSQIG_DEAD + F_live * RHOBEPSQIG_LIVE

                # F is at components 1:6
                M_dead = zero(T)
                for k in 1:4
                    M_dead += fuel_components[fi, mi, k] * M[k]
                end
                momex_dead = clamp(M_dead / mex_dead, zero(T), one(T))
                etam_dead = clamp(one(T) - T(2.59) * momex_dead + T(5.11) * momex_dead^2 - T(3.52) * momex_dead^3, zero(T), one(T))
                IR_dead = GP_WND_ETAS_HOC * etam_dead

                M_live = zero(T)
                for k in 5:6
                    M_live += fuel_components[fi, mi, k] * M[k]
                end
                momex_live = clamp(M_live / mex_live, zero(T), one(T))
                etam_live = clamp(one(T) - T(2.59) * momex_live + T(5.11) * momex_live^2 - T(3.52) * momex_live^3, zero(T), one(T))
                IR_live = GP_WNL_ETAS_HOC * etam_live

                IR_btupft2min = IR_dead + IR_live

                WS_LIMIT = T(0.9) * IR_btupft2min
                wsmf_limited = min(wsmf, WS_LIMIT)
                phiw = phiwterm * wsmf_limited^B
                phis_max = phiwterm * WS_LIMIT^B
                phis = min(phisterm * tanslp2, phis_max)

                vs0 = if RHOBEPSQIG > T(1e-9)
                    spread_rate_adj * IR_btupft2min * xi / RHOBEPSQIG
                else
                    zero(T)
                end
                velocity = vs0 * (one(T) + phis + phiw)

                # --- Elliptical eccentricity ---
                effective_ws_mph = (ws20_ftpmin / T(88)) * waf / T(1.47)
                U = max(effective_ws_mph, zero(T))

                LB = if U < T(0.5)
                    one(T)
                else
                    clamp(T(0.936) * exp(T(0.2566) * U) + T(0.461) * exp(T(-0.1548) * U) - T(0.397), one(T), T(8))
                end

                eccentricity = if LB > T(1.001)
                    LB2 = LB * LB
                    sqrt((LB2 - one(T)) / (LB2 + one(T)))
                else
                    zero(T)
                end

                vel_out[i] = velocity
                ecc_out[i] = eccentricity
            end
        end
    else
        vel_out[i] = zero(T)
        ecc_out[i] = zero(T)
    end
end


#-----------------------------------------------------------------------------#
#                     GPU Direction Kernel (index-list)
#-----------------------------------------------------------------------------#
# Kernel 2: phi gradient + wind → ux, uy

@kernel function direction_kernel!(
    ux, uy, @Const(phi), @Const(active_px), @Const(active_py),
    @Const(vel_in), @Const(ecc_in),
    wind_dir_rad, cellsize
)
    T = eltype(ux)
    i = @index(Global, Linear)
    px = active_px[i]
    py = active_py[i]

    velocity = vel_in[i]
    eccentricity = ecc_in[i]

    if velocity <= zero(T)
        ux[px, py] = zero(T)
        uy[px, py] = zero(T)
    else
        # --- Normal to fire front ---
        rdx2 = T(0.5) / cellsize
        dphidx = (phi[px + 1, py] - phi[px - 1, py]) * rdx2
        dphidy = (phi[px, py + 1] - phi[px, py - 1]) * rdx2
        mag = sqrt(dphidx^2 + dphidy^2)

        normal_x = zero(T)
        normal_y = zero(T)
        if mag > T(1e-10)
            normal_x = dphidx / mag
            normal_y = dphidy / mag
        end

        head = velocity
        back = head * (one(T) - eccentricity) / (one(T) + eccentricity)

        # --- Velocity components ---
        wind_to_x = -sin(wind_dir_rad)
        wind_to_y = -cos(wind_dir_rad)
        cos_theta = normal_x * wind_to_x + normal_y * wind_to_y
        vel = T(0.5) * ((one(T) + cos_theta) * head + (one(T) - cos_theta) * back)

        ux[px, py] = vel * normal_x
        uy[px, py] = vel * normal_y
    end
end


#-----------------------------------------------------------------------------#
#                     GPU CFL Gather Kernel (index-list)
#-----------------------------------------------------------------------------#
# Gathers per-cell max(|ux|, |uy|) into a 1D buffer.
# The actual max reduction is done on CPU (avoids atomic max issues on Metal).

@kernel function cfl_gather_kernel!(
    u_out, @Const(ux), @Const(uy), @Const(active_px), @Const(active_py)
)
    T = eltype(ux)
    i = @index(Global, Linear)
    px = active_px[i]
    py = active_py[i]
    u_out[i] = max(abs(ux[px, py]), abs(uy[px, py]))
end


#-----------------------------------------------------------------------------#
#                     GPU RK2 Kernels (index-list)
#-----------------------------------------------------------------------------#

@kernel function rk2_stage1_kernel!(
    phi, @Const(phi_old), @Const(ux), @Const(uy),
    @Const(active_px), @Const(active_py),
    dt, rcellsize
)
    T = eltype(phi)
    i = @index(Global, Linear)
    px = active_px[i]
    py = active_py[i]

    ux_val = ux[px, py]
    uy_val = uy[px, py]

    EPSILON = T(1e-30)
    CEILING = T(1e3)

    # X-direction gradient
    dphidx = zero(T)
    if ux_val >= zero(T)
        deltaup = phi[px, py] - phi[px - 1, py]
        deltaloc = phi[px + 1, py] - phi[px, py]
        phieast = phi[px, py]
        if abs(deltaloc) > EPSILON
            phieast = phi[px, py] + _half_superbee(deltaup / deltaloc) * deltaloc
        end
        deltaloc_west = -deltaup
        phiwest = phi[px - 1, py]
        if abs(deltaloc_west) > EPSILON
            deltaup_west = phi[px - 2, py] - phi[px - 1, py]
            phiwest = phi[px - 1, py] - _half_superbee(deltaup_west / deltaloc_west) * deltaloc_west
        end
        dphidx = (phieast - phiwest) * rcellsize
    else
        deltaloc = phi[px + 1, py] - phi[px, py]
        phieast = phi[px + 1, py]
        if abs(deltaloc) > EPSILON
            deltaup = phi[px + 2, py] - phi[px + 1, py]
            phieast = phi[px + 1, py] - _half_superbee(deltaup / deltaloc) * deltaloc
        end
        deltaup_west = -deltaloc
        deltaloc_west = phi[px - 1, py] - phi[px, py]
        phiwest = phi[px, py]
        if abs(deltaloc_west) > EPSILON
            phiwest = phi[px, py] + _half_superbee(deltaup_west / deltaloc_west) * deltaloc_west
        end
        dphidx = (phieast - phiwest) * rcellsize
    end

    # Y-direction gradient
    dphidy = zero(T)
    if uy_val > zero(T)
        deltaup = phi[px, py] - phi[px, py - 1]
        deltaloc = phi[px, py + 1] - phi[px, py]
        phinorth = phi[px, py]
        if abs(deltaloc) > EPSILON
            phinorth = phi[px, py] + _half_superbee(deltaup / deltaloc) * deltaloc
        end
        deltaloc_south = -deltaup
        phisouth = phi[px, py - 1]
        if abs(deltaloc_south) > EPSILON
            deltaup_south = phi[px, py - 2] - phi[px, py - 1]
            phisouth = phi[px, py - 1] - _half_superbee(deltaup_south / deltaloc_south) * deltaloc_south
        end
        dphidy = (phinorth - phisouth) * rcellsize
    else
        deltaloc = phi[px, py + 1] - phi[px, py]
        phinorth = phi[px, py + 1]
        if abs(deltaloc) > EPSILON
            deltaup = phi[px, py + 2] - phi[px, py + 1]
            phinorth = phi[px, py + 1] - _half_superbee(deltaup / deltaloc) * deltaloc
        end
        deltaup_south = -deltaloc
        deltaloc_south = phi[px, py - 1] - phi[px, py]
        phisouth = phi[px, py]
        if abs(deltaloc_south) > EPSILON
            phisouth = phi[px, py] + _half_superbee(deltaup_south / deltaloc_south) * deltaloc_south
        end
        dphidy = (phinorth - phisouth) * rcellsize
    end

    dphidx = clamp(dphidx, -CEILING, CEILING)
    dphidy = clamp(dphidy, -CEILING, CEILING)
    if isnan(dphidx); dphidx = zero(T); end
    if isnan(dphidy); dphidy = zero(T); end

    phi_new = phi_old[px, py] - dt * (ux_val * dphidx + uy_val * dphidy)
    if isnan(phi_new)
        phi_new = one(T)
    end
    phi[px, py] = clamp(phi_new, T(-100), T(100))
end


@kernel function rk2_stage2_kernel!(
    phi, @Const(phi_old), @Const(ux), @Const(uy),
    @Const(active_px), @Const(active_py),
    dt, rcellsize
)
    T = eltype(phi)
    i = @index(Global, Linear)
    px = active_px[i]
    py = active_py[i]

    ux_val = ux[px, py]
    uy_val = uy[px, py]

    EPSILON = T(1e-30)
    CEILING = T(1e3)

    # X-direction gradient
    dphidx = zero(T)
    if ux_val >= zero(T)
        deltaup = phi[px, py] - phi[px - 1, py]
        deltaloc = phi[px + 1, py] - phi[px, py]
        phieast = phi[px, py]
        if abs(deltaloc) > EPSILON
            phieast = phi[px, py] + _half_superbee(deltaup / deltaloc) * deltaloc
        end
        deltaloc_west = -deltaup
        phiwest = phi[px - 1, py]
        if abs(deltaloc_west) > EPSILON
            deltaup_west = phi[px - 2, py] - phi[px - 1, py]
            phiwest = phi[px - 1, py] - _half_superbee(deltaup_west / deltaloc_west) * deltaloc_west
        end
        dphidx = (phieast - phiwest) * rcellsize
    else
        deltaloc = phi[px + 1, py] - phi[px, py]
        phieast = phi[px + 1, py]
        if abs(deltaloc) > EPSILON
            deltaup = phi[px + 2, py] - phi[px + 1, py]
            phieast = phi[px + 1, py] - _half_superbee(deltaup / deltaloc) * deltaloc
        end
        deltaup_west = -deltaloc
        deltaloc_west = phi[px - 1, py] - phi[px, py]
        phiwest = phi[px, py]
        if abs(deltaloc_west) > EPSILON
            phiwest = phi[px, py] + _half_superbee(deltaup_west / deltaloc_west) * deltaloc_west
        end
        dphidx = (phieast - phiwest) * rcellsize
    end

    # Y-direction gradient
    dphidy = zero(T)
    if uy_val > zero(T)
        deltaup = phi[px, py] - phi[px, py - 1]
        deltaloc = phi[px, py + 1] - phi[px, py]
        phinorth = phi[px, py]
        if abs(deltaloc) > EPSILON
            phinorth = phi[px, py] + _half_superbee(deltaup / deltaloc) * deltaloc
        end
        deltaloc_south = -deltaup
        phisouth = phi[px, py - 1]
        if abs(deltaloc_south) > EPSILON
            deltaup_south = phi[px, py - 2] - phi[px, py - 1]
            phisouth = phi[px, py - 1] - _half_superbee(deltaup_south / deltaloc_south) * deltaloc_south
        end
        dphidy = (phinorth - phisouth) * rcellsize
    else
        deltaloc = phi[px, py + 1] - phi[px, py]
        phinorth = phi[px, py + 1]
        if abs(deltaloc) > EPSILON
            deltaup = phi[px, py + 2] - phi[px, py + 1]
            phinorth = phi[px, py + 1] - _half_superbee(deltaup / deltaloc) * deltaloc
        end
        deltaup_south = -deltaloc
        deltaloc_south = phi[px, py - 1] - phi[px, py]
        phisouth = phi[px, py]
        if abs(deltaloc_south) > EPSILON
            phisouth = phi[px, py] + _half_superbee(deltaup_south / deltaloc_south) * deltaloc_south
        end
        dphidy = (phinorth - phisouth) * rcellsize
    end

    dphidx = clamp(dphidx, -CEILING, CEILING)
    dphidy = clamp(dphidy, -CEILING, CEILING)
    if isnan(dphidx); dphidx = zero(T); end
    if isnan(dphidy); dphidy = zero(T); end

    phi_rhs = phi[px, py] - dt * (ux_val * dphidx + uy_val * dphidy)
    phi[px, py] = T(0.5) * (phi_old[px, py] + phi_rhs)
end


#-----------------------------------------------------------------------------#
#                     simulate_gpu! Implementation
#-----------------------------------------------------------------------------#

function Elmfire.simulate_gpu!(
    state::Elmfire.FireState{T},
    fuel_ids::AbstractMatrix{Int},
    fuel_array::Elmfire.FuelModelArray{T},
    weather::Elmfire.ConstantWeather{T},
    slope::AbstractMatrix{T},
    aspect::AbstractMatrix{T},
    t_start::T,
    t_stop::T;
    dt_initial::T = one(T),
    target_cfl::T = T(0.9),
    dt_max::T = T(10),
    spread_rate_adj::T = one(T),
    callback::Union{Nothing, Function} = nothing,
    backend::KernelAbstractions.Backend = KernelAbstractions.CPU()
) where {T<:AbstractFloat}
    t = t_start
    dt = dt_initial
    iteration = 0

    # Pre-compute weather scalars
    wind_dir_rad = weather.wind_direction * Elmfire.pio180(T)
    ws20_ftpmin = weather.wind_speed_20ft * T(88)
    live_moisture_class = Int32(clamp(round(Int, T(100) * weather.MLH), 30, 120))

    # Grid dimensions
    nx_pad = state.ncols + 2 * state.padding
    ny_pad = state.nrows + 2 * state.padding

    # Allocate persistent device arrays
    d_ucfl = KernelAbstractions.allocate(backend, T, nx_pad * ny_pad)  # reused for CFL gather
    d_phi = KernelAbstractions.allocate(backend, T, nx_pad, ny_pad)
    d_phi_old = KernelAbstractions.allocate(backend, T, nx_pad, ny_pad)
    d_ux = KernelAbstractions.zeros(backend, T, nx_pad, ny_pad)
    d_uy = KernelAbstractions.zeros(backend, T, nx_pad, ny_pad)

    # Upload static data (once, before the loop)
    # Use Int32 for integer arrays — Metal doesn't support Int64
    d_fuel_ids = Adapt.adapt(backend, Int32.(fuel_ids))
    d_slope = Adapt.adapt(backend, slope)
    d_aspect = Adapt.adapt(backend, aspect)

    d_fuel_id_to_index = Adapt.adapt(backend, Int32.(fuel_array.fuel_id_to_index))

    # Pack fuel model data into two 3D arrays for reduced kernel argument count
    # fuel_scalars[fi, mi, 1:14]: delta, rhob, xi, B, GP_WND, GP_WNL,
    #     phisterm, phiwterm, R_val, mex_dead, mex_live, F_dead, F_live, tr
    fa = fuel_array
    nf = size(fa.delta, 1)
    nm = size(fa.delta, 2)
    h_fuel_scalars = Array{T, 3}(undef, nf, nm, 14)
    h_fuel_scalars[:, :, 1]  .= fa.delta
    h_fuel_scalars[:, :, 2]  .= fa.rhob
    h_fuel_scalars[:, :, 3]  .= fa.xi
    h_fuel_scalars[:, :, 4]  .= fa.B
    h_fuel_scalars[:, :, 5]  .= fa.GP_WND_ETAS_HOC
    h_fuel_scalars[:, :, 6]  .= fa.GP_WNL_ETAS_HOC
    h_fuel_scalars[:, :, 7]  .= fa.phisterm
    h_fuel_scalars[:, :, 8]  .= fa.phiwterm
    h_fuel_scalars[:, :, 9]  .= fa.R_MPRIMEDENOME14SUM_MEX_DEAD
    h_fuel_scalars[:, :, 10] .= fa.mex_dead
    h_fuel_scalars[:, :, 11] .= fa.mex_live
    h_fuel_scalars[:, :, 12] .= fa.F_dead
    h_fuel_scalars[:, :, 13] .= fa.F_live
    h_fuel_scalars[:, :, 14] .= fa.tr
    d_fuel_scalars = Adapt.adapt(backend, h_fuel_scalars)

    # fuel_components[fi, mi, 1:16]: F[1:6], FEPS[1:6], WPRIMENUMER[1:4]
    h_fuel_components = Array{T, 3}(undef, nf, nm, 16)
    h_fuel_components[:, :, 1:6]   .= fa.F
    h_fuel_components[:, :, 7:12]  .= fa.FEPS
    h_fuel_components[:, :, 13:16] .= fa.WPRIMENUMER
    d_fuel_components = Adapt.adapt(backend, h_fuel_components)

    # BitMatrix can't go to GPU, convert to UInt8
    h_fa_nonburnable = Matrix{UInt8}(fa.nonburnable)
    d_fa_nonburnable = Adapt.adapt(backend, h_fa_nonburnable)

    # Upload initial phi (once — d_phi stays on device for the entire simulation)
    copyto!(d_phi, state.phi)
    copyto!(d_phi_old, state.phi_old)

    # Burned array: BitMatrix can't go to GPU, so use UInt8 device copy
    h_burned = zeros(UInt8, state.ncols, state.nrows)
    h_burned .= state.burned
    d_burned = KernelAbstractions.allocate(backend, UInt8, state.ncols, state.nrows)
    copyto!(d_burned, h_burned)

    rcellsize = one(T) / state.cellsize

    # Pre-allocate reusable buffers for index lists and gather (avoids per-iteration allocs)
    # Host buffers stay Int for CPU-side indexing; device buffers use Int32 for Metal compat
    max_active = nx_pad * ny_pad
    h_px = Vector{Int}(undef, max_active)
    h_py = Vector{Int}(undef, max_active)
    h_px32 = Vector{Int32}(undef, max_active)
    h_py32 = Vector{Int32}(undef, max_active)
    d_px = KernelAbstractions.allocate(backend, Int32, max_active)
    d_py = KernelAbstractions.allocate(backend, Int32, max_active)
    d_phi_active = KernelAbstractions.allocate(backend, T, max_active)
    d_vel = KernelAbstractions.allocate(backend, T, max_active)
    d_ecc = KernelAbstractions.allocate(backend, T, max_active)
    h_phi_active = Vector{T}(undef, max_active)
    h_active = Vector{CartesianIndex{2}}(undef, max_active)
    h_cells_to_tag = Vector{CartesianIndex{2}}(undef, max_active)

    while t < t_stop
        iteration += 1

        # Fill active cell list from Set directly (avoids collect allocation)
        n_active = 0
        for idx in state.narrow_band.active
            n_active += 1
            h_active[n_active] = idx
            h_px[n_active] = idx[1]
            h_py[n_active] = idx[2]
        end
        if n_active == 0
            break
        end
        for i in 1:n_active
            h_px32[i] = Int32(h_px[i])
            h_py32[i] = Int32(h_py[i])
        end
        copyto!(d_px, 1, h_px32, 1, n_active)
        copyto!(d_py, 1, h_py32, 1, n_active)

        # Re-upload burned state (modified by CPU burn detection loop)
        h_burned .= state.burned
        copyto!(d_burned, h_burned)

        # Spread rate kernel — Rothermel physics → velocity + eccentricity
        spread_rate_kernel!(backend)(
            d_vel, d_ecc, d_px, d_py, d_burned,
            d_fuel_ids, d_fuel_id_to_index,
            d_fuel_scalars, d_fa_nonburnable, d_fuel_components,
            d_slope,
            ws20_ftpmin,
            weather.M1, weather.M10, weather.M100,
            weather.MLH, weather.MLW,
            live_moisture_class, spread_rate_adj,
            Int32(state.padding), Int32(state.ncols), Int32(state.nrows);
            ndrange = n_active
        )
        KernelAbstractions.synchronize(backend)

        # Direction kernel — phi gradient + wind → ux, uy
        direction_kernel!(backend)(
            d_ux, d_uy, d_phi, d_px, d_py,
            d_vel, d_ecc,
            wind_dir_rad, state.cellsize;
            ndrange = n_active
        )
        KernelAbstractions.synchronize(backend)

        # CFL timestep — launched over n_active cells only
        if iteration > 5
            # Gather per-cell max speed (avoids atomic max which fails on Metal)
            cfl_gather_kernel!(backend)(
                d_ucfl, d_ux, d_uy, d_px, d_py;
                ndrange = n_active
            )
            KernelAbstractions.synchronize(backend)

            copyto!(h_phi_active, 1, d_ucfl, 1, n_active)
            umax = zero(T)
            for k in 1:n_active
                umax = max(umax, h_phi_active[k])
            end
            if umax > T(1e-9)
                cfl = umax * dt / state.cellsize
                dt = min(target_cfl * dt / cfl, dt_max)
            else
                dt = dt_max
            end
        end

        if t + dt > t_stop
            dt = t_stop - t
        end

        # RK2 level set integration — launched over n_active cells only
        # Copy only active cells (RK2 kernels only read phi_old at active positions)
        copy_active_kernel!(backend)(
            d_phi_old, d_phi, d_px, d_py;
            ndrange = n_active
        )
        KernelAbstractions.synchronize(backend)

        rk2_stage1_kernel!(backend)(
            d_phi, d_phi_old, d_ux, d_uy, d_px, d_py, dt, rcellsize;
            ndrange = n_active
        )
        KernelAbstractions.synchronize(backend)

        rk2_stage2_kernel!(backend)(
            d_phi, d_phi_old, d_ux, d_uy, d_px, d_py, dt, rcellsize;
            ndrange = n_active
        )
        KernelAbstractions.synchronize(backend)

        # Gather only active phi values for burn detection (reuse pre-allocated buffers)
        gather_phi_kernel!(backend)(
            d_phi_active, d_phi, d_px, d_py;
            ndrange = n_active
        )
        KernelAbstractions.synchronize(backend)
        copyto!(h_phi_active, 1, d_phi_active, 1, n_active)

        # Update burned cells and narrow band on CPU
        n_to_tag = 0

        for i in 1:n_active
            px, py = h_px[i], h_py[i]
            ix, iy = Elmfire.padded_to_grid(state, px, py)

            if ix < 1 || ix > state.ncols || iy < 1 || iy > state.nrows
                continue
            end

            if h_phi_active[i] <= zero(T) && !state.burned[ix, iy]
                state.burned[ix, iy] = true
                state.time_of_arrival[ix, iy] = t + dt

                fuel_id = fuel_ids[ix, iy]
                fi = fuel_array.fuel_id_to_index[fuel_id]
                mi = live_moisture_class - 29
                waf = Elmfire.wind_adjustment_factor(fuel_array.delta[fi, mi])
                wsmf = ws20_ftpmin * waf
                tanslp2 = Elmfire.calculate_tanslp2(slope[ix, iy])

                velocity, vs0, flin = Elmfire.surface_spread_rate_flat(
                    fuel_array, fi, mi,
                    weather.M1, weather.M10, weather.M100,
                    weather.MLH, weather.MLW,
                    wsmf, tanslp2, spread_rate_adj
                )

                state.spread_rate[ix, iy] = velocity
                state.fireline_intensity[ix, iy] = flin
                if flin > zero(T)
                    state.flame_length[ix, iy] = (T(0.0775) / Elmfire.ft_to_m(T)) * flin^T(0.46)
                end

                n_to_tag += 1
                h_cells_to_tag[n_to_tag] = h_active[i]
            end
        end

        for i in 1:n_to_tag
            Elmfire.tag_band!(state.narrow_band, h_cells_to_tag[i], nx_pad, ny_pad, state.padding)
        end
        Elmfire.untag_isolated!(state.narrow_band, state.phi, state.burned, state.padding)

        t += dt

        if callback !== nothing
            callback(state, t, dt, iteration)
        end
    end

    # Final download — sync device state back to CPU
    copyto!(state.phi, d_phi)
    copyto!(state.ux, d_ux)
    copyto!(state.uy, d_uy)

    nothing
end


#-----------------------------------------------------------------------------#
#                     simulate_gpu_uniform! Implementation
#-----------------------------------------------------------------------------#

function Elmfire.simulate_gpu_uniform!(
    state::Elmfire.FireState{T},
    fuel_id::Int,
    fuel_array::Elmfire.FuelModelArray{T},
    weather::Elmfire.ConstantWeather{T},
    slope_deg::T,
    aspect_deg::T,
    t_start::T,
    t_stop::T;
    kwargs...
) where {T<:AbstractFloat}
    fuel_ids = fill(fuel_id, state.ncols, state.nrows)
    slope = fill(slope_deg, state.ncols, state.nrows)
    aspect = fill(aspect_deg, state.ncols, state.nrows)

    Elmfire.simulate_gpu!(state, fuel_ids, fuel_array, weather, slope, aspect,
                          t_start, t_stop; kwargs...)
end


end # module ElmfireKAExt
