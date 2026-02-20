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
#                     GPU Velocity Kernel
#-----------------------------------------------------------------------------#

@kernel function velocity_kernel!(
    ux, uy, @Const(phi), @Const(mask), @Const(burned),
    @Const(fuel_ids), @Const(fuel_id_to_index),
    @Const(fa_delta), @Const(fa_nonburnable),
    @Const(fa_rhob), @Const(fa_xi), @Const(fa_B),
    @Const(fa_GP_WND_ETAS_HOC), @Const(fa_GP_WNL_ETAS_HOC),
    @Const(fa_phisterm), @Const(fa_phiwterm),
    @Const(fa_R_MPRIMEDENOME14SUM_MEX_DEAD),
    @Const(fa_mex_dead), @Const(fa_mex_live),
    @Const(fa_F_dead), @Const(fa_F_live), @Const(fa_tr),
    @Const(fa_F), @Const(fa_FEPS), @Const(fa_WPRIMENUMER),
    @Const(slope), @Const(aspect),
    ws20_ftpmin, wind_dir_rad, M1, M10, M100, MLH, MLW,
    live_moisture_class, spread_rate_adj,
    cellsize, padding, ncols, nrows
)
    T = eltype(ux)
    px, py = @index(Global, NTuple)

    ix = px - padding
    iy = py - padding
    in_bounds = (ix >= 1) && (ix <= ncols) && (iy >= 1) && (iy <= nrows)

    if mask[px, py] && in_bounds
        if burned[ix, iy]
            # Burned cell: zero velocity
            ux[px, py] = zero(T)
            uy[px, py] = zero(T)
        else
            fuel_id = fuel_ids[ix, iy]
            fi = fuel_id_to_index[fuel_id]
            mi = live_moisture_class - 29

            if fa_nonburnable[fi, mi]
                # Nonburnable: zero velocity
                ux[px, py] = zero(T)
                uy[px, py] = zero(T)
            else
                # --- Wind adjustment factor ---
                delta = fa_delta[fi, mi]
                waf = if delta < T(0.1)
                    T(0.1)
                else
                    clamp(T(1.83) / log((T(20) + T(0.36) * delta) / (T(0.13) * delta)), T(0.1), one(T))
                end
                wsmf = ws20_ftpmin * waf

                # --- Slope factor ---
                slope_rad = slope[ix, iy] * (Elmfire.pi_val(T) / T(180))
                tanslp2 = tan(slope_rad)^2

                # --- Rothermel spread rate (inlined) ---
                rhob = fa_rhob[fi, mi]
                xi = fa_xi[fi, mi]
                B = fa_B[fi, mi]
                GP_WND_ETAS_HOC = fa_GP_WND_ETAS_HOC[fi, mi]
                GP_WNL_ETAS_HOC = fa_GP_WNL_ETAS_HOC[fi, mi]
                phisterm = fa_phisterm[fi, mi]
                phiwterm = fa_phiwterm[fi, mi]
                R_val = fa_R_MPRIMEDENOME14SUM_MEX_DEAD[fi, mi]
                F_dead = fa_F_dead[fi, mi]
                F_live = fa_F_live[fi, mi]
                mex_dead = fa_mex_dead[fi, mi]
                mex_live_base = fa_mex_live[fi, mi]
                tr = fa_tr[fi, mi]

                M = (M1, M10, M100, M1, MLH, MLW)

                SUM_MPRIMENUMER = zero(T)
                for k in 1:4
                    SUM_MPRIMENUMER += fa_WPRIMENUMER[fi, mi, k] * M[k]
                end
                mex_live = mex_live_base * (one(T) - R_val * SUM_MPRIMENUMER) - T(0.226)
                mex_live = max(mex_live, mex_dead)

                RHOBEPSQIG_DEAD = zero(T)
                RHOBEPSQIG_LIVE = zero(T)
                for k in 1:4
                    RHOBEPSQIG_DEAD += fa_FEPS[fi, mi, k] * (T(250) + T(1116) * M[k])
                end
                for k in 5:6
                    RHOBEPSQIG_LIVE += fa_FEPS[fi, mi, k] * (T(250) + T(1116) * M[k])
                end
                RHOBEPSQIG_DEAD *= rhob
                RHOBEPSQIG_LIVE *= rhob
                RHOBEPSQIG = F_dead * RHOBEPSQIG_DEAD + F_live * RHOBEPSQIG_LIVE

                M_dead = zero(T)
                for k in 1:4
                    M_dead += fa_F[fi, mi, k] * M[k]
                end
                momex_dead = clamp(M_dead / mex_dead, zero(T), one(T))
                etam_dead = clamp(one(T) - T(2.59) * momex_dead + T(5.11) * momex_dead^2 - T(3.52) * momex_dead^3, zero(T), one(T))
                IR_dead = GP_WND_ETAS_HOC * etam_dead

                M_live = zero(T)
                for k in 5:6
                    M_live += fa_F[fi, mi, k] * M[k]
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

                # --- Elliptical spread ---
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
    end
end


#-----------------------------------------------------------------------------#
#                     GPU CFL Kernel
#-----------------------------------------------------------------------------#

@kernel function cfl_max_kernel!(umax_out, @Const(ux), @Const(uy), @Const(mask))
    T = eltype(ux)
    px, py = @index(Global, NTuple)

    if mask[px, py]
        u = max(abs(ux[px, py]), abs(uy[px, py]))
        KernelAbstractions.@atomic umax_out[1] = max(umax_out[1], u)
    end
end


#-----------------------------------------------------------------------------#
#                     GPU RK2 Kernels
#-----------------------------------------------------------------------------#

@kernel function rk2_stage1_kernel!(
    phi, @Const(phi_old), @Const(ux), @Const(uy), @Const(mask),
    dt, rcellsize
)
    T = eltype(phi)
    px, py = @index(Global, NTuple)

    if mask[px, py]
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
end


@kernel function rk2_stage2_kernel!(
    phi, @Const(phi_old), @Const(ux), @Const(uy), @Const(mask),
    dt, rcellsize
)
    T = eltype(phi)
    px, py = @index(Global, NTuple)

    if mask[px, py]
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
    live_moisture_class = clamp(round(Int, T(100) * weather.MLH), 30, 120)

    # Grid dimensions
    nx_pad = state.ncols + 2 * state.padding
    ny_pad = state.nrows + 2 * state.padding

    # Allocate backend arrays
    d_mask = KernelAbstractions.zeros(backend, Bool, nx_pad, ny_pad)
    d_umax = KernelAbstractions.zeros(backend, T, 1)

    d_phi = KernelAbstractions.allocate(backend, T, nx_pad, ny_pad)
    d_phi_old = KernelAbstractions.allocate(backend, T, nx_pad, ny_pad)
    d_ux = KernelAbstractions.zeros(backend, T, nx_pad, ny_pad)
    d_uy = KernelAbstractions.zeros(backend, T, nx_pad, ny_pad)
    d_fuel_ids = Adapt.adapt(backend, fuel_ids)
    d_slope = Adapt.adapt(backend, slope)
    d_aspect = Adapt.adapt(backend, aspect)

    # Upload fuel model array fields
    d_fuel_id_to_index = Adapt.adapt(backend, fuel_array.fuel_id_to_index)
    d_fa_delta = Adapt.adapt(backend, fuel_array.delta)
    d_fa_nonburnable = Adapt.adapt(backend, fuel_array.nonburnable)
    d_fa_rhob = Adapt.adapt(backend, fuel_array.rhob)
    d_fa_xi = Adapt.adapt(backend, fuel_array.xi)
    d_fa_B = Adapt.adapt(backend, fuel_array.B)
    d_fa_GP_WND = Adapt.adapt(backend, fuel_array.GP_WND_ETAS_HOC)
    d_fa_GP_WNL = Adapt.adapt(backend, fuel_array.GP_WNL_ETAS_HOC)
    d_fa_phisterm = Adapt.adapt(backend, fuel_array.phisterm)
    d_fa_phiwterm = Adapt.adapt(backend, fuel_array.phiwterm)
    d_fa_R = Adapt.adapt(backend, fuel_array.R_MPRIMEDENOME14SUM_MEX_DEAD)
    d_fa_mex_dead = Adapt.adapt(backend, fuel_array.mex_dead)
    d_fa_mex_live = Adapt.adapt(backend, fuel_array.mex_live)
    d_fa_F_dead = Adapt.adapt(backend, fuel_array.F_dead)
    d_fa_F_live = Adapt.adapt(backend, fuel_array.F_live)
    d_fa_tr = Adapt.adapt(backend, fuel_array.tr)
    d_fa_F = Adapt.adapt(backend, fuel_array.F)
    d_fa_FEPS = Adapt.adapt(backend, fuel_array.FEPS)
    d_fa_WPRIMENUMER = Adapt.adapt(backend, fuel_array.WPRIMENUMER)

    # Upload initial phi
    copyto!(d_phi, state.phi)
    copyto!(d_phi_old, state.phi_old)

    ndrange = (nx_pad, ny_pad)

    while t < t_stop
        iteration += 1

        active_cells = Elmfire.get_active_cells(state.narrow_band)
        if isempty(active_cells)
            break
        end

        # Build and upload active mask
        h_mask = Elmfire.active_mask(state.narrow_band, nx_pad, ny_pad)
        copyto!(d_mask, h_mask)
        copyto!(d_phi, state.phi)

        # Velocity kernel
        velocity_kernel!(backend)(
            d_ux, d_uy, d_phi, d_mask, state.burned,
            d_fuel_ids, d_fuel_id_to_index,
            d_fa_delta, d_fa_nonburnable,
            d_fa_rhob, d_fa_xi, d_fa_B,
            d_fa_GP_WND, d_fa_GP_WNL,
            d_fa_phisterm, d_fa_phiwterm,
            d_fa_R, d_fa_mex_dead, d_fa_mex_live,
            d_fa_F_dead, d_fa_F_live, d_fa_tr,
            d_fa_F, d_fa_FEPS, d_fa_WPRIMENUMER,
            d_slope, d_aspect,
            ws20_ftpmin, wind_dir_rad,
            weather.M1, weather.M10, weather.M100,
            weather.MLH, weather.MLW,
            live_moisture_class, spread_rate_adj,
            state.cellsize, state.padding, state.ncols, state.nrows;
            ndrange = ndrange
        )
        KernelAbstractions.synchronize(backend)

        # CFL timestep
        if iteration > 5
            fill!(d_umax, zero(T))
            cfl_max_kernel!(backend)(d_umax, d_ux, d_uy, d_mask; ndrange = ndrange)
            KernelAbstractions.synchronize(backend)

            h_umax = Array(d_umax)
            umax = h_umax[1]
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

        # RK2 level set integration
        rcellsize = one(T) / state.cellsize
        copyto!(d_phi_old, d_phi)

        rk2_stage1_kernel!(backend)(
            d_phi, d_phi_old, d_ux, d_uy, d_mask, dt, rcellsize;
            ndrange = ndrange
        )
        KernelAbstractions.synchronize(backend)

        rk2_stage2_kernel!(backend)(
            d_phi, d_phi_old, d_ux, d_uy, d_mask, dt, rcellsize;
            ndrange = ndrange
        )
        KernelAbstractions.synchronize(backend)

        # Download results
        copyto!(state.phi, d_phi)
        copyto!(state.ux, d_ux)
        copyto!(state.uy, d_uy)

        # Update burned cells and narrow band on CPU
        cells_to_tag = CartesianIndex{2}[]

        for idx in active_cells
            px, py = idx[1], idx[2]
            ix, iy = Elmfire.padded_to_grid(state, px, py)

            if ix < 1 || ix > state.ncols || iy < 1 || iy > state.nrows
                continue
            end

            if state.phi[px, py] <= zero(T) && !state.burned[ix, iy]
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

                push!(cells_to_tag, idx)
            end
        end

        for idx in cells_to_tag
            Elmfire.tag_band!(state.narrow_band, idx, nx_pad, ny_pad, state.padding)
        end
        Elmfire.untag_isolated!(state.narrow_band, state.phi, state.burned, state.padding)

        t += dt

        if callback !== nothing
            callback(state, t, dt, iteration)
        end
    end

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
