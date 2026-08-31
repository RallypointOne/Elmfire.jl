#-----------------------------------------------------------------------------#
#                     Rothermel Surface Fire Spread Model
#-----------------------------------------------------------------------------#

"""
    SpreadResult{T<:AbstractFloat}

Results from the Rothermel surface fire spread rate calculation.
"""
struct SpreadResult{T<:AbstractFloat}
    velocity::T       # Spread rate (ft/min)
    vs0::T            # Base spread rate without wind/slope (ft/min)
    ir::T             # Reaction intensity (kW/m²)
    hpua::T           # Heat per unit area (kJ/m²)
    flin::T           # Fireline intensity (kW/m)
    phiw::T           # Wind factor
    phis::T           # Slope factor
end

Base.eltype(::SpreadResult{T}) where {T} = T


"""
    moisture_damping(r::T) -> T

Calculate the moisture damping coefficient η_m.
Formula: η_m = 1 - 2.59r + 5.11r² - 3.52r³

Where r = M/M_ex (moisture ratio to extinction moisture).

Implementation follows Rothermel (1972) equation.
"""
@inline function moisture_damping(r::T) where {T<:AbstractFloat}
    r = clamp(r, zero(T), one(T))
    r2 = r * r
    r3 = r2 * r
    eta = one(T) - T(2.59)*r + T(5.11)*r2 - T(3.52)*r3
    return clamp(eta, zero(T), one(T))
end


"""
    surface_spread_rate(
        fm::FuelModel{T},
        M1::T, M10::T, M100::T,
        MLH::T, MLW::T,
        wsmf::T,
        tanslp2::T;
        adj::T = one(T)
    ) -> SpreadResult{T}

Calculate surface fire spread rate using the Rothermel (1972) model.

# Arguments
- `fm`: Fuel model with pre-computed coefficients
- `M1`: 1-hour dead fuel moisture (fraction, e.g., 0.06 for 6%)
- `M10`: 10-hour dead fuel moisture (fraction)
- `M100`: 100-hour dead fuel moisture (fraction)
- `MLH`: Live herbaceous moisture (fraction)
- `MLW`: Live woody moisture (fraction)
- `wsmf`: Mid-flame wind speed (ft/min)
- `tanslp2`: tan²(slope angle)
- `adj`: Adjustment factor (default 1.0)

# Returns
- `SpreadResult` with velocity, reaction intensity, heat per unit area, and fireline intensity

Implementation follows `elmfire_spread_rate.f90:13-132`.
"""
function surface_spread_rate(
    fm::FuelModel{T},
    M1::T, M10::T, M100::T,
    MLH::T, MLW::T,
    wsmf::T,
    tanslp2::T;
    adj::T = one(T)
) where {T<:AbstractFloat}
    # Check for non-burnable
    if isnonburnable(fm)
        return SpreadResult{T}(zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))
    end

    # Moisture array: M1, M10, M100, M_dead_dyn (=M1), MLH, MLW
    M = (M1, M10, M100, M1, MLH, MLW)

    # Calculate live fuel moisture of extinction (dynamic)
    # M_prime numerator for dead classes
    MPRIMENUMER = ntuple(i -> fm.WPRIMENUMER[i] * M[i], 4)
    SUM_MPRIMENUMER = sum(MPRIMENUMER)

    mex_live = fm.mex_live * (one(T) - fm.R_MPRIMEDENOME14SUM_MEX_DEAD * SUM_MPRIMENUMER) - T(0.226)
    mex_live = max(mex_live, fm.mex_dead)

    # Heat of pre-ignition: Qig = 250 + 1116*M
    QIG = ntuple(i -> T(250) + T(1116) * M[i], 6)

    # f * ε * Qig
    FEPSQIG = ntuple(i -> fm.FEPS[i] * QIG[i], 6)

    # ρb * Σ(f * ε * Qig) for dead and live
    RHOBEPSQIG_DEAD = fm.rhob * (FEPSQIG[1] + FEPSQIG[2] + FEPSQIG[3] + FEPSQIG[4])
    RHOBEPSQIG_LIVE = fm.rhob * (FEPSQIG[5] + FEPSQIG[6])
    RHOBEPSQIG = fm.F_dead * RHOBEPSQIG_DEAD + fm.F_live * RHOBEPSQIG_LIVE

    # Dead fuel moisture damping
    # M_dead = Σ(f * M) for dead classes
    FMC = ntuple(i -> fm.F[i] * M[i], 6)
    M_dead = FMC[1] + FMC[2] + FMC[3] + FMC[4]
    momex_dead = M_dead / fm.mex_dead
    etam_dead = moisture_damping(momex_dead)

    # Reaction intensity for dead fuels
    IR_dead = fm.GP_WND_ETAS_HOC * etam_dead

    # Live fuel moisture damping
    M_live = FMC[5] + FMC[6]
    momex_live = M_live / mex_live
    etam_live = moisture_damping(momex_live)

    # Reaction intensity for live fuels
    IR_live = fm.GP_WNL_ETAS_HOC * etam_live

    # Total reaction intensity (Btu/(ft²·min))
    IR_btupft2min = IR_dead + IR_live

    # Wind limit (original Rothermel limit)
    WS_LIMIT = T(0.9) * IR_btupft2min
    wsmf_limited = min(wsmf, WS_LIMIT)

    # Wind factor: φ_w = phiwterm * WSMF^B
    phiw = fm.phiwterm * wsmf_limited^fm.B

    # Maximum slope factor equals max wind factor
    phis_max = fm.phiwterm * WS_LIMIT^fm.B

    # Slope factor: φ_s = min(phisterm * tan²θ, φ_w_max)
    phis = min(fm.phisterm * tanslp2, phis_max)

    # Base spread rate (ft/min)
    vs0 = if RHOBEPSQIG > T(1e-9)
        adj * IR_btupft2min * fm.xi / RHOBEPSQIG
    else
        zero(T)
    end

    # Final spread velocity (ft/min)
    velocity = vs0 * (one(T) + phis + phiw)

    # Convert reaction intensity to SI: kW/m²
    IR = IR_btupft2min * btupft2min_to_kwpm2(T)

    # Heat per unit area: HPUA = IR * tr * 60 (kJ/m²)
    hpua = IR * fm.tr * T(60)

    # Fireline intensity: flin = tr * IR * velocity * 0.3048 (kW/m)
    flin = fm.tr * IR * velocity * ft_to_m(T)

    return SpreadResult{T}(velocity, vs0, IR, hpua, flin, phiw, phis)
end


"""
    surface_spread_rate(
        fm::FuelModel{T},
        moisture::NTuple{5,T},
        wsmf::T,
        tanslp2::T;
        adj::T = one(T)
    ) -> SpreadResult{T}

Convenience method accepting moisture as tuple (M1, M10, M100, MLH, MLW).
"""
function surface_spread_rate(
    fm::FuelModel{T},
    moisture::NTuple{5,T},
    wsmf::T,
    tanslp2::T;
    adj::T = one(T)
) where {T<:AbstractFloat}
    return surface_spread_rate(fm, moisture[1], moisture[2], moisture[3],
                                moisture[4], moisture[5], wsmf, tanslp2; adj=adj)
end


"""
    surface_spread_rate_flat(
        fuel_arr::FuelModelArray{T}, fuel_index::Int, mc_index::Int,
        M1::T, M10::T, M100::T, MLH::T, MLW::T,
        wsmf::T, tanslp2::T, adj::T
    ) -> Tuple{T, T, T}

Compute surface fire spread rate from a `FuelModelArray` using dense array indexing.
This variant is designed for GPU kernels where `Dict`-based `FuelModel` lookup is not available.

Returns `(velocity, vs0, flin)` as a tuple instead of a `SpreadResult` struct.

### Arguments
- `fuel_arr`: Dense fuel model array
- `fuel_index`: Row index in `fuel_arr` (from `fuel_arr.fuel_id_to_index[fuel_id]`)
- `mc_index`: Moisture class index (`live_moisture_class - 29`)
- `M1, M10, M100, MLH, MLW`: Fuel moisture fractions
- `wsmf`: Mid-flame wind speed (ft/min)
- `tanslp2`: tan²(slope angle)
- `adj`: Spread rate adjustment factor
"""
@inline function surface_spread_rate_flat(
    fuel_arr::FuelModelArray{T},
    fuel_index::Int, mc_index::Int,
    M1::T, M10::T, M100::T, MLH::T, MLW::T,
    wsmf::T, tanslp2::T, adj::T
) where {T<:AbstractFloat}
    fi = fuel_index
    mi = mc_index

    # Check nonburnable
    if fuel_arr.nonburnable[fi, mi]
        return (zero(T), zero(T), zero(T))
    end

    # Load coefficients from arrays
    rhob = fuel_arr.rhob[fi, mi]
    xi = fuel_arr.xi[fi, mi]
    B = fuel_arr.B[fi, mi]
    GP_WND_ETAS_HOC = fuel_arr.GP_WND_ETAS_HOC[fi, mi]
    GP_WNL_ETAS_HOC = fuel_arr.GP_WNL_ETAS_HOC[fi, mi]
    phisterm = fuel_arr.phisterm[fi, mi]
    phiwterm = fuel_arr.phiwterm[fi, mi]
    R_MPRIMEDENOME14SUM_MEX_DEAD = fuel_arr.R_MPRIMEDENOME14SUM_MEX_DEAD[fi, mi]
    F_dead = fuel_arr.F_dead[fi, mi]
    F_live = fuel_arr.F_live[fi, mi]
    mex_dead = fuel_arr.mex_dead[fi, mi]
    mex_live_base = fuel_arr.mex_live[fi, mi]
    tr = fuel_arr.tr[fi, mi]

    # Moisture array
    M = (M1, M10, M100, M1, MLH, MLW)

    # Live fuel moisture of extinction (dynamic)
    SUM_MPRIMENUMER = zero(T)
    for k in 1:4
        SUM_MPRIMENUMER += fuel_arr.WPRIMENUMER[fi, mi, k] * M[k]
    end
    mex_live = mex_live_base * (one(T) - R_MPRIMEDENOME14SUM_MEX_DEAD * SUM_MPRIMENUMER) - T(0.226)
    mex_live = max(mex_live, mex_dead)

    # Heat of pre-ignition and effective heating
    RHOBEPSQIG_DEAD = zero(T)
    RHOBEPSQIG_LIVE = zero(T)
    for k in 1:4
        RHOBEPSQIG_DEAD += fuel_arr.FEPS[fi, mi, k] * (T(250) + T(1116) * M[k])
    end
    for k in 5:6
        RHOBEPSQIG_LIVE += fuel_arr.FEPS[fi, mi, k] * (T(250) + T(1116) * M[k])
    end
    RHOBEPSQIG_DEAD *= rhob
    RHOBEPSQIG_LIVE *= rhob
    RHOBEPSQIG = F_dead * RHOBEPSQIG_DEAD + F_live * RHOBEPSQIG_LIVE

    # Moisture damping
    M_dead = zero(T)
    for k in 1:4
        M_dead += fuel_arr.F[fi, mi, k] * M[k]
    end
    etam_dead = moisture_damping(M_dead / mex_dead)
    IR_dead = GP_WND_ETAS_HOC * etam_dead

    M_live = zero(T)
    for k in 5:6
        M_live += fuel_arr.F[fi, mi, k] * M[k]
    end
    etam_live = moisture_damping(M_live / mex_live)
    IR_live = GP_WNL_ETAS_HOC * etam_live

    IR_btupft2min = IR_dead + IR_live

    # Wind/slope factors
    WS_LIMIT = T(0.9) * IR_btupft2min
    wsmf_limited = min(wsmf, WS_LIMIT)
    phiw = phiwterm * wsmf_limited^B
    phis_max = phiwterm * WS_LIMIT^B
    phis = min(phisterm * tanslp2, phis_max)

    # Spread rate
    vs0 = if RHOBEPSQIG > T(1e-9)
        adj * IR_btupft2min * xi / RHOBEPSQIG
    else
        zero(T)
    end
    velocity = vs0 * (one(T) + phis + phiw)

    # Fireline intensity
    IR = IR_btupft2min * btupft2min_to_kwpm2(T)
    flin = tr * IR * velocity * ft_to_m(T)

    return (velocity, vs0, flin)
end


#-----------------------------------------------------------------------------#
#                     Elliptical Fire Spread Model
#-----------------------------------------------------------------------------#

"""
    EllipticalSpread{T<:AbstractFloat}

Elliptical fire spread parameters following Anderson (1982).
"""
struct EllipticalSpread{T<:AbstractFloat}
    head::T           # Head fire spread rate (ft/min)
    back::T           # Backing fire spread rate (ft/min)
    flank::T          # Flanking fire spread rate (ft/min)
    eccentricity::T   # Ellipse eccentricity
    length_to_breadth::T  # Length to breadth ratio
end

Base.eltype(::EllipticalSpread{T}) where {T} = T


"""
    length_to_breadth(wsmfeff::T; lb_cap::T=T(8)) -> T

Length-to-breadth ratio of the elliptical wavelet from the effective mid-flame
wind speed `wsmfeff` (ft/min).

Anderson (1982) as modified by Finney in FARSITE:
`L/B = 0.936·exp(0.2566·U) + 0.461·exp(-0.1548·U) - 0.397`

`U` is `wsmfeff` scaled by `wsmfeff_low_mult` (ELMFIRE's `WSMFEFF_LOW_MULT`,
which converts ft/min to the units the correlation was fit in). `lb_cap`
corresponds to ELMFIRE's `MAX_LOW` (default 8).

Implementation follows `elmfire_level_set.f90:1810`.
"""
@inline function length_to_breadth(wsmfeff::T; lb_cap::T=T(8)) where {T<:AbstractFloat}
    U = max(wsmfeff, zero(T)) * wsmfeff_low_mult(T)
    LB = T(0.936) * exp(T(0.2566) * U) + T(0.461) * exp(T(-0.1548) * U) - T(0.397)
    return clamp(LB, one(T), lb_cap)
end


"""
    elliptical_spread(velocity::T, wsmfeff::T; lb_cap::T=T(8)) -> EllipticalSpread{T}

Calculate elliptical fire spread dimensions from the head fire rate and the
effective mid-flame wind speed.

# Arguments
- `velocity`: Spread rate in the direction of maximum spread (ft/min)
- `wsmfeff`: Effective mid-flame wind speed (**ft/min**), which in ELMFIRE is
  recovered from the combined wind + slope factor by [`effective_wind_speed`](@ref)
  and therefore includes the slope contribution
- `lb_cap`: Maximum length-to-breadth ratio (ELMFIRE `MAX_LOW`, default 8)

The backing rate follows the classical ellipse relation used by ELMFIRE
(`elmfire_level_set.f90:1812-1818`):
`V_back/V_head = (LB - √(LB²-1)) / (LB + √(LB²-1))`, equivalently
`(1-e)/(1+e)` with `e = √(LB²-1)/LB`.
"""
function elliptical_spread(velocity::T, wsmfeff::T; lb_cap::T=T(8)) where {T<:AbstractFloat}
    LB = length_to_breadth(wsmfeff; lb_cap = lb_cap)

    # Ellipse eccentricity for a length-to-breadth (semi-major/semi-minor) ratio
    # of LB: e = c/a = sqrt(1 - b²/a²) = sqrt(LB² - 1) / LB
    eccentricity = if LB > T(1.001)
        sqrt(LB * LB - one(T)) / LB
    else
        zero(T)
    end

    # Head fire rate is the input velocity
    head = velocity

    # Backing fire rate: ELMFIRE's BOH = (LB - sqrt(LB²-1)) / (LB + sqrt(LB²-1))
    back = head * (one(T) - eccentricity) / (one(T) + eccentricity)

    # Flanking fire rate (semi-minor axis rate), equal to a/LB
    flank = head * sqrt(one(T) - eccentricity * eccentricity) / (one(T) + eccentricity)

    return EllipticalSpread{T}(head, back, flank, eccentricity, LB)
end


#-----------------------------------------------------------------------------#
#                     Direction of Maximum Spread
#-----------------------------------------------------------------------------#

"""
    spread_direction(phis::T, phiw::T, aspect_deg::T, wind_direction_deg::T) -> Tuple{T,T,T}

Combine the slope and wind factors as **vectors** and return
`(phimag, dms_x, dms_y)`: the magnitude of the combined factor and the unit
vector pointing in the direction of maximum spread.

Both factors act along the direction the fire is pushed, i.e. 180° from the
aspect (downslope-facing direction) and 180° from the wind's "from" direction:

```
φ_x = φ_w·sin(θ_w - π) + φ_s·sin(θ_a - π)
φ_y = φ_w·cos(θ_w - π) + φ_s·cos(θ_a - π)
```

Implementation follows `elmfire_level_set.f90:1760-1800` and the ELMFIRE technical
reference. Callers must scale `phis`/`phiw` by the acceleration factor beforehand
where ELMFIRE does so.
"""
@inline function spread_direction(
    phis::T, phiw::T, aspect_deg::T, wind_direction_deg::T
) where {T<:AbstractFloat}
    asp = (aspect_deg - T(180)) * pio180(T)
    wnd = (wind_direction_deg - T(180)) * pio180(T)

    phix = phiw * sin(wnd) + phis * sin(asp)
    phiy = phiw * cos(wnd) + phis * cos(asp)

    phimag = max(sqrt(phix * phix + phiy * phiy), T(1e-10))
    if phimag < T(1.1e-10)
        return (phimag, one(T), zero(T))
    end
    rphimag = one(T) / phimag
    return (phimag, phix * rphimag, phiy * rphimag)
end


"""
    effective_wind_speed(phimag::T, wsmfeff_coeff::T, B_inverse::T) -> T

Recover the effective mid-flame wind speed (ft/min) that alone would produce the
combined wind + slope factor `phimag`, by inverting Rothermel (1972) Eq. 47:

`U_mf,e = (|φ| / (C(β/β_op)^-E))^(1/B)`

`wsmfeff_coeff` and `B_inverse` are the pre-computed fuel model terms
`(1/phiwterm)^(1/B)` and `1/B`. Because `|φ|` carries both wind and slope, the
result drives a length-to-breadth ratio that responds to terrain as well as wind.

Implementation follows `elmfire_level_set.f90:1808`.
"""
@inline function effective_wind_speed(phimag::T, wsmfeff_coeff::T, B_inverse::T) where {T<:AbstractFloat}
    return wsmfeff_coeff * phimag^B_inverse
end


"""
    slope_projection_factors(slope_deg::T, aspect_deg::T) -> Tuple{T,T}

Factors that project velocities computed parallel to the local slope onto the
horizontal map plane:

```
U_x / U_x,∥ = 1 - |sin(θ_a)|(1 - cos(γ))
U_y / U_y,∥ = 1 - |cos(θ_a)|(1 - cos(γ))
```

where `γ` is the slope and `θ_a` the aspect. Both factors are 1 on flat ground.

Implementation follows `elmfire_level_set.f90:1770-1771`.
"""
@inline function slope_projection_factors(slope_deg::T, aspect_deg::T) where {T<:AbstractFloat}
    gamma = clamp(slope_deg, zero(T), T(90)) * pio180(T)
    omcos = one(T) - cos(gamma)
    asp = aspect_deg * pio180(T)
    return (one(T) - abs(sin(asp)) * omcos, one(T) - abs(cos(asp)) * omcos)
end


"""
    velocity_at_angle(es::EllipticalSpread{T}, theta::T) -> T

Calculate fire spread rate at angle theta from the wind direction.
theta = 0 is head fire (downwind), theta = π is backing fire (upwind).
"""
function velocity_at_angle(es::EllipticalSpread{T}, theta::T) where {T<:AbstractFloat}
    if es.eccentricity < T(1e-6)
        # Circular spread
        return es.head
    end

    # Ellipse polar equation: r = a(1-e²) / (1 - e*cos(θ))
    # Where a is semi-major axis (head fire distance from center)
    # But we need velocity, which scales with the ellipse dimensions

    # The fire front position traces an ellipse, so velocity at angle θ
    # can be computed from the ellipse geometry
    e = es.eccentricity
    a = (es.head + es.back) / T(2)  # semi-major axis rate

    # Distance from focus (ignition point) at angle θ
    # r = a(1-e²) / (1 - e*cos(θ))
    r = a * (one(T) - e*e) / (one(T) - e * cos(theta))

    return r
end


#-----------------------------------------------------------------------------#
#                     Velocity Components
#-----------------------------------------------------------------------------#

"""
    velocity_components(
        es::EllipticalSpread{T},
        dms_x::T, dms_y::T,
        normal_x::T, normal_y::T
    ) -> Tuple{T, T}

Calculate velocity components (ux, uy) using the Richards (1990) ellipse equation.

Uses semi-major and semi-minor axes derived from head/back velocities and the L/B
ratio to compute the elliptical velocity at any angle relative to the direction of
maximum spread. The returned components are **parallel to the local slope**; apply
[`slope_projection_factors`](@ref) to map them onto the horizontal grid.

# Arguments
- `es`: Elliptical spread parameters (head, back, eccentricity, length_to_breadth)
- `dms_x, dms_y`: Unit vector in the direction of maximum spread, from
  [`spread_direction`](@ref)
- `normal_x, normal_y`: Unit normal vector to the fire front

# Returns
- `(ux, uy)`: Velocity components in grid coordinates (ft/min)
"""
function velocity_components(
    es::EllipticalSpread{T},
    dms_x::T, dms_y::T,
    normal_x::T, normal_y::T
) where {T<:AbstractFloat}
    # Dot and cross products of the front normal against the direction of maximum spread
    cosang = normal_x * dms_x + normal_y * dms_y
    sinang = normal_x * dms_y - normal_y * dms_x

    # Semi-major axis (along wind) and semi-minor axis (across wind)
    a = max(T(0.5) * (es.head + es.back), T(1e-10))
    lb = max(es.length_to_breadth, one(T))
    b = max(a / lb, T(1e-10))

    # Richards (1990) ellipse velocity decomposition
    aa_cosang = a * a * cosang
    bb_sinang = b * b * sinang
    denom = max(sqrt(aa_cosang * cosang + bb_sinang * sinang), T(1e-10))

    # Velocity in the wind-aligned (dydt) and crosswind (dxdt) directions
    dydt = aa_cosang / denom + T(0.5) * (es.head - es.back)
    dxdt = bb_sinang / denom

    # Rotate from DMS-relative to grid coordinates
    ux = dxdt * dms_y + dydt * dms_x
    uy = -dxdt * dms_x + dydt * dms_y

    return (ux, uy)
end


"""
    velocity_components(
        es::EllipticalSpread{T},
        wind_direction_rad::T,
        normal_x::T,
        normal_y::T
    ) -> Tuple{T, T}

Convenience method that takes the direction of maximum spread to be the wind
direction, ignoring slope. Prefer the four-vector method with a direction from
[`spread_direction`](@ref), which accounts for terrain.
"""
function velocity_components(
    es::EllipticalSpread{T},
    wind_direction_rad::T,
    normal_x::T,
    normal_y::T
) where {T<:AbstractFloat}
    # Wind direction: convert FROM (meteorological) to TO (mathematical)
    return velocity_components(es, -sin(wind_direction_rad), -cos(wind_direction_rad),
                               normal_x, normal_y)
end
