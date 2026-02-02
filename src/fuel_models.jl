#-----------------------------------------------------------------------------#
#                               Raw Fuel Model Data
#-----------------------------------------------------------------------------#
"""
    RawFuelModel{T<:AbstractFloat}

Raw fuel model parameters as read from CSV file before coefficient calculation.
"""
struct RawFuelModel{T<:AbstractFloat}
    id::Int
    name::String
    dynamic::Bool

    # Fuel loadings (lb/ft²) - columns 4-9 in CSV
    W0_1hr::T
    W0_10hr::T
    W0_100hr::T
    W0_herb::T       # Live herbaceous (transferred to dead in dynamic models)
    W0_woody::T      # Live woody

    # Surface area to volume ratios (1/ft) - columns 10-12 in CSV
    SIG_1hr::T
    SIG_live_herb::T
    SIG_live_woody::T

    # Fuel bed properties
    delta::T         # Fuel bed thickness (ft)
    mex_dead::T      # Dead fuel moisture of extinction (fraction, not %)
    hoc::T           # Heat of combustion (Btu/lb)
end

# Constructor with automatic type promotion
function RawFuelModel(
    id::Int, name::String, dynamic::Bool,
    W0_1hr, W0_10hr, W0_100hr, W0_herb, W0_woody,
    SIG_1hr, SIG_live_herb, SIG_live_woody,
    delta, mex_dead, hoc
)
    T = promote_type(typeof(W0_1hr), typeof(W0_10hr), typeof(W0_100hr),
                     typeof(W0_herb), typeof(W0_woody), typeof(SIG_1hr),
                     typeof(SIG_live_herb), typeof(SIG_live_woody),
                     typeof(delta), typeof(mex_dead), typeof(hoc))
    RawFuelModel{T}(id, name, dynamic,
                    T(W0_1hr), T(W0_10hr), T(W0_100hr), T(W0_herb), T(W0_woody),
                    T(SIG_1hr), T(SIG_live_herb), T(SIG_live_woody),
                    T(delta), T(mex_dead), T(hoc))
end

Base.eltype(::RawFuelModel{T}) where {T} = T


#-----------------------------------------------------------------------------#
#                               Computed Fuel Model
#-----------------------------------------------------------------------------#
"""
    FuelModel{T<:AbstractFloat}

Complete fuel model with all computed Rothermel coefficients.
Matches ELMFIRE's FUEL_MODEL_TABLE_TYPE structure.

The 6 fuel size classes are:
1. 1-hour dead
2. 10-hour dead
3. 100-hour dead
4. Dynamic dead (transferred herbaceous)
5. Live herbaceous
6. Live woody
"""
struct FuelModel{T<:AbstractFloat}
    # Identification
    id::Int
    name::String
    dynamic::Bool

    # Fuel loadings (lb/ft²) - 6 size classes
    W0::NTuple{6,T}   # Total fuel loading
    WN_dead::T        # Net dead fuel loading
    WN_live::T        # Net live fuel loading

    # Surface area to volume ratios (1/ft)
    SIG::NTuple{6,T}
    SIG_dead::T       # Weighted dead SAV
    SIG_live::T       # Weighted live SAV
    SIG_overall::T    # Overall weighted SAV

    # Fuel bed properties
    delta::T          # Fuel bed thickness (ft)
    mex_dead::T       # Dead fuel moisture of extinction
    mex_live::T       # Live fuel moisture of extinction (initial)
    hoc::T            # Heat of combustion (Btu/lb)

    # Derived physical properties
    rhob::T           # Bulk density (lb/ft³)
    rhop::T           # Particle density (lb/ft³)
    beta::T           # Packing ratio
    betaop::T         # Optimal packing ratio
    xi::T             # Propagating flux ratio
    etas::T           # Mineral damping coefficient

    # Rothermel coefficients (A, B, C, E)
    A::T              # 133.0 / σ^0.7913
    B::T              # 0.02526 * σ^0.54
    C::T              # 7.47 * exp(-0.133 * σ^0.55)
    E::T              # 0.715 * exp(-0.000359 * σ)

    # Reaction velocity
    gammaprime::T     # Γ' (1/min)

    # Pre-computed terms for performance (from ELMFIRE)
    F::NTuple{6,T}           # Area fraction factors
    F_dead::T                # Dead fuel area fraction
    F_live::T                # Live fuel area fraction
    EPS::NTuple{6,T}         # Surface heating numbers
    FEPS::NTuple{6,T}        # f * epsilon
    FMEX::NTuple{4,T}        # f * mex_dead for dead classes
    WPRIMENUMER::NTuple{4,T} # W' numerator for dead classes
    MPRIMEDENOM::NTuple{4,T} # M' denominator for dead classes
    WPRIMEDENOM56::NTuple{2,T} # W' denominator for live classes

    # Pre-computed reaction intensity terms
    GP_WND_ETAS_HOC::T  # Γ' * WN_dead * η_s * HOC
    GP_WNL_ETAS_HOC::T  # Γ' * WN_live * η_s * HOC

    # Pre-computed slope/wind terms
    phisterm::T       # 5.275 * β^(-0.3)
    phiwterm::T       # C * (β/β_op)^(-E)

    # Pre-computed optimization terms
    R_MPRIMEDENOME14SUM_MEX_DEAD::T  # 1/(Σmprimedenom * mex_dead)

    # Residence time (min)
    tr::T             # 384/σ_overall
end

Base.eltype(::FuelModel{T}) where {T} = T


#-----------------------------------------------------------------------------#
#                           Coefficient Computation
#-----------------------------------------------------------------------------#
"""
    compute_fuel_model(raw::RawFuelModel{T}, live_moisture_class::Int) -> FuelModel{T}

Compute all Rothermel coefficients from raw fuel model data.

For dynamic fuel models, the live herbaceous fuel is partitioned between
dead and live classes based on the live moisture class (30-120).

Implementation follows `elmfire_init.f90:786-880`.
"""
function compute_fuel_model(raw::RawFuelModel{T}, live_moisture_class::Int=30) where {T<:AbstractFloat}
    # Initialize W0 and SIG arrays
    W0 = zeros(T, 6)
    SIG = zeros(T, 6)

    # Set fixed SAV values
    SIG[2] = sig_10hr(T)   # 109 1/ft
    SIG[3] = sig_100hr(T)  # 30 1/ft
    SIG[4] = T(9999)       # Dynamic dead (unused unless dynamic model)

    # Copy raw values
    W0[1] = raw.W0_1hr
    W0[2] = raw.W0_10hr
    W0[3] = raw.W0_100hr
    W0[5] = raw.W0_herb
    W0[6] = raw.W0_woody
    SIG[1] = raw.SIG_1hr
    SIG[5] = raw.SIG_live_herb
    SIG[6] = raw.SIG_live_woody

    # Handle dynamic fuel models - transfer herbaceous fuel based on moisture
    if raw.dynamic
        lh = T(live_moisture_class)
        livefrac = clamp((lh - T(30)) / (T(120) - T(30)), zero(T), one(T))
        deadfrac = one(T) - livefrac

        # Transfer portion of herbaceous to dynamic dead
        W0[4] = deadfrac * raw.W0_herb
        W0[5] = livefrac * raw.W0_herb
        SIG[4] = raw.SIG_live_herb

        # Recalculate 1-hr SAV as weighted average of original 1-hr and transferred herb
        denom = SIG[1] * raw.W0_1hr + SIG[4] * W0[4]
        if denom > zero(T)
            SIG[1] = (SIG[1]^2 * raw.W0_1hr + SIG[4]^2 * W0[4]) / denom
        end
        W0[1] = raw.W0_1hr + W0[4]
        W0[4] = zero(T)
        SIG[4] = T(9999)
    else
        W0[4] = zero(T)
        SIG[4] = T(9999)
    end

    # Physical constants
    rhop = rhop_default(T)
    st = st_default(T)
    etas = etas_default(T)

    # Area factors: A(i) = σ(i) * W0(i) / ρp
    A = ntuple(i -> SIG[i] * W0[i] / rhop, 6)

    A_dead = max(sum(A[1:4]), T(1e-9))
    A_live = max(sum(A[5:6]), T(1e-9))
    A_overall = A_dead + A_live

    # Fraction factors: f(i) = A(i) / A_class
    F_arr = zeros(T, 6)
    F_arr[1:4] .= [A[i] / A_dead for i in 1:4]
    F_arr[5:6] .= [A[i] / A_live for i in 5:6]
    F = ntuple(i -> F_arr[i], 6)

    F_dead = A_dead / A_overall
    F_live = A_live / A_overall

    # FMEX: f * mex_dead (dead classes only)
    FMEX = ntuple(i -> F[i] * raw.mex_dead, 4)

    # FW0: f * W0
    FW0 = ntuple(i -> F[i] * W0[i], 6)

    # FSIG: f * σ
    FSIG = ntuple(i -> F[i] * SIG[i], 6)

    # Surface heating numbers: ε = exp(-138/σ)
    EPS = ntuple(i -> exp(T(-138) / SIG[i]), 6)

    # FEPS: f * ε
    FEPS = ntuple(i -> F[i] * EPS[i], 6)

    # W' numerator for dead classes: W0(i) * ε(i)
    WPRIMENUMER = ntuple(i -> W0[i] * EPS[i], 4)

    # W' denominator for live classes: W0(i) * exp(-500/σ(i))
    WPRIMEDENOM56 = (W0[5] * exp(T(-500) / SIG[5]), W0[6] * exp(T(-500) / SIG[6]))

    # M' denominator for dead classes
    MPRIMEDENOM = ntuple(i -> W0[i] * EPS[i], 4)

    # Net fuel loadings
    W0_dead = sum(FW0[1:4])
    W0_live = sum(FW0[5:6])
    WN_dead = W0_dead * (one(T) - st)
    WN_live = W0_live * (one(T) - st)

    # Weighted SAV ratios
    SIG_dead = sum(FSIG[1:4])
    SIG_live = sum(FSIG[5:6])
    SIG_overall = F_dead * SIG_dead + F_live * SIG_live

    # Packing ratios
    total_W0 = sum(W0)
    beta = total_W0 / (raw.delta * rhop)
    betaop = T(3.348) / (SIG_overall^T(0.8189))

    # Bulk density
    rhob = total_W0 / raw.delta

    # Propagating flux ratio
    xi = exp((T(0.792) + T(0.681) * sqrt(SIG_overall)) * (T(0.1) + beta)) /
         (T(192) + T(0.2595) * SIG_overall)

    # Rothermel A, B, C, E coefficients
    A_coeff = T(133) / (SIG_overall^T(0.7913))
    B_coeff = T(0.02526) * SIG_overall^T(0.54)
    C_coeff = T(7.47) * exp(T(-0.133) * SIG_overall^T(0.55))
    E_coeff = T(0.715) * exp(T(-0.000359) * SIG_overall)

    # Reaction velocity
    gammaprime_peak = SIG_overall^T(1.5) / (T(495) + T(0.0594) * SIG_overall^T(1.5))
    beta_ratio = beta / betaop
    gammaprime = gammaprime_peak * beta_ratio^A_coeff * exp(A_coeff * (one(T) - beta_ratio))

    # Residence time
    tr = T(384) / SIG_overall

    # Pre-computed reaction intensity terms
    GP_WND_ETAS_HOC = gammaprime * WN_dead * etas * raw.hoc
    GP_WNL_ETAS_HOC = gammaprime * WN_live * etas * raw.hoc

    # Pre-computed slope/wind terms
    phisterm = T(5.275) * beta^T(-0.3)
    phiwterm = C_coeff * beta_ratio^(-E_coeff)

    # Optimization term
    WPRIMEDENOM56SUM = sum(WPRIMEDENOM56)
    WPRIMENUMER14SUM = sum(WPRIMENUMER)
    MPRIMEDENOM14SUM = sum(MPRIMEDENOM)
    R_MPRIMEDENOME14SUM_MEX_DEAD = one(T) / (MPRIMEDENOM14SUM * raw.mex_dead)

    # Live fuel moisture of extinction (initial)
    mex_live = if WPRIMEDENOM56SUM > T(1e-6)
        T(2.9) * WPRIMENUMER14SUM / WPRIMEDENOM56SUM
    else
        T(100)
    end

    return FuelModel{T}(
        raw.id, raw.name, raw.dynamic,
        ntuple(i -> W0[i], 6), WN_dead, WN_live,
        ntuple(i -> SIG[i], 6), SIG_dead, SIG_live, SIG_overall,
        raw.delta, raw.mex_dead, mex_live, raw.hoc,
        rhob, rhop, beta, betaop, xi, etas,
        A_coeff, B_coeff, C_coeff, E_coeff,
        gammaprime,
        F, F_dead, F_live, EPS, FEPS, FMEX, WPRIMENUMER, MPRIMEDENOM, WPRIMEDENOM56,
        GP_WND_ETAS_HOC, GP_WNL_ETAS_HOC,
        phisterm, phiwterm,
        R_MPRIMEDENOME14SUM_MEX_DEAD,
        tr
    )
end


#-----------------------------------------------------------------------------#
#                           Fuel Model Table
#-----------------------------------------------------------------------------#
"""
    FuelModelTable{T<:AbstractFloat}

2D table of fuel models indexed by (fuel_id, live_moisture_class).
Live moisture class ranges from 30 to 120 (representing 30% to 120% moisture).
"""
struct FuelModelTable{T<:AbstractFloat}
    models::Dict{Tuple{Int,Int}, FuelModel{T}}
    raw_models::Dict{Int, RawFuelModel{T}}
end

function FuelModelTable{T}() where {T<:AbstractFloat}
    FuelModelTable{T}(Dict{Tuple{Int,Int}, FuelModel{T}}(), Dict{Int, RawFuelModel{T}}())
end

FuelModelTable() = FuelModelTable{DefaultFloat}()

Base.eltype(::FuelModelTable{T}) where {T} = T

"""
    add_raw_model!(table::FuelModelTable{T}, raw::RawFuelModel) where {T}

Add a raw fuel model to the table and compute coefficients for all
live moisture classes (30-120).
"""
function add_raw_model!(table::FuelModelTable{T}, raw::RawFuelModel) where {T}
    # Convert raw model to table's precision if needed
    raw_T = RawFuelModel{T}(
        raw.id, raw.name, raw.dynamic,
        T(raw.W0_1hr), T(raw.W0_10hr), T(raw.W0_100hr), T(raw.W0_herb), T(raw.W0_woody),
        T(raw.SIG_1hr), T(raw.SIG_live_herb), T(raw.SIG_live_woody),
        T(raw.delta), T(raw.mex_dead), T(raw.hoc)
    )
    table.raw_models[raw.id] = raw_T
    for ilh in 30:120
        table.models[(raw.id, ilh)] = compute_fuel_model(raw_T, ilh)
    end
    return table
end

"""
    get_fuel_model(table::FuelModelTable, fuel_id::Int, live_moisture::AbstractFloat) -> FuelModel

Get the fuel model for the given fuel ID and live moisture content.
Live moisture is clamped to [30, 120] and rounded to nearest integer.
"""
function get_fuel_model(table::FuelModelTable, fuel_id::Int, live_moisture::AbstractFloat)
    ilh = clamp(round(Int, 100 * live_moisture), 30, 120)
    return table.models[(fuel_id, ilh)]
end

"""
    get_fuel_model(table::FuelModelTable, fuel_id::Int, live_moisture_class::Int) -> FuelModel

Get the fuel model for the given fuel ID and live moisture class (30-120).
"""
function get_fuel_model(table::FuelModelTable, fuel_id::Int, live_moisture_class::Int)
    ilh = clamp(live_moisture_class, 30, 120)
    return table.models[(fuel_id, ilh)]
end


#-----------------------------------------------------------------------------#
#                           Utility Functions
#-----------------------------------------------------------------------------#
"""
    isnonburnable(fm::FuelModel) -> Bool

Check if a fuel model is non-burnable (fuel model 256 or similar).
"""
isnonburnable(fm::FuelModel{T}) where {T} = fm.id == FUEL_MODEL_NONBURNABLE || fm.W0[1] < T(1e-6)
