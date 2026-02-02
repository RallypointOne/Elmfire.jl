#-----------------------------------------------------------------------------#
#                     Crown Fire Model
#-----------------------------------------------------------------------------#
#
# Implements the Cruz (2005) crown fire model for active crown fire spread
# and Van Wagner (1977) model for crown fire initiation.
#
# References:
# - Cruz et al. (2005) "Development and testing of models for predicting
#   crown fire rate of spread in conifer forest stands"
# - Van Wagner (1977) "Conditions for the start and spread of crown fire"
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Canopy Properties
#-----------------------------------------------------------------------------#

"""
    CanopyProperties{T<:AbstractFloat}

Canopy fuel properties for a cell.
"""
struct CanopyProperties{T<:AbstractFloat}
    cbd::T      # Canopy bulk density (kg/m³)
    cbh::T      # Canopy base height (m)
    cc::T       # Canopy cover (fraction, 0-1)
    ch::T       # Canopy height (m)
end

Base.eltype(::CanopyProperties{T}) where {T} = T

# Constructor with defaults
function CanopyProperties{T}(;
    cbd::T = zero(T),
    cbh::T = zero(T),
    cc::T = zero(T),
    ch::T = zero(T)
) where {T<:AbstractFloat}
    CanopyProperties{T}(cbd, cbh, cc, ch)
end


"""
    CrownFireResult{T<:AbstractFloat}

Results from crown fire calculation.
"""
struct CrownFireResult{T<:AbstractFloat}
    crown_fire_type::Int   # 0 = no crown fire, 1 = passive, 2 = active
    spread_rate::T         # Crown fire spread rate (ft/min)
    hpua_canopy::T         # Heat per unit area from canopy (kJ/m²)
    critical_flin::T       # Critical fireline intensity for crown fire initiation (kW/m)
    phiw_crown::T          # Crown fire wind factor
end

Base.eltype(::CrownFireResult{T}) where {T} = T


#-----------------------------------------------------------------------------#
#                     Crown Fire Initiation
#-----------------------------------------------------------------------------#

"""
    critical_fireline_intensity(cbh::T, foliar_moisture::T) -> T

Calculate the critical fireline intensity (kW/m) needed to initiate crown fire.

Uses Van Wagner (1977) criterion:
I_crit = (0.01 * CBH * (460 + 26 * FMC))^1.5

# Arguments
- `cbh`: Canopy base height (m)
- `foliar_moisture`: Foliar moisture content (%, e.g., 100 for 100%)

# Returns
- Critical fireline intensity (kW/m)
"""
function critical_fireline_intensity(cbh::T, foliar_moisture::T) where {T<:AbstractFloat}
    if cbh < zero(T)
        return T(9e9)  # Effectively no crown fire possible
    end

    cbh_eff = max(cbh, T(0.1))
    fmc_term = T(460) + T(26) * foliar_moisture

    return (T(0.01) * cbh_eff * fmc_term)^T(1.5)
end


#-----------------------------------------------------------------------------#
#                     Crown Fire Spread Rate
#-----------------------------------------------------------------------------#

"""
    crown_spread_rate(
        canopy::CanopyProperties{T},
        flin_surface::T,
        ws20::T,
        m1::T,
        vs0::T;
        crown_fire_adj::T = one(T),
        spread_rate_limit::T = T(1000),
        critical_canopy_cover::T = T(0.4)
    ) -> CrownFireResult{T}

Calculate crown fire spread rate using the Cruz (2005) model.

# Arguments
- `canopy`: Canopy fuel properties
- `flin_surface`: Surface fireline intensity (kW/m)
- `ws20`: 20-ft wind speed (mph)
- `m1`: 1-hour dead fuel moisture (fraction)
- `vs0`: Base surface spread rate (ft/min)
- `crown_fire_adj`: Crown fire adjustment factor (default 1.0)
- `spread_rate_limit`: Maximum crown fire spread rate (ft/min)
- `critical_canopy_cover`: Minimum canopy cover for active crown fire

# Returns
- `CrownFireResult` with crown fire type, spread rate, and related values

Implementation follows `elmfire_spread_rate.f90:136-211`.
"""
function crown_spread_rate(
    canopy::CanopyProperties{T},
    flin_surface::T,
    ws20::T,
    m1::T,
    vs0::T;
    crown_fire_adj::T = one(T),
    spread_rate_limit::T = T(1000),
    critical_canopy_cover::T = T(0.4),
    foliar_moisture::T = T(100)
) where {T<:AbstractFloat}
    # Constants
    MPH_20FT_TO_KMPH_10M = T(1.609) / T(0.87)  # Convert 20-ft mph to 10-m km/h

    # Default return for non-crown conditions
    no_crown = CrownFireResult{T}(0, zero(T), zero(T), T(9e9), zero(T))

    # Check if crown fire is possible
    if vs0 <= zero(T) || flin_surface <= zero(T)
        return no_crown
    end
    if canopy.cbd < T(1e-3) || canopy.cc < T(1e-3)
        return no_crown
    end

    # Calculate heat per unit area from canopy fuel
    canopy_depth = max(canopy.ch - canopy.cbh, zero(T))
    hpua_canopy = canopy.cbd * canopy_depth * T(12000)  # kJ/m²

    # Calculate critical fireline intensity
    critical_flin = critical_fireline_intensity(canopy.cbh, foliar_moisture)

    # Check if surface fire intensity exceeds threshold
    if flin_surface <= critical_flin
        return CrownFireResult{T}(0, zero(T), hpua_canopy, critical_flin, zero(T))
    end

    # Crown fire initiated - calculate spread rate
    cbd_eff = max(canopy.cbd, T(0.01))

    # Convert wind speed to 10-m height in km/h
    ws10_kmph = ws20 * MPH_20FT_TO_KMPH_10M

    # Cruz (2005) active crown fire spread rate (m/min originally, convert to ft/min)
    # CROSA = 11.02 * WS10^0.9 * CBD^0.19 * exp(-0.17 * 100 * M1)
    crosa = crown_fire_adj * T(11.02) * ws10_kmph^T(0.9) * cbd_eff^T(0.19) *
            exp(T(-0.17) * T(100) * m1) / ft_to_m(T)  # Convert to ft/min

    crosa = min(crosa, spread_rate_limit)

    # Critical spread rate for active crown fire (R0 = 3/CBD in m/min)
    r0 = (T(3) / cbd_eff) / ft_to_m(T)  # ft/min

    # Crown activity coefficient
    cac = crosa / r0

    crown_fire_type = 0
    cros = zero(T)
    phiw_crown = zero(T)

    if cac > one(T)
        # Active crown fire
        if canopy.cc >= critical_canopy_cover
            crown_fire_type = 2
            cros = crosa
            phiw_crown = min(max(cros / max(vs0, T(0.001)) - one(T), zero(T)), T(200))
        else
            # Passive crown fire (insufficient canopy continuity)
            crown_fire_type = 1
        end
    else
        # Passive crown fire
        crown_fire_type = 1
        if canopy.cc >= critical_canopy_cover
            # Reduced spread rate for passive crown fire
            cros = crosa * exp(-cac)
            phiw_crown = min(max(cros / max(vs0, T(0.001)) - one(T), zero(T)), T(200))
        end
    end

    return CrownFireResult{T}(crown_fire_type, cros, hpua_canopy, critical_flin, phiw_crown)
end


#-----------------------------------------------------------------------------#
#                     Combined Surface + Crown Spread
#-----------------------------------------------------------------------------#

"""
    combined_spread_rate(
        surface_result::SpreadResult{T},
        crown_result::CrownFireResult{T}
    ) -> T

Calculate the combined spread rate considering both surface and crown fire.

For active crown fires, returns the crown fire spread rate.
For passive crown fires, returns surface spread rate (crown fire doesn't add to ROS).
For surface fires only, returns the surface spread rate.
"""
function combined_spread_rate(
    surface_result::SpreadResult{T},
    crown_result::CrownFireResult{T}
) where {T<:AbstractFloat}
    if crown_result.crown_fire_type == 2
        # Active crown fire - use crown fire spread rate
        return crown_result.spread_rate
    else
        # Surface fire or passive crown fire
        return surface_result.velocity
    end
end


"""
    combined_fireline_intensity(
        surface_result::SpreadResult{T},
        crown_result::CrownFireResult{T},
        fm::FuelModel{T}
    ) -> T

Calculate the combined fireline intensity (kW/m) for surface and crown fire.
"""
function combined_fireline_intensity(
    surface_result::SpreadResult{T},
    crown_result::CrownFireResult{T},
    fm::FuelModel{T}
) where {T<:AbstractFloat}
    if crown_result.crown_fire_type >= 1
        # Crown fire contributes additional heat
        # FLIN_total = FLIN_surface + crown contribution
        # Crown FLIN = ROS * HPUA_canopy / 60 (convert to kW/m)
        velocity = combined_spread_rate(surface_result, crown_result)
        crown_flin = velocity * ft_to_m(T) * crown_result.hpua_canopy / T(60)
        return surface_result.flin + crown_flin
    else
        return surface_result.flin
    end
end
