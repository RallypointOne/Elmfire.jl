#-----------------------------------------------------------------------------#
#                               Physical Constants
#-----------------------------------------------------------------------------#

# Default precision type for the package
const DefaultFloat = Float64

# Pi constant - use the appropriate precision
@inline pi_val(::Type{T}) where {T<:AbstractFloat} = T(3.141592653589793)
@inline pio180(::Type{T}) where {T<:AbstractFloat} = pi_val(T) / T(180)

# Convenience for default precision
const PI = pi_val(DefaultFloat)
const PIO180 = pio180(DefaultFloat)

#-----------------------------------------------------------------------------#
#                               Unit Conversions
#-----------------------------------------------------------------------------#

# BTU/(ft²·min) to kW/m²
@inline btupft2min_to_kwpm2(::Type{T}) where {T<:AbstractFloat} = T(1.055) / (T(60) * T(0.3048) * T(0.3048))
const BTUPFT2MIN_TO_KWPM2 = btupft2min_to_kwpm2(DefaultFloat)

# Feet to meters
@inline ft_to_m(::Type{T}) where {T<:AbstractFloat} = T(0.3048)
const FT_TO_M = ft_to_m(DefaultFloat)

# Meters to feet
@inline m_to_ft(::Type{T}) where {T<:AbstractFloat} = T(1) / ft_to_m(T)
const M_TO_FT = m_to_ft(DefaultFloat)

# Chains per hour to feet per minute
@inline chph_to_ftpmin(::Type{T}) where {T<:AbstractFloat} = T(1.1)

# Feet per minute to meters per second
@inline ftpmin_to_mps(::Type{T}) where {T<:AbstractFloat} = ft_to_m(T) / T(60)

# Miles per hour to feet per minute (1 mph = 88 ft/min)
@inline mph_to_ftpmin(::Type{T}) where {T<:AbstractFloat} = T(88)

#-----------------------------------------------------------------------------#
#                               Fuel Model Constants
#-----------------------------------------------------------------------------#

# Fixed surface area to volume ratios (1/ft)
@inline sig_10hr(::Type{T}) where {T<:AbstractFloat} = T(109)
@inline sig_100hr(::Type{T}) where {T<:AbstractFloat} = T(30)
const SIG_10HR = sig_10hr(DefaultFloat)
const SIG_100HR = sig_100hr(DefaultFloat)

# Particle density (lb/ft³)
@inline rhop_default(::Type{T}) where {T<:AbstractFloat} = T(32)
const RHOP_DEFAULT = rhop_default(DefaultFloat)

# Mineral content (lb minerals / lb ovendry mass)
@inline st_default(::Type{T}) where {T<:AbstractFloat} = T(0.055)
const ST_DEFAULT = st_default(DefaultFloat)

# Silica-effective mineral content
@inline se_default(::Type{T}) where {T<:AbstractFloat} = T(0.01)
const SE_DEFAULT = se_default(DefaultFloat)

# Mineral damping coefficient
@inline etas_default(::Type{T}) where {T<:AbstractFloat} = T(0.174) / se_default(T)^T(0.19)
const ETAS_DEFAULT = etas_default(DefaultFloat)

# Non-burnable fuel model ID
const FUEL_MODEL_NONBURNABLE = 256
