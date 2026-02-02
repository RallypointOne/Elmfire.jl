#-----------------------------------------------------------------------------#
#                     Level Set PDE Solver
#-----------------------------------------------------------------------------#
#
# Solves the level set equation for fire front propagation:
#   ∂φ/∂t + F|∇φ| = 0
#
# Where φ is the level set function (φ < 0 inside fire, φ > 0 outside)
# and F is the local fire spread rate in the normal direction.
#
# Uses:
# - 2nd order Runge-Kutta time integration
# - Superbee flux limiter for gradient calculation
# - Narrow band method for efficiency
#-----------------------------------------------------------------------------#


#-----------------------------------------------------------------------------#
#                     Superbee Flux Limiter
#-----------------------------------------------------------------------------#

"""
    half_superbee(r::T) -> T

Superbee flux limiter function (half-superbee variant).
Returns ψ(r)/2 where ψ is the Superbee limiter.

Used for second-order upwind schemes to limit gradients and prevent oscillations.

Implementation matches `elmfire_level_set.f90:1533-1541`.
"""
@inline function half_superbee(r::T) where {T<:AbstractFloat}
    # ψ(r)/2 = max(0, max(min(r/2, 1), min(r, 1/2)))
    return max(zero(T), max(min(T(0.5)*r, one(T)), min(r, T(0.5))))
end


#-----------------------------------------------------------------------------#
#                     Gradient Limiting
#-----------------------------------------------------------------------------#

"""
    limit_gradients(
        phi::AbstractMatrix{T},
        ux::T, uy::T,
        ix::Int, iy::Int,
        rcellsize::T
    ) -> Tuple{T, T}

Calculate flux-limited gradients ∂φ/∂x and ∂φ/∂y at cell (ix, iy).

Uses the Superbee flux limiter with upwind differencing based on the
velocity direction (ux, uy).

Implementation follows `elmfire_level_set.f90:2040-2118`.

# Arguments
- `phi`: Level set field (with boundary padding)
- `ux, uy`: Velocity components at this cell
- `ix, iy`: Cell indices (1-based, assuming 2-cell padding)
- `rcellsize`: Reciprocal of cell size (1/dx)

# Returns
- `(dphidx, dphidy)`: Flux-limited gradients
"""
function limit_gradients(
    phi::AbstractMatrix{T},
    ux::T, uy::T,
    ix::Int, iy::Int,
    rcellsize::T
) where {T<:AbstractFloat}
    EPSILON = T(1e-30)
    CEILING = T(1e3)

    # X-direction gradient
    dphidx = zero(T)
    if ux >= zero(T)
        # Upwind from west
        deltaup = phi[ix, iy] - phi[ix-1, iy]
        deltaloc = phi[ix+1, iy] - phi[ix, iy]

        # East face
        phieast = phi[ix, iy]
        if abs(deltaloc) > EPSILON
            phieast = phi[ix, iy] + half_superbee(deltaup / deltaloc) * deltaloc
        end

        # West face
        deltaloc_west = -deltaup
        phiwest = phi[ix-1, iy]
        if abs(deltaloc_west) > EPSILON
            deltaup_west = phi[ix-2, iy] - phi[ix-1, iy]
            phiwest = phi[ix-1, iy] - half_superbee(deltaup_west / deltaloc_west) * deltaloc_west
        end

        dphidx = (phieast - phiwest) * rcellsize
    else
        # Upwind from east
        deltaloc = phi[ix+1, iy] - phi[ix, iy]

        # East face
        phieast = phi[ix+1, iy]
        if abs(deltaloc) > EPSILON
            deltaup = phi[ix+2, iy] - phi[ix+1, iy]
            phieast = phi[ix+1, iy] - half_superbee(deltaup / deltaloc) * deltaloc
        end

        # West face
        deltaup_west = -deltaloc
        deltaloc_west = phi[ix-1, iy] - phi[ix, iy]
        phiwest = phi[ix, iy]
        if abs(deltaloc_west) > EPSILON
            phiwest = phi[ix, iy] + half_superbee(deltaup_west / deltaloc_west) * deltaloc_west
        end

        dphidx = (phieast - phiwest) * rcellsize
    end

    # Y-direction gradient
    dphidy = zero(T)
    if uy > zero(T)
        # Upwind from south
        deltaup = phi[ix, iy] - phi[ix, iy-1]
        deltaloc = phi[ix, iy+1] - phi[ix, iy]

        # North face
        phinorth = phi[ix, iy]
        if abs(deltaloc) > EPSILON
            phinorth = phi[ix, iy] + half_superbee(deltaup / deltaloc) * deltaloc
        end

        # South face
        deltaloc_south = -deltaup
        phisouth = phi[ix, iy-1]
        if abs(deltaloc_south) > EPSILON
            deltaup_south = phi[ix, iy-2] - phi[ix, iy-1]
            phisouth = phi[ix, iy-1] - half_superbee(deltaup_south / deltaloc_south) * deltaloc_south
        end

        dphidy = (phinorth - phisouth) * rcellsize
    else
        # Upwind from north
        deltaloc = phi[ix, iy+1] - phi[ix, iy]

        # North face
        phinorth = phi[ix, iy+1]
        if abs(deltaloc) > EPSILON
            deltaup = phi[ix, iy+2] - phi[ix, iy+1]
            phinorth = phi[ix, iy+1] - half_superbee(deltaup / deltaloc) * deltaloc
        end

        # South face
        deltaup_south = -deltaloc
        deltaloc_south = phi[ix, iy-1] - phi[ix, iy]
        phisouth = phi[ix, iy]
        if abs(deltaloc_south) > EPSILON
            phisouth = phi[ix, iy] + half_superbee(deltaup_south / deltaloc_south) * deltaloc_south
        end

        dphidy = (phinorth - phisouth) * rcellsize
    end

    # Clamp to prevent numerical issues
    dphidx = clamp(dphidx, -CEILING, CEILING)
    dphidy = clamp(dphidy, -CEILING, CEILING)

    # Handle NaN
    if isnan(dphidx)
        dphidx = zero(T)
    end
    if isnan(dphidy)
        dphidy = zero(T)
    end

    return (dphidx, dphidy)
end


#-----------------------------------------------------------------------------#
#                     CFL Timestep Calculation
#-----------------------------------------------------------------------------#

"""
    compute_cfl_timestep(
        ux::AbstractMatrix{T},
        uy::AbstractMatrix{T},
        active_cells::AbstractVector{CartesianIndex{2}},
        dx::T,
        dt_current::T;
        target_cfl::T = T(0.9),
        dt_max::T = T(10)
    ) -> T

Compute the adaptive timestep based on the CFL condition.

CFL = umax * dt / dx < target_cfl

Implementation follows `elmfire_level_set.f90:2010-2027`.
"""
function compute_cfl_timestep(
    ux::AbstractMatrix{T},
    uy::AbstractMatrix{T},
    active_cells::AbstractVector{CartesianIndex{2}},
    dx::T,
    dt_current::T;
    target_cfl::T = T(0.9),
    dt_max::T = T(10)
) where {T<:AbstractFloat}
    umax = zero(T)
    for idx in active_cells
        u = max(abs(ux[idx]), abs(uy[idx]))
        if u > umax
            umax = u
        end
    end

    if umax > T(1e-9)
        cfl = umax * dt_current / dx
        dt_new = target_cfl * dt_current / cfl
        return min(dt_new, dt_max)
    else
        return dt_max
    end
end


#-----------------------------------------------------------------------------#
#                     RK2 Time Integration
#-----------------------------------------------------------------------------#

"""
    rk2_step!(
        phi::AbstractMatrix{T},
        phi_old::AbstractMatrix{T},
        ux::AbstractMatrix{T},
        uy::AbstractMatrix{T},
        active_cells::AbstractVector{CartesianIndex{2}},
        dt::T,
        dx::T,
        stage::Int
    )

Perform one stage of the 2-stage Runge-Kutta time integration.

Stage 1: φ^(1) = φ^n - dt*(ux*∂φ/∂x + uy*∂φ/∂y)
Stage 2: φ^(n+1) = 0.5*(φ^n + φ^(1) - dt*(ux*∂φ/∂x + uy*∂φ/∂y))

Implementation follows `elmfire_level_set.f90:1957-1983`.
"""
function rk2_step!(
    phi::AbstractMatrix{T},
    phi_old::AbstractMatrix{T},
    ux::AbstractMatrix{T},
    uy::AbstractMatrix{T},
    active_cells::AbstractVector{CartesianIndex{2}},
    dt::T,
    dx::T,
    stage::Int
) where {T<:AbstractFloat}
    rcellsize = one(T) / dx

    if stage == 1
        # Stage 1: Forward Euler step
        for idx in active_cells
            ix, iy = idx[1], idx[2]
            dphidx, dphidy = limit_gradients(phi, ux[idx], uy[idx], ix, iy, rcellsize)
            phi_new = phi_old[idx] - dt * (ux[idx] * dphidx + uy[idx] * dphidy)

            # Clamp and handle NaN
            if isnan(phi_new)
                phi_new = one(T)
            end
            phi[idx] = clamp(phi_new, T(-100), T(100))
        end
    else
        # Stage 2: Average with original
        for idx in active_cells
            ix, iy = idx[1], idx[2]
            dphidx, dphidy = limit_gradients(phi, ux[idx], uy[idx], ix, iy, rcellsize)
            phi_rhs = phi[idx] - dt * (ux[idx] * dphidx + uy[idx] * dphidy)
            phi[idx] = T(0.5) * (phi_old[idx] + phi_rhs)
        end
    end

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Normal Vector Calculation
#-----------------------------------------------------------------------------#

"""
    compute_normal(
        phi::AbstractMatrix{T},
        ix::Int, iy::Int,
        dx::T
    ) -> Tuple{T, T}

Compute the unit normal vector to the level set at cell (ix, iy).
The normal points in the direction of increasing φ (outward from fire).

Uses central differences.
"""
function compute_normal(
    phi::AbstractMatrix{T},
    ix::Int, iy::Int,
    dx::T
) where {T<:AbstractFloat}
    rdx2 = T(0.5) / dx

    dphidx = (phi[ix+1, iy] - phi[ix-1, iy]) * rdx2
    dphidy = (phi[ix, iy+1] - phi[ix, iy-1]) * rdx2

    mag = sqrt(dphidx^2 + dphidy^2)

    if mag > T(1e-10)
        return (dphidx / mag, dphidy / mag)
    else
        return (zero(T), zero(T))
    end
end


#-----------------------------------------------------------------------------#
#                     Narrow Band Management
#-----------------------------------------------------------------------------#

"""
    NarrowBand

Tracks the active cells near the fire front for efficient computation.

The narrow band contains cells within `band_thickness` cells of the fire front
(where φ changes sign).
"""
mutable struct NarrowBand
    active::Set{CartesianIndex{2}}      # Currently active cells
    ever_tagged::Set{CartesianIndex{2}} # Cells that have ever been in the band
    band_thickness::Int                  # Half-width of the narrow band
end

NarrowBand(thickness::Int=5) = NarrowBand(Set{CartesianIndex{2}}(), Set{CartesianIndex{2}}(), thickness)


"""
    tag_band!(nb::NarrowBand, center::CartesianIndex{2}, nx::Int, ny::Int, padding::Int=2)

Add cells within band_thickness of center to the active set.
"""
function tag_band!(nb::NarrowBand, center::CartesianIndex{2}, nx::Int, ny::Int, padding::Int=2)
    bt = nb.band_thickness
    ix, iy = center[1], center[2]

    # Define bounds with padding consideration
    ixstart = max(ix - bt, 1 + padding)
    ixstop = min(ix + bt, nx - padding)
    iystart = max(iy - bt, 1 + padding)
    iystop = min(iy + bt, ny - padding)

    for jx in ixstart:ixstop
        for jy in iystart:iystop
            idx = CartesianIndex(jx, jy)
            push!(nb.active, idx)
            push!(nb.ever_tagged, idx)
        end
    end

    return nothing
end


"""
    untag_isolated!(
        nb::NarrowBand,
        phi::AbstractMatrix{T},
        burned::Union{AbstractMatrix{Bool}, BitMatrix},
        padding::Int = 0
    )

Remove cells from the active set that are:
1. Already burned (φ ≤ 0) and have no unburned neighbors
2. Far from the fire front and won't affect propagation

Note: The active set uses padded coordinates if padding > 0, but the burned
matrix uses grid coordinates. This function handles the conversion.
"""
function untag_isolated!(
    nb::NarrowBand,
    phi::AbstractMatrix{T},
    burned::Union{AbstractMatrix{Bool}, BitMatrix},
    padding::Int = 0
) where {T<:AbstractFloat}
    to_remove = CartesianIndex{2}[]
    ncols, nrows = size(burned)

    for idx in nb.active
        px, py = idx[1], idx[2]

        # Convert to grid coordinates
        ix, iy = px - padding, py - padding

        # Skip if out of bounds for the burned grid
        if ix < 1 || ix > ncols || iy < 1 || iy > nrows
            continue
        end

        # Skip if not burned (use phi to check, or burned grid)
        if !burned[ix, iy]
            continue
        end

        # Check if any neighbor is not burned (in grid coordinates)
        has_unburned_neighbor = false
        for (dx, dy) in ((1,0), (-1,0), (0,1), (0,-1))
            nx, ny = ix + dx, iy + dy
            if 1 <= nx <= ncols && 1 <= ny <= nrows
                if !burned[nx, ny]
                    has_unburned_neighbor = true
                    break
                end
            end
        end

        if !has_unburned_neighbor
            push!(to_remove, idx)
        end
    end

    for idx in to_remove
        delete!(nb.active, idx)
    end

    return nothing
end


"""
    get_active_cells(nb::NarrowBand) -> Vector{CartesianIndex{2}}

Get a vector of active cell indices for iteration.
"""
function get_active_cells(nb::NarrowBand)
    return collect(nb.active)
end


#-----------------------------------------------------------------------------#
#                     Level Set Propagation Step
#-----------------------------------------------------------------------------#

"""
    level_set_step!(
        phi::AbstractMatrix{T},
        phi_old::AbstractMatrix{T},
        ux::AbstractMatrix{T},
        uy::AbstractMatrix{T},
        active_cells::AbstractVector{CartesianIndex{2}},
        dt::T,
        dx::T
    )

Perform one complete level set propagation step using RK2.

1. Copy phi to phi_old
2. Perform RK2 stage 1
3. Perform RK2 stage 2
"""
function level_set_step!(
    phi::AbstractMatrix{T},
    phi_old::AbstractMatrix{T},
    ux::AbstractMatrix{T},
    uy::AbstractMatrix{T},
    active_cells::AbstractVector{CartesianIndex{2}},
    dt::T,
    dx::T
) where {T<:AbstractFloat}
    # Save current state
    for idx in active_cells
        phi_old[idx] = phi[idx]
    end

    # RK2 stage 1
    rk2_step!(phi, phi_old, ux, uy, active_cells, dt, dx, 1)

    # RK2 stage 2
    rk2_step!(phi, phi_old, ux, uy, active_cells, dt, dx, 2)

    return nothing
end


#-----------------------------------------------------------------------------#
#                     Initialize Level Set
#-----------------------------------------------------------------------------#

"""
    initialize_phi!(phi::AbstractMatrix{T}, ignition_points::Vector{Tuple{Int,Int}}, dx::T)

Initialize the level set field φ with signed distance from ignition points.

φ < 0 inside the fire (burned)
φ > 0 outside the fire (unburned)
"""
function initialize_phi!(
    phi::AbstractMatrix{T},
    ignition_points::Vector{Tuple{Int,Int}},
    dx::T
) where {T<:AbstractFloat}
    nx, ny = size(phi)

    # Initialize with large positive value (unburned)
    fill!(phi, T(100))

    # Set ignition points to negative (burned)
    for (ix, iy) in ignition_points
        if 1 <= ix <= nx && 1 <= iy <= ny
            phi[ix, iy] = -one(T)
        end
    end

    # Compute signed distance (approximate) using sweeping
    # This is a simple approximation; for more accuracy use fast marching
    for _ in 1:10  # Multiple sweeps
        for ix in 2:nx-1
            for iy in 2:ny-1
                if phi[ix, iy] > zero(T)
                    # Find minimum neighbor distance
                    min_neighbor = min(
                        phi[ix-1, iy], phi[ix+1, iy],
                        phi[ix, iy-1], phi[ix, iy+1]
                    )
                    phi[ix, iy] = min(phi[ix, iy], min_neighbor + dx)
                end
            end
        end
        for ix in nx-1:-1:2
            for iy in ny-1:-1:2
                if phi[ix, iy] > zero(T)
                    min_neighbor = min(
                        phi[ix-1, iy], phi[ix+1, iy],
                        phi[ix, iy-1], phi[ix, iy+1]
                    )
                    phi[ix, iy] = min(phi[ix, iy], min_neighbor + dx)
                end
            end
        end
    end

    return nothing
end


"""
    initialize_circular_fire!(
        phi::AbstractMatrix{T},
        center_x::Int, center_y::Int,
        radius_cells::T,
        dx::T
    )

Initialize the level set with a circular fire of given radius centered at (center_x, center_y).
"""
function initialize_circular_fire!(
    phi::AbstractMatrix{T},
    center_x::Int, center_y::Int,
    radius_cells::T,
    dx::T
) where {T<:AbstractFloat}
    nx, ny = size(phi)

    for ix in 1:nx
        for iy in 1:ny
            dist = sqrt(T(ix - center_x)^2 + T(iy - center_y)^2) * dx
            phi[ix, iy] = dist - radius_cells * dx
        end
    end

    return nothing
end
