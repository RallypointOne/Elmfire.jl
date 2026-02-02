#-----------------------------------------------------------------------------#
#                           Fuel Model CSV Parsing
#-----------------------------------------------------------------------------#
"""
    parse_fuel_model_line(::Type{T}, line::AbstractString) -> RawFuelModel{T}

Parse a single line from the fuel_models.csv file with specified precision.

CSV format:
fuel_id, name, dynamic, W0_1hr, W0_10hr, W0_100hr, W0_herb, W0_woody,
SIG_1hr, SIG_herb, SIG_woody, delta, mex_dead, hoc

Example line:
1,FBFM01,.FALSE.,0.034,0,0,0,0,3500,9999,9999,1,12,8000
"""
function parse_fuel_model_line(::Type{T}, line::AbstractString) where {T<:AbstractFloat}
    parts = split(strip(line), ',')
    length(parts) >= 14 || error("Invalid fuel model line: expected 14 columns, got $(length(parts))")

    id = parse(Int, parts[1])
    name = String(strip(parts[2]))
    dynamic = uppercase(strip(parts[3])) in [".TRUE.", "TRUE", "T", "1"]

    W0_1hr = parse(T, parts[4])
    W0_10hr = parse(T, parts[5])
    W0_100hr = parse(T, parts[6])
    W0_herb = parse(T, parts[7])
    W0_woody = parse(T, parts[8])

    SIG_1hr = parse(T, parts[9])
    SIG_herb = parse(T, parts[10])
    SIG_woody = parse(T, parts[11])

    delta = parse(T, parts[12])
    mex_dead = parse(T, parts[13]) / T(100)  # Convert from percentage
    hoc = parse(T, parts[14])

    return RawFuelModel{T}(
        id, name, dynamic,
        W0_1hr, W0_10hr, W0_100hr, W0_herb, W0_woody,
        SIG_1hr, SIG_herb, SIG_woody,
        delta, mex_dead, hoc
    )
end

# Default to Float64 for backwards compatibility
parse_fuel_model_line(line::AbstractString) = parse_fuel_model_line(DefaultFloat, line)


"""
    load_fuel_models(::Type{T}, filepath::AbstractString) -> FuelModelTable{T}

Load fuel models from a CSV file and compute all coefficients with specified precision.

The file should be in ELMFIRE fuel_models.csv format.
"""
function load_fuel_models(::Type{T}, filepath::AbstractString) where {T<:AbstractFloat}
    table = FuelModelTable{T}()

    open(filepath, "r") do io
        for line in eachline(io)
            stripped = strip(line)
            # Skip empty lines and comments
            isempty(stripped) && continue
            startswith(stripped, '#') && continue

            try
                raw = parse_fuel_model_line(T, stripped)
                add_raw_model!(table, raw)
            catch e
                @warn "Failed to parse fuel model line: $line" exception=e
            end
        end
    end

    return table
end

# Default to Float64
load_fuel_models(filepath::AbstractString) = load_fuel_models(DefaultFloat, filepath)


"""
    load_fuel_models(::Type{T}) -> FuelModelTable{T}

Load fuel models from the default ELMFIRE fuel_models.csv bundled with the package.
"""
function load_fuel_models(::Type{T}) where {T<:AbstractFloat}
    # Try to find the fuel_models.csv in the package or elmfire submodule
    candidates = [
        joinpath(@__DIR__, "..", "data", "fuel_models.csv"),
        joinpath(@__DIR__, "..", "elmfire", "build", "source", "fuel_models.csv"),
    ]

    for path in candidates
        if isfile(path)
            return load_fuel_models(T, path)
        end
    end

    error("Could not find fuel_models.csv. Please specify the path explicitly.")
end

load_fuel_models() = load_fuel_models(DefaultFloat)


#-----------------------------------------------------------------------------#
#                           Standard Fuel Models
#-----------------------------------------------------------------------------#
"""
    STANDARD_FUEL_MODELS_DATA

Hardcoded standard fuel models (FBFM 1-13 and NB) as tuples.
"""
const STANDARD_FUEL_MODELS_DATA = [
    # id, name, dynamic, W0_1hr, W0_10hr, W0_100hr, W0_herb, W0_woody, SIG_1hr, SIG_herb, SIG_woody, delta, mex_dead, hoc
    (1, "FBFM01", false, 0.034, 0.0, 0.0, 0.0, 0.0, 3500.0, 9999.0, 9999.0, 1.0, 0.12, 8000.0),
    (2, "FBFM02", false, 0.092, 0.046, 0.023, 0.023, 0.0, 3000.0, 1500.0, 9999.0, 1.0, 0.15, 8000.0),
    (3, "FBFM03", false, 0.138, 0.0, 0.0, 0.0, 0.0, 1500.0, 9999.0, 9999.0, 2.5, 0.25, 8000.0),
    (4, "FBFM04", false, 0.23, 0.184, 0.092, 0.0, 0.23, 2000.0, 9999.0, 1500.0, 6.0, 0.20, 8000.0),
    (5, "FBFM05", false, 0.046, 0.023, 0.0, 0.0, 0.092, 2000.0, 9999.0, 1500.0, 2.0, 0.20, 8000.0),
    (6, "FBFM06", false, 0.069, 0.115, 0.092, 0.0, 0.0, 1750.0, 9999.0, 9999.0, 2.5, 0.25, 8000.0),
    (7, "FBFM07", false, 0.052, 0.086, 0.069, 0.0, 0.017, 1750.0, 9999.0, 1550.0, 2.5, 0.40, 8000.0),
    (8, "FBFM08", false, 0.069, 0.046, 0.115, 0.0, 0.0, 2000.0, 9999.0, 9999.0, 0.2, 0.30, 8000.0),
    (9, "FBFM09", false, 0.134, 0.019, 0.007, 0.0, 0.0, 2500.0, 9999.0, 9999.0, 0.2, 0.25, 8000.0),
    (10, "FBFM10", false, 0.138, 0.092, 0.23, 0.0, 0.092, 2000.0, 9999.0, 1500.0, 1.0, 0.25, 8000.0),
    (11, "FBFM11", false, 0.069, 0.207, 0.253, 0.0, 0.0, 1500.0, 9999.0, 9999.0, 1.0, 0.15, 8000.0),
    (12, "FBFM12", false, 0.184, 0.644, 0.759, 0.0, 0.0, 1500.0, 9999.0, 9999.0, 2.3, 0.20, 8000.0),
    (13, "FBFM13", false, 0.322, 1.058, 1.288, 0.0, 0.0, 1500.0, 9999.0, 9999.0, 3.0, 0.25, 8000.0),
    (256, "NB", false, 0.0001, 0.0, 0.0, 0.0, 0.0, 9999.0, 9999.0, 9999.0, 0.01, 0.05, 1.0),
]

"""
    create_standard_fuel_table(::Type{T}) -> FuelModelTable{T}

Create a fuel model table with the standard FBFM 1-13 models plus non-burnable.
"""
function create_standard_fuel_table(::Type{T}) where {T<:AbstractFloat}
    table = FuelModelTable{T}()
    for (id, name, dyn, w1, w10, w100, wh, ww, s1, sh, sw, d, mex, hoc) in STANDARD_FUEL_MODELS_DATA
        raw = RawFuelModel{T}(id, name, dyn,
            T(w1), T(w10), T(w100), T(wh), T(ww),
            T(s1), T(sh), T(sw), T(d), T(mex), T(hoc))
        add_raw_model!(table, raw)
    end
    return table
end

create_standard_fuel_table() = create_standard_fuel_table(DefaultFloat)
