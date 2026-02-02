#=
Validation Metrics for Marshall Fire Calibration
=================================================

Provides metrics for comparing simulated fire perimeters to observed data.
All metrics operate on BitMatrix representations of burned areas.
=#

#-----------------------------------------------------------------------------#
#                           Similarity Metrics
#-----------------------------------------------------------------------------#

"""
    sorensen(simulated::BitMatrix, observed::BitMatrix) -> Float64

Compute Sorensen-Dice coefficient: 2|A∩B| / (|A| + |B|)

Range: [0, 1] where 1 indicates perfect overlap.
Also known as Dice coefficient or F1 score.
"""
function sorensen(simulated::BitMatrix, observed::BitMatrix)
    intersection = count(simulated .& observed)
    total = count(simulated) + count(observed)
    return total > 0 ? 2.0 * intersection / total : 1.0
end

"""
    jaccard(simulated::BitMatrix, observed::BitMatrix) -> Float64

Compute Jaccard index (Intersection over Union): |A∩B| / |A∪B|

Range: [0, 1] where 1 indicates perfect overlap.
More stringent than Sorensen coefficient.
"""
function jaccard(simulated::BitMatrix, observed::BitMatrix)
    intersection = count(simulated .& observed)
    union_count = count(simulated .| observed)
    return union_count > 0 ? intersection / union_count : 1.0
end

"""
    kappa(simulated::BitMatrix, observed::BitMatrix) -> Float64

Compute Cohen's Kappa coefficient for agreement.

Range: [-1, 1] where 1 indicates perfect agreement,
0 indicates chance agreement, and negative values indicate
less agreement than chance.
"""
function kappa(simulated::BitMatrix, observed::BitMatrix)
    n = length(simulated)

    # Confusion matrix elements
    tp = count(simulated .& observed)           # True positives
    tn = count(.!simulated .& .!observed)       # True negatives
    fp = count(simulated .& .!observed)         # False positives
    fn = count(.!simulated .& observed)         # False negatives

    # Observed agreement
    po = (tp + tn) / n

    # Expected agreement by chance
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n^2

    # Kappa
    return pe < 1.0 ? (po - pe) / (1.0 - pe) : 1.0
end

#-----------------------------------------------------------------------------#
#                           Error Metrics
#-----------------------------------------------------------------------------#

"""
    commission_error(simulated::BitMatrix, observed::BitMatrix) -> Float64

Compute commission error (false positive rate): FP / simulated area

Measures over-prediction of fire spread.
Range: [0, 1] where 0 indicates no over-prediction.
"""
function commission_error(simulated::BitMatrix, observed::BitMatrix)
    fp = count(simulated .& .!observed)
    sim_total = count(simulated)
    return sim_total > 0 ? fp / sim_total : 0.0
end

"""
    omission_error(simulated::BitMatrix, observed::BitMatrix) -> Float64

Compute omission error (false negative rate): FN / observed area

Measures under-prediction of fire spread.
Range: [0, 1] where 0 indicates no under-prediction.
"""
function omission_error(simulated::BitMatrix, observed::BitMatrix)
    fn = count(.!simulated .& observed)
    obs_total = count(observed)
    return obs_total > 0 ? fn / obs_total : 0.0
end

"""
    area_error(simulated::BitMatrix, observed::BitMatrix) -> Float64

Compute relative error in burned area: (sim - obs) / obs

Positive values indicate over-prediction, negative indicates under-prediction.
"""
function area_error(simulated::BitMatrix, observed::BitMatrix)
    sim_count = count(simulated)
    obs_count = count(observed)
    return obs_count > 0 ? (sim_count - obs_count) / obs_count : 0.0
end

"""
    absolute_area_error(simulated::BitMatrix, observed::BitMatrix) -> Float64

Compute absolute relative error in burned area: |sim - obs| / obs
"""
function absolute_area_error(simulated::BitMatrix, observed::BitMatrix)
    return abs(area_error(simulated, observed))
end

#-----------------------------------------------------------------------------#
#                           Cell Count Metrics
#-----------------------------------------------------------------------------#

"""
    true_positives(simulated::BitMatrix, observed::BitMatrix) -> Int

Count cells that are correctly predicted as burned.
"""
function true_positives(simulated::BitMatrix, observed::BitMatrix)
    return count(simulated .& observed)
end

"""
    false_positives(simulated::BitMatrix, observed::BitMatrix) -> Int

Count cells that are incorrectly predicted as burned (over-prediction).
"""
function false_positives(simulated::BitMatrix, observed::BitMatrix)
    return count(simulated .& .!observed)
end

"""
    false_negatives(simulated::BitMatrix, observed::BitMatrix) -> Int

Count cells that are incorrectly predicted as unburned (under-prediction).
"""
function false_negatives(simulated::BitMatrix, observed::BitMatrix)
    return count(.!simulated .& observed)
end

"""
    true_negatives(simulated::BitMatrix, observed::BitMatrix) -> Int

Count cells that are correctly predicted as unburned.
"""
function true_negatives(simulated::BitMatrix, observed::BitMatrix)
    return count(.!simulated .& .!observed)
end

#-----------------------------------------------------------------------------#
#                           Summary Statistics
#-----------------------------------------------------------------------------#

"""
    ValidationResult

Container for all validation metrics.
"""
struct ValidationResult
    sorensen::Float64
    jaccard::Float64
    kappa::Float64
    commission_error::Float64
    omission_error::Float64
    area_error::Float64
    true_positives::Int
    false_positives::Int
    false_negatives::Int
    true_negatives::Int
    simulated_cells::Int
    observed_cells::Int
end

"""
    compute_validation_metrics(simulated::BitMatrix, observed::BitMatrix) -> ValidationResult

Compute all validation metrics and return as a summary struct.
"""
function compute_validation_metrics(simulated::BitMatrix, observed::BitMatrix)
    return ValidationResult(
        sorensen(simulated, observed),
        jaccard(simulated, observed),
        kappa(simulated, observed),
        commission_error(simulated, observed),
        omission_error(simulated, observed),
        area_error(simulated, observed),
        true_positives(simulated, observed),
        false_positives(simulated, observed),
        false_negatives(simulated, observed),
        true_negatives(simulated, observed),
        count(simulated),
        count(observed)
    )
end

"""
    print_validation_summary(result::ValidationResult; cellsize_ft::Float64=98.4)

Print a formatted summary of validation metrics.
"""
function print_validation_summary(result::ValidationResult; cellsize_ft::Float64=98.4)
    # Convert cells to acres (1 acre = 43,560 ft²)
    cell_area_ft2 = cellsize_ft^2
    to_acres(cells) = cells * cell_area_ft2 / 43560.0

    println("\n" * "="^60)
    println("Validation Summary")
    println("="^60)

    println("\nSimilarity Metrics:")
    println("  Sorensen-Dice:    $(round(result.sorensen, digits=4))")
    println("  Jaccard Index:    $(round(result.jaccard, digits=4))")
    println("  Cohen's Kappa:    $(round(result.kappa, digits=4))")

    println("\nError Metrics:")
    println("  Commission Error: $(round(100*result.commission_error, digits=2))%")
    println("  Omission Error:   $(round(100*result.omission_error, digits=2))%")
    println("  Area Error:       $(round(100*result.area_error, digits=2))%")

    println("\nConfusion Matrix:")
    println("  True Positives:   $(result.true_positives) cells")
    println("  False Positives:  $(result.false_positives) cells")
    println("  False Negatives:  $(result.false_negatives) cells")
    println("  True Negatives:   $(result.true_negatives) cells")

    println("\nBurned Area:")
    println("  Simulated: $(result.simulated_cells) cells ($(round(to_acres(result.simulated_cells), digits=1)) acres)")
    println("  Observed:  $(result.observed_cells) cells ($(round(to_acres(result.observed_cells), digits=1)) acres)")

    println("="^60)
end

"""
    validate_simulation(state::Elmfire.FireState, observed::BitMatrix; cellsize_ft::Float64=98.4)

Convenience function to validate a simulation result against observed data.
"""
function validate_simulation(state, observed::BitMatrix; cellsize_ft::Float64=98.4)
    result = compute_validation_metrics(state.burned, observed)
    print_validation_summary(result; cellsize_ft=cellsize_ft)
    return result
end

#-----------------------------------------------------------------------------#
#                           Visualization Helpers
#-----------------------------------------------------------------------------#

"""
    create_error_map(simulated::BitMatrix, observed::BitMatrix) -> Matrix{Int}

Create an error map for visualization:
- 0: True negative (neither simulated nor observed)
- 1: True positive (both simulated and observed)
- 2: False positive (simulated but not observed) - commission error
- 3: False negative (observed but not simulated) - omission error
"""
function create_error_map(simulated::BitMatrix, observed::BitMatrix)
    ncols, nrows = size(simulated)
    error_map = zeros(Int, ncols, nrows)

    for ix in 1:ncols
        for iy in 1:nrows
            if simulated[ix, iy] && observed[ix, iy]
                error_map[ix, iy] = 1  # True positive
            elseif simulated[ix, iy] && !observed[ix, iy]
                error_map[ix, iy] = 2  # False positive (commission)
            elseif !simulated[ix, iy] && observed[ix, iy]
                error_map[ix, iy] = 3  # False negative (omission)
            end
            # else 0 = true negative
        end
    end

    return error_map
end
