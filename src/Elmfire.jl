module Elmfire

#-----------------------------------------------------------------------------#
#                               Exports
#-----------------------------------------------------------------------------#

# Constants
export PI, PIO180
export BTUPFT2MIN_TO_KWPM2, FT_TO_M, M_TO_FT

# Fuel Models
export RawFuelModel, FuelModel, FuelModelTable
export compute_fuel_model, add_raw_model!, get_fuel_model, isnonburnable
export load_fuel_models, create_standard_fuel_table, parse_fuel_model_line

# Rothermel Model
export SpreadResult, surface_spread_rate, moisture_damping
export EllipticalSpread, elliptical_spread, velocity_at_angle, velocity_components

# Level Set Solver
export half_superbee, limit_gradients
export compute_cfl_timestep, rk2_step!, level_set_step!
export compute_normal
export NarrowBand, tag_band!, untag_isolated!, get_active_cells
export initialize_phi!, initialize_circular_fire!

# Simulation
export wind_adjustment_factor, calculate_tanslp2
export FireState, grid_to_padded, padded_to_grid
export ignite!, ignite_point!, ignite_circle!
export simulate!, simulate_uniform!
export get_fire_perimeter, get_burned_area, get_burned_area_acres
export reset!  # Phase 3: For ensemble support

# Phase 2: Extended Simulation
export SimulationConfig, CanopyGrid, get_canopy_properties
export simulate_full!, simulate_full_uniform!

# Crown Fire Model
export CanopyProperties, CrownFireResult
export critical_fireline_intensity, crown_spread_rate
export combined_spread_rate, combined_fireline_intensity

# Spotting / Ember Transport
export SpottingParameters, SpotFire
export lognormal_params, sample_lognormal, sardoy_parameters
export compute_num_embers, transport_ember, generate_spot_fires
export SpotFireTracker, add_spot_fires!, get_ready_ignitions!

# Weather Interpolation
export ConstantWeather
export WeatherGrid, WeatherTimeSeries, WeatherInterpolator
export find_time_indices, interpolate_wind_direction
export create_grid_mapping, get_weather_at, create_constant_interpolator

# Phase 3: Geospatial I/O
export GeoMetadata, GeoRaster, LandscapeData
export read_geotiff, read_fuel_raster, read_dem
export compute_slope_aspect, read_landscape
export write_geotiff, write_fire_perimeter
export grid_to_geo, geo_to_grid, resample_to_match

# Phase 3: Monte Carlo Ensemble
export PerturbationConfig, EnsembleConfig
export EnsembleMember, EnsembleResult
export perturb_weather, perturb_ignition
export compute_burn_probability, compute_mean_arrival_time
export aggregate_ensemble_statistics!, run_ensemble!
export check_convergence, get_exceedance_probability, get_percentile_fire

# Phase 3: Parallel Execution
export ParallelConfig, ThreadLocalState
export create_thread_local_states, run_ensemble_threaded!
export BatchSimulationJob, BatchResult, run_batch_simulations
export parallel_map, parallel_reduce

# Phase 3: WUI Models
export WUIBuilding, WUIGrid, HamadaParameters, BuildingIgnitionResult
export compute_radiative_heat_flux, compute_view_factor
export building_ignition_probability, hamada_spread_probability
export update_wui_state!, get_wui_statistics
export create_building_grid

# Phase 3: Suppression Models
export SuppressionResource, ContainmentLine, SuppressionState
export add_resource!, construct_containment_line!
export apply_containment!, update_suppression_state!
export assign_resource!, plan_indirect_attack, plan_direct_attack
export simulate_with_suppression!, get_suppression_statistics


#-----------------------------------------------------------------------------#
#                               Includes
#-----------------------------------------------------------------------------#

using Random

include("constants.jl")
include("fuel_models.jl")
include("io.jl")
include("rothermel.jl")
include("level_set.jl")
include("crown_fire.jl")
include("spotting.jl")
include("weather.jl")
include("simulation.jl")

# Phase 3: New modules (order matters - geospatial first, then others that depend on simulation)
include("geospatial.jl")
include("ensemble.jl")
include("parallel.jl")
include("wui.jl")
include("suppression.jl")

end # module
