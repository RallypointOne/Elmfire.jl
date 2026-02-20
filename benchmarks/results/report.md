# Elmfire.jl Benchmark Report

**Date:** 2026-02-20 11:29

## System

| Property | Value |
|----------|-------|
| Julia | 1.12.5 |
| OS | macOS 26.2 |
| CPU | Apple M1 Pro |
| CPU Cores | 8 |
| RAM | 16.0 GB |
| GPU | Metal.MTL.MTLDeviceInstance (object of type AGXG13XDevice) |
| Elmfire | 1.0.0-DEV |
| KernelAbstractions | 0.9.40 |
| Metal | 1.9.2 |

## Configuration

| Parameter | Value |
|-----------|-------|
| Fuel Model | FBFM01 |
| Wind Speed | 15 mph (from west) |
| Sim Duration | 10 min |
| Cell Size | 10 ft |
| Runs per config | 5 (median reported) |

## Results

| Grid | Precision | Backend | Median (s) | Min (s) | Max (s) | Burned Cells |
|------|-----------|---------|------------|---------|---------|--------------|
| 100 | Float64 | CPU | 0.073 | 0.073 | 0.147 | 3800 |
| 100 | Float32 | CPU | 0.060 | 0.060 | 0.105 | 3941 |
| 100 | Float64 | KA.CPU | 0.133 | 0.125 | 0.150 | 3955 |
| 100 | Float32 | KA.CPU | 0.110 | 0.104 | 0.147 | 3850 |
| 100 | Float32 | Metal | 0.577 | 0.568 | 0.593 | 3710 |
| 256 | Float64 | CPU | 0.130 | 0.126 | 0.179 | 9052 |
| 256 | Float32 | CPU | 0.110 | 0.108 | 0.110 | 8946 |
| 256 | Float64 | KA.CPU | 0.152 | 0.141 | 0.153 | 9152 |
| 256 | Float32 | KA.CPU | 0.157 | 0.135 | 0.232 | 9064 |
| 256 | Float32 | Metal | 0.633 | 0.605 | 0.859 | 8787 |
| 512 | Float64 | CPU | 0.139 | 0.137 | 0.166 | 9704 |
| 512 | Float32 | CPU | 0.115 | 0.114 | 0.194 | 9334 |
| 512 | Float64 | KA.CPU | 0.183 | 0.167 | 0.223 | 8975 |
| 512 | Float32 | KA.CPU | 0.181 | 0.157 | 0.404 | 8878 |
| 512 | Float32 | Metal | 0.835 | 0.760 | 0.949 | 8752 |
| 1024 | Float64 | CPU | 0.146 | 0.136 | 0.185 | 9554 |
| 1024 | Float32 | CPU | 0.112 | 0.110 | 0.154 | 9451 |
| 1024 | Float64 | KA.CPU | 0.350 | 0.292 | 0.377 | 9393 |
| 1024 | Float32 | KA.CPU | 0.333 | 0.292 | 0.361 | 9144 |
| 1024 | Float32 | Metal | 0.800 | 0.765 | 0.910 | 9015 |

## Float32 vs Float64 Speedup

| Grid | Backend | Speedup (×) |
|------|---------|-------------|
| 100 | CPU | 1.22 |
| 100 | KA.CPU | 1.21 |
| 256 | CPU | 1.18 |
| 256 | KA.CPU | 0.97 |
| 512 | CPU | 1.20 |
| 512 | KA.CPU | 1.01 |
| 1024 | CPU | 1.31 |
| 1024 | KA.CPU | 1.05 |

## Metal GPU vs CPU Speedup (Float32)

| Grid | CPU (s) | Metal (s) | Speedup (×) |
|------|---------|-----------|-------------|
| 100 | 0.060 | 0.577 | 0.10 |
| 256 | 0.110 | 0.633 | 0.17 |
| 512 | 0.115 | 0.835 | 0.14 |
| 1024 | 0.112 | 0.800 | 0.14 |

## Plots

### Simulation Time by Grid Size
![Time by Grid Size](time_by_grid.png)

### Burned Cells by Grid Size
![Burned Cells](burned_cells.png)

### Float32 Speedup
![Float32 Speedup](float32_speedup.png)

## Notes

- **KA.CPU** benchmarks the GPU code path running on CPU threads via
  `KernelAbstractions.CPU()`, not an actual GPU.
- **Metal** runs on the Apple Silicon GPU. Only Float32 is supported.
- CPU and GPU paths use different update strategies (serial vs parallel RK2),
  so burned cell counts may differ. Both are valid.
