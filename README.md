# Elmfire.jl tutorial data

Cached input data for `docs/tutorials/marshall-fire.qmd`. This branch is an orphan
branch (no shared history with `main`), following the same pattern as
`benchmark-results`.

These files exist so the docs render is reproducible and does not depend on live
federal data services being available — the LANDFIRE Product Service in particular
fails intermittently, and one failed page aborts the entire Quarto project render.

## Files

| File | Source | Product |
|------|--------|---------|
| `marshall_fbfm13.tif` | [LANDFIRE Product Service](https://lfps.usgs.gov) | `LF2024_FBFM13` — 13 Anderson Fire Behavior Fuel Models |
| `marshall_elev.tif` | [LANDFIRE Product Service](https://lfps.usgs.gov) | `LF2020_Elev` — Elevation |
| `marshall_perimeter.geojson` | [WFIGS Interagency Fire Perimeters](https://data-nifc.opendata.arcgis.com/datasets/nifc::wfigs-interagency-fire-perimeters/about) | Marshall Fire final perimeter, US-CO |

Area of interest: `-105.260 39.924 -105.126 39.992` (Boulder County, Colorado).
Both rasters are 382 x 253 cells at 30 m in NAD83 CONUS Albers, and align exactly.

Retrieved 2026-08-31.

## Regenerating

`LF2025_FBFM13` covers only the SW and NW geoAreas, so the newest FBFM13 product
does not cover Colorado — pin the version explicitly rather than taking the latest.
The elevation layer is named `LF2020_Elev`, not `ELEV`.

```julia
using Landfire   # requires LANDFIRE_EMAIL
aoi = "-105.260 39.924 -105.126 39.992"
fbfm13 = get(Landfire.Dataset(Landfire.products(false; layer="LF2024_FBFM13"), aoi))
elev   = get(Landfire.Dataset(Landfire.products(false; layer="LF2020_Elev"), aoi))
```

Then commit the resulting GeoTIFFs to this branch and update the date above.
