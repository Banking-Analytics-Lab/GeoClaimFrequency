
# Script Description

This folder contains the script:

- `osm14corine2000_feature_extract.py`

This script constructs environmental features for each postcode using **OpenStreetMap (OSM) 2014** data combined with **CORINE Land Cover 2000**. Features are computed at multiple spatial radii.

---

## Data Sources

### OpenStreetMap (OSM 2014)

- Dataset: `belgium-140101.osm.pbf`
- Source: https://download.geofabrik.de/europe/belgium.html

OSM features are used to extract geographic and infrastructure-related information.

Useful references:
- https://wiki.openstreetmap.org/wiki/Map_features
- https://wiki.openstreetmap.org/wiki/Key:highway

---

### CORINE Land Cover 2000

- Source: https://land.copernicus.eu/en/products/corine-land-cover/clc-2000
- Viewer: https://land.copernicus.eu/en/map-viewer?dataset=6704f90ca82e4f228a46111519f8978e

CORINE 2000 is a land cover inventory with 44 thematic classes for the reference year 2000.


