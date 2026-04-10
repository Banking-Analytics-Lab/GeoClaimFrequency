"""
Accident-feature extraction (Belgium 1997)

- Vectors:      OSM snapshot 2014  (roads/POIs/buildings)
- Temporal mask: CORINE 2000 (raster, 100 m)  artificial surfaces
- Radii:        0.5 km, 1 km, 3 km, 5km
- CRS:          EPSG:31370 (Belgian Lambert 72)  
- Output:       One wide CSV with per-km^2 metrics and suffixes per radius
"""

import os
import re
import time
import warnings
from typing import Iterable, Dict, List

import numpy as np
import pandas as pd
import geopandas as gpd
import functools
from shapely import contains
from shapely.geometry import Point, shape
from pyrosm import OSM
from shapely.prepared import prep

import rasterio
import rasterio.features

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
print = functools.partial(print, flush=True)

PATH_LOCATIONS_CSV = "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/OSM_2014_CORINE_2000/unique_locations_BeMTPL97.csv"
PATH_OSM_PBF       = "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/OMS/belgium-140101.osm.pbf"         # ~2014
PATH_CLC2000_TIF   = "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/OSM_2014_CORINE_2000/U2006_CLC2000_V2020_20u1.tif"      # EPSG:3035
OUT_CSV            = "osm_features_OSM2014_maskCLC2000_faster3.csv"

# parameters
CRS_METRIC = 31370                            # Belgian Lambert 72 (meters)
RADII_M    = [500, 1000, 3000, 5000]                # 0.5 km, 1 km, 3 km, 5km
# “stable” road classes (safer wrt 1997 reality)
STABLE_HWY = {"motorway", "trunk", "primary", "secondary", "tertiary"}

# CORINE artificial-surface codes (full set)
CLC_ARTIFICIAL_ALL = set(range(1, 12)) #{111, 112, 121, 122, 123, 124, 131, 132, 133, 141, 142}
#https://clc.gios.gov.pl/doc/clc/CLC_Legend_EN.pd
CLC_BUILTUP_CORE   = set(range(1, 6)) #{111, 112, 121, 122, 123, 124} # Core urban & transport

# select which mask to use:
USE_CORE_BUILTUP_ONLY = False
ARTIF_CODES = CLC_BUILTUP_CORE if USE_CORE_BUILTUP_ONLY else CLC_ARTIFICIAL_ALL


# small helpers
def first_number(x):
    """Extract first integer from a string (for 'maxspeed', 'lanes', etc.).
    This function will select the first number in  OSM tag
    e.g., "50", "70 km/h", "2 lanes"."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"\d+", str(x))
    return float(m.group(0)) if m else np.nan


def per_km2(value: float, area_km2: float) -> float:
    """Safe division to normalize counts by km²."""
    return value / max(area_km2, 1e-9)

# This is transforming all the geometries to points in the OSM file
def pois_as_points(pois_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a POI GeoDataFrame where all geometries are points.
       Non-points (lines/polys) become representative points (always inside)."""
    gdf = pois_gdf[pois_gdf.geometry.notna()].copy()

    is_point = gdf.geom_type.isin(["Point", "MultiPoint"])
    pts = gdf[is_point].copy()

    nonpts = gdf[~is_point].copy()
    if len(nonpts):
        nonpts["geometry"] = nonpts.representative_point()

    out = pd.concat([pts, nonpts], ignore_index=True)

    # explode any MultiPoint into single points
    if "MultiPoint" in out.geom_type.unique():
        out = out.explode(index_parts=False, ignore_index=True)

    return out



# load locations
t0 = time.time()
print("Loading locations…")
df_loc = pd.read_csv(PATH_LOCATIONS_CSV)  #file with postcode, lat, long
# A GeoDataFrame is basically like a pandas DataFrame that contains at least one dedicated column for storing geometries
gdf_loc = gpd.GeoDataFrame(
    df_loc,
    geometry=gpd.points_from_xy(df_loc["long"], df_loc["lat"]),
    crs=4326,                      # WGS84 input, this is lat and long
).to_crs(CRS_METRIC)               # -> Belgian meters
print(f"  locations: {len(gdf_loc)} rows")
# verify that the projection was done 
assert gdf_loc.crs.to_epsg() == CRS_METRIC, "Locations must be in EPSG:31370"

MAX_R = max(RADII_M)
#Union of all the geometris of the 583 points and extend a maximum of 5000 m
study_extent = gdf_loc.unary_union.buffer(MAX_R).envelope  #

# CORINE 2000 mask 
print("Building CORINE 2000 artificial-surface mask…")
with rasterio.open(PATH_CLC2000_TIF) as src:
    clc_ma = src.read(1, masked=True)
    transform = src.transform
    crs_clc = src.crs
    nodata    = src.nodata

# For reporting, look only at valid pixels
vals_valid = np.unique(clc_ma.compressed().astype(int))
print("CLC codes (sample):", vals_valid[:20], "... min:", vals_valid.min(), "max:", vals_valid.max(), "nodata:", nodata)

# Detect coding scheme on valid pixels only
lvl3_artificial = {111,112,121,122,123,124,131,132,133,141,142}
index44_all     = set(range(1, 45))
lvl1_all        = {1,2,3,4,5}

if (len(set(vals_valid) & lvl3_artificial) >= 3) or (vals_valid.max() >= 100):
    scheme = "lvl3"
elif (vals_valid.min() >= 1) and (vals_valid.max() <= 44) and set(vals_valid).issubset(index44_all):
    scheme = "index44"
elif set(vals_valid).issubset(lvl1_all):
    scheme = "lvl1"
else:
    # fall back to index44 if it is like 1..44 
    raise ValueError("Unrecognized CORINE coding after removing NODATA; inspect printed codes")

print("Detected CORINE scheme:", scheme)
#This was to recognize the codes
if scheme == "lvl3":
    CLC_ARTIFICIAL_ALL = lvl3_artificial
    CLC_BUILTUP_CORE   = {111,112,121,122,123,124}
elif scheme == "index44":
    # CLC Level-3 classes are reindexed to 1..44  (1..11 = artificial surfaces)
    CLC_ARTIFICIAL_ALL = set(range(1, 12))   # 1..11
    CLC_BUILTUP_CORE   = set(range(1, 6))    # 1..5 = core built-up/transport
elif scheme == "lvl1":
    CLC_ARTIFICIAL_ALL = {1}
    CLC_BUILTUP_CORE   = {1}

ARTIF_CODES = CLC_BUILTUP_CORE if USE_CORE_BUILTUP_ONLY else CLC_ARTIFICIAL_ALL

# Build mask strictly on valid pixels, NODATA stays 0
mask_bool = (~clc_ma.mask) & np.isin(clc_ma.filled(0).astype(int), list(ARTIF_CODES)) & (clc_ma.filled(0) != 0)
mask = mask_bool.astype(np.uint8)
print("Mask coverage fraction:", float(mask.mean()))

# Polygonize only once
print("Vectorizing CORINE mask…")
shapes = rasterio.features.shapes(mask, transform=transform)
artificial_polys = [shape(geom) for geom, val in shapes if val == 1]
gdf_art = gpd.GeoDataFrame(geometry=artificial_polys, crs=crs_clc)

# reproject to 31370 
gdf_art = gdf_art.to_crs(CRS_METRIC)

# clip to the study extent to remove 99% of polygons outside Belgium of interest
gdf_art = gpd.clip(gdf_art, study_extent)

# dissolve to a single multipolygon
gdf_art = gdf_art.dissolve()
gdf_art["geometry"] = gdf_art.buffer(0)
gdf_art = gdf_art.explode(index_parts=False)
gdf_art["geometry"] = gdf_art.simplify(5)  
gdf_art = gdf_art[~gdf_art.geometry.is_empty].set_crs(CRS_METRIC)
art_union = gdf_art.unary_union  
art_prep = prep(art_union)
print("  CORINE mask ready (simplified & prepared).")
## reusable GeoDataFrame for mask operations
gdf_mask = gpd.GeoDataFrame(geometry=[art_union], crs=CRS_METRIC)

# OSM vectors
print("Loading OSM (roads / POIs / buildings)…")
osm = OSM(PATH_OSM_PBF)

# Load minimal columns only
roads_full = osm.get_network(network_type="all")
#pois_full  = osm.get_pois()
blds_full  = osm.get_buildings()
traffic_filters = {
    "custom_filter": {
        "highway": ["traffic_signals"],   # nodes
        "traffic_calming": True,          # bumps, humps, tables...
        "traffic_sign": True,             # includes 'stop'
        "amenity": True,                  # parking, fuel, hospital, school...
        "shop": True,
        "tourism": True
    },
    "filter_type": "keep",
    "keep_nodes": True,
    "keep_ways": True,
    "keep_relations": False
}
raw = osm.get_data_by_custom_criteria(**traffic_filters)
pois_full = gpd.GeoDataFrame(raw, geometry="geometry", crs=4326)

# Project once
roads_full = roads_full.to_crs(CRS_METRIC)
pois_full  = pois_full.to_crs(CRS_METRIC)
pois_pts = pois_as_points(pois_full) # convert all to points
blds_full  = blds_full.to_crs(CRS_METRIC)
# keeping only polygonal features
blds_full = blds_full[blds_full.geom_type.isin(["Polygon","MultiPolygon"])].copy()

# fixing invalids
blds_full["geometry"] = blds_full.buffer(0)
blds_full = blds_full.explode(index_parts=False)
blds_full = blds_full[~blds_full.geometry.is_empty]

# Shrink to study extent early
roads_full = gpd.clip(roads_full, study_extent)
pois_full  = gpd.clip(pois_pts,  study_extent)
blds_full  = gpd.clip(blds_full,  study_extent)

# Keep the interested columns
roads = roads_full[["geometry","highway","oneway","lit","lanes","maxspeed","junction"]].copy()
pois = pois_full.copy()
print(f'Columns of the points')
print(sorted(pois_full.columns.tolist()))
print(f'Columns of the blds')
print(sorted(blds_full.columns.tolist()))
blds_full  = blds_full[["geometry","building","building:levels","height","building:use","building:material"]].copy()

# Keep only stable highway classes
roads = roads[roads["highway"].isin(STABLE_HWY)].copy()
###
print("Before CORINE mask:")
print("  Roads:", len(roads_full))
print("  POIs :", len(pois_full))
print("  Blds :", len(blds_full))

# Premask once
print("Applying CORINE mask to OSM layers once… (this is the big win)")
# Intersect with art_union once so downstream only clips by buffers
roads = gpd.overlay(roads, gdf_mask, how="intersection")
r_time = time.time() - t0
print(f'Done the masking for the roads in {r_time:.2f} sec')
pois  = gpd.overlay(pois_full, gdf_mask, how="intersection")
p_time = time.time() - t0
print(f'Done the masking for the points in {p_time:.2f} sec')
blds_pts = blds_full.copy()
blds_pts["geometry"] = blds_pts.geometry.representative_point()

try:
    blds = gpd.sjoin(blds_pts, gdf_mask, predicate="intersects", how="inner").drop(columns=["index_right"])
except TypeError:
    # GeoPandas < 0.10 fallback
    blds = gpd.sjoin(blds_pts, gdf_mask, op="intersects", how="inner").drop(columns=["index_right"])

print("Buildings before mask:", len(blds_pts), "after mask:", len(blds))

# Build spatial index once (used later in buffers)
_ = blds.sindex
b_time = time.time() - t0
print(f'Done the masking for the buldings in {b_time:.2f} sec')

# after masking
print("After CORINE mask:")
print("  Roads:", len(roads))
print("  POIs :", len(pois))
print("  Blds :", len(blds))

print("Masked OSM ready. Main loop will only clip by buffers now.")
# Verify CRS and bounds
print("CRS:", roads.crs, pois.crs, blds.crs, gdf_art.crs, gdf_loc.crs)
print("roads bounds:", roads.total_bounds)
print("pois bounds:", pois.total_bounds)
print("blds bounds:", blds.total_bounds)
print("gdf_art bounds:", gdf_art.total_bounds)
print("locs bounds:", gdf_loc.total_bounds)
ver_t = time.time() - t0
print(f'Time until here {ver_t:.2f} sec')

# feature engine 
def features_for_buffer(center_geom, radius_m) -> Dict[str, float]:
    """
    Compute all features inside:
      - circular buffer (center_geom buffered by radius_m), and
      - inside CORINE-2000 artificial surfaces (mask), to “backdate” OSM2014.
    All densities normalized per km² where appropriate.
    """
    buf = center_geom.buffer(radius_m)                 # circular neighborhood
    area_km2 = buf.area / 1_000_000.0                  # m² -> km²

    # Clip each layer to the buffer first (fast), then intersect with Corin mask
    R = gpd.clip(roads, buf)  
    #P = gpd.clip(pois_pts, buf)
    P = gpd.clip(pois, buf)
    #B = gpd.clip(blds, buf)  
    idx = list(blds.sindex.query(buf, predicate="within"))
    B = blds.iloc[idx]

    #Roads 
    if len(R) == 0:
        # densities per km² -> 0.0 (no roads => zero density)
        road_len_km_per_km2 = 0.0
        intersection_count_per_km2 = 0.0
        roundabout_count_per_km2 = 0.0
        
        # shares/means over road length -> NaN (undefined when no roads)
        pct_len_major = np.nan
        pct_len_minor = np.nan
        oneway_share_len = np.nan
        lanes_mean = np.nan
        lit_share_len  = np.nan
        maxspeed_mean_kmh = np.nan
        
    else:
        R["len_km"] = R.length / 1000.0 
        total_len_km = R["len_km"].sum()
        road_len_km_per_km2 = per_km2(total_len_km, area_km2)

        # Percentage of major road density
        hi = R[R["highway"].isin({"motorway", "trunk", "primary"})]
        pct_len_major = hi["len_km"].sum() / max(total_len_km, 1e-9)

        # Percentage of minor road density
        mi = R[R["highway"].isin({"secondary", "tertiary"})]
        pct_len_minor = mi["len_km"].sum() / max(total_len_km, 1e-9)

        # oneway
        oneway_vals = R.get("oneway", pd.Series(index=R.index)).astype(str).str.lower()
        oneway_len_km = R.loc[oneway_vals.isin({"yes","true","1","-1"}), "len_km"].sum()
        oneway_share_len = oneway_len_km / total_len_km if total_len_km else np.nan

        # lit yes/true/1 as lit
        lit_vals = R.get("lit", pd.Series(index=R.index)).astype(str).str.lower()
        lit_len_km = R.loc[lit_vals.isin({"yes","true","1"}), "len_km"].sum()
        lit_share_len = lit_len_km / total_len_km if total_len_km else np.nan

        # Mean of the lanes
        R["lanes_num"] = pd.to_numeric(R.get("lanes", pd.Series(index=R.index)), errors="coerce")
        has_lanes = R["lanes_num"].notna()
        lanes_mean = np.average(R.loc[has_lanes, "lanes_num"], weights=R.loc[has_lanes, "len_km"]) if has_lanes.any() else np.nan
        maxspeed_mean_kmh = R.get("maxspeed", pd.Series(index=R.index)).apply(first_number).mean()

        # This takes all the vertices and look if they are repited 3 or more times...
        exploded = R.explode(index_parts=False)
        coords = exploded.geometry.apply(lambda g: list(g.coords) if g is not None else []).explode()
        if not coords.empty and isinstance(coords.iloc[0], (tuple, list)):
            coord_df = pd.DataFrame(coords.tolist(), columns=["x", "y"])
            intersection_count = int((coord_df.groupby(["x", "y"]).size() >= 3).sum())
        else:
            intersection_count = 0
        intersection_count_per_km2 = per_km2(intersection_count, area_km2)

        roundabout_count_per_km2 = per_km2(int(R.get("junction", pd.Series(index=R.index)).astype(str).str.lower().eq("roundabout").sum()), area_km2)

    #POIs 
    if len(P) == 0:
        traffic_signal_count_per_km2 = 0.0
        stop_sign_count_per_km2      = 0.0
        speed_bump_count_per_km2     = 0.0
        retail_count_per_km2         = 0.0
        tourism_count_per_km2        = 0.0
        parking_count_per_km2        = 0.0
        has_education = has_healthcare = has_fuel_station = 0
        school_count_per_km2 = 0.0
        healthcare_count_per_km2 = 0.0
        fuel_count_per_km2 = 0.0

    else:
        def tag_series(df, key):
            if key in df.columns:
                return df[key].astype(str).str.lower()
            if "tags" in df.columns:
                return df["tags"].apply(lambda d: str(d.get(key, "")).lower() if isinstance(d, dict) else "")
            return pd.Series("", index=df.index)

        highway          = tag_series(P, "highway")
        crossing         = tag_series(P, "crossing")
        traffic_sign     = tag_series(P, "traffic_sign")
        traffic_calming  = tag_series(P, "traffic_calming")
        amenity          = tag_series(P, "amenity")
        shop             = tag_series(P, "shop")
        tourism          = tag_series(P, "tourism")

        # traffic signals (avoid double-counting)
        traffic_signal_count = int(((highway == "traffic_signals") | (crossing == "traffic_signals")).sum())
        stop_sign_count = int((highway == "stop").sum()) + int(traffic_sign.str.split(";").apply(lambda xs: "stop" in [s.strip() for s in xs]).sum())
        # vertical calming only
        vertical_calming = {"bump", "hump", "table", "cushion", "rumble_strip", "mini_bumps"}
        speed_bump_count = int(traffic_calming.isin(vertical_calming).sum())

        retail_count  = int(shop.ne("").sum())
        tourism_count = int(tourism.ne("").sum())

        parking_count = int(amenity.isin(["parking", "parking_space"]).sum())

        # Education
        edu_vals = {"library", "kindergarten", "college", "school", "university"}
        education_count = int(amenity.isin(edu_vals).sum())
        has_education = int(education_count > 0)

        # Healthcare
        health_vals = {"hospital", "clinic"}
        healthcare_count = int(amenity.isin(health_vals).sum())
        has_healthcare = int(healthcare_count > 0)

        #Fuel stations
        fuel_vals = {"fuel"}
        fuel_count = int(amenity.isin(fuel_vals).sum())
        has_fuel_station = int(fuel_count > 0)

        traffic_signal_count_per_km2 = per_km2(traffic_signal_count, area_km2)
        stop_sign_count_per_km2      = per_km2(stop_sign_count, area_km2)
        speed_bump_count_per_km2     = per_km2(speed_bump_count, area_km2)
        retail_count_per_km2         = per_km2(retail_count, area_km2)
        tourism_count_per_km2        = per_km2(tourism_count, area_km2)
        parking_count_per_km2        = per_km2(parking_count, area_km2)
        school_count_per_km2 = per_km2(education_count, area_km2)
        healthcare_count_per_km2 = per_km2(healthcare_count, area_km2)
        fuel_count_per_km2 = per_km2(fuel_count, area_km2)

    # A more robust version with lower case cleaning and also managing info like 12m as 12
    # Buildings 
    if len(B) == 0:
        building_count_per_km2 = 0.0
        avg_building_levels = np.nan
        avg_building_height = np.nan
        building_use_diversity = 0
        building_material_diversity = 0
        # residential, commercial, industrial, mixed
        res_bld_per_km2 = com_bld_per_km2 = ind_bld_per_km2 = mix_bld_per_km2 = 0.0

    else:
        B = B.copy()

        # Count & density
        building_count = int(len(B))
        building_count_per_km2 = per_km2(building_count, area_km2)

        # Levels (numeric, tolerant)
        levels_ser = pd.to_numeric(B.get("building:levels", pd.Series(index=B.index)), errors="coerce")
        avg_building_levels = levels_ser.mean()

        # Height: parse numbers (handles e.g. "12 m")
        height_raw = B.get("height", pd.Series(index=B.index)).astype(str)
        height_num = pd.to_numeric(height_raw.str.extract(r"([-+]?\d*\.?\d+)")[0], errors="coerce")
        avg_building_height = height_num.mean()

        # Diversity: simple counts of distinct non-empty values
        use_ser = B.get("building:use", pd.Series(index=B.index)).astype(str).str.lower().replace({"": np.nan})
        building_use_diversity = int(use_ser.dropna().nunique())

        mat_ser = B.get("building:material", pd.Series(index=B.index)).astype(str).str.lower().replace({"": np.nan})
        building_material_diversity = int(mat_ser.dropna().nunique())

        # Broad building type buckets (robust to common aliases)
        bld = B.get("building", pd.Series(index=B.index)).astype(str).str.lower()

        res_keys = {"residential", "house", "detached", "semidetached_house", "terrace", "apartments"}
        com_keys = {"commercial", "retail", "office", "supermarket", "mall"}
        ind_keys = {"industrial", "warehouse", "factory", "workshop", "hangar"}

        res_bld = int(bld.isin(res_keys).sum())
        com_bld = int(bld.isin(com_keys).sum())
        ind_bld = int(bld.isin(ind_keys).sum())

        # Mixed useexplicit 'mixed' OR building:use looks mixed (e.g., "residential;retail")
        use_has_mixed = use_ser.fillna("").str.contains(r"\bmixed\b|;", regex=True)
        mix_bld = int((bld.eq("mixed") | use_has_mixed).sum())

        # Per-km² outputs
        res_bld_per_km2 = per_km2(res_bld, area_km2)
        com_bld_per_km2 = per_km2(com_bld, area_km2)
        ind_bld_per_km2 = per_km2(ind_bld, area_km2)
        mix_bld_per_km2 = per_km2(mix_bld, area_km2)

    # collect results
    return dict(
        road_len_km_per_km2=road_len_km_per_km2,
        pct_len_major=pct_len_major,
        pct_len_minor=pct_len_minor,
        oneway_share_len=oneway_share_len,
        lit_share_len=lit_share_len,
        lanes_mean=lanes_mean,
        maxspeed_mean_kmh=maxspeed_mean_kmh,
        intersection_count_per_km2=intersection_count_per_km2,
        roundabout_count_per_km2=roundabout_count_per_km2,
        traffic_signal_count_per_km2=traffic_signal_count_per_km2,
        stop_sign_count_per_km2=stop_sign_count_per_km2,
        speed_bump_count_per_km2=speed_bump_count_per_km2,
        building_count_per_km2=building_count_per_km2,
        avg_building_levels=avg_building_levels,
        avg_building_height=avg_building_height,
        building_use_diversity=building_use_diversity,
        building_material_diversity=building_material_diversity,
        residential_building_count_per_km2=res_bld_per_km2,
        commercial_building_count_per_km2=com_bld_per_km2,
        industrial_building_count_per_km2=ind_bld_per_km2,
        mixed_use_building_count_per_km2=mix_bld_per_km2,
        retail_count_per_km2=retail_count_per_km2,
        tourism_count_per_km2=tourism_count_per_km2,
        parking_count_per_km2=parking_count_per_km2,
        has_education=has_education,
        has_healthcare=has_healthcare,
        has_fuel_station=has_fuel_station,
        school_count_per_km2 = school_count_per_km2,
        healthcare_count_per_km2 = healthcare_count_per_km2,
        fuel_count_per_km2 = fuel_count_per_km2
    )


# main loop
print("Computing features…")
rows = []
for i, loc in gdf_loc.iterrows():
    base = {"postcode": loc["postcode"], "lat": loc["lat"], "long": loc["long"]}
    for r in RADII_M:
        feats = features_for_buffer(loc.geometry, r)
        dt = time.time() - t0  # elapsed seconds
        print(f"[{i}/{len(gdf_loc)}] radius {r} done in {dt:.2f} sec")        
        base.update({f"{k}_r{r}": v for k, v in feats.items()})
    rows.append(base)
# assemble to DataFrame
df = pd.DataFrame(rows)

# add provenance columns
df["osm_snapshot_year"] = 2014
df["corine_epoch"]      = 2000
df["corine_mask_codes"] = "core" if USE_CORE_BUILTUP_ONLY else "artificial_all"

# save
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

# Look the summary statiatics of the new features!
print('\n--- Summary ---')
print(df.info(max_cols=1000, show_counts=True, memory_usage="deep"))
#print(df.info())
print(df.describe(include='all'))

columns_int = df.columns
for col in columns_int:
    print(f'\nValue counts for column: {col}')
    print(df[col].value_counts())


print(f"Total time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-t0))}")
