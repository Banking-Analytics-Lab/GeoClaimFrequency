import pandas as pd
import sys
import warnings
import functools

pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")
print = functools.partial(print, flush=True)

def merge_osm_features(base_df, osm_df, save_path=None, precision=5, selected_columns=None):
    base_df = base_df.copy()
    if "Unnamed: 0" in base_df.columns:
        base_df = base_df.drop("Unnamed: 0", axis=1)

    osm_df = osm_df.copy()
    # join key types match
    if "postcode" not in base_df.columns or "postcode" not in osm_df.columns:
        raise ValueError("Both base and OSM CSVs must have column 'postcode'")
    base_df["postcode"] = base_df["postcode"].astype(str)
    osm_df["postcode"]  = osm_df["postcode"].astype(str)

    # choose OSM columns
    if selected_columns is not None:
        cols_to_use = ["postcode"] + [c for c in selected_columns if c in osm_df.columns]
        missing = sorted(set(selected_columns) - set(cols_to_use))
        if missing:
            print(f"[warn] {len(missing)} OSM columns missing. Example: {missing[:5]}")
        osm_df = osm_df[cols_to_use]

    merged = pd.merge(base_df, osm_df, on="postcode", how="left")

    if save_path:
        merged.to_csv(save_path, index=False)
        print(f"[write] {save_path} (rows={len(merged)})")

    return merged

if __name__ == "__main__":
    # CLI: python merge_one.py <input_csv> <split_tag> <radius|ALL> <osm_csv> <output_csv>
    path_base_df = sys.argv[1]
    split_tag    = sys.argv[2]     # e.g., Train, Test, X_train, X_val, or all for no distinction
    radius       = sys.argv[3]     # 500|1000|3000|5000|ALL
    osm_csv      = sys.argv[4]
    out_csv      = sys.argv[5]

    base_df = pd.read_csv(path_base_df)
    osm_df  = pd.read_csv(osm_csv)

    base_features = [
        "road_len_km_per_km2_r500","intersection_count_per_km2_r500","roundabout_count_per_km2_r500",
        "traffic_signal_count_per_km2_r500","retail_count_per_km2_r500","tourism_count_per_km2_r500",
        "parking_count_per_km2_r500","has_education_r500","has_healthcare_r500","has_fuel_station_r500",
        "school_count_per_km2_r500","healthcare_count_per_km2_r500","fuel_count_per_km2_r500"
    ]
    if radius == "ALL":
        feats = (base_features
                 + [c.replace("r500","r1000") for c in base_features]
                 + [c.replace("r500","r3000") for c in base_features]
                 + [c.replace("r500","r5000") for c in base_features])
    else:
        feats = [c.replace("r500", f"r{radius}") for c in base_features]

    merge_osm_features(base_df, osm_df, selected_columns=feats, save_path=out_csv)

