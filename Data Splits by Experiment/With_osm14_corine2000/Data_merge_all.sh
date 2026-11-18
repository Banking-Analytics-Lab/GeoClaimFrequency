#!/bin/bash
#SBATCH --job-name=merge_all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=<your-allocation>
#SBATCH --output=merge_all.%j.out

module load python/3.11 proj gdal geos
source ~/p3_env_nvl_test/bin/activate

IN_ROOT="${INPUT_DIR:-data}/BeMTPL_Agg.csv" # path where train and test are
OUT_ROOT="${WORK_DIR:-data/Data_creation}/With_osm14_corine2000"
OSM="${INPUT_DIR:-data}/OSM_2014_CORINE_2000/osm_features_OSM2014_maskCLC2000_faster3.csv"

mkdir -p "$OUT_ROOT/all" 

declare -a RADII=("ALL")

# Train
for r in "${RADII[@]}"; do
  python "${CODE_DIR:-.}/Data Splits by Experiment/Data_merge.py" \
    "$IN_ROOT" "all" "$r" "$OSM" \
    "$OUT_ROOT/all/Data_all_osm14_clc2000_${r}.csv"
done
