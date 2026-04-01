#!/bin/bash
#SBATCH --job-name=merge_all
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=def-cbravo
#SBATCH --output=merge_all.%j.out

module load python/3.11 proj gdal geos
source ~/p3_env_nvl_test/bin/activate

IN_ROOT='/home/salfonso/projects/def-cbravo/salfonso/Belgian/Study_by_zones/Preprocessing_byzones/Aggregating_data_BeMTPL97/BeMTPL_Agg.csv' # path where train and test are
OUT_ROOT="/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/With_osm14_corine2000"
OSM="/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/OSM_2014_CORINE_2000/osm_features_OSM2014_maskCLC2000_faster3.csv"

mkdir -p "$OUT_ROOT/all" 

declare -a RADII=("ALL")

# Train
for r in "${RADII[@]}"; do
  python /home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_6_5CV/Data_osm14_corine2000/Data_merge.py \
    "$IN_ROOT" "all" "$r" "$OSM" \
    "$OUT_ROOT/all/Data_all_osm14_clc2000_${r}.csv"
done
