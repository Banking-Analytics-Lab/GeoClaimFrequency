#!/bin/bash
#SBATCH --job-name=Refit_out0_R3
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=results/With_lat_long_osm14corine2000/Image_DNN/Neigh_3/Refit_model/outer0/Refit_out0.out
set -euo pipefail

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate

python "${CODE_DIR:-.}/Zone Level Frequency Models/Models with Tabular and Images/MResNet18_refit_single_config.py"  \
  --data_withfolds_id "${WORK_DIR:-data/Data_creation}/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv "${INPUT_DIR:-data}/unique_locations_BeMTPL97.csv" \
  --outer_fold 0 \
  --out_dir "${RESULTS_DIR:-results}/With_lat_long_osm14corine2000/Image_DNN/Neigh_3/Refit_model/outer0" \
  --img_root "${IMAGES_DIR:-data/Images_Ortho_95}" \
  --radius_km "3" \
  --num_vars "['ageph_mean','ageph_median','ageph_std','bm_mean','bm_median','bm_std','power_mean','power_median','power_std','agec_mean','agec_median','agec_std','coverage_TPL_prop','coverage_TPL+_prop','sex_female_prop','fuel_diesel_prop','use_private_prop','fleet_0_prop','lat','long','road_len_km_per_km2_r5000', 'intersection_count_per_km2_r5000', 'roundabout_count_per_km2_r5000', 'traffic_signal_count_per_km2_r5000', 'retail_count_per_km2_r5000', 'tourism_count_per_km2_r5000', 'parking_count_per_km2_r5000', 'has_education_r5000', 'has_healthcare_r5000', 'has_fuel_station_r5000', 'school_count_per_km2_r5000', 'healthcare_count_per_km2_r5000', 'fuel_count_per_km2_r5000']" \
  --cat_vars "None" \
  --weights_path "${INPUT_DIR:-data}/resnet18_weights.pth" \
  --batch_size 32 \
  --hidden_dim 128 \
  --epochs 60 \
  --lr 0.0001 \
  --dropout 0.3 \
  --optimizer adamw \
  --weight_decay 0.0001 \
  --seed 611
