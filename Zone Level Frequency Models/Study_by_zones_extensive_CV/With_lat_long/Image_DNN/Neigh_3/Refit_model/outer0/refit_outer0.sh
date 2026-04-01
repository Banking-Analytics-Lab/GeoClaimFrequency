#!/bin/bash
#SBATCH --job-name=Refit_out0_R3
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/With_lat_long/Image_DNN/Neigh_3/Refit_model/outer0/Refit_out0.out
set -euo pipefail

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate

python /home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/Imagenes_DNN/MResNet18_refit_single_config.py  \
  --data_withfolds_id "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/unique_locations_BeMTPL97.csv" \
  --outer_fold 0 \
  --out_dir "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/With_lat_long/Image_DNN/Neigh_3/Refit_model/outer0" \
  --img_root "/home/salfonso/scratch/Belgian/Images_Ortho_95" \
  --radius_km "3" \
  --num_vars "['ageph_mean','ageph_median','ageph_std','bm_mean','bm_median','bm_std','power_mean','power_median','power_std','agec_mean','agec_median','agec_std','coverage_TPL_prop','coverage_TPL+_prop','sex_female_prop','fuel_diesel_prop','use_private_prop','fleet_0_prop','lat','long']" \
  --cat_vars "None" \
  --weights_path "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Models_frequency/Poisson_Assumption/DNN_images_osm_standard/ResNet18/resnet18_weights.pth" \
  --batch_size 16 \
  --hidden_dim 128 \
  --epochs 15 \
  --lr 0.001 \
  --dropout 0.1 \
  --optimizer adamw \
  --weight_decay 0.0001 \
  --seed 611
