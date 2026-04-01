#!/bin/bash
#SBATCH --job-name=refit_img_out0_RALL
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:30:00
#SBATCH --output=/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/ONLY_images/ALL_radii/Refit_model/outer0/Refit_out0.out
set -euo pipefail

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate

python /home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/Imagenes_DNN/MResNet18_multiRadii_ONLY_images_refit.py \
  --data_withfolds_id "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/unique_locations_BeMTPL97.csv" \
  --outer_fold 0 \
  --out_dir /home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/ONLY_images/ALL_radii/Refit_model/outer0 \
  --img_root "/home/salfonso/scratch/Belgian/Images_Ortho_95" \
  --radii "0.5,1,3" \
  --weights_path "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Models_frequency/Poisson_Assumption/DNN_images_osm_standard/ResNet18/resnet18_weights.pth" \
  --batch_size 16 --hidden_dim 128 --epochs 60 \
  --lr 0.0001 --dropout 0.3 --optimizer adamw --weight_decay 0.0001 \
  --seed 611
