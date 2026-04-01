#!/bin/bash
#SBATCH --job-name=Embedding_creation_out1_R3
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/Embeddings_ResNet18/Augmented_data_with_embeddings/outer1/Dataug_out1.out
set -euo pipefail

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate

python /home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/Image_Emb_ResNet18/Augmented_data_by_folds.py  \
  --data_withfolds_id "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/unique_locations_BeMTPL97.csv" \
  --outer_fold 1 \
  --img_root "/home/salfonso/scratch/Belgian/Images_Ortho_95" \
  --radius_km "3" \
  --weights_path "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/ONLY_images/Neigh_3/Refit_model/outer1/refit_model_outer1.pth" \
  --out_path "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/Embeddings_ResNet18/Augmented_data_with_embeddings/outer1/data_withfolds_id_withEmb_outer1.csv" \
  --norm_json "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/ONLY_images/Neigh_3/Refit_model/outer1/image_norm_outer1.json"
  
