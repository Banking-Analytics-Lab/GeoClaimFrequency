#!/bin/bash
#SBATCH --job-name=nomic_creation_emb
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=def-cbravo
#SBATCH --output=Nomic_creation_emb.out

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate 

python creation_nomic3km_emb.py\
  --data_withfolds_csv "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv     "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/unique_locations_BeMTPL97.csv" \
  --img_root           "/home/salfonso/scratch/Belgian/Images_Ortho_95" \
  --local_model_dir    "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Embeddings_nomic_v15/nomic_v15_local" \
  --batch_size 128 --num_workers 1 \
  --skip_missing \
  --out_parquet "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Embeddings_nomic_v15/augmented_with_nomic3km.parquet"  \
  --also_csv
