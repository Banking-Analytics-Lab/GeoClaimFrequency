#!/bin/bash
#SBATCH --job-name=nomic_creation_emb
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=<your-allocation>
#SBATCH --output=Nomic_creation_emb.out

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate 

python creation_nomic3km_emb.py\
  --data_withfolds_csv "${WORK_DIR:-data/Data_creation}/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv     "${INPUT_DIR:-data}/unique_locations_BeMTPL97.csv" \
  --img_root           "${IMAGES_DIR:-data/Images_Ortho_95}" \
  --local_model_dir    "${WORK_DIR:-data/Data_creation}/Embeddings_nomic_v15/nomic_v15_local" \
  --batch_size 128 --num_workers 1 \
  --skip_missing \
  --out_parquet "${WORK_DIR:-data/Data_creation}/Embeddings_nomic_v15/augmented_with_nomic3km.parquet"  \
  --also_csv
