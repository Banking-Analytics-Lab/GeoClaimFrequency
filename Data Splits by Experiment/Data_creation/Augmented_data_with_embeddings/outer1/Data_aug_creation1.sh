#!/bin/bash
#SBATCH --job-name=Embedding_creation_out1_R3
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=results/Embeddings_ResNet18/Augmented_data_with_embeddings/outer1/Dataug_out1.out
set -euo pipefail

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate

python "${CODE_DIR:-.}/Data Splits by Experiment/Augmented_data_by_folds.py"  \
  --data_withfolds_id "${WORK_DIR:-data/Data_creation}/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv "${INPUT_DIR:-data}/unique_locations_BeMTPL97.csv" \
  --outer_fold 1 \
  --img_root "${IMAGES_DIR:-data/Images_Ortho_95}" \
  --radius_km "3" \
  --weights_path "${RESULTS_DIR:-results}/ONLY_images/Neigh_3/Refit_model/outer1/refit_model_outer1.pth" \
  --out_path "${RESULTS_DIR:-results}/Embeddings_ResNet18/Augmented_data_with_embeddings/outer1/data_withfolds_id_withEmb_outer1.csv" \
  --norm_json "${RESULTS_DIR:-results}/ONLY_images/Neigh_3/Refit_model/outer1/image_norm_outer1.json"
  
