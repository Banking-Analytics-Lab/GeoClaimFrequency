#!/bin/bash
#SBATCH --job-name=refit_img_out0_R0.5
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=results/ONLY_images/Neigh_0.5/Refit_model/outer0/Refit_out0.out
set -euo pipefail

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate

python "${CODE_DIR:-.}/Zone Level Frequency Models/Models with ONLY images/MResNet18_refit_images_ONLY.py" \
  --data_withfolds_id "${WORK_DIR:-data/Data_creation}/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv "${INPUT_DIR:-data}/unique_locations_BeMTPL97.csv" \
  --outer_fold 0 \
  --out_dir "${RESULTS_DIR:-results}/ONLY_images/Neigh_0.5/Refit_model/outer0" \
  --img_root "${IMAGES_DIR:-data/Images_Ortho_95}" \
  --radius_km "0.5" \
  --weights_path "${INPUT_DIR:-data}/resnet18_weights.pth" \
  --batch_size 32 --hidden_dim 128 --epochs 15 \
  --lr 0.001 --dropout 0.3 --optimizer adamw --weight_decay 0.0001 \
  --seed 611
