#!/bin/bash
#SBATCH --job-name=refit_img_out0_RALL
#SBATCH --gpus=a100_1g.5gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:30:00
#SBATCH --output=results/ONLY_images/ALL_radii/Refit_model/outer0/Refit_out0.out
set -euo pipefail

module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate

python "${CODE_DIR:-.}/Zone Level Frequency Models/Models with ONLY images/MResNet18_multiRadii_ONLY_images_refit.py" \
  --data_withfolds_id "${WORK_DIR:-data/Data_creation}/Stratified_division/alldata_with_fold_id.csv" \
  --unique_loc_csv "${INPUT_DIR:-data}/unique_locations_BeMTPL97.csv" \
  --outer_fold 0 \
  --out_dir "${RESULTS_DIR:-results}/ONLY_images/ALL_radii/Refit_model/outer0" \
  --img_root "${IMAGES_DIR:-data/Images_Ortho_95}" \
  --radii "0.5,1,3" \
  --weights_path "${INPUT_DIR:-data}/resnet18_weights.pth" \
  --batch_size 16 --hidden_dim 128 --epochs 60 \
  --lr 0.0001 --dropout 0.3 --optimizer adamw --weight_decay 0.0001 \
  --seed 611
