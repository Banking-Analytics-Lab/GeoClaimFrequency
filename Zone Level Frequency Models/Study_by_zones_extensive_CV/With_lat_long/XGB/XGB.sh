#!/bin/bash
#SBATCH --job-name=xgb_ext
#SBATCH --array=0-5
#SBATCH --cpus-per-task=3
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --account=<your-allocation>
#SBATCH --output=XGB_extended.%A_%a.out

module load gcc arrow/19.0.1
source ~/p3_env_gnn/bin/activate

OUTER_FOLD=${SLURM_ARRAY_TASK_ID}
DATA_DIR="${WORK_DIR:-data/Data_creation}/Stratified_division/alldata_with_fold_id.csv"
OUT_DIR="${RESULTS_DIR:-results}/With_lat_long/XGB/Results"
SCRIPT="${CODE_DIR:-.}/Zone Level Frequency Models/Models with Only tabular data/XGB.py"

python "$SCRIPT" \
  --data_withfolds_id "$DATA_DIR" \
  --outer_fold ${OUTER_FOLD} \
  --out_dir "$OUT_DIR" \
  --num_vars "['ageph_mean','ageph_median','ageph_std','bm_mean','bm_median','bm_std','power_mean','power_median','power_std','agec_mean','agec_median','agec_std','coverage_TPL_prop','coverage_TPL+_prop','sex_female_prop','fuel_diesel_prop','use_private_prop','fleet_0_prop','lat','long']" \
  --cat_vars "None" \
  --has_osm14 "no"
