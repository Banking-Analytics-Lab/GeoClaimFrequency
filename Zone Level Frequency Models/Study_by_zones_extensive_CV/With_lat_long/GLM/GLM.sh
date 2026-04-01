#!/bin/bash
#SBATCH --job-name=glm_ext
#SBATCH --array=0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --account=def-cbravo
#SBATCH --output=Poisson_GLM_extended.%A_%a.out

module load gcc arrow/19.0.1
source ~/p3_env_gnn/bin/activate #using this environment in Nibi

OUTER_FOLD=${SLURM_ARRAY_TASK_ID}
DATA_DIR="/home/salfonso/projects/rrg-cbravo/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Stratified_division/alldata_with_fold_id.csv"
OUT_DIR="/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/With_lat_long/GLM/Results"
SCRIPT="/home/salfonso/projects/rrg-cbravo/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/GLM/GLM.py"

python "$SCRIPT" --data_withfolds_id "$DATA_DIR" --outer_fold ${OUTER_FOLD} --out_dir "$OUT_DIR" \
    --num_vars "['ageph_mean','ageph_median','ageph_std','bm_mean','bm_median','bm_std','power_mean','power_median','power_std','agec_mean','agec_median','agec_std','coverage_TPL_prop','coverage_TPL+_prop','sex_female_prop','fuel_diesel_prop','use_private_prop','fleet_0_prop','lat','long']"
