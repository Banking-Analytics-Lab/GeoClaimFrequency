#!/bin/bash
#SBATCH --job-name=xgb_resnet18emb
#SBATCH --array=0-5 #0-5
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:30:00
#SBATCH --account=def-cbravo
#SBATCH --output=XGB_resnet18emb.%A_%a.out

set -euo pipefail

module load gcc arrow/19.0.1
source ~/p3_env_gnn/bin/activate

OUTER_FOLD=${SLURM_ARRAY_TASK_ID}

# Per-fold CSV produced earlier
BASE_DIR="/home/salfonso/projects/rrg-cbravo/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Embeddings_Resnet18"
DATA_DIR="${BASE_DIR}/outer${OUTER_FOLD}/data_withfolds_id_withEmb_outer${OUTER_FOLD}.csv"

OUT_DIR="/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/With_lat_long/XGB_ResNet18Emb/Results"
SCRIPT="/home/salfonso/projects/rrg-cbravo/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/XGB/XGB.py"

mkdir -p "$OUT_DIR"

# Quick sanity check
if [[ ! -r "$DATA_DIR" ]]; then
  echo "ERROR: CSV not readable: $DATA_DIR" >&2
  exit 1
fi

# Base numeric features
BASE_NUM_VARS="['ageph_mean','ageph_median','ageph_std','bm_mean','bm_median','bm_std','power_mean','power_median','power_std','agec_mean','agec_median','agec_std','coverage_TPL_prop','coverage_TPL+_prop','sex_female_prop','fuel_diesel_prop','use_private_prop','fleet_0_prop','lat','long']"

# Dynamically collect embedding columns for this fold
NUM_VARS_STR=$(
python - "$DATA_DIR" "$BASE_NUM_VARS" <<'PY'
import sys, json, ast, pandas as pd, re
csv_path = sys.argv[1]
base_list = ast.literal_eval(sys.argv[2])

cols = pd.read_csv(csv_path, nrows=0).columns.tolist()

# Detect embeddings like img_embR3km_0..511
pat = re.compile(r'^img_embR3km_(\d+)$')
emb = sorted([c for c in cols if pat.match(c)], key=lambda s:int(pat.match(s).group(1)))

# Optional: warn if none found (helps catch wrong file or prefix)
if not emb:
    print(json.dumps(base_list))
else:
    print(json.dumps(base_list + emb))
PY
)

# Run model
python "$SCRIPT" \
  --data_withfolds_id "$DATA_DIR" \
  --outer_fold ${OUTER_FOLD} \
  --out_dir "$OUT_DIR" \
  --num_vars "$NUM_VARS_STR" \
  --has_osm14 'no'
