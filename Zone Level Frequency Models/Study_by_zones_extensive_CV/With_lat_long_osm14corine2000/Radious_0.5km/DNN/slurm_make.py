import os
from textwrap import dedent

outers = [0,1,2,3,4,5]              #6 outer folds
epoch_options = [15, 30, 60]        # emit one .sh per epoch value
slurm_array_range = "0-7"           # match rows-1 in grid CSV no header

# walltime per epoch 
walltime_for_epoch = {
    15: "01:00:00",
    30: "02:35:00",
    60: "03:10:00",
}

# Fixed single feature set 
NUM_VARS = "['ageph_mean','ageph_median','ageph_std','bm_mean','bm_median','bm_std','power_mean','power_median','power_std','agec_mean','agec_median','agec_std','coverage_TPL_prop','coverage_TPL+_prop','sex_female_prop','fuel_diesel_prop','use_private_prop','fleet_0_prop','lat','long','road_len_km_per_km2_r500', 'intersection_count_per_km2_r500', 'roundabout_count_per_km2_r500', 'traffic_signal_count_per_km2_r500', 'retail_count_per_km2_r500', 'tourism_count_per_km2_r500', 'parking_count_per_km2_r500', 'has_education_r500', 'has_healthcare_r500', 'has_fuel_station_r500', 'school_count_per_km2_r500', 'healthcare_count_per_km2_r500', 'fuel_count_per_km2_r500']"
CAT_VARS = "None"   

# Paths
DATA_CSV = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Stratified_division/alldata_with_fold_id.csv"
BASE_OUT_ROOT = "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/DNN/Radious_0.5km"   
GRID_CSV_TEMPLATE = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/Imagenes_DNN/hyperparam_grid_ep{epochs}.csv"
PYTHON_SCRIPT = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/DNN/TabularDNN.py"

MODULE_LINES = """\
module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate
"""
SBATCH_GPU_LINE = "#SBATCH --gpus=a100_1g.5gb:1"                   
SBATCH_CPUS_LINE = "#SBATCH --cpus-per-task=2"
SBATCH_MEM_LINE  = "#SBATCH --mem=8G"

# Where to drop .sh files
OUTPUT_FOLDER = "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/DNN/Radious_0.5km"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

template = dedent("""\
    #!/bin/bash
    #SBATCH --job-name=TabDNN_ep{epochs}_out{{outer}}
    {gpu_line}
    {cpus_line}
    {mem_line}
    #SBATCH --time={walltime}
    #SBATCH --array={array_range}
    #SBATCH --output={base_out}/outer{{outer}}/out{{outer}}_ep{epochs}_%a.out

    set -euo pipefail

    {module_block}

    DATA_CSV="{data_csv}"
    BASE_OUT="{base_out}"
    OUTER={{outer}}

    # Fixed feature set for this run
    NUM_VARS="{num_vars}"
    CAT_VARS="{cat_vars}"

    OUT_DIR="${{BASE_OUT}}/outer${{OUTER}}"
    mkdir -p "${{OUT_DIR}}"


    GRID_CSV="{grid_csv}"

    # Pick this task's hyperparam row from the epoch-specific grid
    IDX=${{SLURM_ARRAY_TASK_ID}}
    HPARAM_LINE=$(tail -n +2 "${{GRID_CSV}}" | sed -n "$((IDX+1))p")
    IFS=',' read -r CONFIG_IDX BATCH_SIZE HIDDEN_DIM EPOCHS LR DROPOUT OPTIMIZER WEIGHT_DECAY SEED CONFIG_ID <<< "${{HPARAM_LINE}}"

    echo "[ep {epochs}] outer=${{OUTER}} task=${{IDX}} cfg_id=${{CONFIG_ID}} bs=${{BATCH_SIZE}} hd=${{HIDDEN_DIM}} ep=${{EPOCHS}} lr=${{LR}} dr=${{DROPOUT}} opt=${{OPTIMIZER}} wd=${{WEIGHT_DECAY}} seed=${{SEED}}"

    python {python_script} \\
        --data_withfolds_id "${{DATA_CSV}}" \\
        --outer_fold $OUTER \\
        --out_dir "${{OUT_DIR}}" \\
        --num_vars "${{NUM_VARS}}" \\
        --cat_vars "{cat_vars_cli}" \\
        --seed $SEED \\
        --batch_size $BATCH_SIZE \\
        --hidden_dim $HIDDEN_DIM \\
        --epochs $EPOCHS \\
        --lr $LR \\
        --dropout $DROPOUT \\
        --optimizer $OPTIMIZER \\
        --weight_decay $WEIGHT_DECAY
""")

for epochs in epoch_options:
    walltime = walltime_for_epoch[epochs]
    grid_csv = GRID_CSV_TEMPLATE.format(epochs=epochs)

    script_body = template.format(
        epochs=epochs,
        gpu_line=SBATCH_GPU_LINE,
        cpus_line=SBATCH_CPUS_LINE,
        mem_line=SBATCH_MEM_LINE,
        walltime=walltime,
        array_range=slurm_array_range,
        module_block=MODULE_LINES.strip(),
        base_out=BASE_OUT_ROOT,
        data_csv=DATA_CSV,
        num_vars=NUM_VARS.replace('"','\\"'),     # escape quotes
        cat_vars=(CAT_VARS if CAT_VARS!="None" else "None"),
        cat_vars_cli=(CAT_VARS if CAT_VARS!="None" else "None"),
        grid_csv=grid_csv,
        python_script=PYTHON_SCRIPT,
    )

    for outer in outers:
        fname = f"TabDNN_ep{epochs}_out{outer}.sh"
        fpath = os.path.join(OUTPUT_FOLDER, fname)
        with open(fpath, "w") as f:
            f.write(script_body.replace("{outer}", str(outer)))
        os.chmod(fpath, 0o750)

print(f"Done. Scripts in {OUTPUT_FOLDER}")
