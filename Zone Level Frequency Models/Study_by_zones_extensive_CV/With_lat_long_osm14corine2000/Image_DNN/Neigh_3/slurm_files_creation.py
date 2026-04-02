import os
from textwrap import dedent


radii = ["3"]  
outers = [0, 1, 2, 3, 4, 5]  # outer folds
epoch_options = [15, 30, 60]

walltime_for_epoch = {
    15: "01:00:00",
    30: "02:00:00",
    60: "04:00:00",
}


slurm_array_range = "0-7"

# static paths  base script
DATA_CSV = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Stratified_division/alldata_with_fold_id.csv"
UNIQUE_LOC = "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/unique_locations_BeMTPL97.csv"
IMG_ROOT = "/home/salfonso/scratch/Belgian/Images_Ortho_95"
WEIGHTS_PATH = "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Models_frequency/Poisson_Assumption/DNN_images_osm_standard/ResNet18/resnet18_weights.pth"
NUM_VARS = "['ageph_mean','ageph_median','ageph_std','bm_mean','bm_median','bm_std','power_mean','power_median','power_std','agec_mean','agec_median','agec_std','coverage_TPL_prop','coverage_TPL+_prop','sex_female_prop','fuel_diesel_prop','use_private_prop','fleet_0_prop','lat','long','road_len_km_per_km2_r5000', 'intersection_count_per_km2_r5000', 'roundabout_count_per_km2_r5000', 'traffic_signal_count_per_km2_r5000', 'retail_count_per_km2_r5000', 'tourism_count_per_km2_r5000', 'parking_count_per_km2_r5000', 'has_education_r5000', 'has_healthcare_r5000', 'has_fuel_station_r5000', 'school_count_per_km2_r5000', 'healthcare_count_per_km2_r5000', 'fuel_count_per_km2_r5000']"
CAT_VARS = "None"
BASE_OUT_ROOT = "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/Image_DNN"
GRID_CSV_TEMPLATE = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/Imagenes_DNN/hyperparam_grid_ep{epochs}.csv"

PYTHON_SCRIPT = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/Imagenes_DNN/MResNet18_extcv_single_config.py"


MODULE_LINES = """\
module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate
"""

# GPU 
SBATCH_GPU_LINE = "#SBATCH --gpus=a100_1g.5gb:1"
SBATCH_CPUS_LINE = "#SBATCH --cpus-per-task=2"
SBATCH_MEM_LINE = "#SBATCH --mem=8G"

# Folder to drop all generated .sh files
OUTPUT_FOLDER = "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/With_lat_long_osm14corine2000/Image_DNN"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --------- TEMPLATE ----------
template = dedent("""\
    #!/bin/bash
    #SBATCH --job-name=N{radius}_out{outer}_sc_ep{epochs}
    {gpu_line}
    {cpus_line}
    {mem_line}
    #SBATCH --time={walltime}
    #SBATCH --array={array_range}
    #SBATCH --output={base_out}/Neigh_{radius}/N{radius}_outer{outer}_ep{epochs}_%a.out
    set -euo pipefail

    {module_block}

    # Radius and outer fold
    RADIUS="{radius}"
    OUTER={outer}

    DATA_CSV="{data_csv}"
    UNIQUE_LOC="{unique_loc}"
    IMG_ROOT="{img_root}"
    WEIGHTS_PATH="{weights_path}"

    NUM_VARS="{num_vars}"
    CAT_VARS="{cat_vars}"

    BASE_OUT="{base_out}/Neigh_{radius}/Results"
    OUT_DIR="${{BASE_OUT}}/R${{RADIUS}}km/outer${{OUTER}}"
    mkdir -p "${{OUT_DIR}}"

    GRID_CSV="{grid_csv}"

    # selecting the hyperparameter tuning
    IDX=${{SLURM_ARRAY_TASK_ID}}

    HPARAM_LINE=$(tail -n +2 "${{GRID_CSV}}" | sed -n "$((IDX+1))p")

    IFS=',' read -r CONFIG_IDX BATCH_SIZE HIDDEN_DIM EPOCHS LR DROPOUT OPTIMIZER WEIGHT_DECAY SEED CONFIG_ID <<< "${{HPARAM_LINE}}"

    echo "SLURM task $IDX using config_idx=${{CONFIG_IDX}}, config_id=${{CONFIG_ID}}"
    echo "bs=${{BATCH_SIZE}} hd=${{HIDDEN_DIM}} ep=${{EPOCHS}} lr=${{LR}} dr=${{DROPOUT}} opt=${{OPTIMIZER}} wd=${{WEIGHT_DECAY}} seed=${{SEED}}"

    python {python_script} \\
        --data_withfolds_id "${{DATA_CSV}}" \\
        --unique_loc_csv "${{UNIQUE_LOC}}" \\
        --outer_fold $OUTER \\
        --out_dir "${{OUT_DIR}}" \\
        --img_root "${{IMG_ROOT}}" \\
        --radius_km "${{RADIUS}}" \\
        --num_vars "${{NUM_VARS}}" \\
        --cat_vars "${{CAT_VARS}}" \\
        --seed $SEED \\
        --weights_path "${{WEIGHTS_PATH}}" \\
        --batch_size $BATCH_SIZE \\
        --hidden_dim $HIDDEN_DIM \\
        --epochs $EPOCHS \\
        --lr $LR \\
        --dropout $DROPOUT \\
        --optimizer $OPTIMIZER \\
        --weight_decay $WEIGHT_DECAY
    """)

# Generate the files
for radius in radii:
    for outer in outers:
        for epochs in epoch_options:
            walltime = walltime_for_epoch[epochs]

            # where logs and results go
            base_out = os.path.join(
                BASE_OUT_ROOT,
                f"Neigh_{radius}"
            )

            # which grid csv to use for this epochs value
            grid_csv = GRID_CSV_TEMPLATE.format(epochs=epochs)

            script_text = template.format(
                radius=radius,
                outer=outer,
                epochs=epochs,
                walltime=walltime,
                array_range=slurm_array_range,
                gpu_line=SBATCH_GPU_LINE,
                cpus_line=SBATCH_CPUS_LINE,
                mem_line=SBATCH_MEM_LINE,
                module_block=MODULE_LINES.strip(),
                base_out=BASE_OUT_ROOT,
                data_csv=DATA_CSV,
                unique_loc=UNIQUE_LOC,
                img_root=IMG_ROOT,
                weights_path=WEIGHTS_PATH,
                num_vars=NUM_VARS.replace('"', '\\"'),
                cat_vars=CAT_VARS,
                grid_csv=grid_csv,
                python_script=PYTHON_SCRIPT,
            )

            # filename like N0.5_out0_sc_ep15.sh
            fname = f"N{radius}_out{outer}_sc_ep{epochs}.sh"
            fpath = os.path.join(OUTPUT_FOLDER, fname)
            with open(fpath, "w") as f:
                f.write(script_text)

            # make it executable right away
            os.chmod(fpath, 0o750)

print(f"Done. Scripts are in {OUTPUT_FOLDER}")

