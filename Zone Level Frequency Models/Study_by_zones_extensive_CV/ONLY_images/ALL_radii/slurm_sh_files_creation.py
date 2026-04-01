import os
from textwrap import dedent

# Including all the radii in the imahges
radii_groups = ["0.5,1,3"]

outers = [0, 1, 2, 3, 4, 5]
epoch_options = [15, 30, 60]

walltime_for_epoch = {
    15: "03:30:00",
    30: "04:30:00",
    60: "07:00:00",
}

slurm_array_range = "0-7"  

# static paths
DATA_CSV = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Data_creation/Stratified_division/alldata_with_fold_id.csv"
UNIQUE_LOC = "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Preprocessing/Images_extract_unique_options/unique_locations_BeMTPL97.csv"
IMG_ROOT = "/home/salfonso/scratch/Belgian/Images_Ortho_95"
WEIGHTS_PATH = "/home/salfonso/projects/def-cbravo/salfonso/Belgian/Models_frequency/Poisson_Assumption/DNN_images_osm_standard/ResNet18/resnet18_weights.pth"


# Base folder for multi-radii outputs (kept separate from every Neigh_x singles)
BASE_OUT_ROOT = "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/ONLY_images/ALL_radii"
GRID_CSV_TEMPLATE = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/Imagenes_DNN/hyperparam_grid_ep{epochs}.csv"


PYTHON_SCRIPT = "/home/salfonso/projects/rrg-cbravo/salfonso/Project3/Study_by_zones_extensive_CV_1025/Models_frequency/Imagenes_DNN/MResNet18_multiRadii_extcv_ONLY_images.py"

MODULE_LINES = """\
module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate
"""

# GPU/CPU/MEM
SBATCH_GPU_LINE  = "#SBATCH --gpus=a100_1g.5gb:1"
SBATCH_CPUS_LINE = "#SBATCH --cpus-per-task=2"
SBATCH_MEM_LINE  = "#SBATCH --mem=8G"


OUTPUT_FOLDER = "/home/salfonso/scratch/Belgian/Study_by_zones_extensive_CV/ONLY_images/ALL_radii"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Produce the template
template = dedent("""\
    #!/bin/bash
    #SBATCH --job-name=ALLR_out{outer}_ep{epochs}
    {gpu_line}
    {cpus_line}
    {mem_line}
    #SBATCH --time={walltime}
    #SBATCH --array={array_range}
    #SBATCH --output={base_out}/outer{outer}/ALLR_outer{outer}_ep{epochs}_%a.out
    set -euo pipefail

    {module_block}

    OUTER={outer}

    DATA_CSV="{data_csv}"
    UNIQUE_LOC="{unique_loc}"
    IMG_ROOT="{img_root}"
    WEIGHTS_PATH="{weights_path}"

    # radii list (comma separated). Keep quoted to preserve the commas.
    RADII="{radii_csv}"

    BASE_OUT="{base_out}"
    OUT_DIR="${{BASE_OUT}}/outer${{OUTER}}"
    mkdir -p "${{OUT_DIR}}"

    GRID_CSV="{grid_csv}"

    # 2. Selecting hyperparameters
   
    IDX=${{SLURM_ARRAY_TASK_ID}}
    HPARAM_LINE=$(tail -n +2 "${{GRID_CSV}}" | sed -n "$((IDX+1))p")
    IFS=',' read -r CONFIG_IDX BATCH_SIZE HIDDEN_DIM EPOCHS LR DROPOUT OPTIMIZER WEIGHT_DECAY SEED CONFIG_ID <<< "${{HPARAM_LINE}}"

    echo "SLURM task $IDX using config_idx=${{CONFIG_IDX}}, config_id=${{CONFIG_ID}}"
    echo "bs=${{BATCH_SIZE}} hd=${{HIDDEN_DIM}} ep=${{EPOCHS}} lr=${{LR}} dr=${{DROPOUT}} opt=${{OPTIMIZER}} wd=${{WEIGHT_DECAY}} seed=${{SEED}}"
    echo "radii=${{RADII}}"

    # 3. run the inner cv for the given config
    python {python_script} \
      --data_withfolds_id "${{DATA_CSV}}" \
      --unique_loc_csv "${{UNIQUE_LOC}}" \
      --outer_fold $OUTER \
      --out_dir "${{OUT_DIR}}" \
      --img_root "${{IMG_ROOT}}" \
      --radii "${{RADII}}" \
      --seed $SEED \
      --weights_path "${{WEIGHTS_PATH}}" \
      --batch_size $BATCH_SIZE \
      --hidden_dim $HIDDEN_DIM \
      --epochs $EPOCHS \
      --lr $LR \
      --dropout $DROPOUT \
      --optimizer $OPTIMIZER \
      --weight_decay $WEIGHT_DECAY
""")

# generate 
for radii_csv in radii_groups:
    # a compact tag for filenames like "0p5_1_3"
    tag = radii_csv.replace(".", "p").replace(",", "_")
    for outer in outers:
        for epochs in epoch_options:
            walltime = walltime_for_epoch[epochs]
            base_out = BASE_OUT_ROOT  # keep shared; each outer has its own subdir
            grid_csv = GRID_CSV_TEMPLATE.format(epochs=epochs)

            script_text = template.format(
                outer=outer,
                epochs=epochs,
                walltime=walltime,
                array_range=slurm_array_range,
                gpu_line=SBATCH_GPU_LINE,
                cpus_line=SBATCH_CPUS_LINE,
                mem_line=SBATCH_MEM_LINE,
                module_block=MODULE_LINES.strip(),
                base_out=base_out,
                data_csv=DATA_CSV,
                unique_loc=UNIQUE_LOC,
                img_root=IMG_ROOT,
                weights_path=WEIGHTS_PATH,
                grid_csv=grid_csv,
                python_script=PYTHON_SCRIPT,
                radii_csv=radii_csv
            )

            fname = f"ALLR_{tag}Onlyimg_out{outer}_ep{epochs}.sh"
            fpath = os.path.join(OUTPUT_FOLDER, fname)
            with open(fpath, "w") as f:
                f.write(script_text)
            os.chmod(fpath, 0o750)

print(f"Done. Scripts are in {OUTPUT_FOLDER}")
