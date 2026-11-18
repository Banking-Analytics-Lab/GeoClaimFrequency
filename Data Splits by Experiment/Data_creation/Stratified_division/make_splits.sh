#!/bin/bash
#
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8GB
#SBATCH --time=0:30:00
#SBATCH --account=<your-allocation>
#SBATCH --output=Making_6_divisions.out
module load gcc arrow/19.0.1
source ~/p3_env_nvl_test/bin/activate

python Division_6.py 
