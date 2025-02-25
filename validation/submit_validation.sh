#!/bin/bash

# User info
#SBATCH --mail-user=ubelix.que.pos@gmail.com
#SBATCH --mail-type=begin,end,fail

# Job name
#SBATCH --job-name="ormir_xct_autocontour"

# Runtime and memory
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --tmp=50G
#SBATCH --array=1-10%50

# Workdir
#SBATCH --chdir=/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/TOOLS/ORMIR_XCT/validation/
#SBATCH --output=out/autocontour_%A_%a.out
#SBATCH --error=out/autocontour_%A_%a.err

##############################################################################################################
### Load modules
HPC_WORKSPACE=hpc_abaqus module load Workspace
module load Anaconda3

unset SLURM_GTIDS

### greyscale_filenames.txt contains lines with 1 greyscale_filename per line.
greyscale_filenames=/storage/workspaces/artorg_msb/hpc_abaqus/poncioni/TOOLS/ORMIR_XCT/validation/validation_utils/filenames.txt

### Line <i> contains greyscale_filename for run <i>
# Get greyscale_filename                                                                                                                                  
greyscale_filename=$(cat $greyscale_filenames | awk -v var=$SLURM_ARRAY_TASK_ID 'NR==var{print $1}')

### Zero pad the task ID to match the numbering of the input files
n=$(printf "%04d" $SLURM_ARRAY_TASK_ID)

# Run command
eval "$(conda shell.bash hook)" && conda activate ormir_xct && python run_validation.py images.grayscale_filenames=$greyscale_filename
