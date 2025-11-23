#!/bin/bash
#SBATCH --job-name=obj_outofdist_test # Job name
#SBATCH --output=logs/%j.out  # Output file (%x is job name, %j is jobID)
#SBATCH --error=logs/%j.err   # Error file
#SBATCH --time=72:00:00                # walltime
#SBATCH --ntasks=4                     # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                      # number of nodes
#SBATCH --gpus=2                      # request 4 GPUs
#SBATCH --mem-per-cpu=16G              # memory per CPU core (adjusted to standard format)
#SBATCH --mail-user=vv382@scarletmail.rutgers.edu # email address
#SBATCH --mail-type=BEGIN,END,FAIL     # combined mail-type options

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#module load cuda

# Change to the directory where your script is located
# cd $SLURM_SUBMIT_DIR # This uses the directory from which you submitted the job

# Make your script executable if it's not already
torchrun --nproc_per_node=2 --master_port 12345 main.py --model shvit_s4 --data-path /research/projects/mllab/vv382/ --dist-eval --weight-decay 0.025 --save_freq=10 --epochs=20  --data-set=MEDMNIST > output.txt

