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

MODEL=${1:-shvit_s4}
DATASET=${2:-MEDMNIST}
DATA_PATH=${3:-/research/projects/mllab/vv382/}
OUTPUT_DIR=${4:-""}
EPOCHS=${5:-100}

echo "=========================================="
echo "Training SHViT"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Data Path: $DATA_PATH"
echo "Output Dir: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Auto-detect checkpoint and resume if exists
RESUME_ARG=""
if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
    LATEST_CKPT=$(ls -t ${OUTPUT_DIR}/checkpoint_*.pth 2>/dev/null | head -n 1)
    if [ -n "$LATEST_CKPT" ]; then
        echo "Found checkpoint: ${LATEST_CKPT}"
        echo "Resuming training..."
        RESUME_ARG="--resume ${LATEST_CKPT}"
    fi
fi

OUTPUT_ARG=""
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_ARG="--output_dir ${OUTPUT_DIR}"
fi

torchrun --nproc_per_node=2 --master_port 12345 main.py \
    --model $MODEL \
    --data-set $DATASET \
    --data-path $DATA_PATH \
    $OUTPUT_ARG \
    --epochs $EPOCHS \
    --batch-size 256 \
    --save_freq 50 \
    $RESUME_ARG \
    --dist-eval

