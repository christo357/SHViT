#!/bin/bash
#SBATCH --job-name=shvit_learning_curve # Job name
#SBATCH --output=logs/%j.out  # Output file (%x is job name, %j is jobID)
#SBATCH --error=logs/%j.err   # Error file
#SBATCH --time=72:00:00                # walltime
#SBATCH --ntasks=4                     # number of processor cores (i.e. tasks)
#SBATCH --nodes=1                      # number of nodes
#SBATCH --gpus=2                      # request 2 GPUs
#SBATCH --mem-per-cpu=16G              # memory per CPU core (adjusted to standard format)
#SBATCH --mail-user=vv382@scarletmail.rutgers.edu # email address
#SBATCH --mail-type=BEGIN,END,FAIL     # combined mail-type options

# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
#module load cuda

# Learning Curve Analysis for SHViT
# Trains models on 10%, 32.5%, 55%, 77.5%, and 100% of training data
# All models are evaluated on the same full test set

MODEL=${1:-shvit_s1}
DATASET=${2:-CIFAR}
DATA_PATH=${3:-/research/projects/mllab/vv382/}
BASE_OUTPUT=${4:-learning_curve_results}
EPOCHS=${5:-50}

echo "=========================================="
echo "Learning Curve Analysis for SHViT"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Data Path: $DATA_PATH"
echo "Base Output: $BASE_OUTPUT"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Five training data fractions: 10%, 32.5%, 55%, 77.5%, 100%
FRACTIONS=(0.1 0.325 0.55 0.775 1.0)

for fraction in "${FRACTIONS[@]}"
do
    echo ""
    echo "=========================================="
    echo "Training with ${fraction} of training data"
    echo "=========================================="
    
    OUTPUT_DIR="${BASE_OUTPUT}/${MODEL}_${DATASET}_frac${fraction}"
    
    # Auto-detect latest checkpoint and resume if exists
    RESUME_ARG=""
    if [ -d "$OUTPUT_DIR" ]; then
        # Find the latest checkpoint
        LATEST_CKPT=$(ls -t ${OUTPUT_DIR}/checkpoint_*.pth 2>/dev/null | head -n 1)
        if [ -n "$LATEST_CKPT" ]; then
            echo "Found existing checkpoint: ${LATEST_CKPT}"
            echo "Resuming training from this checkpoint..."
            RESUME_ARG="--resume ${LATEST_CKPT}"
        else
            echo "No checkpoint found, starting from scratch"
        fi
    else
        echo "No existing output directory, starting from scratch"
    fi
    
    torchrun --nproc_per_node=2 --master_port 12345 main.py \
        --model $MODEL \
        --data-set $DATASET \
        --data-path $DATA_PATH \
        --output_dir $OUTPUT_DIR \
        --train-subset-fraction $fraction \
        --subset-seed 42 \
        --epochs $EPOCHS \
        --batch-size 256 \
        --weight-decay 0.025 \
        --save_freq 15 \
        $RESUME_ARG \
        --dist-eval >output.txt
    
    echo "Completed training with fraction ${fraction}"
    echo "Results saved to ${OUTPUT_DIR}"
done

echo ""
echo "=========================================="
echo "Learning Curve Experiments Complete!"
echo "=========================================="
echo "Results directory: ${BASE_OUTPUT}"
echo ""
echo "To analyze results, run:"
echo "python analyze_learning_curve.py --results-dir ${BASE_OUTPUT} --model ${MODEL} --dataset ${DATASET}"
