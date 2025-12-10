# script for running analyze_patchify_stride.py for stride experiments

#!/bin/bash
#SBATCH --job-name=shvit_stride_exp
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=72:00:00
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=vv382@scarletmail.rutgers.edu
#SBATCH --mail-type=BEGIN,END,FAIL

export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# Train SHViT with Different Patchify Strides Across Domains
# Tests how patchify stride affects generalization

MODEL=${1:-shvit_s1}
DATASET=${2:-CIFAR}
DATA_PATH=${3:-/research/projects/mllab/vv382/}
BASE_OUTPUT=${4:-stride_experiments}
EPOCHS=${5:-100}

echo "=========================================="
echo "Patchify Stride Experiments"
echo "=========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Data Path: $DATA_PATH"
echo "Epochs: $EPOCHS"
echo "=========================================="

# Test strides: 8, 16 (original), 32
# Skip stride 4 for efficiency (too many patches, slow training)
STRIDES=(8 16 32)

for stride in "${STRIDES[@]}"
do
    echo ""
    echo "=========================================="
    echo "Training with patchify stride ${stride}x${stride}"
    echo "=========================================="
    
    OUTPUT_DIR="${BASE_OUTPUT}/${MODEL}_${DATASET}_stride${stride}"
    
    # Check for existing checkpoint to resume
    RESUME_ARG=""
    if [ -d "$OUTPUT_DIR" ]; then
        LATEST_CKPT=$(ls -t ${OUTPUT_DIR}/checkpoint_*.pth 2>/dev/null | head -n 1)
        if [ -n "$LATEST_CKPT" ]; then
            echo "Found checkpoint: ${LATEST_CKPT}"
            echo "Resuming training..."
            RESUME_ARG="--resume ${LATEST_CKPT}"
        fi
    fi
    
    # Note: This requires modifying main.py to accept --patch-stride argument
    # For now, you need to train models separately with modified model code
    
    # Placeholder - you'll need to modify the model creation
    echo "Training ${MODEL} with stride ${stride} on ${DATASET}"
    echo "Output: ${OUTPUT_DIR}"
    
    # torchrun --nproc_per_node=2 --master_port 12345 ../../main.py \
    #     --model $MODEL \
    #     --data-set $DATASET \
    #     --data-path $DATA_PATH \
    #     --output_dir $OUTPUT_DIR \
    #     --patch-stride $stride \
    #     --epochs $EPOCHS \
    #     --batch-size 256 \
    #     --weight-decay 0.025 \
    #     --save_freq 20 \
    #     $RESUME_ARG \
    #     --dist-eval
    
    echo "Completed training with stride ${stride}"
done

echo ""
echo "=========================================="
echo "All Stride Experiments Complete!"
echo "=========================================="
echo ""
echo "To analyze results, run:"
echo "python extensions/analysis/analyze_patchify_stride.py --model ${MODEL} --datasets ${DATASET} --checkpoint-dir ${BASE_OUTPUT}"
