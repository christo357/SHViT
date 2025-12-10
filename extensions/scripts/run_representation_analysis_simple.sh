# script for running analyze_representations.py

#!/bin/bash

# Simple Representation Analysis Runner (no SLURM required)
# Compares learned features between SHViT and DeiT models

# Configuration
SHVIT_MODEL="shvit_s2"
DEIT_MODEL="deit_tiny_patch16_224"
DATASET="CIFAR"
NUM_CLASSES=100
DATA_PATH="dataset/"

# Model checkpoints
SHVIT_CHECKPOINT="results/shvit_s2_CIFAR_frac1.0/checkpoint_99.pth"
DEIT_CHECKPOINT="results/deit_tiny_patch16_224_CIFAR_frac1.0/checkpoint_99.pth"

# Output directory
OUTPUT_DIR="representation_analysis/${SHVIT_MODEL}_vs_${DEIT_MODEL}_${DATASET}"

echo "=========================================="
echo "Representation Analysis"
echo "=========================================="
echo "SHViT: $SHVIT_MODEL"
echo "DeiT: $DEIT_MODEL"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "=========================================="
echo ""
echo "NOTE: Make sure you have uploaded the DeiT checkpoint to:"
echo "      $DEIT_CHECKPOINT"
echo ""
read -p "Press Enter to continue..."

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run analysis
python extensions/analysis/analyze_representations.py \
    --shvit-checkpoint "$SHVIT_CHECKPOINT" \
    --deit-checkpoint "$DEIT_CHECKPOINT" \
    --shvit-model "$SHVIT_MODEL" \
    --deit-model "$DEIT_MODEL" \
    --data-path "$DATA_PATH" \
    --data-set "$DATASET" \
    --nb-classes $NUM_CLASSES \
    --input-size 224 \
    --batch-size 256 \
    --max-batches 50 \
    --num-workers 4 \
    --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - similarity_results.json    (numerical results)"
echo "  - cka_matrix.png            (heatmap of layer similarities)"
echo "  - feature_statistics.png    (feature distributions)"
echo "  - similarity_progression.png (depth-wise analysis)"
echo "=========================================="
