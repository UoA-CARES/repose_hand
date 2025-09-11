#!/bin/bash

# Shell script to run play_views.py from different camera angles
# Author: Generated for repose_hand project
# Date: $(date)

# Base command configuration
SCRIPT_PATH="python scripts/rl_games/play_views.py"
TASK="Template-Repose-Hand-v0"

# Define base directory for logs
BASE_DIR="/home/lee/code/repose_hand/logs/rl_games/Shadow_hands/2025-09-01_17-06-51"

# Ensure the directory exists before attempting to remove its contents
if [ -d "$BASE_DIR/videos/play" ]; then
    rm -rf "$BASE_DIR/videos/play/*"
else
    echo "Directory $BASE_DIR/videos/play does not exist. Skipping removal."
fi

# Update checkpoint path
CHECKPOINT="$BASE_DIR/nn/Shadow_hands.pth"

# 30 * 20 = 600 steps for a 20-second video at 30 FPS
VIDEO_LENGTH="1800"
NUM_ENVS="1"

# Define different camera views (x,y,z coordinates)
# Close views optimized for small robot hand details
# "-0.12, -0.16, 0.4"  # Top-down view - hand configuration detail
VIEWS=(
    "-1.12, -0.16, 0.4"  # back
    "0.88, -0.16, 0.4"  # front
    "-0.12, -0.16, 2"  # top_down
    "-0.12, -1.16, 0.4"  # right_side
    "-0.12, 0.84, 0.4"  # left_side
    "-0.12, -1.16, 1"  # diagonal_right
    "-0.72, 0.44, 1"  # diagonal_left_back
    "0.48, -0.76, 1"  # diagonal_right_front
    "0.48, 0.44, 1"  # diagonal_left_front
)

# View descriptions for logging
VIEW_DESCRIPTIONS=(
    "back"
    "front"
    "top_down"
    "right_side"
    "left_side"
    "diagonal_right"
    "diagonal_left_back"
    "diagonal_right_front"
    "diagonal_left_front"
)

echo "=========================================="
echo "Running play_views.py from multiple angles"
echo "Task: $TASK"
echo "Checkpoint: $CHECKPOINT"
echo "Video Length: $VIDEO_LENGTH steps"
echo "Number of Environments: $NUM_ENVS"
echo "=========================================="

# Create output directory for this run
OUTPUT_DIR="./video_outputs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$OUTPUT_DIR"

# Loop through each view
for i in "${!VIEWS[@]}"; do
    view="${VIEWS[$i]}"
    description="${VIEW_DESCRIPTIONS[$i]}"
    
    echo ""
    echo "----------------------------------------"
    echo "Running view $((i+1))/${#VIEWS[@]}: $description (${view})"
    echo "----------------------------------------"
    
    # Run the command
    $SCRIPT_PATH \
        --task "$TASK" \
        --checkpoint "$CHECKPOINT" \
        --video_length "$VIDEO_LENGTH" \
        --num_envs "$NUM_ENVS" \
        --views="$view" \
        --video
    

        
    # Rename the generated video to prevent overwriting
    VIDEO_SOURCE="$BASE_DIR/videos/play/rl-video-step-0.mp4"
    VIDEO_DEST="$BASE_DIR/videos/play/rl-video-step-0_${description}_${VIDEO_LENGTH}.mp4"

    if [ -f "$VIDEO_SOURCE" ]; then
        mv "$VIDEO_SOURCE" "$VIDEO_DEST"
        echo "✓ Video renamed to: rl-video-step-0_${description}_${VIDEO_LENGTH}.mp4"
    else
        echo "⚠ Warning: Video file not found at expected location"
    fi

    
    # Optional: Add a brief pause between runs
    echo "Waiting 3 seconds before next view..."
    sleep 3
done

echo ""
echo "=========================================="
echo "All views completed!"
echo "Check the outputs/ directory for generated videos"
echo "=========================================="
