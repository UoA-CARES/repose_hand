# Play with 10 Different Camera Views

The `play_10_views.py` script is an enhanced version of the standard RL-Games play script that automatically cycles through 10 different camera viewpoints during agent playback.

## Features

- **Automatic Camera Switching**: Cycles through 10 predefined camera viewpoints
- **Configurable Intervals**: Set how often the camera view changes
- **Video Recording**: Record videos with multiple views or separate videos for each view
- **Fallback Support**: Works even when camera automation is not available

## Usage

### Basic Usage
```bash
python scripts/rl_games/play_10_views.py --task Template-Repose-Hand-v0 --checkpoint path/to/checkpoint.pth
```

### With Video Recording
```bash
python scripts/rl_games/play_10_views.py \
    --task Template-Repose-Hand-v0 \
    --checkpoint path/to/checkpoint.pth \
    --video \
    --video_length 600
```

### Custom View Change Interval
```bash
python scripts/rl_games/play_10_views.py \
    --task Template-Repose-Hand-v0 \
    --checkpoint path/to/checkpoint.pth \
    --view_change_interval 100
```

### Record Separate Videos for Each View
```bash
python scripts/rl_games/play_10_views.py \
    --task Template-Repose-Hand-v0 \
    --checkpoint path/to/checkpoint.pth \
    --video \
    --record_all_views \
    --video_length 300
```

### Real-time Playback
```bash
python scripts/rl_games/play_10_views.py \
    --task Template-Repose-Hand-v0 \
    --checkpoint path/to/checkpoint.pth \
    --real-time
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--view_change_interval` | int | 50 | Number of simulation steps between camera view changes |
| `--record_all_views` | flag | False | Record separate videos for each of the 10 views |
| `--video` | flag | False | Enable video recording |
| `--video_length` | int | 200 | Length of recorded video in steps |
| `--real-time` | flag | False | Run simulation in real-time |
| `--num_envs` | int | None | Number of environments to simulate |

All other arguments from the standard `play.py` script are also supported.

## Camera Viewpoints

The script cycles through these 10 viewpoints:

1. **Front View**: Direct front view of the hand and object
2. **Right Side View**: Profile view from the right side
3. **Back View**: View from behind the setup
4. **Left Side View**: Profile view from the left side  
5. **Top View**: Bird's eye view from above
6. **Bottom-Right Angled**: Lower angle from bottom-right
7. **Top-Left Angled**: Higher angle from top-left
8. **Close-up Front**: Zoomed-in front view for detail
9. **Diagonal High**: High diagonal perspective
10. **Low Angled**: Low angle view for ground perspective

## Manual Camera Control

If automatic camera switching is not available, you can manually control the camera using Isaac Sim's viewport controls:

- **Left click + drag**: Rotate camera around target
- **Middle click + drag**: Pan camera position
- **Right click + drag**: Zoom camera in/out
- **Mouse wheel**: Quick zoom in/out

## Video Output

### Single Video Mode (default with `--video`)
- Creates one video file with automatic view changes
- Saved to: `logs/rl_games/{task_name}/{run_dir}/videos/play_multi_view/`

### Multiple Videos Mode (with `--record_all_views`)
- Creates 10 separate video files, one for each camera view
- Saved to: `logs/rl_games/{task_name}/{run_dir}/videos/play_10_views/view_01/`, `view_02/`, etc.
- Each video shows the same simulation from a different viewpoint

## Example Workflow

1. Train your agent using the standard training script
2. Use `play_10_views.py` to visualize performance from multiple angles
3. Record videos to share or analyze the agent's behavior
4. Use different view change intervals to focus on specific aspects

```bash
# Train the agent first
python scripts/rl_games/train.py --task Template-Repose-Hand-v0

# Play with multiple views and record video
python scripts/rl_games/play_10_views.py \
    --task Template-Repose-Hand-v0 \
    --checkpoint logs/rl_games/Shadow_hands/latest/nn/Shadow_hands.pth \
    --video \
    --video_length 600 \
    --view_change_interval 60 \
    --real-time
```

## Troubleshooting

- **Camera switching not working**: The script includes manual control instructions and fallback modes
- **Import errors**: Make sure you're running in the Isaac Lab environment with all dependencies
- **Video recording issues**: Ensure sufficient disk space and proper write permissions
- **Performance issues**: Try reducing `--num_envs` or increasing `--view_change_interval`
