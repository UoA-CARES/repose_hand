### Purpose of this task
# TODO: emphasis this is an manager based env from isaaclab
This repository contains an Isaac Lab extension and example tasks for the "repose_hand" project — a minimal Isaac Lab-based environment and tooling for experimenting with hand/robot reposing and manipulation tasks. It includes example scripts to:
- run environments in GUI or headless mode,
- collect and annotate demonstrations,
- run simple agents (zero/random),
- and run training using Isaac Lab wrappers.

### Run with Docker
We provide two useful docker commands for the official Isaac Lab image. Adapt host paths as needed (cache, logs, data volumes).

Machine with screen (X11 forwarding):
```bash
# Launch container with X11 rendering enabled (use on workstation with display)
xhost +
docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v $(pwd):/workspace/repose_hand:rw \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   nvcr.io/nvidia/isaac-lab:2.2.0
```

Headless machine (no display):
```bash
# Launch container for headless use (no DISPLAY forwarding)
docker run --name isaac-lab --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -v $(pwd):/workspace/repose_hand:rw \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   nvcr.io/nvidia/isaac-lab:2.2.0
```

# TODO: change all the Template-Repose-Hand-v0  to Template-Repose-Hand-v0 
# TODO: I have really change all the isaaclab launcher to normal python, pls change the commit to align with my new change
### How to run training
Use the isaaclab launcher script to invoke training scripts inside the repo. Replace Template-Repose-Hand-v0  and other args as needed.

Example (robomimic BC training):
```bash
# run training (headless)
python scripts/imitation_learning/robomimic/train.py --task Template-Repose-Hand-v0  --algo bc --dataset <DATASET_PATH> --headless
```

Example (RL training using RL-Games or another trainer):
```bash
python scripts/reinforcement_learning/rl_games/train.py --task Template-Repose-Hand-v0  --headless
```

### How to run / play an environment interactively
To run a small demo in GUI (use the docker X11 command above or run locally with Isaac Sim installed):
```bash
# A simple GUI example that spawns a scene
python scripts/reinforcement_learning/rl_games/play.py --task Template-Repose-Hand-v0  --num_envs 4
```

For teleoperation or recording demos, use the mimic consolidated demo script (keyboard teleop by default):
```bash
python scripts/imitation_learning/isaaclab_mimic/consolidated_demo.py --task Template-Repose-Hand-v0  --teleop_device keyboard
```

### Zero-action agent
Quick test agent that applies zero actions to validate environment setup:
```bash
python scripts/zero_agent.py --task Template-Repose-Hand-v0 
```

### Random-action agent
Quick test agent that applies random actions:
```bash
python scripts/random_agent.py --task Template-Repose-Hand-v0 
```

### Listing the available tasks
A helper script lists registered tasks/environments available in the current Python environment:
```bash
# If using the isaaclab launcher:
python scripts/list_envs.py
```