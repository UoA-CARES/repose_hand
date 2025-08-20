# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with an interactive agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import select
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Interactive agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import repose_hand.tasks  # noqa: F401


def main():
    """Interactive agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    # create default actions
    actions = -torch.ones(env.action_space.shape, device=env.unwrapped.device)
    # simulate environment
    print("Enter a joint angle value between -1 and 1 and press Enter. Press 'q' and Enter to quit.")
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # check for user input
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                user_input = sys.stdin.readline().strip()
                if user_input.lower() == "q":
                    break
                try:
                    angle = float(user_input)
                    # if not (-1.0 <= angle <= 1.0):
                    #     print("Invalid input. Please enter a value between -1 and 1.")
                    # else:
                        # create action tensor
                    actions = torch.full(env.action_space.shape, angle, device=env.unwrapped.device)
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
