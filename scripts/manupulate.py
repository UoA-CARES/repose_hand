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
    
    # Define joint names list (27 joints)
    joint_names = [
        'thumb_carpal_1_thumb_abd0', 'middle_1_m_pp_1_middle_mcp0', 'palm_2_1_palm_abd0', 
        'index_1_i_pp_1_index_mcp0', 'thumb_1_t_pp_1_thumb_mcp0', 'middle_1_m_pp_1_middle_mcp1', 
        'ring_1_r_pp_1_ring_mcp0', 'pinky_1_p_pp_1_pinky_mcp0', 'index_1_i_pp_1_index_mcp1', 
        'thumb_1_t_pp_1_thumb_mcp1', 'middle_1_m_pp_1_middle_mcp2', 'ring_1_r_pp_1_ring_mcp1', 
        'pinky_1_p_pp_1_pinky_mcp1', 'index_1_i_pp_1_index_mcp2', 'thumb_1_t_pp_1_thumb_mcp2', 
        'middle_1_m_ip_1_m_pp_ip0', 'ring_1_r_pp_1_ring_mcp2', 'pinky_1_p_pp_1_pinky_mcp2', 
        'index_1_i_ip_1_i_pp_ip0', 'thumb_1_t_ip_1_t_pp_ip0', 'middle_1_m_dp_1_m_ip_pp0', 
        'ring_1_r_ip_1_r_pp_ip0', 'pinky_1_p_ip_1_p_pp_ip0', 'index_1_i_dp_1_i_ip_dp0', 
        'thumb_1_t_dp_1_t_ip_dp0', 'ring_1_r_dp_1_r_ip_dp0', 'pinky_1_p_dp_1_p_ip_dp0'
    ]
    
    # reset environment
    env.reset()

    # create default actions (all joints start at 0)
    actions = - torch.ones(env.action_space.shape, device=env.unwrapped.device)
    
    # Print instructions
    print("\n" + "="*60)
    print("JOINT CONTROL INTERFACE")
    print("="*60)
    print("Commands:")
    print("  <joint_name> <value>  - Set specific joint to value (-1 to 1)")
    print("  list                  - Show all joint names with indices")
    print("  status                - Show current joint values")
    print("  reset                 - Reset all joints to 0")
    print("  q                     - Quit")
    print("="*60)
    print("Example: thumb_mcp0 0.5")
    print("Note: You can use partial joint names (e.g., 'thumb_mcp0' instead of full name)")
    print("="*60 + "\n")
    
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # check for user input
            if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                user_input = sys.stdin.readline().strip()
                if user_input.lower() == "q":
                    break
                elif user_input.lower() == "list":
                    print("\nJoint Names (Index: Name):")
                    for i, joint_name in enumerate(joint_names):
                        print(f"  {i:2d}: {joint_name}")
                    print("")
                elif user_input.lower() == "status":
                    print("\nCurrent Joint Values:")
                    for i, joint_name in enumerate(joint_names):
                        print(f"  {i:2d}: {joint_name:<35} = {actions[0, i].item():6.3f}")
                    print("")
                elif user_input.lower() == "reset":
                    actions = - torch.ones(env.action_space.shape, device=env.unwrapped.device)
                    print("All joints reset to 0.0")
                else:
                    # Parse joint name and value
                    parts = user_input.split()
                    if len(parts) == 2:
                        joint_input, value_str = parts
                        try:
                            value = float(value_str)
                            # Find matching joint
                            matching_joints = []
                            for i, joint_name in enumerate(joint_names):
                                if joint_input.lower() in joint_name.lower():
                                    matching_joints.append((i, joint_name))
                            
                            if len(matching_joints) == 1:
                                joint_idx, joint_name = matching_joints[0]
                                actions[0, joint_idx] = value
                                print(f"Set {joint_name} (index {joint_idx}) to {value:.3f}")
                            elif len(matching_joints) > 1:
                                print(f"Ambiguous joint name '{joint_input}'. Matches:")
                                for i, joint_name in matching_joints:
                                    print(f"  {i:2d}: {joint_name}")
                                print("Please be more specific.")
                            else:
                                # Try to parse as index
                                try:
                                    joint_idx = int(joint_input)
                                    if 0 <= joint_idx < len(joint_names):
                                        actions[0, joint_idx] = value
                                        print(f"Set {joint_names[joint_idx]} (index {joint_idx}) to {value:.3f}")
                                    else:
                                        print(f"Joint index {joint_idx} out of range (0-{len(joint_names)-1})")
                                except ValueError:
                                    print(f"No joint found matching '{joint_input}'. Use 'list' to see all joints.")
                        except ValueError:
                            print("Invalid value. Please enter a number between -1 and 1.")
                    elif len(parts) == 1 and parts[0]:
                        print("Invalid format. Use: <joint_name> <value> or <joint_index> <value>")
                    # Empty input is ignored (allows continuous simulation)

            # apply actions
            env.step(actions)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
