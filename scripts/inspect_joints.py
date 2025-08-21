#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to inspect the joint names and action space of the robot in the environment.

This script helps you find the exact names of each joint in the action space.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Inspect robot joint names and action space.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times")
parser.add_argument("--task", type=str, default="Template-Repose-Hand-v0", help="Name of the task")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import repose_hand.tasks  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict


def main():
    """Inspect the joint names and action space of the robot."""
    
    # create environment configuration
    env_cfg = gym.make(args_cli.task, num_envs=1, render_mode=None).unwrapped.cfg
    
    # create environment
    env = gym.make(args_cli.task, num_envs=1, render_mode=None)
    
    print("=" * 80)
    print(f"Environment: {args_cli.task}")
    print("=" * 80)
    
    # Get robot asset from the environment
    robot = env.unwrapped.scene["robot"]
    
    # Print robot information
    print(f"Robot asset: {robot}")
    print(f"Number of joints: {robot.num_joints}")
    print(f"Number of actions: {robot.num_actions}")
    
    print("\n" + "=" * 50)
    print("JOINT NAMES AND INFORMATION:")
    print("=" * 50)
    
    # Get all joint names
    joint_names = robot.joint_names
    print(f"Total joints: {len(joint_names)}")
    print("\nAll joint names:")
    for i, joint_name in enumerate(joint_names):
        print(f"  {i:2d}: {joint_name}")
    
    # Get actuated joint names (these are the ones in the action space)
    actuated_joint_names = robot.actuated_joint_names
    print(f"\n\nActuated joints (action space): {len(actuated_joint_names)}")
    print("These are the joints you can control:")
    for i, joint_name in enumerate(actuated_joint_names):
        print(f"  Action {i:2d}: {joint_name}")
    
    # Print joint limits
    print("\n" + "=" * 50)
    print("JOINT LIMITS:")
    print("=" * 50)
    
    joint_pos_limits = robot.data.soft_joint_pos_limits[0]  # [0] for first environment
    joint_vel_limits = robot.data.soft_joint_vel_limits[0]
    
    print(f"{'Joint Name':<35} {'Pos Min':<10} {'Pos Max':<10} {'Vel Limit':<10}")
    print("-" * 70)
    
    for i, joint_name in enumerate(actuated_joint_names):
        pos_min = joint_pos_limits[i, 0].item()
        pos_max = joint_pos_limits[i, 1].item()
        vel_limit = joint_vel_limits[i].item()
        print(f"{joint_name:<35} {pos_min:<10.3f} {pos_max:<10.3f} {vel_limit:<10.3f}")
    
    # Print action space information
    print("\n" + "=" * 50)
    print("ACTION SPACE INFORMATION:")
    print("=" * 50)
    
    action_space = env.action_space
    print(f"Action space type: {type(action_space)}")
    print(f"Action space shape: {action_space.shape}")
    print(f"Action space low: {action_space.low}")
    print(f"Action space high: {action_space.high}")
    
    # Print observation space information
    print("\n" + "=" * 50)
    print("OBSERVATION SPACE INFORMATION:")
    print("=" * 50)
    
    obs_space = env.observation_space
    print(f"Observation space type: {type(obs_space)}")
    if hasattr(obs_space, 'spaces'):
        print("Observation space components:")
        for key, space in obs_space.spaces.items():
            print(f"  {key}: {space.shape}")
    else:
        print(f"Observation space shape: {obs_space.shape}")
    
    # Test a random action to verify action space
    print("\n" + "=" * 50)
    print("TESTING ACTION SPACE:")
    print("=" * 50)
    
    # Reset environment
    obs, _ = env.reset()
    print("Environment reset successfully!")
    
    # Sample a random action
    action = env.action_space.sample()
    print(f"Random action shape: {action.shape}")
    print(f"Random action values: {action}")
    
    # Take a step
    obs, reward, terminated, truncated, info = env.step(action)
    print("Action executed successfully!")
    
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Task: {args_cli.task}")
    print(f"Total joints: {len(joint_names)}")
    print(f"Actuated joints (action space): {len(actuated_joint_names)}")
    print(f"Action space shape: {action_space.shape}")
    
    print("\nActuated joint names (copy this list for your use):")
    print("actuated_joint_names = [")
    for joint_name in actuated_joint_names:
        print(f'    "{joint_name}",')
    print("]")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
