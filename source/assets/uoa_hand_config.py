# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# (-0.07, -0.16, 0.5)
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Allegro Hand robots from Wonik Robotics.

The following configurations are available:

* :obj:`ALLEGRO_HAND_CFG`: Allegro Hand with implicit actuator model.

Reference:

* https://www.wonikrobotics.com/robot-hand

"""

import math
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

UOA_HAND_CONFIG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lee/code/repose_hand/source/assets/uoa_hand_v5.usd",
        # VERY IMPORTANT
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        # Not using explicit tendon properties since we're simulating with joints
        activate_contact_sensors=False,  # Enable to get better feedback during interactions
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.05,  # Increased damping for tendon-like behavior
            max_linear_velocity=0.8,  # Lower max velocity for tendon-driven system
            max_angular_velocity=25.0,  # Lower max angular velocity
            max_depenetration_velocity=5.0,  # Lower for more stable contacts
            max_contact_impulse=1e4,  # Reduced for more realistic contact forces
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=12,  # Increased for better convergence
            solver_velocity_iteration_count=4,   # Added velocity iterations for tendon-like dynamics
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            fix_root_link=True,  # Fix the base in space
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.008,  # Increased for softer contacts
            rest_offset=0.002      # Added small rest offset for compliance
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.12, -0.16, 0.4),
        rot=(1., 0, 0, 0),
        # joint_pos={".*": 0.001},
        joint_pos={
            "thumb_carpal_1_thumb_abd0": 0.0,
            "middle_1_m_pp_1_middle_mcp0": -0.174,
            "palm_2_1_palm_abd0": 0.0,
            "index_1_i_pp_1_index_mcp0": -0.174,
            "thumb_1_t_pp_1_thumb_mcp0": -0.174,
            "middle_1_m_pp_1_middle_mcp1": 0.0,
            "ring_1_r_pp_1_ring_mcp0": -0.174,
            "pinky_1_p_pp_1_pinky_mcp0": -0.174,
            "index_1_i_pp_1_index_mcp1": 0.0,
            "thumb_1_t_pp_1_thumb_mcp1": -0.349,
            "middle_1_m_pp_1_middle_mcp2": -0.349,
            "ring_1_r_pp_1_ring_mcp1": 0.0,
            "pinky_1_p_pp_1_pinky_mcp1": 0.0,
            "index_1_i_pp_1_index_mcp2": -0.349,
            "thumb_1_t_pp_1_thumb_mcp2": -0.523,
            "middle_1_m_ip_1_m_pp_ip0": 0.0,
            "ring_1_r_pp_1_ring_mcp2": -0.349,
            "pinky_1_p_pp_1_pinky_mcp2": -0.349,
            "index_1_i_ip_1_i_pp_ip0": 0.0,
            "thumb_1_t_ip_1_t_pp_ip0": 0.0,
            "middle_1_m_dp_1_m_ip_pp0": 0.0,
            "ring_1_r_ip_1_r_pp_ip0": 0.0,
            "pinky_1_p_ip_1_p_pp_ip0": 0.0,
            "index_1_i_dp_1_i_ip_dp0": 0.0,
            "thumb_1_t_dp_1_t_ip_dp0": 0.0,
            "ring_1_r_dp_1_r_ip_dp0": 0.0,
            "pinky_1_p_dp_1_p_ip_dp0": 0.0,
        },
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=30.0,  # Reduced for tendon-like behavior
            # Realistic values for tendon-driven system modeled with joints:
            stiffness=4.5,           # Lower stiffness to simulate tendon elasticity
            damping=0.35,            # Higher damping to model energy dissipation in tendons
            friction=0.45,           # Moderate friction to simulate tendon routing friction
            dynamic_friction=0.35,   # Reduced dynamic friction
            effort_limit_sim=1.8,    # Lower effort limit to better match tendon force transmission
        ),
    },
    soft_joint_pos_limit_factor=0.92,  # More conservative limit for tendon system
)