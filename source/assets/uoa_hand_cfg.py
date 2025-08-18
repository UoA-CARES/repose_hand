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
# UOA_HAND_CONFIG = ArticulationCfg(
#     # prim_path="{ENV_REGEX_NS}/Robot",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/lee/code/repose_cube/source/assets/uoa_hand_test_3.usd"),
#     actuators={
#         "fingers": ImplicitActuatorCfg(
#             joint_names_expr=[".*"],
#             velocity_limit_sim=60.0,  # deg/s (deg because of USD convention)
#             stiffness=1.0,
#             damping=0.1,
#             friction=0.1,
#             effort_limit_sim=1.0,
#         ),
#     },
#     soft_joint_pos_limit_factor=0.95,  # Use 95% of joint limits (default is 1.0)
# )
UOA_HAND_CONFIG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lee/code/repose_hand/source/assets/uoa_hand_v5.usd",
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        # fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1.0,
            max_angular_velocity=40,
            max_depenetration_velocity=10.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
            fix_root_link=True,  # Fix the base in space
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.07, -0.16, 0.5),
        rot=(1., 0, 0, 0),
        joint_pos={".*": 0.001},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=50.0,  # deg/s (deg because of USD convention)
            stiffness=1.0,
            damping=0.5,
            friction=0.1,
            effort_limit_sim=0.9,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)