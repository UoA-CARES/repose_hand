# # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from isaaclab.utils import configclass
# import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg
# from isaaclab_assets import ALLEGRO_HAND_CFG  # isort: skip
from isaaclab_assets.robots.shadow_hand import SHADOW_HAND_CFG
from isaaclab_assets import ALLEGRO_HAND_CFG

import sys
sys.path.append("/home/lee/code/repose_cube")
from source.assets.uoa_hand_cfg import UOA_HAND_CONFIG  # isort: skip
# from source.assets import UOA_HAND_CFG
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveGaussianNoiseCfg as Gnoise

import isaaclab_tasks.manager_based.manipulation.inhand.mdp as mdp

##
# Scene definition
##


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    object_pose = mdp.InHandReOrientationCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=0.1,
        make_quat_unique=False,
        marker_pos_offset=(-0.2, -0.06, 0.08),
        debug_vis=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        alpha=0.95,
        rescale_to_limits=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class KinematicObsGroupCfg(ObsGroup):
        """Observations with full-kinematic state information.

        This does not include acceleration or force information.
        """

        # observation terms (order preserved)
        # -- robot terms
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, noise=Gnoise(std=0.005))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, noise=Gnoise(std=0.01))

        # -- object terms
        object_pos = ObsTerm(
            func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False}
        )
        object_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w,
            scale=0.2,
            noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )

        # -- command terms
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
        )

        # -- action terms
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoVelocityKinematicObsGroupCfg(KinematicObsGroupCfg):
        """Observations with partial kinematic state information.

        In contrast to the full-kinematic state group, this group does not include velocity information
        about the robot joints and the object root frame. This is useful for tasks where velocity information
        is not available or has a lot of noise.
        """

        def __post_init__(self):
            # call parent post init
            super().__post_init__()
            # set unused terms to None
            self.joint_vel = None
            self.object_lin_vel = None
            self.object_ang_vel = None

    # observation groups
    policy: KinematicObsGroupCfg = KinematicObsGroupCfg()


@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",  # Changed from startup to reset to match shadow hands tasks
        # min_step_count_between_reset=720,  # Added parameter from shadow hands tasks
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),  # Updated to match shadow hands tasks
            "restitution_range": (1.0, 1.0),  # Updated to match shadow hands tasks
            "num_buckets": 250,
        },
    )
    
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",  # Changed from startup to reset
        # min_step_count_between_reset=720,  # Added parameter
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),  # Updated to match shadow hands tasks
            "damping_distribution_params": (0.3, 3.0),  # Updated to match shadow hands tasks
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    
    # Added joint position limits randomization from shadow hands tasks
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        # min_step_count_between_reset=720,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),  # Fixed typo
            "operation": "add",
            "distribution": "gaussian",
        },
    )
    
    # Added tendon properties randomization from shadow hands tasks
    robot_tendon_properties = EventTerm(
        func=mdp.randomize_fixed_tendon_parameters,
        # min_step_count_between_reset=720,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", fixed_tendon_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",  # Changed from startup to reset
        min_step_count_between_reset=720,  # Added parameter
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),  # Updated to match shadow hands tasks
            "restitution_range": (1.0, 1.0),  # Updated to match shadow hands tasks
            "num_buckets": 250,
        },
    )
    
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",  # Changed from startup to reset
        min_step_count_between_reset=720,  # Added parameter
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),  # Updated to match shadow hands tasks
            "operation": "scale",
            "distribution": "uniform",  # Added distribution parameter
        },
    )
    
    # # Added gravity randomization from shadow hands tasks
    # reset_gravity = EventTerm(
    #     func=mdp.randomize_physics_scene_gravity,
    #     mode="interval",
    #     is_global_time=True,
    #     interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
    #     params={
    #         "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
    #         "operation": "add",
    #         "distribution": "gaussian",
    #     },
    # )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            "position_range": {".*": [0.2, 0.2]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # track_pos_l2 = RewTerm(
    #     func=mdp.track_pos_l2,
    #     weight=-10.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    # )
    track_orientation_inv_l2 = RewTerm(
        func=mdp.track_orientation_inv_l2,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.1, "command_name": "object_pose"},
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=250.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    )

    # -- penalties
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- optional penalties (these are disabled by default)
    # object_away_penalty = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=-0.0,
    #     params={"term_keys": "object_out_of_reach"},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success, params={"num_success": 50, "command_name": "object_pose"}
    )

    object_out_of_reach = DoneTerm(func=mdp.object_away_from_robot, params={"threshold": 0.6})

    # object_out_of_reach = DoneTerm(
    #     func=mdp.object_away_from_goal, params={"threshold": 0.24, "command_name": "object_pose"}
    # )

@configclass
class InHandObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # robot configuration updated to match shadow hands tasks
    robot: ArticulationCfg = UOA_HAND_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")
    # .replace(
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(0.0, 0.0, 0.5),
    #         rot=(1.0, 0.0, 0.0, 0.0),
    #         joint_pos={".*": 0.0},
    #     )
    # )

    # object configuration updated with more physics properties from shadow hands tasks
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        # 
        # 0.0, -0.19, 0.5
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.95), intensity=1000.0),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.02, 0.02, 0.02), intensity=1000.0),
    )

    # Update scene spacing to match shadow hands tasks
    def __post_init__(self):
        super().__post_init__()
        self.env_spacing = 0.75
        self.replicate_physics = True

##
# Environment configuration
##

@configclass
class InHandObjectEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the in hand reorientation environment."""

    # Scene settings
    scene: InHandObjectSceneCfg = InHandObjectSceneCfg(num_envs=8192, env_spacing=0.6)
    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        # change viewer settings
        self.viewer.eye = (2.0, 2.0, 2.0)


@configclass
class ReposeCubeEnvCfg(InHandObjectEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to allegro hand
        # self.scene.robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ReposeCubeEnvCfg_PLAY(ReposeCubeEnvCfg):
    """Smaller environment configuration for playing/testing."""
    
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None

##
# Environment configuration with no velocity observations.
##
