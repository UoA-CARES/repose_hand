import math
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg



# UOA_HAND_CONFIG = ArticulationCfg(
#     # prim_path="{ENV_REGEX_NS}/Robot",
#     spawn=sim_utils.UsdFileCfg(
#         usd_path="/home/lee/code/repose_cube/source/assets/uoa_hand_test_2.usd"),
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
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/lee/code/repose_cube/source/assets/uoa_hand_test_2.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # TODO
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(0, 0, 0, 0),
        joint_pos={".*": 0.01},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit_sim=60.0,  # deg/s (deg because of USD convention)
            stiffness=1.0,
            damping=0.1,
            friction=0.1,
            effort_limit_sim=1.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)