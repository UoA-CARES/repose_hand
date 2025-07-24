import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg



UOA_HAND_CONFIG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lee/code/repose_cube/source/assets/uoa_hand_test_2.usd"),
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
    soft_joint_pos_limit_factor=0.95,  # Use 95% of joint limits (default is 1.0)
)