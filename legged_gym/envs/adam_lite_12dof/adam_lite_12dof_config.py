from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class AdamLite12dofRoughCfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.95]  # x,y,z [m]
        # default_joint_angles = {  # = target angles [rad] when action = 0.0
        #     'hipPitch_Left': -0.32,     #0
        #     'hipRoll_Left': 0.0,        #1
        #     'hipYaw_Left': -0.18,       #2
        #     'kneePitch_Left': 0.66,     #3
        #     'anklePitch_Left': -0.29,   #4
        #     'ankleRoll_Left': -0.0,     #5

        #     'hipPitch_Right': -0.32,    #6
        #     'hipRoll_Right': -0.,       #7
        #     'hipYaw_Right': 0.18,       #8
        #     'kneePitch_Right': 0.66,    #9
        #     'anklePitch_Right': -0.29,  #10
        #     'ankleRoll_Right': 0.0,     #11
        # }

        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'hipPitch_Left': -0.1,
            'hipRoll_Left': 0,
            'hipYaw_Left': 0.,
            'kneePitch_Left': 0.3,
            'anklePitch_Left': -0.2,
            'ankleRoll_Left': 0,
            'hipPitch_Right': -0.1,
            'hipRoll_Right': 0,
            'hipYaw_Right': 0.,
            'kneePitch_Right': 0.3,
            'anklePitch_Right': -0.2,
            'ankleRoll_Right': 0,
        }

    class env(LeggedRobotCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12
        episode_length_s = 10
        
    class safety:
        # safety factors
        pos_limit = 0.9
        vel_limit = 0.9
        torque_limit = 0.85
    
    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-0.05, 0.05]
        push_robots = True  # Enable robot pushing for disturbance training
        push_interval_s = 5
        max_push_vel_xy = 0.2
        # Curriculum learning for push_robots
        curriculum = True  # Enable curriculum learning for push disturbances
        max_push_vel_xy_curriculum = 0.5  # Maximum push velocity to reach through curriculum
        initial_push_vel_xy = 0.0  # Initial push velocity (easier at start)

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        # PD Drive parameters - tuned for stability and tracking
        stiffness = {
            'hipPitch': 305.0,   # 230 Nm effort in URDF
            'hipRoll': 700.0,    # 160 Nm effort in URDF
            'hipYaw': 405.0,      # 105 Nm effort in URDF
            'kneePitch': 305.0,  # 230 Nm effort in URDF
            'anklePitch': 25.0,  # 40 Nm effort in URDF
            'ankleRoll': 0.0,   # 12 Nm effort in URDF
        }  # [N*m/rad]
        damping = {
            'hipPitch': 6.1,
            'hipRoll': 30.0,
            'hipYaw': 6.1,
            'kneePitch': 6.1,
            'anklePitch': 2.55,
            'ankleRoll': 0.35,
        }  # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25  # 0.5 corresponds to about 30 degrees range
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/adam_lite/adam_lite_12dof.urdf'
        # file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/adam_pro/adam_pro_12dof.urdf'
        name = "adam_lite_12dof"
        foot_name = "toe"  # The foot links are toeLeft and toeRight
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["pelvis", "knee", "hip", "torso"]
        penalize_contacts_on = ['pelvis', 'shin', 'shoulder', 'elbow', 'thigh', 'torso']
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.9  # Target height for pelvis
        min_dist = 0.2
        max_dist = 0.5
        class scales(LeggedRobotCfg.rewards.scales):
            # Velocity Tracking
            tracking_lin_vel = 1.0      # Track commanded linear velocity
            tracking_ang_vel = 0.5      # Track commanded angular velocity
            lin_vel_z = -2.0            # Penalize vertical velocity
            ang_vel_xy = -0.05          # Penalize roll/pitch angular velocity
            
            # Pose Stability
            base_height = -2.0          # Maintain target height
            
            # Joint Control
            dof_acc = -2.5e-7           # Penalize joint accelerations
            dof_vel = -1e-3             # Penalize high joint velocities
            dof_pos_limits = -5.0       # Penalize approaching joint limits
            # hip_pos = -0.5              # Regularize hip positions
            ankle_pos = -1.0            # Keep ankles stable (fix toe-up & roll issues)
            action_rate = -0.01         # Smooth actions (penalize rapid changes)
            
            # Foot Contact & Gait
            feet_air_time = 0.05        # Small value to lift feet without jumping
            feet_swing_height = -30.0   # Ensure proper swing clearance
            contact = 1.0               # Reward foot contact配合实现相位同步）
            contact_no_vel = -0.1       # Penalize foot contact while moving
            feet_distance = 1.5       # Penalize feet getting too close or too far away
            orientation = 1.0

            action_smoothness = -0.01    # Encourage smooth actions
            # Symmetry & Gait Quality
            feet_lateral_deviation = -5.0  # Keep feet pointing forward
            
            # Safety & Survival
            collision = -1.0            # Penalize collisions with forbidden body parts
            alive = 0.15                # Encourage staying alive
            
            # Optional (currently disabled)
            # torques = -0.00001        # Penalize high torques (encourage efficiency)
            # foot_slip = -0.1          # Penalize lateral foot slip during stance

class AdamLite12dofRoughCfgPPO(LeggedRobotCfgPPO):
    class policy:
        init_noise_std = 0.5
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 128
        rnn_num_layers = 1
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        schedule = 'adaptive'
        num_learning_epochs = 5
        num_mini_batches = 4
        desired_kl = 0.01
        clip_param = 0.2
    
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        max_iterations = 6000
        run_name = 'action_scale_0.25_high_pos_terrian'
        experiment_name = 'adam_lite_12dof'
        num_steps_per_env = 32
        
        # logging
        save_interval = 100 # check for potential saves every this many iterations
        # load and resume
        resume = False
        load_run = -1 #"Oct17_14-14-05_push*" #"Oct16_22-05-20_" #"Oct16_12-36-25_" #"Oct14_21-46-15_*" #"Oct14_21-46-15_*" #"Oct14_21-46-15_" #"Oct15_01-10-49_" #"Oct14_02-52-45_" #"Oct13_02-15-48_" # -1 = last run
        checkpoint = -1# -1 = last saved model

        
        # WandB Configuration
        use_wandb = True  # Enable WandB logging
        wandb_project = 'pnd_adam_lite_12dof_locomotion'  # WandB project name
        wandb_entity = None  # WandB entity (optional)
        wandb_tags = ['adam_lite_12dof', 'rough_terrain', 'lstm']  # Tags

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 0.4


