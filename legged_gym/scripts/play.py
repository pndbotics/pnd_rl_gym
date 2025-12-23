import sys
from legged_gym import LEGGED_GYM_ROOT_DIR
import os
import sys
from legged_gym import LEGGED_GYM_ROOT_DIR

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
# 设置matplotlib使用非交互式后端，避免PIL依赖问题
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict


def plot_data(data_log, dof_names, log_dir):
    """绘制机器人运动数据"""
    time = np.array(data_log['timestep'])
    num_dofs = len(dof_names)
    
    os.makedirs(log_dir, exist_ok=True)
    
    # 为每个关节创建转矩-转速相图
    for i in range(num_dofs):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 设置标题
        joint_name = dof_names[i]
        fig.suptitle(f'{joint_name} - Torque-Velocity Phase Plot', fontsize=14, fontweight='bold')
        
        # 获取转矩和转速数据
        torque = np.array(data_log[f'torque_{i}'])
        velocity = np.array(data_log[f'dof_vel_{i}'])
        
        # 绘制转矩-转速相图（散点图，按时间着色）
        scatter = ax.scatter(torque, velocity, c=time, cmap='viridis', 
                            alpha=0.6, s=10, edgecolors='none')
        
        # 添加颜色条（表示时间）
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Time (s)', fontsize=11)
        
        # 设置坐标轴
        ax.set_xlabel('Torque (N·m)', fontsize=12)
        ax.set_ylabel('Velocity (rad/s)', fontsize=12)
        ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        ax.axvline(x=0.0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        save_path = os.path.join(log_dir, f'torque_velocity_{i}_{joint_name}.png')
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Joint {i} ({joint_name}) torque-velocity plot saved to: {save_path}')
    
    # 创建基座速度、姿态和脚部接触力的综合图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Base Dynamics and Foot Contact Forces', fontsize=16, fontweight='bold')
    
    # 1. 基座线速度
    ax = axes[0, 0]
    ax.plot(time, data_log['lin_vel_x'], 'r-', label='vx (forward)', linewidth=2)
    ax.plot(time, data_log['lin_vel_y'], 'g-', label='vy (lateral)', linewidth=2)
    ax.plot(time, data_log['lin_vel_z'], 'b-', label='vz (vertical)', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='target vx=1.0')
    ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Base Linear Velocity (Body Frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 基座角速度
    ax = axes[0, 1]
    ax.plot(time, data_log['ang_vel_x'], 'r-', label='ωx (roll rate)', linewidth=2)
    ax.plot(time, data_log['ang_vel_y'], 'g-', label='ωy (pitch rate)', linewidth=2)
    ax.plot(time, data_log['ang_vel_z'], 'b-', label='ωz (yaw rate)', linewidth=2)
    ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Base Angular Velocity (Body Frame)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 基座姿态角
    ax = axes[1, 0]
    ax.plot(time, np.rad2deg(data_log['roll']), 'r-', label='Roll', linewidth=2)
    ax.plot(time, np.rad2deg(data_log['pitch']), 'g-', label='Pitch', linewidth=2)
    ax.plot(time, np.rad2deg(data_log['yaw']), 'b-', label='Yaw', linewidth=2)
    ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Base Orientation (Roll-Pitch-Yaw)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 脚部接触力（垂直方向）
    ax = axes[1, 1]
    num_feet = len([k for k in data_log.keys() if k.startswith('foot_force_z_')])
    colors = ['r', 'g', 'b', 'orange', 'purple', 'cyan']
    for i in range(num_feet):
        foot_name = f'Foot {i}'
        ax.plot(time, data_log[f'foot_force_z_{i}'], 
                color=colors[i % len(colors)], linewidth=2, label=foot_name, alpha=0.8)
    ax.axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Foot Contact Forces (Vertical)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存基座和脚部分析图
    save_path_base = os.path.join(log_dir, 'base_foot_analysis.png')
    fig.savefig(save_path_base, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Base and foot analysis plot saved to: {save_path_base}')
    
    # 打印统计信息
    print('\n=== Motion Statistics ===')
    print(f'Average forward velocity: {np.mean(data_log["lin_vel_x"]):.3f} m/s (target: 1.0 m/s)')
    print(f'Average lateral velocity: {np.mean(data_log["lin_vel_y"]):.3f} m/s')


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.curriculum = False  # Disable curriculum to use max_push_vel_xy directly
    env_cfg.domain_rand.push_interval_s = 3
    env_cfg.domain_rand.max_push_vel_xy = 1.5

    env_cfg.env.test = True
 
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    
    # load policy (skip if only testing default pose)
    policy = None
    if args.test_default_pose:
        print('Skipping policy loading in test_default_pose mode...')
    else:
        train_cfg.runner.resume = True
        ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=None)
        policy = ppo_runner.get_inference_policy(device=env.device)
        
        # export policy as a jit module (used to run it from C++)
        if EXPORT_POLICY:
            path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
            export_policy_as_jit(ppo_runner.alg.actor_critic, path)
            print('Exported policy as jit script to: ', path)

    # Set fixed velocity commands for all robots
    # commands: [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
    # env.commands[:, 0] = 0.0  # x方向速度 (m/s)
    # env.commands[:, 1] = 0.0  # y方向速度 (m/s)  
    # env.commands[:, 2] = 0.0  # yaw角速度 (rad/s)
    print(f'Set fixed velocity command: vx={env.commands[0, 0]:.2f} m/s, vy={env.commands[0, 1]:.2f} m/s, vyaw={env.commands[0, 2]:.2f} rad/s')
    
    # Print mode information
    if args.test_default_pose:
        print('\n=== TEST DEFAULT POSE MODE ===')
        print('Actions will be set to zero to test default joint angles.')
        print('Target angles: default_joint_angles from config file.')
        print('==============================\n')
    else:
        print('\n=== POLICY INFERENCE MODE ===')
        print('Using trained policy network to generate actions.')
        print('============================\n')

    # Data recording for plotting
    robot_id = 0  # 记录第一个机器人的数据
    data_log = defaultdict(list)
    
    # 获取关节名称和脚部索引
    dof_names = env.dof_names
    num_dofs = len(dof_names)
    feet_indices = env.feet_indices
    num_feet = len(feet_indices)
    
    print(f'DOF names: {dof_names}')
    print(f'Number of DOFs: {num_dofs}')
    print(f'Number of feet: {num_feet}')
    
    num_steps = 5 * int(env.max_episode_length)
    print(f'Running simulation for {num_steps} steps...')
    
    for i in range(num_steps):
        if args.test_default_pose:
            # Use zero actions to test default joint angles
            # Target angle = action_scale * action + default_angle
            # When action = 0, target angle = default_angle
            actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        else:
            actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        # Keep the velocity commands fixed (prevent automatic resampling)
        # env.commands[:, 0] = 0.6
        # env.commands[:, 1] = 0.0
        # env.commands[:, 2] = 0.0
        
        # Record data for the first robot (every step)
        # 基座线速度 (body frame)
        data_log['lin_vel_x'].append(env.base_lin_vel[robot_id, 0].cpu().item())
        data_log['lin_vel_y'].append(env.base_lin_vel[robot_id, 1].cpu().item())
        data_log['lin_vel_z'].append(env.base_lin_vel[robot_id, 2].cpu().item())
        
        # 基座角速度 (body frame)
        data_log['ang_vel_x'].append(env.base_ang_vel[robot_id, 0].cpu().item())
        data_log['ang_vel_y'].append(env.base_ang_vel[robot_id, 1].cpu().item())
        data_log['ang_vel_z'].append(env.base_ang_vel[robot_id, 2].cpu().item())
        
        # 基座姿态 (roll, pitch, yaw)
        data_log['roll'].append(env.rpy[robot_id, 0].cpu().item())
        data_log['pitch'].append(env.rpy[robot_id, 1].cpu().item())
        data_log['yaw'].append(env.rpy[robot_id, 2].cpu().item())
        
        # 关节转矩
        for j in range(num_dofs):
            data_log[f'torque_{j}'].append(env.torques[robot_id, j].cpu().item())
        
        # 关节速度
        for j in range(num_dofs):
            data_log[f'dof_vel_{j}'].append(env.dof_vel[robot_id, j].cpu().item())
        
        # 脚部接触力（垂直方向）
        for j in range(num_feet):
            foot_idx = feet_indices[j]
            force_z = env.contact_forces[robot_id, foot_idx, 2].cpu().item()
            data_log[f'foot_force_z_{j}'].append(force_z)
        
        # 时间步
        data_log['timestep'].append(i * env.dt)
    
    print('Simulation completed. Generating plots...')
    # 获取日志目录路径
    if args.test_default_pose:
        # In test_default_pose mode, save plots to a dedicated directory
        from datetime import datetime
        timestamp = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.task, f'test_default_pose_{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
        print(f'Saving plots to: {log_dir}')
    else:
        # Use the loaded model's directory
        from legged_gym.utils import get_load_path
        log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
        model_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
        log_dir = os.path.dirname(model_path)  # 获取模型所在的运行目录
    plot_data(data_log, dof_names, log_dir)
    print('Plots saved!')

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
