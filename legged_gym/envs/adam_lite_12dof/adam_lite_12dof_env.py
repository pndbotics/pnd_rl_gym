
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse

class AdamLite12dofRobot(LeggedRobot):
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize push curriculum if enabled
        if self.cfg.domain_rand.curriculum:
            self.push_vel_xy = self.cfg.domain_rand.initial_push_vel_xy
        else:
            self.push_vel_xy = self.cfg.domain_rand.max_push_vel_xy
    
    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0.  # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0.  # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0.  # sin/cos phase
        
        return noise_vec

    def _init_foot(self):
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_buffers(self):
        super()._init_buffers()
        self._init_foot()

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        self.update_feet_state()

        period = 0.8
        offset = 0.5
        self.phase = (self.episode_length_buf * self.dt) % period / period
        self.phase_left = self.phase
        self.phase_right = (self.phase + offset) % 1
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)
        
        return super()._post_physics_step_callback()
    
    
    def compute_observations(self):
        """ Computes observations
        """
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        self.obs_buf = torch.cat((  self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ), dim=-1)
        self.privileged_obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    sin_phase,
                                    cos_phase
                                    ), dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        
    def _reward_contact(self):
        res = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            is_stance = self.leg_phase[:, i] < 0.55
            contact = self.contact_forces[:, self.feet_indices[i], 2] > 1
            
            # print 10 envs' contact and is_stance
            # if self.common_step_counter % 100 == 0:
            #     print(f"contact: {contact}, is_stance: {is_stance}")
            res += ~(contact ^ is_stance)
        return res
    
    def _reward_feet_swing_height(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        pos_error = torch.square(self.feet_pos[:, :, 2] - 0.1) * ~contact
        return torch.sum(pos_error, dim=(1))
    
    def _reward_alive(self):
        # Reward for staying alive
        return 1.0
    
    def _reward_contact_no_vel(self):
        # Penalize contact with no velocity
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        contact_feet_vel = self.feet_vel * contact.unsqueeze(-1)
        penalize = torch.square(contact_feet_vel[:, :, :3])
        return torch.sum(penalize, dim=(1, 2))
    
    def _reward_hip_pos(self):
        # Penalize excessive hip roll and hip yaw deviations
        # For V-shaped hip, allow moderate deviations for coordination
        # Indices for hipRoll_Left, hipYaw_Left, hipRoll_Right, hipYaw_Right
        # Based on joint order: hipPitch_L(0), hipRoll_L(1), hipYaw_L(2), kneePitch_L(3), anklePitch_L(4), ankleRoll_L(5),
        #                       hipPitch_R(6), hipRoll_R(7), hipYaw_R(8), kneePitch_R(9), anklePitch_R(10), ankleRoll_R(11)
        return torch.sum(torch.square(self.dof_pos[:, [1, 2, 7, 8]]), dim=1)
    
    def _reward_ankle_pos(self):
        # Penalize ankle deviations from default position
        # Problem: anklePitch pointing up too much, ankleRoll moving randomly
        # Solution: Keep ankles close to default position
        # Indices: anklePitch_L(4), ankleRoll_L(5), anklePitch_R(10), ankleRoll_R(11)
        ankle_deviation = self.dof_pos[:, [4, 5, 10, 11]] - self.default_dof_pos[:, [4, 5, 10, 11]]
        return torch.sum(torch.square(ankle_deviation), dim=1)
    
    def _reward_foot_slip(self):
        # Penalize lateral (Y-axis) foot movement during stance phase
        # This encourages straight-line leg motion despite V-shaped hip structure
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # Lateral velocity (Y component in world frame)
        foot_lateral_vel = self.feet_vel[:, :, 1]  # Y-axis velocity
        # Only penalize during contact
        slip = torch.square(foot_lateral_vel) * contact
        return torch.sum(slip, dim=1)
    
    def _reward_feet_lateral_deviation(self):
        """
        Penalize foot orientation to keep feet pointing forward (prevent 外八字)
        This measures the actual orientation of the foot link in body frame
        The foot orientation is determined by all leg joints working together
        """
        # Get base quaternion
        base_quat = self.root_states[:, 3:7]  # (num_envs, 4) - w, x, y, z
        
        # Get foot orientations from rigid body states (quaternions at indices 3:7)
        feet_quat = self.feet_state[:, :, 3:7]  # (num_envs, num_feet, 4) - w, x, y, z
        
        # Get body's forward direction in world frame
        forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, 3)
        body_forward_world = quat_apply(base_quat, forward_vec)  # (num_envs, 3)
        body_forward_world = body_forward_world.unsqueeze(1).expand(-1, self.feet_num, -1)  # (num_envs, num_feet, 3)
        
        # Reshape for batch quat_apply: (num_envs * num_feet, 4)
        feet_quat_flat = feet_quat.reshape(-1, 4)
        
        # Foot's local forward direction is X-axis
        foot_local_forward = torch.tensor([1.0, 0.0, 0.0], device=self.device)
        foot_local_forward_flat = foot_local_forward.unsqueeze(0).expand(self.num_envs * self.feet_num, 3)
        
        # Apply foot quaternion to get foot's forward in world frame
        foot_forward_world_flat = quat_apply(feet_quat_flat, foot_local_forward_flat)  # (num_envs * num_feet, 3)
        foot_forward_world = foot_forward_world_flat.reshape(self.num_envs, self.feet_num, 3)
        
        # Calculate yaw angle difference in world frame (project to XY plane)
        body_forward_xy = body_forward_world[:, :, :2]  # (num_envs, num_feet, 2)
        foot_forward_xy = foot_forward_world[:, :, :2]
        
        # Normalize
        body_forward_xy = body_forward_xy / (torch.norm(body_forward_xy, dim=2, keepdim=True) + 1e-8)
        foot_forward_xy = foot_forward_xy / (torch.norm(foot_forward_xy, dim=2, keepdim=True) + 1e-8)
        
        # Calculate angle using atan2
        body_yaw = torch.atan2(body_forward_xy[:, :, 1], body_forward_xy[:, :, 0])
        foot_yaw = torch.atan2(foot_forward_xy[:, :, 1], foot_forward_xy[:, :, 0])
        
        # Angle difference
        yaw_diff = foot_yaw - body_yaw
        # Normalize to [-pi, pi]
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        
        # Check contact for debug info
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        
        # Penalize yaw deviation from zero (feet should point same direction as body)
        # Use exp penalty for stronger effect on large deviations
        # For small angles (< 5 deg), penalty is mild; for large angles (> 15 deg), penalty grows rapidly
        yaw_error = torch.square(yaw_diff)
        penalty = torch.sum(yaw_error, dim=1)
        
        # Debug print: only show first env, every 100 steps
        # if self.common_step_counter % 100 == 0:
        #     for i in range(self.feet_num):
        #         foot_name = "Left" if i == 0 else "Right"
        #         contact_str = "Contact" if contact[0, i].item() else "Swing"
        #         yaw_deg = yaw_diff[0, i].item() * 180.0 / 3.14159
        #         print(f"[Env 0] {foot_name} foot ({contact_str}): yaw={yaw_deg:.2f}°, error={yaw_error[0, i].item():.6f}")
        
        return penalty
    
    def _reward_feet_swing_lateral_vel(self):
        """
        Penalize lateral velocity during swing phase (when foot is in air)
        Encourages feet to move in sagittal plane only
        """
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # During swing phase (not in contact), penalize lateral velocity
        foot_lateral_vel = self.feet_vel[:, :, 1]  # Y-axis velocity in world frame
        swing_lateral_penalty = torch.square(foot_lateral_vel) * (~contact)
        return torch.sum(swing_lateral_penalty, dim=1)
    
    def _reward_action_smoothness(self):
        """
        Penalize the second derivative of actions (jerk)
        Encourages smooth, gradual changes in motion
        """
        if not hasattr(self, 'last_last_actions'):
            self.last_last_actions = torch.zeros_like(self.actions)
            return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        # Calculate second derivative: (a[t] - a[t-1]) - (a[t-1] - a[t-2])
        action_diff_1 = self.actions - self.last_actions
        action_diff_2 = self.last_actions - self.last_last_actions
        action_jerk = action_diff_1 - action_diff_2
        
        # Update history
        self.last_last_actions = self.last_actions.clone()
        
        return torch.sum(torch.square(action_jerk), dim=1)
    
    def update_push_curriculum(self, env_ids):
        """ Implements curriculum learning for push disturbances.
        Gradually increases push velocity as the robot becomes more stable.
        Uses hysteresis mechanism to prevent oscillation.
        
        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if not self.cfg.domain_rand.curriculum:
            return
        
        # Initialize curriculum tracking variables
        if not hasattr(self, 'push_curriculum_counter'):
            self.push_curriculum_counter = 0
            self.push_performance_history = []
        
        # Only update curriculum every N resets to avoid oscillation
        # 每100次reset才评估一次，避免频繁调整
        self.push_curriculum_counter += 1
        if self.push_curriculum_counter % 100 != 0:
            return
        
        # Calculate stability metric: combination of tracking performance and survival
        if len(env_ids) > 0:
            # Use tracking reward and alive reward as indicators of stability
            tracking_reward = self.episode_sums.get("tracking_lin_vel", torch.zeros(self.num_envs, device=self.device))[env_ids]
            alive_reward = self.episode_sums.get("alive", torch.zeros(self.num_envs, device=self.device))[env_ids]
            
            # Normalize by episode length
            avg_tracking = torch.mean(tracking_reward) / self.max_episode_length
            avg_alive = torch.mean(alive_reward) / self.max_episode_length
            
            # Store in history for moving average (keep last 5 evaluations)
            performance_score = avg_tracking.item() + avg_alive.item()
            self.push_performance_history.append(performance_score)
            if len(self.push_performance_history) > 5:
                self.push_performance_history.pop(0)
            
            # Use moving average to smooth out noise
            smoothed_performance = np.mean(self.push_performance_history)
            
            # Hysteresis thresholds: 不同的增加和减少阈值，防止震荡
            # 增加难度需要更高的性能（保守）
            if "tracking_lin_vel" in self.reward_scales:
                increase_threshold = 0.85 * self.reward_scales["tracking_lin_vel"]  # 需要达到85%才增加
                decrease_threshold = 0.40 * self.reward_scales["tracking_lin_vel"]  # 低于40%才减少
            else:
                increase_threshold = 0.85
                decrease_threshold = 0.40
            
            if "alive" in self.reward_scales:
                alive_increase_threshold = 0.90 * self.reward_scales["alive"]
                alive_decrease_threshold = 0.50 * self.reward_scales["alive"]
            else:
                alive_increase_threshold = 0.135  # 0.9 * 0.15
                alive_decrease_threshold = 0.075  # 0.5 * 0.15
            
            old_push_vel = self.push_vel_xy
            
            # 增加难度：需要同时满足tracking和alive的高阈值
            if avg_tracking > increase_threshold and avg_alive > alive_increase_threshold:
                # 小步增加，避免突然变难
                self.push_vel_xy = np.clip(
                    self.push_vel_xy + 0.1,  # 从0.1改为0.05，更平滑
                    self.cfg.domain_rand.initial_push_vel_xy,
                    self.cfg.domain_rand.max_push_vel_xy_curriculum
                )
                if self.push_vel_xy != old_push_vel:
                    print(f"✅ [Curriculum] Push velocity increased: {old_push_vel:.2f} → {self.push_vel_xy:.2f} m/s "
                          f"(tracking: {avg_tracking:.3f}, alive: {avg_alive:.3f})")
            
            # 减少难度：任一指标过低就减少（更快响应困难）
            elif avg_tracking < decrease_threshold or avg_alive < alive_decrease_threshold:
                # 减少时步长更大，快速降低难度帮助恢复
                self.push_vel_xy = np.clip(
                    self.push_vel_xy - 0.1,  # 减少时稍快一些
                    self.cfg.domain_rand.initial_push_vel_xy,
                    self.cfg.domain_rand.max_push_vel_xy_curriculum
                )
                if self.push_vel_xy != old_push_vel:
                    print(f"⚠️  [Curriculum] Push velocity decreased: {old_push_vel:.2f} → {self.push_vel_xy:.2f} m/s "
                          f"(tracking: {avg_tracking:.3f}, alive: {avg_alive:.3f})")
            
            # 在稳定区间内（40%-85%）不做调整，避免震荡
    
    def _push_robots(self):
        """ Random pushes the robots with curriculum-based velocity.
        Overrides base class method to use curriculum learning.
        """
        if not self.cfg.domain_rand.push_robots:
            return
            
        env_ids = torch.arange(self.num_envs, device=self.device)
        push_env_ids = env_ids[self.episode_length_buf[env_ids] % int(self.cfg.domain_rand.push_interval) == 0]
        if len(push_env_ids) == 0:
            return
        
        # Use curriculum-adjusted push velocity instead of fixed max_push_vel_xy
        max_vel = self.push_vel_xy
        self.root_states[push_env_ids, 7:9] = torch_rand_float(
            -max_vel, max_vel, (len(push_env_ids), 2), device=self.device
        )  # lin vel x/y
        
        env_ids_int32 = push_env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32), 
            len(env_ids_int32)
        )
    
    def reset_idx(self, env_ids):
        """ Reset some environments and update curriculum.
        Overrides base class to add push curriculum updates.
        
        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        # Update push curriculum before resetting
        if self.cfg.domain_rand.curriculum and len(env_ids) > 0:
            self.update_push_curriculum(env_ids)
        
        # Call parent reset
        super().reset_idx(env_ids)
        
        # Add push curriculum info to logging
        if self.cfg.domain_rand.curriculum:
            if "episode" not in self.extras:
                self.extras["episode"] = {}
            self.extras["episode"]["push_vel_xy"] = self.push_vel_xy

    def _process_dof_props(self, props, env_id):
        """Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(
                self.num_dof,
                2,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
            self.dof_vel_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            self.torque_limits = torch.zeros(
                self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
            )
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = (
                    props["lower"][i].item() * self.cfg.safety.pos_limit
                )
                self.dof_pos_limits[i, 1] = (
                    props["upper"][i].item() * self.cfg.safety.pos_limit
                )
                self.dof_vel_limits[i] = (
                    props["velocity"][i].item() * self.cfg.safety.vel_limit
                )
                self.torque_limits[i] = (
                    props["effort"][i].item() * self.cfg.safety.torque_limit
                )
        return props
    
    def _reward_feet_distance(self):
        """
        Calculates the reward based on the distance between the feet. Penalize feet get close to each other or too far away.
        """
        foot_pos = self.rigid_state[:, self.feet_indices, :2]
        foot_dist = torch.norm(foot_pos[:, 0, :] - foot_pos[:, 1, :], dim=1)
        fd = self.cfg.rewards.min_dist
        max_df = self.cfg.rewards.max_dist
        d_min = torch.clamp(foot_dist - fd, -0.5, 0.0)
        d_max = torch.clamp(foot_dist - max_df, 0.0, 0.5)
        return (
            torch.exp(-torch.abs(d_min) * 100) + torch.exp(-torch.abs(d_max) * 100)
        ) / 2
    
    def _reward_orientation(self):
        """
        Calculates the reward for maintaining a flat base orientation. It penalizes deviation
        from the desired base orientation using the base euler angles and the projected gravity vector.
        """
        quat_mismatch = torch.exp(
            -torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10
        )
        orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
        return (quat_mismatch + orientation) / 2.0