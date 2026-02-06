from legged_gym import LEGGED_GYM_ROOT_DIR
from typing import Union
import numpy as np
import time
import torch

from pndbotics_sdk_py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from pndbotics_sdk_py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from pndbotics_sdk_py.idl.default import pnd_adam_msg_dds__LowCmd_, pnd_adam_msg_dds__LowState_, pnd_adam_msg_dds__HandCmd_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowCmd_
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import LowState_ 
from pndbotics_sdk_py.idl.pnd_adam.msg.dds_ import HandCmd_

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_adam, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data, ypr_to_quaternion
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.policy = torch.jit.load(config.policy_path)
        # Initializing process variables
        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros(config.num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
       
        if config.msg_type == "adam_lite":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(23)
            self.low_state = pnd_adam_msg_dds__LowState_(23)
        elif config.msg_type == "adam_pro":
            self.low_cmd = pnd_adam_msg_dds__LowCmd_(31)
            self.low_state = pnd_adam_msg_dds__LowState_(31)
            self.hand_cmd = pnd_adam_msg_dds__HandCmd_()
            self.close_hand = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500], dtype=int)
            self.hand_pub = ChannelPublisher("rt/handcmd", HandCmd_)
            self.hand_pub.Init()
        else:
            raise ValueError("Invalid msg_type")
        self.mode_pr_ = MotorMode.PR
        self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmd_)
        self.lowcmd_publisher_.Init()


        self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowState_)
        self.lowstate_subscriber.Init(self.LowState_Handler, 10)

        # wait for the subscriber to receive data
        self.wait_for_low_state()
        init_cmd_adam(self.low_cmd, self.mode_pr_)

    def LowState_Handler(self, msg: LowState_):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmd_):
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            print("wait for low state")
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            # create_zero_cmd(self.low_cmd)
            # self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):

        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx

        kps = self.config.kps + self.config.arm_waist_kps
        kds = self.config.kds + self.config.arm_waist_kds
        default_pos = np.concatenate((self.config.default_angles, self.config.arm_waist_target), axis=0)
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            # hand publisher
            if config.msg_type != "adam_lite":
                for i in range(12):
                    self.hand_cmd.position[i] = self.close_hand[i]
                self.hand_pub.Write(self.hand_cmd)

            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
    



    def default_pos_state(self):
            
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            # hand publisher
            if config.msg_type != "adam_lite":
                for i in range(12):
                    self.hand_cmd.position[i] = self.close_hand[i]
                self.hand_pub.Write(self.hand_cmd)
            
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def run(self):

        # hand publisher
        if config.msg_type != "adam_lite":
            for i in range(12):
                self.hand_cmd.position[i] = self.close_hand[i]
            self.hand_pub.Write(self.hand_cmd)
            
        self.counter += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq

        # imu_state quaternion: w, x, y, z
        # quat = self.low_state.imu_state.quaternion
        quat = ypr_to_quaternion(self.low_state.imu_state.ypr[0],self.low_state.imu_state.ypr[1],self.low_state.imu_state.ypr[2])
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        if self.config.imu_type == "torso":
            # imu data needs to be transformed to the pelvis frame
            waist_yaw = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].q
            waist_yaw_omega = self.low_state.motor_state[self.config.arm_waist_joint2motor_idx[0]].dq
            quat, ang_vel = transform_imu_data(waist_yaw=waist_yaw, waist_yaw_omega=waist_yaw_omega, imu_quat=quat, imu_omega=ang_vel)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)
        qj_obs = self.qj.copy()
        dqj_obs = self.dqj.copy()
        qj_obs = (qj_obs - self.config.default_angles) * self.config.dof_pos_scale
        dqj_obs = dqj_obs * self.config.dof_vel_scale
        ang_vel = ang_vel * self.config.ang_vel_scale
        period = 0.8
        count = self.counter * self.config.control_dt
        phase = count % period / period
        sin_phase = np.sin(2 * np.pi * phase)
        cos_phase = np.cos(2 * np.pi * phase)

        self.cmd[0] = self.remote_controller.get_walk_x_direction_speed()
        self.cmd[1] = self.remote_controller.get_walk_y_direction_speed()
        self.cmd[2] = self.remote_controller.get_walk_yaw_direction_speed()

        num_actions = self.config.num_actions
        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.config.cmd_scale * self.config.max_cmd
        self.obs[9 : 9 + num_actions] = qj_obs
        self.obs[9 + num_actions : 9 + num_actions * 2] = dqj_obs
        self.obs[9 + num_actions * 2 : 9 + num_actions * 3] = self.action
        self.obs[9 + num_actions * 3] = sin_phase
        self.obs[9 + num_actions * 3 + 1] = cos_phase

        # Get the action from the policy network
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
            
        # transform action to target_dof_pos
        target_dof_pos = self.config.default_angles + self.action * self.config.action_scale

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = self.config.arm_waist_target[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.arm_waist_kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.arm_waist_kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command
        self.low_cmd.motor_cmd[5].q = self.low_state.motor_state[5].q  # keep ankle motor position
        self.low_cmd.motor_cmd[11].q = self.low_state.motor_state[11].q  # keep ankle motor position
        self.send_cmd(self.low_cmd)

        time.sleep(self.config.control_dt)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="adam_lite.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_real/configs/{args.config}"
    config = Config(config_path)

    # Initialize DDS communication
    ChannelFactoryInitialize(1, args.net)

    controller = Controller(config)

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.B] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
