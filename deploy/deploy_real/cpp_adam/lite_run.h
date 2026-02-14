#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <unitree/idl/pnd_adam/LowCmd_.hpp>
#include <unitree/idl/pnd_adam/LowState_.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/common/time/time_tool.hpp>

#include "torch/script.h"

#include <eigen3/Eigen/Eigen>

#include "joystick.h"
#include "DataBuffer.h"
#include <string>

#include "remote_controller.hpp"
class Controller
{
	public:
		Controller(const std::string &net_interface);
		~Controller();
		void low_state_message_handler(const void *message);
		void move_to_default_pos();
		void run();
		void damp();
		void zero_torque_state();
		void compute_obs();
		void compute_act();
	private:
		void low_cmd_write_handler();

		unitree::common::ThreadPtr low_cmd_write_thread_ptr;

		DataBuffer<pnd_adam::msg::dds_::LowCmd_> mLowCmdBuf;
		DataBuffer<pnd_adam::msg::dds_::LowState_> mLowStateBuf;

		unitree::robot::ChannelPublisherPtr<pnd_adam::msg::dds_::LowCmd_> lowcmd_publisher;
		unitree::robot::ChannelSubscriberPtr<pnd_adam::msg::dds_::LowState_> lowstate_subscriber;

		// joystick
		// xRockerBtnDataStruct joy;
		RemoteController joystick;
		// yaml config
		std::vector<float> joint_idx;
		std::vector<float> kps_;
		std::vector<float> kds_;
		std::vector<float> default_angles_;
		
		float ang_vel_scale;
		float dof_pos_scale;
		float dof_vel_scale;
		float action_scale;
		std::vector<float> cmd_scale;

		int num_actions;
		int num_obs;
		int delta_num;
		int frame_stack;
		int latent_frame_stack;
		int latent_size;

		float cycle_time;
		float gait_phase;
		float sin_phase;
		float cos_phase;

		std::vector<float> max_cmd;
		Eigen::Vector3f command = Eigen::Vector3f::Zero();
		Eigen::Vector3f ang_vel = Eigen::Vector3f::Zero();

		Eigen::VectorXf obs;
		Eigen::VectorXf act;
		Eigen::VectorXf act_v;

		// history for estimator / policy (对应 StateMLP 中的 hist_obs / vae_obs)
		Eigen::VectorXf hist_obs;
		Eigen::VectorXf vae_obs;

		// 对应 StateMLP 中的 input_array / est_input_array
		float *input_array;
		float *est_input_array;
		float avg_yaw_vel = 0.0;
		int obs_delta_num;
		int est_input_num;
		int policy_input_num;
		int rl_counter = 0;
		// 对应 StateMLP 中的 input_data_mlp_humanoid / output_data_mlp 等
		Eigen::VectorXf input_data_mlp_humanoid;
		Eigen::VectorXf output_data_mlp;
		Eigen::VectorXf action_last;
		Eigen::VectorXf last_action_d;
		Eigen::VectorXf last_action_dot_d;
  		float predictive_time = 0.02;
		// para
		Eigen::VectorXf para_0;
		Eigen::VectorXf para_1;
		Eigen::VectorXf para_2;
		Eigen::VectorXf para_3;
		bool gait_d = 1;
		bool gait_a = 1;
  		float timer_plan = 0.0;
		torch::jit::script::Module policy;
		torch::jit::script::Module estimator;
		int fre_count = 0;
};

#endif
