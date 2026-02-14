#include "lite_run.h"
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <thread>
#include <cstdlib>
#include "utilities.h"

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

Controller::Controller(const std::string &net_interface)
{
	// yaml config
	YAML::Node yaml_node = YAML::LoadFile("../../configs/lite_run.yaml");

	joint_idx = yaml_node["joint_idx"].as<std::vector<float>>();
	kps_ = yaml_node["kps"].as<std::vector<float>>();
	kds_ = yaml_node["kds"].as<std::vector<float>>();
	default_angles_ = yaml_node["default_angles"].as<std::vector<float>>();
	ang_vel_scale = yaml_node["ang_vel_scale"].as<float>();
	dof_pos_scale = yaml_node["dof_pos_scale"].as<float>();
	dof_vel_scale = yaml_node["dof_vel_scale"].as<float>();
	action_scale = yaml_node["action_scale"].as<float>();
	cmd_scale = yaml_node["cmd_scale"].as<std::vector<float>>();
	num_actions = yaml_node["num_actions"].as<float>();
	num_obs = yaml_node["num_obs"].as<float>();
	delta_num = yaml_node["delta_num"].as<int>();
	frame_stack = yaml_node["frame_stack"].as<int>();
	latent_frame_stack = yaml_node["latent_frame_stack"].as<int>();
	latent_size = yaml_node["latent_size"].as<int>();
	cycle_time = yaml_node["cycle_time_run"].as<float>();
	max_cmd = yaml_node["max_cmd"].as<std::vector<float>>();

	// history and buffers
	// 这里 obs_delta_num 对应训练时的 MLP 单帧观测维度（例如 93），
	// 可能大于当前实际使用的 num_obs（例如 81），多出的维度用 0 填充。
	obs_delta_num = 93;
	est_input_num = obs_delta_num * latent_frame_stack;
	// policy 输入为 (obs_delta_num * frame_stack + latent_size)
	policy_input_num = obs_delta_num * frame_stack + latent_size;
	
	input_array = new float[policy_input_num];
	est_input_array = new float[est_input_num];

	obs = Eigen::VectorXf::Zero(num_obs);
	act = Eigen::VectorXf::Zero(num_actions);
	act_v = Eigen::VectorXf::Zero(num_actions);
	hist_obs = Eigen::VectorXf::Zero(obs_delta_num * frame_stack);
	vae_obs = Eigen::VectorXf::Zero(obs_delta_num * latent_frame_stack);

	input_data_mlp_humanoid = Eigen::VectorXf::Zero(obs_delta_num);
	output_data_mlp = Eigen::VectorXf::Zero(num_actions + delta_num);
	action_last = Eigen::VectorXf::Zero(num_actions + delta_num);
	last_action_d = Eigen::VectorXf::Zero(num_actions + delta_num);
	last_action_dot_d = Eigen::VectorXf::Zero(num_actions + delta_num);

	para_0 = Eigen::VectorXf::Zero(num_actions + delta_num);
	para_1 = Eigen::VectorXf::Zero(num_actions + delta_num);
	para_2 = Eigen::VectorXf::Zero(num_actions + delta_num);
	para_3 = Eigen::VectorXf::Zero(num_actions + delta_num);

	policy = torch::jit::load("../../../pre_train/adam_lite/policy_aug5.pt");
	estimator = torch::jit::load("../../../pre_train/adam_lite/estimator_aug5.pt");
	std::cout << "Model loaded\n";

	// dds init
	unitree::robot::ChannelFactory::Instance()->Init(1, net_interface);

	pnd_adam::msg::dds_::LowCmd_(0, std::vector<pnd_adam::msg::dds_::MotorCmd_>(23), 0);
	pnd_adam::msg::dds_::LowState_(0, 0, pnd_adam::msg::dds_::IMUState_(), std::vector<pnd_adam::msg::dds_::MotorState_>(23), std::array<float,19>{}, 0);
	lowcmd_publisher.reset(new unitree::robot::ChannelPublisher<pnd_adam::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
	lowstate_subscriber.reset(new unitree::robot::ChannelSubscriber<pnd_adam::msg::dds_::LowState_>(TOPIC_LOWSTATE));

	lowcmd_publisher->InitChannel();
	lowstate_subscriber->InitChannel(std::bind(&Controller::low_state_message_handler, this, std::placeholders::_1));

	while (!mLowStateBuf.GetDataPtr())
	{
		usleep(100000);
	}
	
	low_cmd_write_thread_ptr = unitree::common::CreateRecurrentThreadEx("low_cmd_write", UT_CPU_ID_NONE, 2000, &Controller::low_cmd_write_handler, this);
	std::cout << "Controller init done!\n";
}

Controller::~Controller()
{
	// low_cmd_write_thread_ptr->Stop();
	delete[] input_array;
	delete[] est_input_array;
}

void Controller::zero_torque_state()
{
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	std::cout << "zero_torque_state, press start\n";
	while (!joystick.button[KeyMap::start])
	{
		auto low_cmd = std::make_shared<pnd_adam::msg::dds_::LowCmd_>(0, std::vector<pnd_adam::msg::dds_::MotorCmd_>(23), 0);

		for (auto &cmd : low_cmd->motor_cmd())
		{
			cmd.q() = 0;
			cmd.dq() = 0;
			cmd.kp() = 0;
			cmd.kd() = 0;
			cmd.tau() = 0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);
		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}

void Controller::move_to_default_pos()
{
	std::cout << "move_to_default_pos, press A\n";
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	auto low_state = mLowStateBuf.GetDataPtr();	
	std::array<float, 23> jpos;
	for (int i = 0; i < 23; i++)
	{
		jpos[i] = low_state->motor_state()[i].q();
	}

	int num_steps = 100;
	int count = 0;

	while (count <= num_steps || !joystick.button[KeyMap::A]) 
	{
		auto low_cmd = std::make_shared<pnd_adam::msg::dds_::LowCmd_>(0, std::vector<pnd_adam::msg::dds_::MotorCmd_>(23), 0);
		float phase = std::clamp<float>(float(count++) / num_steps, 0, 1);
		
		// leg
		for (int i = 0; i < 23; i++)
		{
			low_cmd->motor_cmd()[i].q() = (1 - phase) * jpos[i] + phase * default_angles_[i];
			low_cmd->motor_cmd()[i].kp() = kps_[i];
			low_cmd->motor_cmd()[i].kd() = kds_[i];
			low_cmd->motor_cmd()[i].tau() = 0.0;
			low_cmd->motor_cmd()[i].dq() = 0.0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}

void Controller::compute_obs()
{
	auto low_state = mLowStateBuf.GetDataPtr();
	Eigen::Matrix3f R = Eigen::Quaternionf(low_state->imu_state().quaternion()[0], low_state->imu_state().quaternion()[1], low_state->imu_state().quaternion()[2], low_state->imu_state().quaternion()[3]).toRotationMatrix();
	command[0] = joystick.get_walk_x_direction_speed() * max_cmd[0] * cmd_scale[0];
	command[1] = joystick.get_walk_y_direction_speed() * max_cmd[1] * cmd_scale[1];
	command[2] = joystick.get_walk_yaw_direction_speed() * max_cmd[2] * cmd_scale[2];
	ang_vel = Eigen::Vector3f(low_state->imu_state().gyroscope()[0], low_state->imu_state().gyroscope()[1], low_state->imu_state().gyroscope()[2]);
    avg_yaw_vel = (1.0 * 0.01 / cycle_time) * ang_vel(2) + (1.0 - 1.0 * 0.01 / cycle_time) * avg_yaw_vel;  
	gait_phase = rl_counter * 0.01 / cycle_time; 
	sin_phase = sin(2.0 * M_PI * gait_phase);
	cos_phase = cos(2.0 * M_PI * gait_phase);
	
	// Input Obs for MLP Humanoid
	input_data_mlp_humanoid.segment(0, 6) << command[0], command[1], command[2],-R(2,0), -R(2,1), -R(2,2);
	for(int i = 0; i < 23; i++)
	{
		input_data_mlp_humanoid(6 + i) = (low_state->motor_state()[i].q() - default_angles_[i]) * dof_pos_scale;
		input_data_mlp_humanoid(29 + i) = low_state->motor_state()[i].dq() * dof_vel_scale;
	}
	input_data_mlp_humanoid.segment(52, 35) =  action_last;
	for(int i = 0; i < 3; i++){
		input_data_mlp_humanoid(87 + i) = ang_vel_scale * low_state->imu_state().gyroscope()[i];
	}
	input_data_mlp_humanoid(90) = avg_yaw_vel * dof_vel_scale; //  1/91
	input_data_mlp_humanoid(91) = sin_phase; //  1/92
	input_data_mlp_humanoid(92) = cos_phase; //  1/93

//   input_data_mlp_humanoid.segment(87, 3) = omega_filter->mFilter(input_data_mlp_humanoid.segment(87, 3));

	input_data_mlp_humanoid =
		input_data_mlp_humanoid.cwiseMax(-18.0).cwiseMin(18.0);

  // est_input_array fresh
  vae_obs.head(vae_obs.size() - obs_delta_num) = vae_obs.tail(vae_obs.size() - obs_delta_num);
  vae_obs.tail(obs_delta_num) = input_data_mlp_humanoid.segment(0, obs_delta_num);
  for (int i = 0; i < vae_obs.size(); i++)
    est_input_array[i] = vae_obs[i];

  // input_array fresh
  hist_obs.head(hist_obs.size() - obs_delta_num) = hist_obs.tail(hist_obs.size() - obs_delta_num);
  hist_obs.tail(obs_delta_num) = input_data_mlp_humanoid.segment(0, obs_delta_num);
  for (int i = 0; i < hist_obs.size(); i++)
  input_array[i] = hist_obs[i];
  std::cout << "g:" << input_data_mlp_humanoid.segment(3,3).transpose() << std::endl;
}

void Controller::compute_act()
{
 std::vector<torch::jit::IValue> inputs_est;
  auto input_data_est = torch::zeros(est_input_num).toType(torch::kFloat);
  input_data_est = torch::from_blob(est_input_array, est_input_num);
  input_data_est.to(torch::kCPU);
  inputs_est.emplace_back(input_data_est);
  // 打印 inputs 中的张量数据

  torch::Tensor output_data_est = estimator.forward(inputs_est).toTensor();
  

  for (int i = 0; i < latent_size; i++)
  {
    input_array[est_input_num + i] = output_data_est[i].item<float>();
    
  }
  // std::cout << "vel_x_e:" << input_array[est_input_num] << std::endl;

  std::vector<torch::jit::IValue> inputs;
  auto input_data = torch::zeros(policy_input_num).toType(torch::kFloat);
  input_data = torch::from_blob(input_array, policy_input_num);
  input_data.to(torch::kCPU);
  inputs.emplace_back(input_data);

  torch::Tensor output_data = policy.forward(inputs).toTensor();
  std::vector<float> out(output_data.data_ptr<float>(),
                         output_data.data_ptr<float>() + output_data.numel());
  for (int i = 0; i < 23 + delta_num; i++)
  {
    if (gait_a)
    {
      output_data_mlp(i) = std::max(std::min(out[i], (float)18.0), (float)-18.0);
      para_0 = last_action_d;
      para_1 = last_action_dot_d;
      para_2 = 3.0 *
                   (output_data_mlp - last_action_d -
                    last_action_dot_d * predictive_time) /
                   predictive_time / predictive_time -
               (-last_action_dot_d) / predictive_time;
      para_3 = -2.0 *
                   (output_data_mlp - last_action_d -
                    last_action_dot_d * predictive_time) /
                   predictive_time / predictive_time / predictive_time +
               (-last_action_dot_d) / predictive_time / predictive_time;
      timer_plan = 0.0;
    }
  }

    if(rl_counter < 34){
      rl_counter += 1;
    }else{
      rl_counter += 1;
    }
  action_last = output_data_mlp;
}

void Controller::run()
{
	std::cout << "run controller, press select\n";

	using namespace std::chrono;

	const microseconds cycle_time(2500);   // 2.5ms = 400Hz
	auto next_cycle = steady_clock::now();
	auto low_cmd = std::make_shared<pnd_adam::msg::dds_::LowCmd_>(0,std::vector<pnd_adam::msg::dds_::MotorCmd_>(23),0);

	while (!joystick.button[KeyMap::B])
	{
		auto loop_start = steady_clock::now();

		// ===== 读取状态 =====
		auto low_state = mLowStateBuf.GetDataPtr();

		// ===== 100Hz策略 =====
		if (fre_count % 4 == 0)
		{
			compute_obs();
			compute_act();
		}
		Eigen::VectorXf mlp_out = output_data_mlp;
		Eigen::VectorXf mlp_out_dot = Eigen::VectorXf::Zero(23 + delta_num);
		if (gait_a)
		{
			timer_plan += 0.0025;
			mlp_out = para_0 + para_1 * timer_plan + para_2 * timer_plan * timer_plan +
					para_3 * timer_plan * timer_plan * timer_plan;
			mlp_out_dot = para_1 + 2.0 * para_2 * timer_plan +
						3.0 * para_3 * timer_plan * timer_plan;
		}
		act = mlp_out.segment(0, 23) * action_scale;
		// act_v.setZero();
		act_v.segment(0, 12) = mlp_out.segment(23, 12) * action_scale;

		// ===== 构造控制指令 =====

		for (int i = 0; i < 23; ++i)
		{
			low_cmd->motor_cmd()[i].q()  = act(i) + default_angles_[i];
			low_cmd->motor_cmd()[i].kp() = kps_[i];
			low_cmd->motor_cmd()[i].kd() = kds_[i];
			low_cmd->motor_cmd()[i].dq() = act_v(i);
			low_cmd->motor_cmd()[i].tau() = 0;
		}

		// mLowCmdBuf.SetDataPtr(low_cmd);
		last_action_d = mlp_out;
		last_action_dot_d = mlp_out_dot;
		// ===== 精确周期控制 =====
		next_cycle += cycle_time;

		// 如果已经落后太多，直接重置时间基准（防止死循环追赶）
		if (steady_clock::now() > next_cycle + cycle_time)
		{
			next_cycle = steady_clock::now();
		}

		std::this_thread::sleep_until(next_cycle);

		fre_count++;
	}

}

void Controller::damp()
{
	std::cout << "damping\n";
	const std::chrono::milliseconds cycle_time(20);
	auto next_cycle = std::chrono::steady_clock::now();

	while (true)
	{
		auto low_cmd = std::make_shared<pnd_adam::msg::dds_::LowCmd_>();
		for (auto &cmd : low_cmd->motor_cmd())
		{
			cmd.kp() = 0;
			cmd.kd() = 8;
			cmd.dq() = 0;
			cmd.tau() = 0;
		}

		mLowCmdBuf.SetDataPtr(low_cmd);

		next_cycle += cycle_time;
		std::this_thread::sleep_until(next_cycle);
	}
}


void Controller::low_state_message_handler(const void *message)
{
	pnd_adam::msg::dds_::LowState_* ptr = (pnd_adam::msg::dds_::LowState_*)message;
	mLowStateBuf.SetData(*ptr);
	joystick.set(ptr->wireless_remote());
}

void Controller::low_cmd_write_handler()
{
	if (auto lowCmdPtr = mLowCmdBuf.GetDataPtr())
	{
		// lowCmdPtr->mode_machine() = mLowStateBuf.GetDataPtr()->mode_machine();
		lowCmdPtr->mode_pr() = 0;
		for (auto &cmd : lowCmdPtr->motor_cmd())
		{
			cmd.mode() = 1;
		}
		// lowCmdPtr->crc() = crc32_core((uint32_t*)(lowCmdPtr.get()), (sizeof(pnd_adam::msg::dds_::LowCmd_) >> 2) - 1);
		lowcmd_publisher->Write(*lowCmdPtr);
	}
}
