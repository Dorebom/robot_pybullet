import time
from env import reward
from env.enviroment import Env
from controller.sac import Sac, ReplayBuffer

if __name__ == '__main__':

    total_step = 100

    _reward = reward.BasicReward(success_reward = 10000,
                                range_out_pos = 0.2,
                                range_out_orn = 0.8,
                                total_step = total_step,
                                scale_pos_coef = 1000,
                                scale_orn_coef = 1000)
    _env = Env(reward = _reward)

    robot_base_pose = [0, 0, 0, 0, 0, 0]
    robot_tool_pose = [0, 0, -0.1, 0, 0, 0]
    robot_tcp_pose = [0.5, 0, 0.05, 0, 0, 0]

    work_base_pose = [0.5, 0, 0, 0, 0, 0]
    _env.load(robot_tool_pose = robot_tool_pose, \
                    robot_base_pose = robot_base_pose, \
                    robot_tcp_pose = robot_tcp_pose, \
                    work_base_pose = work_base_pose)

    _sac = Sac(_env)



    time.sleep(1)