import sys
import time
import numpy as np

from env import reward
from env.enviroment import Env

if __name__ == '__main__':

    robot_base_pose = [0, 0, 0, 0, 0, 0]
    robot_tool_pose = [0, 0, -0.2, 0, 0, 0]
    robot_tcp_pose = [0.5, 0, 0.2, 0, 0, 0]

    work_base_pose = [0.4, 0, 0, 0, 0, 0]
    total_step = 100

    _reward = reward.BasicReward(success_reward = 10000,
                                range_out_pos = 0.2,
                                range_out_orn = 0.8,
                                total_step = total_step,
                                scale_pos_coef = 1000,
                                scale_orn_coef = 1000)
    _env = Env(reward = _reward)
    _env.load(robot_tool_pose = robot_tool_pose, \
                    robot_base_pose = robot_base_pose, \
                    robot_tcp_pose = robot_tcp_pose, \
                    work_base_pose = work_base_pose)
    time.sleep(1)

    robot_base_pose = [0, 0, 0, 0, 0, 0]
    robot_tool_pose = [0.0, -0.4, -0.2, 0, 0, 0]
    robot_tcp_pose = [0.5, 0, 0.2, 0, 0, 0]

    _env.reset(tool_pose = robot_tool_pose, \
                base_pose = robot_base_pose, \
                tcp_pose = robot_tcp_pose, \
                work_pose = work_base_pose)

    time.sleep(1)

    act_tcp_pose, act_force = _env.observe_state()

    #cmd_tcp_pose = np.array(act_tcp_pose) + np.array([0.0005, 0.0, 0.0, 0.0, 0.0, 0.0])

    #_env.robot.move_to_pose(cmd_tcp_pose)

    time.sleep(1)

    _env.destory()