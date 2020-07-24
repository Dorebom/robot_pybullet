import sys
import time
import numpy as np

from env.enviroment import Env

if __name__ == '__main__':

    robot_base_pose = [0, 0, 0, 0, 0, 0]
    robot_tool_pose = [0, 0, -0.2, 0, 0, 0]
    robot_tcp_pose = [0.5, 0, 0.2, 0, 0, 0]

    work_base_pose = [0.4, 0, 0, 0, 0, 0]

    _env = Env()
    _env.load_robot(tool_pose = robot_tool_pose, \
                    base_pose = robot_base_pose, \
                    tcp_pose = robot_tcp_pose)
    _env.load_work(work_base_pose)
    time.sleep(3)

    robot_base_pose = [0, 0, 0, 0, 0, 0]
    robot_tool_pose = [0.0, -0.4, -0.2, 0, 0, 0]
    robot_tcp_pose = [0.5, 0, 0.2, 0, 0, 0]

    _env.reset(tool_pose = robot_tool_pose, \
               base_pose = robot_base_pose, \
               tcp_pose = robot_tcp_pose, \
               work_pose = work_base_pose)

    time.sleep(3)

    act_tcp_pose, act_force = _env.robot.get_state()

    cmd_tcp_pose = np.array(act_tcp_pose) + np.array([0.0005, 0.0, 0.0, 0.0, 0.0, 0.0])

    _env.robot.move_to_pose(cmd_tcp_pose)

    time.sleep(3)

    _env.destory()