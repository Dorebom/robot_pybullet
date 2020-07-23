import sys
import time
import numpy as np

from env.enviroment import Env

if __name__ == '__main__':

    robot_base_pose = [0, 0, 0, 0, 0, 0]
    robot_tool_pose = [0, 0, 0, 0, 0, 0]
    robot_tcp_pose = [0.3, 0, 0.2, 0, 0, 0]

    _env = Env()
    _env.load_robot(tool_pose = robot_tool_pose, \
                    base_pose = robot_base_pose, \
                    tcp_pose = robot_tcp_pose)

    time.sleep(10)