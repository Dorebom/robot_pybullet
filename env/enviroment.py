import numpy as np
import pybullet as p

from env.robot import Manipulator


class Env():
    def __init__(self):
        p.connect(p.GUI)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setRealTimeSimulation(False)


    def load_robot(self, tcp_pose = [0, 0, 0, 0, 0, 0], \
                    base_pose = [0, 0, 0, 0, 0, 0], \
                    tool_pose = [0, 0, 0, 0, 0, 0]):

        robot = Manipulator(tool_pose=tool_pose, base_pose=base_pose)
        robot.reset_pose(tcp_pose=tcp_pose)

    