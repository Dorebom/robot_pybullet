import numpy as np
import pybullet as p

from env.robot import Manipulator
from env.work import Work

class Env():
    def __init__(self):
        p.connect(p.GUI)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setRealTimeSimulation(False)
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

        # Plane
        p.loadURDF("urdf/plane/plane.urdf", [0, 0, -0.1])

    def load_robot(self, tcp_pose = [0, 0, 0, 0, 0, 0], \
                    base_pose = [0, 0, 0, 0, 0, 0], \
                    tool_pose = [0, 0, 0, 0, 0, 0]):

        self.robot = Manipulator(tool_pose=tool_pose, base_pose=base_pose)
        self.robot.reset_pose(tcp_pose=tcp_pose)

    def load_work(self, base_pose = [0, 0, 0, 0, 0, 0]):
        self.work = Work(base_pose = base_pose)

    def reset(self, tcp_pose = [0, 0, 0, 0, 0, 0], \
                    base_pose = [0, 0, 0, 0, 0, 0], \
                    tool_pose = [0, 0, 0, 0, 0, 0], \
                    work_pose = [0, 0, 0, 0, 0, 0]):
        self.robot.remove()
        p.resetSimulation()
        # Plane
        p.loadURDF("urdf/plane/plane.urdf", [0, 0, -0.1])

        self.work.reset(base_pose = work_pose)

        self.robot.reset(tcp_pose = tcp_pose, \
                         base_pose = base_pose, \
                         tool_pose = tool_pose)

    def destory(self):
        p.disconnect()

