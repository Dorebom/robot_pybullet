import numpy as np
import pybullet as p
import gym
from gym import spaces

from env.robot import Manipulator
from env.work import Work

class Env():
    def __init__(self, reward):
        p.connect(p.GUI)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.setRealTimeSimulation(False)
        p.setGravity(0, 0, -9.8)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

        # Plane
        p.loadURDF("urdf/plane/plane.urdf", [0, 0, -0.1])

        # for learning
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(6,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=1,
            shape=(12,),
            dtype=np.float32
        )
        self.reward = reward


    def load(self, robot_tcp_pose = [0, 0, 0, 0, 0, 0], \
            robot_base_pose = [0, 0, 0, 0, 0, 0], \
            robot_tool_pose = [0, 0, 0, 0, 0, 0], \
            work_base_pose = [0, 0, 0, 0, 0, 0]):

        self._load_robot(tcp_pose = robot_tcp_pose, \
                        base_pose = robot_base_pose, \
                        tool_pose = robot_tool_pose)

        self.work = Work(base_pose = work_base_pose)

    def _load_robot(self, tcp_pose = [0, 0, 0, 0, 0, 0], \
                    base_pose = [0, 0, 0, 0, 0, 0], \
                    tool_pose = [0, 0, 0, 0, 0, 0]):

        self.robot = Manipulator(tool_pose=tool_pose, base_pose=base_pose)
        self.robot.reset_pose(tcp_pose=tcp_pose)

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

    def observe_state(self):
        act_tcp_pose, act_force = self.robot.get_state()
        act_work_pose = self.work.get_state()

        rel_tcp_pose = np.array(act_tcp_pose) - np.array(act_work_pose)

        '''
        ノイズ処理
        '''

        return rel_tcp_pose, act_force

    def step(self, action):
        pass

    def get_reward(self, relative_pose, success, act_step):
        return self.reward.reward_function(relative_pose, success, act_step)