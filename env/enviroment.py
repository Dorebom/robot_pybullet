from copy import deepcopy
import numpy as np
import pybullet as p
import gym
from gym import spaces

from env.robot import Manipulator
from env.work import Work

class Env():
    def __init__(self, reward, step_max_pos = 0.0005, step_max_orn = 0.01):
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
        self.step_max_pos = step_max_pos
        self.step_max_orn = step_max_orn
        self.inv_scaled_force_coef = 5000
        self.reward = reward
        self.step_counter = 0
        self.epoch_counter = 0

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

    def reset_epoch(self):
        self.step_counter   = 0
        self.epoch_counter += 1

    def step(self, action):
        self.step_counter += 1

        cmd_abs_tcp_pose = np.zeros(6)
        cmd_abs_tcp_pose[:3] = np.array(self.act_abs_tcp_pose[:3]) + np.array(action[:3])
        cmd_abs_tcp_pose[3:6] = np.array(self.act_abs_tcp_pose[3:6]) + np.array(action[3:6])

        self.robot.move_to_pose(cmd_abs_tcp_pose)

        pose, force, success, out_range = self.decision_process()

        r = self.calc_reward(relative_pose = pose,
                            success = success,
                            out_range = out_range,
                            act_step = self.step_counter)

        done = success or out_range

        return np.concatenate([pose, force]), r, done, success

    def decision_process(self):
        '''
        observe
        act_abs_tcp_pose
        act_rel_tcp_pose
        act_abs_work_pose
        act_force
        '''
        self.observe_state()
        scaled_act_force = self.act_force / self.inv_scaled_force_coef

        success_range_of_pos = 0.003
        success_range_of_orn = 0.02
        success = (np.linalg.norm(self.act_rel_tcp_pose[:3]) <= success_range_of_pos and \
                    np.linalg.norm(self.act_rel_tcp_pose[3:]) <= success_range_of_orn) 

        out_range_of_pos = 0.05
        out_range_of_orn = 0.8
        out_range = any([abs(pos) > out_range_of_pos for pos in self.act_rel_tcp_pose[:3]]) \
                or any([abs(orn) > out_range_of_orn for orn in self.act_rel_tcp_pose[3:6]])

        return self.act_rel_tcp_pose, scaled_act_force, success, out_range

    def observe_state(self):
        self.act_abs_tcp_pose, self.act_force = self.robot.get_state()
        self.act_abs_work_pose = self.work.get_state()

        self.act_rel_tcp_pose = np.array(self.act_abs_tcp_pose) - np.array(self.act_abs_work_pose)
        '''
        ノイズ処理
        '''
        return self.act_rel_tcp_pose, self.act_force

    def calc_reward(self, relative_pose, success, out_range, act_step):
        return self.reward.reward_function(relative_pose, success, out_range, act_step)

    def scale_action(self, action):
        scaled_action = deepcopy(action)
        scaled_action[:3]*=self.step_max_pos
        scaled_action[3:]*=self.step_max_orn
        return scaled_action