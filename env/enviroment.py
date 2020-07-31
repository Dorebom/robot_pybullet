from copy import deepcopy
import numpy as np
import pybullet as p
import gym
from gym import spaces

from env.robot import Manipulator
from env.work import Work

class Env():
    def __init__(self, reward,
                step_max_pos = 0.0005,
                step_max_orn = 0.01,
                initial_pos_noise = 0.001,
                initial_orn_noise = 0.001,
                step_pos_noise = 0.0002,
                step_orn_noise = 0.0002):
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

        self.max_initial_pos_noise = initial_pos_noise
        self.max_initial_orn_noise = initial_orn_noise
        self.max_step_pos_noise = step_pos_noise
        self.max_step_orn_noise = step_orn_noise

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

    def reset(self, mode = 'abs',
                    tcp_pose = [0, 0, 0, 0, 0, 0],
                    base_pose = [0, 0, 0, 0, 0, 0],
                    tool_pose = [0, 0, 0, 0, 0, 0],
                    work_pose = [0, 0, 0, 0, 0, 0]):

        self.robot.remove()
        p.resetSimulation()
        # Plane
        p.loadURDF("urdf/plane/plane.urdf", [0, 0, -0.1])

        self.work.reset(base_pose = work_pose)

        if mode == 'abs':
            self.robot.reset(tcp_pose = tcp_pose, \
                            base_pose = base_pose, \
                            tool_pose = tool_pose)
        elif mode == 'rel':
            abs_tcp_pose = np.array(work_pose) + np.array(tcp_pose)
            self.robot.reset(tcp_pose = abs_tcp_pose, \
                            base_pose = base_pose, \
                            tool_pose = tool_pose)

        return self.decision_process()

    def destory(self):
        p.disconnect()

    def step(self, action, step):

        # ここは指令値生成なので，真値が良い
        cmd_abs_tcp_pose = np.zeros(6)
        cmd_abs_tcp_pose[:3] = np.array(self.act_abs_tcp_pose[:3]) + np.array(action[:3])
        cmd_abs_tcp_pose[3:6] = np.array(self.act_abs_tcp_pose[3:6]) + np.array(action[3:6])

        self.robot.move_to_pose(cmd_abs_tcp_pose)

        pose, force, success, out_range = self.decision_process()

        r = self.calc_reward(relative_pose = pose,
                            success = success,
                            out_range = out_range,
                            act_step = step)

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
        mode = 'rel'

        act_pose_noisy, act_force = self.observe_state(mode)
        scaled_act_force = act_force / self.inv_scaled_force_coef

        # [Note] ここは真値で評価
        success_range_of_pos = 0.003
        success_range_of_orn = 0.02
        success = (np.linalg.norm(self._act_rel_tcp_pose[:3]) <= success_range_of_pos and \
                    np.linalg.norm(self._act_rel_tcp_pose[3:]) <= success_range_of_orn) 

        # [Note] ここは真値で評価は正しくない気がする．
        out_range_of_pos = 0.05
        out_range_of_orn = 0.8
        out_range = any([abs(pos) > out_range_of_pos for pos in act_pose_noisy[:3]]) \
                or any([abs(orn) > out_range_of_orn for orn in act_pose_noisy[3:6]])

        return act_pose_noisy, scaled_act_force, success, out_range

    def observe_state(self, mode = 'abs'):
        self._act_abs_tcp_pose, self._act_force = self.robot.get_state()
        self._act_abs_work_pose = self.work.get_state()

        self._act_rel_tcp_pose = np.array(self._act_abs_tcp_pose) - np.array(self._act_abs_work_pose)
        '''
        ノイズ処理
        '''
        act_return_tcp_pose_noisy = np.zeros(6)
        if mode == 'rel':
            act_return_tcp_pose_noisy = self._act_rel_tcp_pose
        elif mode == 'abs':
            act_return_tcp_pose_noisy = self._act_abs_tcp_pose

        act_return_tcp_pose_noisy[:3] += self.initial_pos_noise
        act_return_tcp_pose_noisy[3:6] += self.initial_orn_noise
        act_return_tcp_pose_noisy[:3] += np.random.uniform(-self.max_step_pos_noise,
                                                        self.max_step_pos_noise, 3)
        act_return_tcp_pose_noisy[3:6] += np.random.uniform(-self.max_step_orn_noise,
                                                        self.max_step_orn_noise, 3)

        return act_return_tcp_pose_noisy, self.act_force

    def calc_reward(self, relative_pose, success, out_range, act_step):
        return self.reward.reward_function(relative_pose, success, out_range, act_step)

    def scale_action(self, action):
        scaled_action = deepcopy(action)
        scaled_action[:3]*=self.step_max_pos
        scaled_action[3:]*=self.step_max_orn
        return scaled_action