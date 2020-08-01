from copy import deepcopy
from datetime import datetime
import collections
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time

class Trainer():
    """
    __init__
    _tarin
    _reset
    train
    """

    def __init__(self, agent,
                start_steps = 1000,
                steps_per_episode=150,
                episodes=20):
        self.agent = agent
        self.start_steps = start_steps
        self.total_step_counter = 0
        self.steps_per_episode = steps_per_episode
        self.episodes = episodes

        self.step_from_start_episode = 0
        self.ep = 0
        self.act_step = 0

        # Create a logger
        str_date = datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
        self.writer = SummaryWriter(log_dir='runs/'+str_date)

    def _init(self):
        # env init
        self.robot_base_pose = np.array([0, 0, 0, 0, 0, 0])
        self.work_base_pose = np.array([0.5, 0, 0, 0, 0, 0])
        self.robot_tool_pose = np.array([0.0, 0.0, -0.15, 0, 0, 0])
        self.rel_robot_tcp_pose = np.array([0.0, 0.0, 0.08, 0, 0, 0])
        #self.rel_robot_tcp_pose[:3] += np.random.uniform(-0.02, 0.02, 3)
        #self.rel_robot_tcp_pose[3:] += np.random.uniform(-0.02, 0.02, 3)

        act_rel_tcp_pose, act_force = self.agent.env.init_env(mode='rel',
                                        robot_base_pose=self.robot_base_pose,
                                        robot_tool_pose=self.robot_tool_pose,
                                        robot_tcp_pose=self.rel_robot_tcp_pose,
                                        work_base_pose=self.work_base_pose)

        return np.concatenate([act_rel_tcp_pose, act_force])

    def _reset(self):

        self.robot_base_pose = np.array([0, 0, 0, 0, 0, 0])
        self.work_base_pose = np.array([0.5, 0, 0, 0, 0, 0])
        self.robot_tool_pose = np.array([0.0, 0.0, -0.15, 0, 0, 0])
        self.rel_robot_tcp_pose = np.array([0.0, 0.0, 0.08, 0, 0, 0])
        #self.rel_robot_tcp_pose[:3] += np.random.uniform(-0.02, 0.02, 3)
        #self.rel_robot_tcp_pose[3:] += np.random.uniform(-0.02, 0.02, 3)

        act_rel_tcp_pose, act_force = self.agent.env.reset(mode='rel',
                            tcp_pose=self.rel_robot_tcp_pose,
                            base_pose=self.robot_base_pose,
                            tool_pose=self.robot_tool_pose,
                            work_pose=self.work_base_pose)

        return np.concatenate([act_rel_tcp_pose, act_force])


    def _train(self, obs):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.

        # [要確認]最初のstepだけobsを取得

        if self.total_step_counter > self.start_steps:
            a = self.agent.get_action(obs)
        else:
            a = self.agent.env.action_space.sample()

        scaled_a = self.agent.env.scale_action(a)

        scaled_a = [0, 0, -0.001, 0, 0, 0]

        new_obs, r, done, success = self.agent.env.step(action = scaled_a, step = self.act_step)

        self.agent.replay_buffer.store(obs, a, r, new_obs, done)

        self.obs = new_obs

        # Update handling
        if self.total_step_counter >= self.agent.update_after \
            and self.total_step_counter % self.agent.update_every == 0:
            for j in range(self.agent.update_every):
                batch = self.agent.replay_buffer.sample_batch(self.agent.batch_size)
                self.agent.update(data=batch)

        return obs, r, done, success

    def train(self):
        # initialize
        all_rewards, episode_success, episodes_reward = [], [], []
        tmp_episode_reward = 0
        step_from_start_episode = 0

        obs = self._init()

        while self.ep < self.episodes:

            obs, r, done, success = self._train(obs)

            tmp_episode_reward += r

            self.act_step += 1

            # End of episode handling and next starting
            if done or (self.act_step + 1) % self.steps_per_episode == 0 :
                # End process
                step_from_start_episode = 0
                self.ep += 1
                episodes_reward.append(deepcopy(tmp_episode_reward))
                self.writer.add_scalar('reward', tmp_episode_reward, self.ep)
                tmp_episode_reward = 0

                distance = np.linalg.norm(obs[:3])
                self.writer.add_scalar('distance', distance, self.ep)

                episode_success.append(success)
                self.writer.add_scalar('success', sum(episode_success), self.ep)

                # Next starting
                self.act_step = 0
                obs = self._reset()