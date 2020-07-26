from datetime import datetime
import collections
from torch.utils.tensorboard import SummaryWriter

class Trainer():
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

        # Create a logger
        self.writer = SummaryWriter("runs/"+datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))

    def _train(self):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.

        # [要確認]最初のstepだけobsを取得

        if self.total_step_counter > self.start_steps:
            a = self.agent.get_action(self.obs)
        else:
            a = self.agent.env.action_space.sample()

        scaled_a = self.agent.env.scale_action(a)
        new_obs, r, done, success = self.agent.env.step(scaled_a)

        self.agent.replay_buffer.store(obs, a, r, new_obs, done)

        self.obs = new_obs

        # Update handling
        if total_step_counter >= self.agent.update_after \
            and self.total_step_counter % self.agent.update_every == 0:
            for j in range(self.agent.update_every):
                batch = self.agent.replay_buffer.sample_batch(self.agent.batch_size)
                self.agent.update(data=batch)

        return obs, r, done, success

    def init_episode(self):
        pass


    def train(self):
        # initialize
        all_rewards, episode_success, episodes_reward = [], [], []
        tmp_episode_reward = 0
        step_from_start_episode = 0

        while ep < self.episodes:
            obs, r, done, success = self._train()

            tmp_episode_reward += r

            # End of episode handling and next starting
            if done or (step_from_start_episode+1) % self.steps_per_episode == 0 :
                # End process
                step_from_start_episode = 0
                ep += 1
                episodes_reward.append(copy.deepcopy(tmp_episode_reward))
                self.writer.add_scalar('reward', tmp_episode_reward, ep)
                tmp_episode_reward = 0

                distance = np.linalg.norm(obs[:3])
                sac.writer.add_scalar('distance', distance, ep)

                episode_success.append(success)
                self.writer.add_scalar('success', sum(episode_success), ep)

                # Next starting
                base_pose = np.array([0.0,0,0.1,0,0,0])
                base_pose[:3]+=np.random.uniform(-0.02, 0.02, 3)
                base_pose[3:]+=np.random.uniform(-0.02, 0.02, 3)
                obs = sac.env.reset(base_pose)