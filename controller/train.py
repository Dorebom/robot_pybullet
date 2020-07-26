

class Trainer():
    def __init__(self, agent,
                epochs = 50,
                start_steps = 1000):
        self.agent = agent
        self.epochs = epochs
        self.start_steps = start_steps
        self.total_step_counter = 0

    def train(self):
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
        if total_step_counter >= self.agent.update_after and self.total_step_counter % self.agent.update_every == 0:
            for j in range(self.agent.update_every):
                batch = self.agent.replay_buffer.sample_batch(self.agent.batch_size)
                self.agent.update(data=batch)

        # End of episode handling



    def init_epoch(self)

