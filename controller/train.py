

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
            a = self.agent.get_action(obs)
        else:
            a = self.agent.env.action_space.sample()

        scaled_a = self.agent.env.scale_action(a)
        obs, r, done, success = self.agent.env.step(scaled_a)

        # [要確認]状態量の捉え方が甘い気がする．確認しよう．