import time
from env import reward
from env.enviroment import Env
from controller.sac import SoftActorCritic, ReplayBuffer
from controller.train import Trainer

if __name__ == '__main__':

        total_step = 100

        _reward = reward.BasicReward(success_reward = 10000,
                                range_out_pos = 0.05,
                                range_out_orn = 0.2,
                                total_step = total_step,
                                scale_pos_coef = 1000,
                                scale_orn_coef = 1000)
        _env = Env(reward = _reward,
                step_max_pos = 0.001,
                step_max_orn = 0.005,
                initial_pos_noise= 0.0,
                initial_orn_noise= 0.0,
                step_pos_noise=0.0,
                step_orn_noise=0.0)
        _sac = SoftActorCritic(_env, hidden_sizes=(300, 300, 200), alpha=10, batch_size = 300)
        _trainer = Trainer(agent=_sac, episodes=1000)

        _trainer.train()

        _sac.env.destory()
