import time
from env import reward
from env.enviroment import Env
from controller.sac import SoftActorCritic, ReplayBuffer
from controller.train import Trainer

if __name__ == '__main__':

    total_step = 100

    _reward = reward.BasicReward(success_reward = 10000,
                                range_out_pos = 0.2,
                                range_out_orn = 0.5,
                                total_step = total_step,
                                scale_pos_coef = 1000,
                                scale_orn_coef = 1000)
    _env = Env(reward = _reward, 
            step_max_pos = 0.002,
            step_max_orn = 0.01)
    _sac = SoftActorCritic(_env)
    _trainer = Trainer(agent=_sac, episodes=500)

    _trainer.train()

    _sac.env.destory()
