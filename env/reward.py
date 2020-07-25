import numpy as np



class BasicReward():
    def __init__(self, success_reward = 1,
                range_out_pos = 0.05,
                range_out_orn = 0.8,
                total_step = 150,
                scale_pos_coef = 1,
                scale_orn_coef = 1):
        self.success_reward = success_reward
        self.range_out_pos = range_out_pos
        self.range_out_orn = range_out_orn
        self.total_step = total_step
        self.range_out_cost = success_reward / total_step
        self.scale_pos_coef = scale_pos_coef
        self.scale_orn_coef = scale_orn_coef

    def reward_function(self, relative_pose, success, out_range, act_step):
        if success:
            return self.success_reward
        else:
            # 1
            range_out_cost = 0
            if out_range:
                range_out_cost = self.range_out_cost * (self.total_step - act_step)
            # 2
            relative_pose_cost = np.linalg.norm(relative_pose[:3]) * self.scale_pos_coef \
                                + np.linalg.norm(relative_pose[3:6]) * self.scale_orn_coef

            # 3 [後日追加予定]スタック状態維持は負の報酬、解除は正の報酬

            return -range_out_cost - relative_pose_cost