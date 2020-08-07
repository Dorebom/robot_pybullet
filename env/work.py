'''
Add: 2020.7.25.
Author: dorebom.b
'''

import time
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

from utils.modify_urdf import UrdfDesigner


class Work():
    def __init__(self, base_pose=[0, 0, 0, 0, 0, 0]):
        self.load(base_pose = base_pose)

    def load(self, base_pose):
        orn_q = p.getQuaternionFromEuler(base_pose[3:6])
        self.work_id = p.loadURDF("urdf/work/basic_work.urdf", basePosition=base_pose[:3], baseOrientation=orn_q, useFixedBase=True)
        #print('work joint num:',p.getNumJoints(self.work_id))

    def remove(self):
        pass

    def set_form(self):
        pass

    def reset(self, base_pose = [0, 0, 0, 0, 0, 0]):
        #self.remove()
        self.load(base_pose = base_pose)

    def get_state(self):
        work_states = p.getLinkState(self.work_id, 6)
        self.act_abs_work_pose = np.zeros(6)
        self.act_abs_work_pose[:3] = np.array(work_states[0])
        self.act_abs_work_pose[3:6] = np.array(p.getEulerFromQuaternion(work_states[1]))

        return self.act_abs_work_pose