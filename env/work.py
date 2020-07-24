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
        self.work_id = p.loadURDF("urdf/work/basic_work.urdf", basePosition=base_pose[:3], baseOrientation=orn_q)
        print('work joint num:',p.getNumJoints(self.work_id))

    def remove(self):
        pass

    def set_form(self):
        pass

    def reset(self, base_pose = [0, 0, 0, 0, 0, 0]):
        #self.remove()
        self.load(base_pose = base_pose)