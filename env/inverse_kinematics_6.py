import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

class InverseKinematics6():

    def __init__(self, robot_id, link_param, tool_pose):
        self.robot_id = robot_id
        self.link_param = link_param
        self.tool_pose = tool_pose

    def set_tool_pose(self, tool_pose):
        self.tool_pose = tool_pose

    def calc_ik(self, tcp_pose):
        a1 = self.link_param[0]
        a2 = self.link_param[1]
        b  = self.link_param[2]
        c1 = self.link_param[3]
        c2 = self.link_param[4]
        c3 = self.link_param[5]
        c4 = self.link_param[6]

        wrist_pos, wrist_orn_m = self._calc_wrist_pose(tcp_pose=tcp_pose)
        print('pos_w: ', wrist_pos)
        print('orn_w: ', wrist_orn_m)

    def _calc_wrist_pose(self, tcp_pose):
        c4 = self.link_param[6]

        tcp_pos = np.array(tcp_pose[:3])
        tcp_orn_m = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tcp_pose[3:6]))).reshape([3,3])

        wrist_pos = tcp_pos + np.dot(tcp_orn_m, (np.array(self.tool_pose[:3]) + np.array([0.0, 0.0, c4]) ))
        inv_uc_orn_M = R.from_matrix(np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.tool_pose[3:6]) )).reshape([3,3])).inv().as_matrix()
        wrist_orn_m = np.dot(inv_uc_orn_M, tcp_orn_m)

        #tmp_1 = -np.array(self.tool_pose[:3]) + np.array([0.0, 0.0, c4])
        #tmp_1[1] *= -1.0

        rot = R.from_dcm(wrist_orn_m)
        rot_e = np.array(rot.as_euler('XYZ'))
        rot_e[0] *= -1.0
        rot_e[2] *= -1.0
        rot_3 = R.from_euler('XYZ', rot_e)
        wrist_orn_m2 = rot_3.as_dcm()

        return wrist_pos, wrist_orn_m2
