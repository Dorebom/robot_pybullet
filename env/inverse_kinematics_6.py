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
        rot = R.from_dcm(wrist_orn_m)
        rot_e = np.array(rot.as_euler('XYZ'))
        print('orn_w: ', rot_e)

    def _calc_wrist_pose(self, tcp_pose):
        c4 = self.link_param[6]

        print("tcp_pos:", tcp_pose[:3])
        print("tcp_orn(input):", tcp_pose[3:6])
        tcp_pos = np.array(tcp_pose[:3])
        rot_tcp_orn = R.from_euler('XYZ', tcp_pose[3:6])
        inv_rot_tcp_orn = R.from_matrix(rot_tcp_orn.as_matrix()).inv().as_matrix()
        print("tcp_orn(scipy):", rot_tcp_orn.as_euler('XYZ'))
        print("inv_tcp_orn:", R.from_matrix(inv_rot_tcp_orn).as_euler('XYZ'))

        tool_pos = np.array(self.tool_pose[:3]) + np.array([0.0, 0.0, c4])
        rot_tool_orn = R.from_euler('XYZ', self.tool_pose[3:6])
        inv_rot_tool_orn = R.from_matrix(rot_tool_orn.as_matrix()).inv().as_matrix()
        print("tool_pos:", tool_pos)
        print("tool_orn:", self.tool_pose[3:6])
        print("inv_tool_orn:", R.from_matrix(inv_rot_tool_orn).as_euler('XYZ'))

        rot_w_prime_orn = R.from_matrix(np.dot(rot_tcp_orn.as_matrix(), inv_rot_tool_orn))
        # ここから
        print("w_prime_orn:", rot_w_prime_orn.as_euler('XYZ'))

        rot_w_orn = R.from_matrix(rot_w_prime_orn.as_matrix(), )

        wrist_pos = tcp_pos - np.dot(rot_tcp_orn.as_matrix(), tmp_1)
        print('wrist_pos:', wrist_pos)


        inv_uc_orn_M = R.from_matrix(np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(self.tool_pose[3:6]) )).reshape([3,3])).inv().as_matrix()
        wrist_orn_m = np.dot(inv_uc_orn_M, tcp_orn_m)

        rot_wr = R.from_euler('XYZ', [0.0, 0, 0])
        rot_wr_m = rot_wr.as_matrix()
        inv_rot_wr_m = R.from_matrix(rot_wr_m).inv().as_matrix()
        #print('aaa', rot_wr_m)

        wrist_orn_m2 = np.dot(inv_rot_wr_m, wrist_orn_m)

        rot = R.from_dcm(wrist_orn_m2)
        rot_e = np.array(rot.as_euler('XYZ'))
        #rot_e[0] *= -1.0
        #rot_e[2] *= -1.0
        rot_3 = R.from_euler('XYZ', rot_e)
        wrist_orn_m2 = rot_3.as_dcm()

        return wrist_pos, wrist_orn_m2
