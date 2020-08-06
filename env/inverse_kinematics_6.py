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

    def calc_ik(self, tcp_pose, current_joint_pos):
        a1 = self.link_param[0]
        a2 = self.link_param[1]
        b  = self.link_param[2]
        c1 = self.link_param[3]
        c2 = self.link_param[4]
        c3 = self.link_param[5]
        c4 = self.link_param[6]

        # 把持しているワークの座標を被Work座標からRobot座標への変換もやってる
        wrist_pos, wrist_orn_m = self._calc_wrist_pose(tcp_pose=tcp_pose)

        # A.
        c0 = np.array(wrist_pos)
        nx1 = np.sqrt(c0[0] ** 2 + c0[1] ** 2) - a1
        s1_sq = nx1 ** 2 + (c0[2] - c1) ** 2
        s2_sq = (nx1 + 2 * a1) ** 2 + (c0[2] - c1) ** 2
        k_sq = a2 ** 2 + c3 ** 2

        # B.1(J1)
        q1_ = []
        q1_.append(np.arctan2(c0[1], c0[0]))
        if (q1_[0] < 0):
            q1_.append(np.arctan2(c0[1], c0[0]) + np.pi)
        else:
            q1_.append(np.arctan2(c0[1], c0[0]) - np.pi)
        q1_idx = self.getNearestValue(q1_, current_joint_pos[0])
        q1 = q1_[q1_idx]

        # B.2(J2)
        q2_ = []
        if q1_idx == 0:
            tmp = (s1_sq + c2 ** 2 - k_sq) / (2 * np.sqrt(s1_sq) * c2)
            if abs(tmp) > 1:
                return current_joint_pos, False
            q2_.append(- np.arccos(tmp) + np.arctan2(nx1, c0[2] - c1) - np.pi / 2)
            q2_.append(np.arccos(tmp) + np.arctan2(nx1, c0[2] - c1) - np.pi / 2)
        else:
            tmp = (s2_sq + c2 ** 2 - k_sq) / (2 * np.sqrt(s2_sq) * c2)
            if abs(tmp) > 1:
                return current_joint_pos, False
            q2_.append(- np.arccos(tmp) - np.arctan2(nx1 + 2 * a1, c0[2] - c1) - np.pi / 2)
            q2_.append(np.arccos(tmp) - np.arctan2(nx1 + 2 * a1, c0[2] - c1) - np.pi / 2)
        q2_idx = self.getNearestValue(q2_, current_joint_pos[1])
        q2 = q2_[q2_idx]

        # B.3(J3)
        q3_ = []
        if q1_idx == 0:
            tmp = (s1_sq - c2 ** 2 - k_sq) / (2 * c2 * np.sqrt(k_sq))
            if abs(tmp) > 1:
                return current_joint_pos, False
            q3_.append(np.arccos(tmp) - np.arctan2(a2, c3) + np.pi / 2)
            q3_.append(- np.arccos(tmp) - np.arctan2(a2, c3) + np.pi / 2)
        else:
            tmp = (s2_sq - c2 ** 2 - k_sq) / (2 * c2 * np.sqrt(k_sq))
            if abs(tmp) > 1:
                return current_joint_pos, False
            q3_.append(np.arccos(tmp) - np.arctan2(a2, c3) + np.pi / 2)
            q3_.append(- np.arccos(tmp) - np.arctan2(a2, c3) + np.pi / 2)
        q3 = q3_[q2_idx]

        # C.
        e11 = wrist_orn_m[0][0]
        e12 = wrist_orn_m[0][1]
        e13 = wrist_orn_m[0][2]
        e21 = wrist_orn_m[1][0]
        e22 = wrist_orn_m[1][1]
        e23 = wrist_orn_m[1][2]
        e31 = wrist_orn_m[2][0]
        e32 = wrist_orn_m[2][1]
        e33 = wrist_orn_m[2][2]

        s1p = np.sin(q1)
        s23p = np.sin(q2 + q3)
        c1p = np.cos(q1)
        c23p = np.cos(q2 + q3)
        mp = e13 * s23p * c1p + e23 * s23p * s1p + e33 * c23p

        # D.1(J4)
        q4_p = np.arctan2(e23 * c1p - e13 * s1p, e13 * c23p * c1p + e23 * c23p * s1p - e33 * s23p)
        if abs(current_joint_pos[3] - q4_p) > np.pi * 1.9:
            if q4_p > 0:
                q4_p = -np.pi * 2.0 + q4_p
            else:
                q4_p = np.pi * 2.0 + q4_p
        if q4_p > 0:
            q4_q = q4_p - np.pi
        elif q4_p <= 0:
            q4_q = q4_p + np.pi

        # D.2(J5)
        q5_p = np.arctan2(np.sqrt(1 - mp ** 2), mp)
        q5_q = - q5_p
        # D.3(J6)
        q6_p = np.arctan2(e12 * s23p * c1p + e22 * s23p * s1p + e32 * c23p,
                          -e11 * s23p * c1p - e21 * s23p * s1p - e31 * c23p)
        if abs(current_joint_pos[5] - q6_p) > np.pi * 1.9:
            if q6_p > 0:
                q6_p = -np.pi * 2.0 + q6_p
            else:
                q6_p = np.pi * 2.0 + q6_p
        if q6_p > 0:
            q6_q = q6_p - np.pi
        elif q6_p <= 0:
            q6_q = q6_p + np.pi
        # E.
        if abs(current_joint_pos[3] - q4_p) + abs(current_joint_pos[5] - q6_p) <= \
            abs(current_joint_pos[3] - q4_q) + abs(current_joint_pos[5] - q6_p):
            q4 = q4_p
            q5 = q5_p
            q6 = q6_p
            q4_idx = 0
        else:
            q4 = q4_q
            q5 = q5_q
            q6 = q6_q
            q4_idx = 1

        if abs(q4) > np.pi * (190.0/180.0):
            return current_joint_pos, False

        return [q1, q2, q3, q4, q5, q6], True

    def _calc_wrist_pose(self, tcp_pose):
        c4 = self.link_param[6]

        tcp_pos = np.array(tcp_pose[:3])
        tcp_orn = R.from_euler('XYZ', tcp_pose[3:6])
        inv_rot_tcp_orn = R.from_matrix(tcp_orn.as_matrix()).inv()

        tool_pos = np.array(self.tool_pose[:3]) - np.array([0.0, 0.0, c4])
        rot_tool_orn = R.from_euler('XYZ', self.tool_pose[3:6])
        inv_rot_tool_orn = R.from_matrix(rot_tool_orn.as_matrix()).inv()

        rot_world_link_orn = R.from_matrix(np.dot(tcp_orn.as_matrix(), inv_rot_tool_orn.as_matrix()))

        rot_change_rw = R.from_euler('XYZ', [0, np.pi, 0])
        inv_rot_change_rw = R.from_matrix(rot_change_rw.as_matrix()).inv()
        rot_robot_link_orn = R.from_matrix(np.dot(rot_world_link_orn.as_matrix(), inv_rot_change_rw.as_matrix()))

        robot_link_pos = tcp_pos - np.dot(rot_world_link_orn.as_matrix(), tool_pos)

        return robot_link_pos, rot_robot_link_orn.as_matrix()

    def getNearestValue(self, datalist, num):
        idx = np.abs(np.asarray(datalist) - num).argmin()
        return idx