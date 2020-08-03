import sys
import time
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':

    p.connect(p.GUI)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setRealTimeSimulation(False)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

    # Plane
    plane_pos = [0, 0, -0.1]
    p.loadURDF("urdf/plane/plane.urdf", plane_pos)

    base_pose = [0, 0, 0, 0, 0, 0]
    orn_q = p.getQuaternionFromEuler(base_pose[3:6])
    robot_id = p.loadURDF("urdf/kuka_iiwa_6/basic_model.urdf", basePosition=base_pose[:3], baseOrientation=orn_q)

    p.resetJointState(robot_id, 0, targetValue=np.pi * 0.25)
    p.resetJointState(robot_id, 1, targetValue=np.pi * 0.25)
    p.stepSimulation()

    print('robot joint num:',p.getNumJoints(robot_id))

    end_effector_link_index = 6
    wrist_link_index = 4


    states = p.getLinkState(robot_id, end_effector_link_index)
    act_abs_tcp_pose = np.zeros(6)
    act_abs_tcp_pose[:3] = states[0]
    act_abs_tcp_pose[3:6] = p.getEulerFromQuaternion(states[1])
    print("pos_tcp: ", act_abs_tcp_pose[:3])
    print("orn_tcp: ", act_abs_tcp_pose[3:6])

    states = p.getLinkState(robot_id, wrist_link_index)
    act_abs_tcp_pose2 = np.zeros(6)
    act_abs_tcp_pose2[:3] = states[4]
    act_abs_tcp_pose2[3:6] = p.getEulerFromQuaternion(states[5])

    print("pos_w: ", act_abs_tcp_pose2[:3])
    print("orn_w: ", act_abs_tcp_pose2[3:6])

    act_joint_pos = np.zeros(6)
    for i in range(6):
        act_joint_pos[i] = p.getJointState(robot_id, i)[0]
        if abs(act_joint_pos[i]) < 1e-6:
            act_joint_pos[i] = 0.0
    print('j_pos: ', act_joint_pos)

    a1 = 0
    a2 = 0
    b  = 0
    c1 = 0.36
    c2 = 0.42
    c3 = 0.4
    c4 = 0.081

    tcp_pose = act_abs_tcp_pose

    tcp_pos = np.array(tcp_pose[:3])
    tcp_orn_m = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tcp_pose[3:6]))).reshape([3,3])

    print(tcp_orn_m)

    tool_pose = [0, 0, 0.045, 0, 0, 0]

    tmp_1 = tool_pose[:3] + np.array([0.0, 0.0, c4])
    tmp_1[1] *= -1.0

    wrist_pos = tcp_pos - np.dot(tcp_orn_m, tmp_1)
    inv_uc_orn_M = R.from_matrix(np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(tool_pose[3:6]) )).reshape([3,3])).inv().as_matrix()
    wrist_orn_m = np.dot(inv_uc_orn_M, tcp_orn_m)

    rot = R.from_dcm(wrist_orn_m)
    rot_e = np.array(rot.as_euler('XYZ'))
    rot_e[0] *= -1.0
    rot_e[2] *= -1.0
    rot_3 = R.from_euler('XYZ', rot_e)
    wrist_orn_m2 = rot_3.as_dcm()

    print("pos_w: ", wrist_pos)
    print("orn_w: ", wrist_orn_m2)

    time.sleep(3)