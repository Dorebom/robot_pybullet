import sys
import time
import numpy as np
import pybullet as p

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

    #p.resetJointState(robot_id, 3, targetValue=np.pi * 0.25)
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
    act_abs_tcp_pose = np.zeros(6)
    act_abs_tcp_pose[:3] = states[4]
    act_abs_tcp_pose[3:6] = p.getEulerFromQuaternion(states[5])

    print("pos_w: ", act_abs_tcp_pose[:3])
    print("orn_w: ", act_abs_tcp_pose[3:6])

    act_joint_pos = np.zeros(6)
    for i in range(6):
        act_joint_pos[i] = p.getJointState(robot_id, i)[0]
    print('j_pos: ', act_joint_pos)



    time.sleep(3)