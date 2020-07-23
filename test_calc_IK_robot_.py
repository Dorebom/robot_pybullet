import sys
import time
import numpy as np
import pybullet as p

from utils.modify_urdf import UrdfDesigner
from utils.kinematics import Kinematics

p.connect(p.GUI)
p.setRealTimeSimulation(False)

# Plane
p.loadURDF("urdf/plane/plane.urdf", [0, 0, -0.1])

# Modify robot urdf
ud = UrdfDesigner()
ud.load('robot')
ud.export()

# Load robot(kuka_iiwa)
robot_id = p.loadURDF("urdf/kuka_iiwa/modified_model.urdf", [0.0, 0.0, 0.0])
robot_joint_count = p.getNumJoints(robot_id)

time.sleep(1.0)

robot_param = [0.0, 0.0, 0.0, 0.36, 0.42, 0.4, 0.101]
tool_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

q = [0.0, 0.0, 0.0, -0.5 * np.pi, 0.0, 0.5 * np.pi, 0.0]
q_1 = [0.0, 0.0, -0.5 * np.pi, 0.0, 0.5 * np.pi, 0.0]

k = Kinematics(robot_id, robot_param, tool_pose, q_1)
k.reset_joint_angle(q_1)

link_states = p.getLinkState(robot_id, robot_joint_count-1)
current_pos = link_states[0]
current_orn_q = link_states[1]
current_orn_e = p.getEulerFromQuaternion(current_orn_q)
current_pose = np.concatenate([current_pos, current_orn_e])
print('current_pose:', current_pose)

current_angle, _, _ = k.calc_inverse_kinematics(current_pose, q_1)

print('current_angle:', current_angle)

for i in range(3):
    p.resetJointState(robot_id, i, targetValue=current_angle[i])
for i in range(3):
    p.resetJointState(robot_id, i+3, targetValue=current_angle[i+3])
time.sleep(0.1)
p.stepSimulation()

time.sleep(10)