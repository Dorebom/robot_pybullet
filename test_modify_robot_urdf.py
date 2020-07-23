import sys
import time
import pybullet as p

from utils.modify_urdf import UrdfDesigner

p.connect(p.GUI)
p.setRealTimeSimulation(False)

# Plane
p.loadURDF("urdf/plane/plane.urdf", [0, 0, -0.1])

# Modify robot urdf
ud = UrdfDesigner()
ud.load('robot')
ud.export()

# Load robot(kuka_iiwa)
p.loadURDF("urdf/kuka_iiwa/basic_model.urdf", [0.0, 0.0, 0.0])

time.sleep(10)