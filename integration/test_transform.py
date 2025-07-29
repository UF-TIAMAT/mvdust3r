import numpy as np
import pickle
import itertools

import numpy as np
import pickle
import itertools
from scipy.spatial.transform import Rotation

OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
from scipy.spatial.transform import Rotation

rot = np.eye(4)
rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()

with open("/blue/prabhat/duminduaelamurem/wd/repo_tests/aaai/mvdust3r/rendering_results/axis_rotation/debug.pkl", "rb") as f:
    data = pickle.load(f)
    print(data.keys())

initial_1 = data["input_camera_matrices"][0][:3, :3]
r = Rotation.from_matrix(initial_1)
yaw, pitch, roll = r.as_euler('zxy', degrees=True)
print(yaw, pitch, roll)

world2cam_1 = np.linalg.inv(data["cams2world"][0][:3, :3])
r = Rotation.from_matrix(np.linalg.inv(data["cams2world"][0][:3, :3]))
yaw, pitch, roll = r.as_euler('yxz', degrees=True)
print(yaw, pitch, roll)

# transform_1 = world2cam_1 @ initial_1.T 

initial_2 = data["input_camera_matrices"][1][:3, :3]
world2cam_2 = np.linalg.inv(data["cams2world"][1][:3, :3])
# transform_2 = world2cam_2 @ initial_2.T

r = Rotation.from_matrix(initial_2)
yaw, pitch, roll = r.as_euler('zxy', degrees=True)
print(yaw, pitch, roll)

r = Rotation.from_matrix(np.linalg.inv(data["cams2world"][1][:3, :3]))
yaw, pitch, roll = r.as_euler('yxz', degrees=True)
print(yaw, pitch, roll)


r = R.from_euler(convention, angles, degrees=degrees)
r.as_matrix()


