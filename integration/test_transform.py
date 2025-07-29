import numpy as np

input_camera_matrix = np.array([[0.866, -0.5, 0, 0],
                          [0.5, 0.866, 0, 0],
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]], dtype=np.float32) 

rot = np.array([[1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float32)

input_camera_matrix_w2c = np.linalg.inv(input_camera_matrix)  # Assuming the first camera is the reference

world_2_world_transform = np.eye(4, dtype=np.float32)
world_2_world_transform[3, :3] = rot @ input_camera_matrix_w2c[:3, 3]  # Set the translation part
world_2_world_transform[:3, :3] = rot @ input_camera_matrix_w2c[:3, :3]  # Set the rotation part

print("Finalized")