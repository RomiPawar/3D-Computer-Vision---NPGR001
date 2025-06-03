import numpy as np
from scipy.optimize import fsolve

K = np.array([
    [2649.4768, 0.0000, 1611.4155],
    [0.0000, 2649.4768, 936.2453],
    [0.0000, 0.0000, 1.0000]
])

R = np.array([
    [0.166762435107263, 0.985996139504843, 0.001379535661650],
    [0.767058936399130, -0.128854026286462, -0.628503960210278],
    [-0.619524719706430, 0.105869036036574, -0.777805161259139]
])


C = np.array([-263.4389, -148.7411, 460.3024])

top = np.array([1520, 255])
bottom = np.array([1540, 900])

def pixel_to_camera(pixel, K):
    u, v = pixel
    pixel_h = np.array([u, v, 1])
    return np.linalg.inv(K) @ pixel_h

def ray_to_world(camera_point, R, C):
    return R.T @ (camera_point)

# Get ray in camera coordinate system
bottom_camera = pixel_to_camera(bottom, K)
top_camera = pixel_to_camera(top, K)

# Get ray in world coordinate system
bottom_world = ray_to_world(bottom_camera, R, C)
top_world = ray_to_world(top_camera, R, C)

# Scale of bottom ray to get z=0
lambda_bottom = -C[2]/bottom_world[2]

bottom_world_pos = C + lambda_bottom * bottom_world

# Scale of top ray to get the same x coordinate as for bottom position
lambda_top = (bottom_world_pos-C)/top_world
lambda_top = lambda_top[0]


top_world_pos = C + lambda_top * top_world

height_vector = top_world_pos - bottom_world_pos
height = abs(height_vector[2])
print(f"Height of the gray object: {height}")
############################ Output ####################################
# Height of the gray object: 175.83294258464696
