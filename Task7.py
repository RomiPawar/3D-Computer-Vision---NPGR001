import numpy as np
import matplotlib.pyplot as plt

# Left camera matrix
P1 = np.array([
    [1325.0, 0.0, 805.0, 0.0],
    [0.0, 1325.0, 468.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

# Right camera matrix
P2 = np.array([
    [1126.67010454, -249.71530121, 1035.32929249, -1347.97882937],
    [150.95878897, 1299.48285727, 513.03484063, -381.74973338],
    [-0.18907694, -0.00556371, 0.98194651, -0.09044673]
])

# Load verified points
left_points = np.loadtxt('3D CV/7_left_points_verified.txt')  # Shape (N, 2)
right_points = np.loadtxt('3D CV/7_right_points_verified.txt')  # Shape (N, 2)

def construct_D(xi, yi, P1, P2):
    """Constructs the D matrix for a given correspondence."""
    D = np.vstack([
        xi[0] * P1[2, :] - P1[0, :],
        xi[1] * P1[2, :] - P1[1, :],
        yi[0] * P2[2, :] - P2[0, :],
        yi[1] * P2[2, :] - P2[1, :]
    ])
    return D

def triangulate_point(D):
    """Triangulates a single 3D point from matrix D."""
    # Apply numerical conditioning
    S = np.diag(1 / np.max(np.abs(D), axis=0))
    D_conditioned = D @ S

    # Solve DSX' = 0 using SVD
    _, _, vt = np.linalg.svd(D_conditioned.T @ D_conditioned)
    X_prime = vt[-1]

    # Compute the 3D point X = SX'
    X = S @ X_prime

    # Normalize by the last element
    X /= X[-1]
    return X

# Triangulate all points
points_3D = []
for left, right in zip(left_points, right_points):
    D = construct_D(left, right, P1, P2)
    X = triangulate_point(D)
    # Ensure points are in front of both cameras
    if (P1 @ X)[2] > 0 and (P2 @ X)[2] > 0:
        points_3D.append(X)

points_3D = np.array(points_3D)  # Shape (M, 4), homogeneous coordinates

# Convert to Euclidean coordinates for visualization
points_3D = points_3D[:, :3]  # Drop the homogeneous scale

# Plot the 3D points
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], c='blue', marker='o')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Reconstructed 3D Points')
plt.show()
