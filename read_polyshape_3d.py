# read roofshape from .polyshape file
# plot the entire building
# extract the rooftop

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
from mpl_toolkits.mplot3d import Axes3D


# Converted vertices array (22x3)
# Example 1000: vertices, the coordinates of vertices on the plot
ex_verts = np.array([
    [245.5097, 366.4517, 75.0000],
    [290.6382, 28.5419, 75.0000],
    [243.8408, 63.2476, 111.7922],
    [240.9064, 79.8072, 112.4230],
    [199.7914, 111.0648, 144.6584],
    [177.2708, 276.7621, 145.0000],
    [207.5955, 17.4195, 75.0000],
    [204.2353, 32.3258, 75.0000],
    [153.4020, 294.4090, 126.9071],
    [129.5602, 22.4224, 75.0000],
    [101.2336, 227.1593, 75.0000],
    [79.5783, 284.2168, 127.2086],
    [31.0913, 217.1453, 75.0000],
    [15.2045, 335.7384, 75.0000],
    [245.5097, 366.4517, 0.0],
    [290.6382, 28.5419, 0.0],
    [207.5955, 17.4195, 0.0],
    [204.2353, 32.3258, 0.0],
    [129.5602, 22.4224, 0.0],
    [101.2336, 227.1593, 0.0],
    [31.0913, 217.1453, 0.0],
    [15.2045, 335.7384, 0.0]
])

# Converted faces with 0-based indexing
# Each face is composed by different vertices. Faces records the labels of the vertices
ex_faces = [
    [9, 4, 3, 7],
    [7, 6, 2, 3],
    [6, 1, 2],
    [1, 0, 5, 4, 3, 2],
    [0, 13, 11, 8, 5],
    [11, 12, 13],
    [12, 10, 8, 11],
    [10, 9, 4, 5, 8],
    [14, 15, 16, 17, 18, 19, 20, 21],
    [0, 14, 15, 1],
    [1, 15, 16, 6],
    [6, 16, 17, 7],
    [7, 17, 18, 9],
    [9, 18, 19, 10],
    [10, 19, 20, 12],
    [12, 20, 21, 13],
    [13, 21, 14, 0]
]

# read the .polyshape file, into verts and faces
import numpy as np
import re


def read_polyshape(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    # Parse header information
    num_verts = int(re.search(r'(\d+)', lines[0]).group())
    num_faces = int(re.search(r'(\d+)', lines[1]).group())

    # Read vertices (next num_verts lines after header)
    verts = []
    for line in lines[2:2 + num_verts]:
        x, y, z = map(float, line.split(','))
        verts.append([x, y, z])

    # Read faces (remaining lines)
    faces = []
    for line in lines[2 + num_verts:2 + num_verts + num_faces]:
        parts = list(map(int, line.split(',')))
        face = [p - 1 for p in parts[:-1]]  # Convert to 0-based index
        faces.append(face)

    return np.array(verts), faces

# Calculate planarity error
def err_planarity(X, F):
    fval = 0.0
    for face in F:
        verts = X[face, :]
        A = np.cov(verts, rowvar=False)
        eigenvalues = np.linalg.eigh(A)[0]
        fval += eigenvalues[0]

    range_x = np.max(X[:, 0]) - np.min(X[:, 0])
    range_y = np.max(X[:, 1]) - np.min(X[:, 1])
    range_z = np.max(X[:, 2]) - np.min(X[:, 2])
    max_range = max(range_x, range_y, range_z)

    return fval / max_range


# error = err_planarity(verts, faces)
# print(f"Planarity error = {error:.6f}")

# plot the 3d building geometry
def plot_building(verts, faces):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each face
    for face in faces:
        # Get vertices for this face (convert to 0-based index)
        polygon = [verts[i] for i in face]
        # Create polygonal patch
        poly = Poly3DCollection([polygon], alpha=0.5, edgecolor='k', linewidths=1)
        poly.set_facecolor(np.random.rand(3, ))
        ax.add_collection3d(poly)

    min_vals = verts.min(axis=0)
    max_vals = verts.max(axis=0)
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=60, azim=-30)
    err  = err_planarity(verts, faces)
    plt.title('3D Building Model with err: ' + str(err))
    plt.show()

def identify_rooftops(verts, faces):
    """
    Identify rooftop faces based on:
    1. Faces containing no ground-level vertices (Z=0)
    2. Non-horizontal faces are filtered out
    """
    # Find ground vertices (Z=0)
    # With a precision tolerance, to deal with coplanarity
    ground_mask = np.isclose(verts[:, 2], 0, atol=1e-15)
    ground_indices = set(np.where(ground_mask)[0])

    # Collect faces with no ground vertices
    roof_faces = []
    for face in faces:
        if not any(idx in ground_indices for idx in face):
            roof_faces.append(face)


    # First pass: exclude faces with any ground vertices
    roof_candidates = []
    for face in faces:
        if not any(idx in ground_indices for idx in face):
            roof_candidates.append(face)

    # Second pass: filter horizontal faces
    roof_faces = []
    for face in roof_candidates:
        # Calculate face normal
        pts = verts[face]
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # Consider faces with normals pointing upward (Z > 0.5)
        if abs(normal[2]) > 0.5:  # ~45 degree threshold, loose the criteria from 0.7, otherwise, not inclusive
            roof_faces.append(face)
    
    return roof_faces

def plot_rooftops(verts, roof_faces):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all vertices and rooftop faces
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], c='grey', alpha=0.3)
    for face in roof_faces:
        poly = verts[face]
        ax.plot_trisurf(poly[:, 0], poly[:, 1], poly[:, 2], alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# Usage
if __name__ == "__main__":
    verts, faces = read_polyshape("C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/test.txt")

    # Result verification
    print("Vertices shape:", verts.shape)
    print("Vertices:")
    print(verts[:])
    print("\nFaces :")
    for i, f in enumerate(faces[:]):
        print(f"Face {i + 1}: {f}")
    plot_building(verts, faces)
    print()

    # identify the rooftops
    roof_faces = identify_rooftops(verts, faces)
    print(f"Found {len(roof_faces)} rooftop faces:")
    for i, face in enumerate(roof_faces[:]):
        print(f"Rooftop {i + 1}: {face}")
    plot_rooftops(verts, roof_faces)