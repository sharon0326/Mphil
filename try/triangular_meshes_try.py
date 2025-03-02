import numpy as np
import read_polyshape_3d
from coplanarity_mesh import RoofSolarPanel
from cartesian_lonlat import convert_coordinate_system
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def ray_triangle_intersection(ray_origin, ray_dir, triangle, epsilon=1e-6):
    """Möller–Trumbore intersection algorithm."""
    v0, v1, v2 = triangle
    edge1 = v1 - v0
    edge2 = v2 - v0

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if -epsilon < a < epsilon:
        return False  # Ray parallel to triangle

    f = 1.0 / a
    s = ray_origin - v0
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, edge1)
    v = f * np.dot(ray_dir, q)

    if v < 0.0 or u + v > 1.0:
        return False

    t = f * np.dot(edge2, q)
    return t > epsilon


def is_shaded(centroid, solar_azimuth, solar_zenith, all_triangles, current_mesh_idx):
    """Check if mesh is shaded by any other mesh at given solar position."""
    ray_origin = np.array(centroid)

    # Convert solar angles to direction vector
    az_rad = np.radians(solar_azimuth)
    zen_rad = np.radians(solar_zenith)

    dx = np.sin(zen_rad) * np.sin(az_rad)
    dy = np.sin(zen_rad) * np.cos(az_rad)
    dz = np.cos(zen_rad)
    ray_dir = np.array([dx, dy, dz])
    ray_dir /= np.linalg.norm(ray_dir)

    # Check intersections with all triangles except current mesh
    for tri_data in all_triangles:
        if tri_data['mesh_idx'] == current_mesh_idx:
            continue

        tri = [np.array(v) for v in tri_data['triangle']]
        if ray_triangle_intersection(ray_origin, ray_dir, tri):
            return True
    return False


def extract_meshes(nested_list):
    """Flatten nested structure to extract individual meshes (each with 4 points)."""
    meshes = []
    for item in nested_list:
        if isinstance(item, list):
            if len(item) == 4 and all(len(sub) == 3 for sub in item):
                meshes.append(item)
            else:
                meshes.extend(extract_meshes(item))
    return meshes


def compute_centroid(mesh):
    """Compute centroid (average of coordinates) of each mesh."""
    n = len(mesh)
    avg_lon = sum(p[0] for p in mesh) / n
    avg_lat = sum(p[1] for p in mesh) / n
    avg_alt = sum(p[2] for p in mesh) / n
    return (avg_lon, avg_lat, avg_alt)

# currently it is based on the normal vector, but with real building's coordinates,
# it should calculate according to the exact azimuth instead of the mesh grid
def calculate_tilt_azimuth(mesh):
    """Calculate tilt and azimuth angles for a mesh based on its normal vector."""
    A = np.array(mesh[0])
    B = np.array(mesh[1])
    C = np.array(mesh[2])
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    norm_length = np.linalg.norm(normal)

    if norm_length == 0:
        return 0.0, 180.0  # Default for degenerate mesh

    normal = normal / norm_length
    if normal[2] < 0:
        normal = -normal  # Ensure normal points upward

    tilt = np.degrees(np.arccos(normal[2]))
    azimuth = np.degrees(np.arctan2(normal[0], normal[1])) % 360
    return tilt, azimuth


if __name__ == '__main__':
    # Configuration (same as original)
    CONVERSION_PARAMS = {
        'input_file': "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/test2.txt",
        'earth_radius': 6378137.0,  # WGS84 in meters
        'local_centroid': (12.516, 21.043),  # (x, y) in meters
        'geo_centroid': (52.1986125198786, 0.11358089726501427),  # (lat, lon)
        'unit_scaling': (10.0, 10.0, 10.0),  # (x, y, z) scale factors
        'panel_config': {
            'origin_lat': 37.7749,
            'origin_lon': -122.4194,
            'panel_dx': 2.0,
            'panel_dy': 1.0,
            'max_panels': 10,
            'b_scale_x': 1.0,
            'b_scale_y': 1.0,
            'b_scale_z': 1.0,
            'grid_size': 10.0
        },
        'plot_style': {
            'face_color': plt.cm.viridis(0.5),
            'edge_color': 'k',
            'alpha': 0.5,
            'labels': ('Longitude', 'Latitude', 'Elevation (m)')
        }
    }


    vertices, faces = read_polyshape_3d.read_polyshape(CONVERSION_PARAMS['input_file'])
    roof = RoofSolarPanel(V=vertices, F=faces, **CONVERSION_PARAMS['panel_config'])
    print(roof.roof_faces)

    converted_mesh = convert_coordinate_system(CONVERSION_PARAMS, roof.mesh_objects,
                                               *CONVERSION_PARAMS['local_centroid'],
                                               *CONVERSION_PARAMS['geo_centroid'],
                                               *CONVERSION_PARAMS['unit_scaling'])
    meshes = extract_meshes(converted_mesh)

    # Preprocess all meshes into triangles
    all_triangles = []
    for mesh_idx, mesh in enumerate(meshes):
        # Split quad into two triangles (assuming 4-point convex mesh)
        tri1 = [mesh[0], mesh[1], mesh[2]]
        tri2 = [mesh[0], mesh[2], mesh[3]]
        all_triangles.append({'mesh_idx': mesh_idx, 'triangle': tri1})
        all_triangles.append({'mesh_idx': mesh_idx, 'triangle': tri2})

    #print(all_triangles)

    # Assuming 'all_triangles' is your input list (no need to parse)
    data = all_triangles  # Directly use the list instead of parsing

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract all triangles from the data
    triangles = [item['triangle'] for item in data]

    # Create a Poly3DCollection and add it to the plot
    mesh = Poly3DCollection(triangles, alpha=0.5, edgecolor='k', facecolor='cyan')
    ax.add_collection3d(mesh)

    # Set axis labels and display
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude')
    ax.view_init(elev=20, azim=-45)
    plt.show()