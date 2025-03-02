import datetime
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import read_polyshape_3d
from coplanarity_mesh import RoofSolarPanel
from cartesian_lonlat import convert_coordinate_system, visualize_3d_mesh, convert_coordinate_system_building
import matplotlib.pyplot as plt

def compute_centroid(triangle):
    """Calculate centroid with robust type checking"""
    # Convert to numpy array if needed
    if not isinstance(triangle, np.ndarray):
        try:
            triangle = np.array(triangle, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"Invalid triangle structure: {triangle}") from e

    # Verify triangle shape (3 vertices, 3 coordinates each)
    if triangle.shape != (3, 3):
        raise ValueError(f"Invalid triangle dimensions: {triangle.shape}. Should be (3,3)")

    return np.mean(triangle, axis=0)

def ray_triangle_intersection(ray_origin, ray_dir, triangle, epsilon=1e-100):
    """Möller–Trumbore algorithm with dynamic epsilon"""
    v0, v1, v2 = [np.array(p) for p in triangle]
    edge1 = v1 - v0
    edge2 = v2 - v0

    # Auto-calculate epsilon if not provided
    if epsilon is None:
        avg_edge = (np.linalg.norm(edge1) + np.linalg.norm(edge2) + np.linalg.norm(v2 - v1)) / 3
        epsilon = max(1e-6, avg_edge * 0.01)

    h = np.cross(ray_dir, edge2)
    a = np.dot(edge1, h)

    if -epsilon < a < epsilon:
        return False

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

def calculate_tilt_azimuth(triangle):
    """Calculate surface orientation from triangle geometry with type safety"""
    # Convert to numpy array if needed
    if not isinstance(triangle, np.ndarray):
        triangle = np.array(triangle, dtype=np.float64)

    # Ensure we have valid 3D coordinates
    if triangle.shape != (3, 3):
        raise ValueError(f"Invalid triangle shape {triangle.shape}. Expected 3 vertices with 3 coordinates each")

    # Calculate normal vector
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    normal = np.cross(v1, v2)

    # Normalize and calculate angles
    normal /= np.linalg.norm(normal)
    tilt = np.degrees(np.arccos(normal[2]))
    azimuth = np.degrees(np.arctan2(normal[1], normal[0])) % 360

    return tilt, azimuth

def solar_vector(azimuth, zenith):
    """Convert solar angles to 3D direction vector"""
    az_rad = np.radians(azimuth)
    zen_rad = np.radians(zenith)
    return np.array([
        np.sin(zen_rad) * np.sin(az_rad),
        np.sin(zen_rad) * np.cos(az_rad),
        np.cos(zen_rad)
    ])


def is_shaded(triangle, solar_dir, building_triangles, num_samples=1):
    """Check shading using multiple rays across the triangle"""
    sample_points = generate_sample_points(triangle, num_samples)

    for point in sample_points:
        ray_origin = point + solar_dir * 1e-6  # Offset to avoid self-intersection

        # Check against building triangles
        for tri in building_triangles:
            if ray_triangle_intersection(ray_origin, solar_dir, tri):
                return True  # Shaded if any ray hits

    return False  # Not shaded if all rays clear

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


def calculate_mesh_averages(results):
    """Calculate average radiance for each mesh."""
    return {mesh_id: np.mean(list(data.values())) for mesh_id, data in results.items()}

def create_mesh_coordinate_map(mesh_triangles):
    return {
        idx: {
            'vertices': geometry,  # First element of tuple
            'centroid': compute_centroid(geometry),  # Process geometry only
            'mesh_index': mesh_idx  # Second element of tuple
        }
        for idx, (geometry, mesh_idx) in enumerate(mesh_triangles)  # Unpack tuple here
    }


def plot_solar_access(shaded, unshaded, solar_azimuth, solar_zenith):
    """
    Visualizes 3D solar access analysis by plotting shaded and unshaded areas.

    Parameters:
    - shaded (list of arrays): List of polygons representing shaded areas.
    - unshaded (list of arrays): List of polygons representing unshaded areas.
    - solar_azimuth (float): Solar azimuth angle in degrees for the title.
    - solar_zenith (float): Solar zenith angle in degrees for the title.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot UNSHADED areas (green)
    if unshaded:
        unshaded_collection = Poly3DCollection(
            unshaded,
            facecolors='#00ff00',  # Bright green
            edgecolors='#003300',  # Dark green edges
            linewidths=0.3,
            alpha=0.9,
            zorder=2
        )
        ax.add_collection3d(unshaded_collection)

    # Plot SHADED areas (red) on top
    if shaded:
        shaded_collection = Poly3DCollection(
            shaded,
            facecolors='#ff3300',  # Bright orange-red
            edgecolors='#660000',  # Dark red edges
            linewidths=0.3,
            alpha=0.8,
            zorder=3
        )
        ax.add_collection3d(shaded_collection)

    # Set axes limits based on combined points
    if shaded or unshaded:
        all_points = np.concatenate(shaded + unshaded) if shaded else np.concatenate(unshaded)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        padding = 0.1 * (max_vals - min_vals)
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])
        ax.set_zlim(min_vals[2] - padding[2], max_vals[2] + padding[2])

    # Configure view and labels
    ax.view_init(elev=45, azim=-45)
    ax.set_xlabel('X Axis', fontsize=10, labelpad=10)
    ax.set_ylabel('Y Axis', fontsize=10, labelpad=10)
    ax.set_zlabel('Elevation', fontsize=10, labelpad=10)
    ax.set_title(
        f'Solar Access Map\nAzimuth: {solar_azimuth}°, Zenith: {solar_zenith}°',
        fontsize=12, pad=15
    )

    # Add legend
    legend_elements = [
        plt.matplotlib.patches.Patch(facecolor='#00ff00', alpha=0.9, label='Direct Sunlight'),
        plt.matplotlib.patches.Patch(facecolor='#ff3300', alpha=0.8, label='Shaded Areas')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Optimize 3D rendering
    plt.tight_layout()
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95))
    ax.zaxis.set_pane_color((0.97, 0.97, 0.97))
    ax.grid(False)

    plt.show()



def create_comprehensive_results(averages, coordinate_map):
    results = {}
    for mesh_id, avg in averages.items():
        try:
            idx = int(mesh_id.split('_')[1]) - 1
            if idx not in coordinate_map:
                raise KeyError(f"No coordinate data for index {idx}")

            results[mesh_id] = {
                'average_radiance': avg,
                'original_coordinates': coordinate_map[idx]['vertices'],
                'centroid': coordinate_map[idx]['centroid'],
                'mesh_index': coordinate_map[idx]['mesh_index']
            }
        except (ValueError, IndexError) as e:
            print(f"Skipping invalid mesh ID {mesh_id}: {str(e)}")

    return results

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def load_and_process_building(params):
    """Load and process building geometry data"""
    vertices, faces = read_polyshape_3d.read_polyshape(params['input_file'])
    roof = RoofSolarPanel(
        V=vertices,
        F=faces,
        **params['panel_config']
    )

    # Convert building coordinates
    converted_building = convert_coordinate_system_building(
        params, roof.V,
        *params['local_centroid'],
        *params['geo_centroid'],
        *params['unit_scaling']
    )

    # Generate building triangles
    building_triangles = []
    for face in roof.triangular_F:
        building_triangles.append([converted_building[i] for i in face])

    return roof, converted_building, building_triangles

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def process_solar_meshes(roof, params):
    """Process and triangulate solar panel meshes"""
    converted_mesh = convert_coordinate_system(
        params, roof.mesh_objects,
        *params['local_centroid'],
        *params['geo_centroid'],
        *params['unit_scaling']
    )

    # Extract and triangulate meshes
    mesh_triangles = []
    for mesh_idx, square in enumerate(extract_meshes(converted_mesh)):
        tri1 = [square[0], square[1], square[2]]
        tri2 = [square[0], square[2], square[3]]
        mesh_triangles.append((tri1, mesh_idx))
        mesh_triangles.append((tri2, mesh_idx))

    return converted_mesh, mesh_triangles


def generate_sample_points(triangle, num_samples=4):
    """Generate multiple sample points on a triangle (centroid + edge midpoints)."""
    samples = []
    # Centroid
    centroid = compute_centroid(triangle)
    samples.append(centroid)
    # Edge midpoints
    triangle = np.array(triangle)  # Ensure it's a numpy array
    for i in range(3):
        midpoint = (triangle[i] + triangle[(i+1) % 3]) / 2.0
        samples.append(midpoint)
    return samples[:num_samples]

def calculate_shading_status(mesh_triangles, building_triangles, solar_azimuth, solar_zenith):
    """Simplified shading calculation using enhanced is_shaded"""
    solar_dir = solar_vector(solar_azimuth, solar_zenith)
    shaded = []
    unshaded = []

    for idx, (tri, mesh_idx) in enumerate(mesh_triangles):
        if is_shaded(tri, solar_dir, building_triangles):
            shaded.append(tri)
        else:
            unshaded.append(tri)

    return shaded, unshaded


if __name__ == '__main__':
    """Main execution flow for solar potential analysis"""
    # Load configuration parameters
    CONVERSION_PARAMS = {
    'input_file': "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/test2.txt",
    'earth_radius': 6378137.0,
    'local_centroid': (12.516, 21.043),
    'geo_centroid': (52.1986125198786, 0.11358089726501427),
    'unit_scaling': (10.0, 10.0, 10.0),
    'timezone': 'Europe/London',
    'panel_config': {
        'origin_lat': 37.7749,
        'origin_lon': -122.4194,
        'panel_dx': 2.0,
        'panel_dy': 1.0,
        'max_panels': 10,
        'b_scale_x': 1.0,
        'b_scale_y': 1.0,
        'b_scale_z': 1.0,
        'grid_size': 20.0
    },
    'simulation_params': {
        'start': datetime.datetime(2023, 6, 1, 8, 0),
        'end': datetime.datetime(2023, 6, 1, 16, 0),
        'resolution': 'hourly'
    },
    'visualization': {
        'face_color': plt.cm.viridis(0.5),
        'edge_color': 'k',
        'alpha': 0.5,
        'labels': ('Longitude', 'Latitude', 'Elevation (m)')
    }
    }

    # for a specific solar asimuth and zenith, calculate and plot the unshaded and shaded mesh
    roof, converted_building, building_triangles = load_and_process_building(CONVERSION_PARAMS)
    converted_mesh, mesh_triangles = process_solar_meshes(roof, CONVERSION_PARAMS)
    shaded, unshaded = calculate_shading_status(
        mesh_triangles, building_triangles,
        solar_azimuth=180, solar_zenith=45
    )
    plot_solar_access(
        shaded=shaded,
        unshaded=unshaded,
        solar_azimuth=180,
        solar_zenith=45
    )