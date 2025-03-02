import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import read_polyshape_3d
from coplanarity_mesh import RoofSolarPanel

# Configuration constants
INPUT_FILE_PATH = "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/BJ39_500_099048_0010.polyshape"
LOCAL_REFERENCE = (118.56912906633572, 232.99662254850713)
GEOGRAPHIC_REFERENCE = (52.1986125198786, 0.11358089726501427)
PANEL_PARAMS = {
    'origin_lat': 37.7749,
    'origin_lon': -122.4194,
    'panel_dx': 2.0,
    'panel_dy': 1.0,
    'max_panels': 10,
    'b_scale_x': 1.0,
    'b_scale_y': 1.0,
    'b_scale_z': 1.0,
    'grid_size': 10.0
}

def convert_nested_coordinates(nested_mesh, x_ref, y_ref, lat_ref, lon_ref):
    """
    Converts nested mesh coordinates from local Cartesian to geographic system.

    Args:
        nested_mesh: Nested structure containing [x, y, z] coordinate triplets
        x_ref: Local reference x-coordinate
        y_ref: Local reference y-coordinate
        lat_ref: Geographic reference latitude (degrees)
        lon_ref: Geographic reference longitude (degrees)

    Returns:
        Nested structure with converted [lat, lon, z] coordinate triplets
    """
    if isinstance(nested_mesh, list):
        if len(nested_mesh) == 3 and all(isinstance(v, (int, float)) for v in nested_mesh):
            x, y, z = nested_mesh
            lat_rad = math.radians(lat_ref)
            meters_per_degree = 111319.488

            delta_lat = (y - y_ref) / meters_per_degree
            delta_lon = (x - x_ref) / (meters_per_degree * math.cos(lat_rad))

            return [
                lat_ref + delta_lat,
                lon_ref + delta_lon,
                z  # Maintain original elevation
            ]
        return [convert_nested_coordinates(item, x_ref, y_ref, lat_ref, lon_ref)
                for item in nested_mesh]
    return nested_mesh


def plot_3d_structure(nested_mesh):
    """Visualizes 3D mesh structure using matplotlib."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set(xlabel='X Axis', ylabel='Y Axis', zlabel='Z Axis',
           title='3D Mesh Grid Visualization')

    # Configure visualization parameters
    face_color = plt.cm.viridis(0.5)
    edge_color = 'k'
    alpha = 0.5

    for mesh_object in nested_mesh:
        for square in mesh_object:
            vertices = [[*point] for point in square]
            poly = Poly3DCollection(
                [vertices],
                alpha=alpha,
                edgecolor=edge_color,
                facecolor=face_color
            )
            ax.add_collection3d(poly)

    ax.autoscale_view()
    ax.set_box_aspect([1, 1, 1])  # Maintain aspect ratio
    plt.show()


def main():
    # Load building data
    vertices, faces = read_polyshape_3d.read_polyshape(INPUT_FILE_PATH)

    # Initialize solar panel layout
    roof = RoofSolarPanel(V=vertices, F=faces, **PANEL_PARAMS)

    # Convert coordinate systems
    converted_mesh = convert_nested_coordinates(
        roof.mesh_objects,
        *LOCAL_REFERENCE,
        *GEOGRAPHIC_REFERENCE
    )

    # Visualize results
    plot_3d_structure(converted_mesh)


if __name__ == "__main__":
    main()