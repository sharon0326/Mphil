import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import read_polyshape_3d
from coplanarity_mesh import RoofSolarPanel

import math

def convert_coordinate_system_building(CONVERSION_PARAMS, mesh_grid, centroid_x, centroid_y,
                              centroid_lat, centroid_lon,
                              x_scale=1.0, y_scale=1.0, z_scale=1.0):
    """
    Convert flat list of [x, y, z] points to geographic coordinates.
    Handles input format like [[x1,y1,z1], [x2,y2,z2], ...].
    """
    lat_rad = math.radians(centroid_lat)
    earth_r = CONVERSION_PARAMS['earth_radius']

    def convert_point(point):
        raw_x, raw_y, raw_z = point
        # Apply unit scaling
        x_m = raw_x / x_scale
        y_m = raw_y / y_scale
        z_m = raw_z / z_scale

        delta_x = x_m - centroid_x
        delta_y = y_m - centroid_y

        delta_lat = math.degrees(delta_y / earth_r)
        delta_lon = math.degrees(delta_x / (earth_r * math.cos(lat_rad)))

        return [
            centroid_lon + delta_lon,
            centroid_lat + delta_lat,
            z_m  # Z remains in meters
        ]

    # Process all points in the flat list
    return np.array([convert_point(point) for point in mesh_grid])

# clear the code and maybe integrate to be a single RoofSolarPanel object
def convert_coordinate_system(CONVERSION_PARAMS, mesh_grid, centroid_x, centroid_y,
                              centroid_lat, centroid_lon,
                              x_scale=1.0, y_scale=1.0, z_scale=1.0):
    """
    Convert nested mesh coordinates to geographic system with configurable scaling.

    Args:
        mesh_grid: Nested structure of [x, y, z] points
        centroid_x: Local X reference in target units (meters)
        centroid_y: Local Y reference in target units (meters)
        centroid_lat: Geographic latitude reference (degrees)
        centroid_lon: Geographic longitude reference (degrees)
        x_scale: Scaling factor for X-axis (input units/meter)  because I assume cartesian in dm, so scale = 10
        y_scale: Scaling factor for Y-axis (input units/meter)
        z_scale: Scaling factor for Z-axis (input units/meter)

    Returns:
        Nested structure of [lon, lat, z] points in geographic coordinates
    """
    lat_rad = math.radians(centroid_lat)
    earth_r = CONVERSION_PARAMS['earth_radius']

    def convert_point(point):
        raw_x, raw_y, raw_z = point
        # Apply unit scaling
        x_m = raw_x / x_scale
        y_m = raw_y / y_scale
        z_m = raw_z / z_scale

        delta_x = x_m - centroid_x
        delta_y = y_m - centroid_y

        delta_lat = math.degrees(delta_y / earth_r)
        delta_lon = math.degrees(delta_x / (earth_r * math.cos(lat_rad)))

        return [
            centroid_lon + delta_lon,
            centroid_lat + delta_lat,
            z_m  # Maintain z-unit as meters
        ]

    def process_structure(data):
        if isinstance(data, list):
            if len(data) == 3 and all(isinstance(i, (int, float)) for i in data):
                return convert_point(data)

            return [process_structure(item) for item in data]

        return data

    return process_structure(mesh_grid)


def visualize_3d_mesh(nested_mesh, CONVERSION_PARAMS):
    """3D visualization with configurable labels"""
    style = CONVERSION_PARAMS['plot_style']

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set(xlabel=style['labels'][0],
           ylabel=style['labels'][1],
           zlabel=style['labels'][2],
           title='Converted 3D Mesh Visualization')

    for mesh_object in nested_mesh:
        for square in mesh_object:
            vertices = [[*point] for point in square]
            poly = Poly3DCollection(
                [vertices],
                alpha=style['alpha'],
                edgecolor=style['edge_color'],
                facecolor=style['face_color']
            )
            ax.add_collection3d(poly)

    ax.autoscale_view()
    ax.set_box_aspect([1, 1, 1])
    plt.show()

if __name__ == "__main__":
    # Configuration
    # Scale factors assume decimeter
    CONVERSION_PARAMS = {
        'input_file': "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/BJ39_500_100047_0010.polyshape",
        'earth_radius': 6378137.0,  # WGS84 in meters
        'geo_centroid': (52.1986125198786, 0.11358089726501427),  # (lat, lon)
        'unit_scaling': (1.0, 1.0, 1.0),  # (x, y, z) scale factors
        'panel_config': {
            'panel_dx': 2.0,
            'panel_dy': 1.0,
            'max_panels': 10,
            'b_scale_x': 0.05,
            'b_scale_y': 0.05,
            'b_scale_z': 0.05,
            'grid_size': 1.0
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
    ground_centroid = roof.get_ground_centroid()[:2]
    converted_mesh = convert_coordinate_system(CONVERSION_PARAMS,
        roof.mesh_objects,
        *ground_centroid,
        *CONVERSION_PARAMS['geo_centroid'],
        *CONVERSION_PARAMS['unit_scaling']
    )
    roof.display_building_and_rooftops()
    visualize_3d_mesh(converted_mesh, CONVERSION_PARAMS)
    print(converted_mesh)