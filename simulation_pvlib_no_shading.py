import pandas as pd
import numpy as np
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance
import read_polyshape_3d
from coplanarity_mesh import RoofSolarPanel
from cartesian_lonlat import convert_coordinate_system
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


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


def simulate_period_for_mesh(centroid, tilt, azimuth, timezone_str, start_time, end_time, time_base='daily'):
    """Simulate solar flux for a mesh using PVLib."""
    location = {
        'latitude': centroid[1],
        'longitude': centroid[0],
        'timezone': timezone_str
    }
    site = Location(location['latitude'], location['longitude'], tz=location['timezone'])

    # Generate time range
    start = pd.Timestamp(start_time).tz_localize(timezone_str)
    end = pd.Timestamp(end_time).tz_localize(timezone_str)

    if time_base == 'hourly':
        times = pd.date_range(start=start, end=end, freq='h', tz=timezone_str)
    elif time_base == 'daily':
        start_date = start.floor('D')
        end_date = end.ceil('D') - pd.Timedelta(seconds=1)
        times = pd.date_range(start=start_date, end=end_date, freq='h', tz=timezone_str)
    elif time_base == 'weekly':
        start_date = start.floor('D')
        end_date = end.ceil('D') - pd.Timedelta(seconds=1)
        times = pd.date_range(start=start_date, end=end_date, freq='h', tz=timezone_str)
    else:
        raise ValueError("Invalid time_base")

    times = times[(times >= start) & (times <= end)]
    if times.empty:
        return {}

    # Get solar data
    solar_pos = site.get_solarposition(times)
    clearsky = site.get_clearsky(times, model='ineichen', linke_turbidity=3.0)

    # Compute irradiance
    poa = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_pos['apparent_zenith'],
        solar_azimuth=solar_pos['azimuth']
    )
    total_flux = poa['poa_global'].clip(lower=0)

    # Aggregate results
    if time_base == 'hourly':
        aggregated = total_flux
    elif time_base == 'daily':
        aggregated = total_flux.resample('D').sum()
    elif time_base == 'weekly':
        aggregated = total_flux.resample('W').sum()

    return {ts.to_pydatetime(): val for ts, val in aggregated.items()}


def calculate_mesh_averages(results):
    """Calculate average radiance for each mesh."""
    return {mesh_id: np.mean(list(data.values())) for mesh_id, data in results.items()}


def create_mesh_coordinate_map(meshes):
    """Map each mesh to its coordinates and centroid."""
    return {
        f"Mesh_{idx + 1}": {
            'original_coordinates': mesh,
            'centroid': compute_centroid(mesh)
        }
        for idx, mesh in enumerate(meshes)
    }


def create_comprehensive_results(averages, coordinate_map):
    """Combine averages with coordinate data."""
    return {
        mesh_id: {
            'average_radiance': avg,
            'original_coordinates': coordinate_map[mesh_id]['original_coordinates'],
            'centroid': coordinate_map[mesh_id]['centroid']
        }
        for mesh_id, avg in averages.items()
    }


if __name__ == '__main__':
    # Configuration (same as original)
    CONVERSION_PARAMS = {
        'input_file': "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/BJ39_500_098051_0020.polyshape",
        'earth_radius': 6378137.0,  # WGS84 in meters
        #'local_centroid': (12.516, 21.043),  # (x, y) in meters
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
    converted_mesh = convert_coordinate_system(CONVERSION_PARAMS, roof.mesh_objects,
                                               *ground_centroid,
                                               *CONVERSION_PARAMS['geo_centroid'],
                                               *CONVERSION_PARAMS['unit_scaling'])


    meshes = extract_meshes(converted_mesh)

    # Compute orientations
    mesh_orientations = [calculate_tilt_azimuth(mesh) for mesh in meshes]
    centroids = [compute_centroid(mesh) for mesh in meshes]
    mesh_coordinate_map = create_mesh_coordinate_map(meshes)

    # Simulation parameters
    location_config = {
        'timezone': 'Europe/London',
        'start_time': datetime.datetime(2023, 1, 1, 0, 0),
        'end_time': datetime.datetime(2023, 12, 31, 23, 0),
        'time_base': 'hourly'
    }

    # Run simulation for each mesh
    results = {}
    for idx, (centroid, (tilt, azimuth)) in enumerate(zip(centroids, mesh_orientations)):
        mesh_id = f"Mesh_{idx + 1}"
        results[mesh_id] = simulate_period_for_mesh(
            centroid, tilt, azimuth,
            location_config['timezone'],
            location_config['start_time'],
            location_config['end_time'],
            location_config['time_base']
        )

    # Process results
    print(results)
    averages = calculate_mesh_averages(results)
    comprehensive_results = create_comprehensive_results(averages, mesh_coordinate_map)
    #print(comprehensive_results)

    # Plotting (same as original)
    radiances = [mesh['average_radiance'] for mesh in comprehensive_results.values()]
    norm = Normalize(vmin=min(radiances), vmax=max(radiances))
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for mesh in comprehensive_results.values():
        coords = np.array(mesh['original_coordinates'])
        polygon = Poly3DCollection([coords], alpha=0.8)
        polygon.set_facecolor(cmap(norm(mesh['average_radiance'])))
        ax.add_collection3d(polygon)

    ax.set_xlabel('X');
    ax.set_ylabel('Y');
    ax.set_zlabel('Z')
    sm = ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, label='Average Radiance (W/m²)')
    plt.show()