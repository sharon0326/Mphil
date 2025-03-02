import datetime
import pandas as pd
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance
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


def is_shaded(centroid, solar_dir, mesh_triangles, building_triangles, current_idx):
    """Check shading from both meshes and building structure"""
    ray_origin = centroid + solar_dir * 1e-6  # Offset to avoid self-intersection
    #print(ray_origin)

    # Check against other mesh triangles
    #for i, tri in enumerate(mesh_triangles):
    #    if i == current_idx:
    #        continue
    #    if ray_triangle_intersection(ray_origin, solar_dir, tri):
    #        return True

    # Check against building triangles
    for tri in building_triangles:
        if ray_triangle_intersection(ray_origin, solar_dir, tri):
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


def simulate_period_with_shading(centroid, tilt, azimuth, mesh_triangles, building_triangles, current_idx, timezone_str,
                                 start_time, end_time, time_base='daily'):
    """Simulate solar flux for a mesh triangle considering shading from other meshes and buildings."""
    # Set up location and site
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

    # Get solar position and clearsky data
    solar_pos = site.get_solarposition(times)
    clearsky = site.get_clearsky(times, model='ineichen', linke_turbidity=3.0)

    # Precompute shading status for each time point
    shaded_mask = np.zeros(len(times), dtype=bool)
    for i, (ts, pos) in enumerate(solar_pos.iterrows()):
        solar_azimuth = pos['azimuth']
        solar_zenith = pos['apparent_zenith']
        solar_dir = solar_vector(solar_azimuth, solar_zenith)

        # Check if the current triangle is shaded at this time
        shaded_mask[i] = is_shaded(
            centroid, solar_dir,
            mesh_triangles, building_triangles,
            current_idx
        )

    # Calculate irradiance for all timesteps
    poa = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_pos['apparent_zenith'],
        solar_azimuth=solar_pos['azimuth']
    )

    # Apply shading mask (set shaded timesteps to zero)
    total_flux = poa['poa_global'].clip(lower=0)
    total_flux[shaded_mask] = 0

    # Aggregate results based on time_base
    if time_base == 'hourly':
        aggregated = total_flux
    elif time_base == 'daily':
        aggregated = total_flux.resample('D').sum()
    elif time_base == 'weekly':
        aggregated = total_flux.resample('W').sum()

    print(times)
    print(poa)
    return {ts.to_pydatetime(): val for ts, val in aggregated.items()}


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

def calculate_shading_status(mesh_triangles, building_triangles, solar_azimuth, solar_zenith):
    """Calculate shading status for all mesh triangles"""
    solar_dir = solar_vector(solar_azimuth, solar_zenith)
    shaded = []
    unshaded = []

    for idx, (tri, mesh_idx) in enumerate(mesh_triangles):
        centroid = compute_centroid(tri)
        if is_shaded(centroid, solar_dir,
                     [t[0] for t in mesh_triangles],
                     building_triangles, idx):
            shaded.append(tri)
        else:
            unshaded.append(tri)

    return shaded, unshaded




def validate_mesh_integrity(mesh_triangles):
    """Validate mesh triangle structure"""
    for idx, (tri, mesh_idx) in enumerate(mesh_triangles):
        if len(tri) != 3 or any(len(vertex) != 3 for vertex in tri):
            raise ValueError(f"Invalid triangle at index {idx}")
        if not all(isinstance(coord, (int, float)) for vertex in tri for coord in vertex):
            raise TypeError(f"Non-numeric coordinates in triangle {idx}")


def run_solar_simulation(building_data, solar_meshes, config):
    """Run full solar simulation across specified time period"""
    # Prepare simulation components
    mesh_orientations = [calculate_tilt_azimuth(tri) for tri, _ in solar_meshes['mesh_triangles']]
    centroids = [compute_centroid(tri) for tri, _ in solar_meshes['mesh_triangles']]
    mesh_coordinate_map = create_mesh_coordinate_map(solar_meshes['mesh_triangles'])

    # Initialize results storage
    results = {}

    # Main simulation loop
    for idx, (centroid, (tilt, azimuth)) in enumerate(zip(centroids, mesh_orientations)):
        results[f"Mesh_{idx + 1}"] = simulate_period_with_shading(
            centroid=centroid,
            tilt=tilt,
            azimuth=azimuth,
            mesh_triangles=[tri for tri, _ in solar_meshes['mesh_triangles']],
            building_triangles=building_data['building_triangles'],
            current_idx=idx,
            timezone_str=config['timezone'],
            start_time=config['simulation_period']['start'],
            end_time=config['simulation_period']['end'],
            time_base=config['simulation_period']['resolution']
        )

    return process_results(results, mesh_coordinate_map)


def process_results(raw_results, coordinate_map):
    """Process and enrich simulation results"""
    averages = calculate_mesh_averages(raw_results)
    return create_comprehensive_results(averages, coordinate_map)


def visualize_simulation_results(results):
    """Visualize final results with 3D colormap"""
    radiances = [mesh['average_radiance'] for mesh in results.values()]
    norm = Normalize(vmin=min(radiances), vmax=max(radiances))
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for mesh in results.values():
        coords = np.array(mesh['original_coordinates'])
        polygon = Poly3DCollection([coords], alpha=0.8)
        polygon.set_facecolor(cmap(norm(mesh['average_radiance'])))
        ax.add_collection3d(polygon)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Elevation')

    sm = ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, label='Average Radiance (W/m²)')
    plt.show()

def initialize_components(config):
    """Initialize and return building and solar mesh data"""
    # Process building geometry
    roof, converted_building, building_triangles = load_and_process_building(config)
    # Process solar panel meshes
    converted_mesh, mesh_triangles = process_solar_meshes(roof, config)

    # Create coordinate map during initialization
    coordinate_map = create_mesh_coordinate_map(mesh_triangles)

    return (
        {
            'building_triangles': building_triangles,
            'roof': roof,
            'converted_building': converted_building
        },
        {
            'mesh_triangles': mesh_triangles,
            'converted_mesh': converted_mesh,
            'coordinate_map': coordinate_map
        }
    )


def run_complete_simulation(building_data, solar_meshes, config):
    """Execute full solar potential simulation"""
    # Prepare simulation data
    mesh_data = prepare_mesh_data(solar_meshes['mesh_triangles'])

    # Run simulation for each mesh element
    results = {}
    for idx, (centroid, orientation) in enumerate(zip(mesh_data['centroids'], mesh_data['orientations'])):
        results[f"Mesh_{idx + 1}"] = execute_single_simulation(
            centroid=centroid,
            tilt=orientation[0],
            azimuth=orientation[1],
            building_data=building_data,
            solar_meshes=solar_meshes,
            config=config,
            mesh_idx=idx
        )

    return results


def prepare_mesh_data(mesh_triangles):
    """Prepare mesh data for simulation"""
    return {
        'orientations': [calculate_tilt_azimuth(tri) for tri, _ in mesh_triangles],
        'centroids': [compute_centroid(tri) for tri, _ in mesh_triangles]
        # Removed coordinate map creation from here
    }

def execute_single_simulation(centroid, tilt, azimuth, building_data, solar_meshes, config, mesh_idx):
    """Run simulation for a single mesh element"""
    return simulate_period_with_shading(
        centroid=centroid,
        tilt=tilt,
        azimuth=azimuth,
        mesh_triangles=[tri for tri, _ in solar_meshes['mesh_triangles']],
        building_triangles=building_data['building_triangles'],
        current_idx=mesh_idx,
        timezone_str=config['timezone'],
        start_time=config['simulation_params']['start'],
        end_time=config['simulation_params']['end'],
        time_base=config['simulation_params']['resolution']
    )


def process_and_visualize_results(raw_results, solar_meshes):
    """Process results and create visualizations"""
    # Calculate averages and create comprehensive dataset
    averages = calculate_mesh_averages(raw_results)

    # Use coordinate map from solar_meshes
    comprehensive_results = create_comprehensive_results(averages, solar_meshes['coordinate_map'])

    # Generate visualization
    create_3d_visualization(comprehensive_results)

    return comprehensive_results

def create_3d_visualization(results_data):
    """Generate 3D visualization of results"""
    radiances = [mesh['average_radiance'] for mesh in results_data.values()]
    norm = Normalize(vmin=min(radiances), vmax=max(radiances))
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for mesh in results_data.values():
        coords = np.array(mesh['original_coordinates'])
        polygon = Poly3DCollection([coords], alpha=0.8)
        polygon.set_facecolor(cmap(norm(mesh['average_radiance'])))
        ax.add_collection3d(polygon)

    ax.set_xlabel('X Axis', fontsize=9)
    ax.set_ylabel('Y Axis', fontsize=9)
    ax.set_zlabel('Elevation', fontsize=9)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, label='Average Radiance (W/m²)')
    plt.tight_layout()
    plt.show()

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
        'resolution': 'weekly'
    },
    'visualization': {
        'face_color': plt.cm.viridis(0.5),
        'edge_color': 'k',
        'alpha': 0.5,
        'labels': ('Longitude', 'Latitude', 'Elevation (m)')
    }
    }

    # Initialize building and solar components
    building_data, solar_meshes = initialize_components(CONVERSION_PARAMS)
    # Run full solar simulation
    simulation_results = run_complete_simulation(building_data, solar_meshes, CONVERSION_PARAMS)
    # Process and visualize final results
    comprehensive_results = process_and_visualize_results(simulation_results, solar_meshes)
    print(comprehensive_results)


    # Configuration (same as original)
    CONVERSION_PARAMS = {
        'input_file': "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/test2.txt",
        'earth_radius': 6378137.0,  # WGS84 in meters
        'local_centroid': (12.516, 21.043),  # (x, y) in meters
        'geo_centroid': (52.1986125198786, 0.11358089726501427),  # (lat, lon)
        'unit_scaling': (10.0, 10.0, 10.0),  # (x, y, z) scale factors
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
            'grid_size': 4.0
        },
        'plot_style': {
            'face_color': plt.cm.viridis(0.5),
            'edge_color': 'k',
            'alpha': 0.5,
            'labels': ('Longitude', 'Latitude', 'Elevation (m)')
        }
    }

    # for a specific solar asimuth and zenith, calculate and plot the unshaded and shaded mesh
    roof, converted_building, building_triangles = load_and_process_building(CONVERSION_PARAMS)
    # Process solar meshes
    converted_mesh, mesh_triangles = process_solar_meshes(roof, CONVERSION_PARAMS)

    # Calculate shading
    shaded, unshaded = calculate_shading_status(
        mesh_triangles, building_triangles,
        solar_azimuth=180, solar_zenith=45
    )
    # Visualize results
    plot_solar_access(
        shaded=shaded,
        unshaded=unshaded,
        solar_azimuth=180,
        solar_zenith=45
    )

    #-----------------all above is fine-----------------------
    # Updated processing loop with shading integration
    mesh_orientations = [calculate_tilt_azimuth(tri[0]) for tri in mesh_triangles]
    centroids = [compute_centroid(geom) for geom, idx in mesh_triangles]
    #print(centroids)

    mesh_coordinate_map = create_mesh_coordinate_map(mesh_triangles)
    #print(mesh_coordinate_map)

    # Generate building triangles from original structure (should be precomputed)
    building_triangles = []
    for face in roof.triangular_F:  # Use pre-triangulated faces
        building_triangles.append([converted_building[i] for i in face])

    # Convert and extract all mesh triangles (should be precomputed)
    converted_mesh = convert_coordinate_system(CONVERSION_PARAMS, roof.mesh_objects,
                                               *CONVERSION_PARAMS['local_centroid'],
                                               *CONVERSION_PARAMS['geo_centroid'],
                                               *CONVERSION_PARAMS['unit_scaling'])
    all_mesh_triangles = []
    for square in extract_meshes(converted_mesh):
        tri1 = [square[0], square[1], square[2]]
        tri2 = [square[0], square[2], square[3]]
        all_mesh_triangles.extend([tri1, tri2])

    # Simulation parameters
    location_config = {
        'timezone': 'Europe/London',
        'start_time': datetime.datetime(2023,1 , 1, 8, 0),
        'end_time': datetime.datetime(2023, 12, 31, 16, 0),
        'time_base': 'weekly'
    }

    # Main simulation loop with shading
    results = {}
    for idx, (centroid, (tilt, azimuth)) in enumerate(zip(centroids, mesh_orientations)):
        mesh_id = f"Mesh_{idx + 1}"

        results[mesh_id] = simulate_period_with_shading(
            centroid=centroid,
            tilt=tilt,
            azimuth=azimuth,
            mesh_triangles=all_mesh_triangles,
            building_triangles=building_triangles,
            current_idx=idx,
            timezone_str=location_config['timezone'],
            start_time=location_config['start_time'],
            end_time=location_config['end_time'],
            time_base=location_config['time_base']
        )

    # Post-processing
    averages = calculate_mesh_averages(results)
    #print(averages)
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

