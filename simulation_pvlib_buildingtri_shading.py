import datetime
import time
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
    if not isinstance(triangle, np.ndarray):
        try:
            triangle = np.array(triangle, dtype=np.float64)
        except ValueError as e:
            raise ValueError(f"Invalid triangle structure: {triangle}") from e

    if triangle.shape != (3, 3):
        raise ValueError(f"Invalid triangle dimensions: {triangle.shape}. Should be (3,3)")

    return np.mean(triangle, axis=0)

def ray_triangle_intersection(ray_origin, ray_dir, triangle, epsilon=1e-10):
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

def is_shaded(triangle, solar_dir, building_triangles, num_samples):
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
    Visualizes 3D solar access analysis by plotting shaded and unshaded areas with compass.

    Parameters:
    - shaded (list of arrays): List of polygons representing shaded areas.
    - unshaded (list of arrays): List of polygons representing unshaded areas.
    - solar_azimuth (float): Solar azimuth angle in degrees for the title.
    - solar_zenith (float): Solar zenith angle in degrees for the title.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot UNSHADED areas (yellow)
    if unshaded:
        unshaded_collection = Poly3DCollection(
            unshaded,
            facecolors='#ffff00',  # Yellow
            edgecolors='#999900',  # Dark yellow edges
            linewidths=0.3,
            alpha=0.9,
            zorder=2
        )
        ax.add_collection3d(unshaded_collection)

    # Plot SHADED areas (grey) on top
    if shaded:
        shaded_collection = Poly3DCollection(
            shaded,
            facecolors='#b0b0b0',  # Grey
            edgecolors='#505050',  # Darker grey edges
            linewidths=0.3,
            alpha=0.8,
            zorder=3
        )
        ax.add_collection3d(shaded_collection)

    if shaded or unshaded:
        all_points = np.concatenate(shaded + unshaded) if shaded else np.concatenate(unshaded)
        min_vals = np.min(all_points, axis=0)
        max_vals = np.max(all_points, axis=0)

        padding = 0.1 * (max_vals - min_vals)
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(min_vals[1] - padding[1], max_vals[1] + padding[1])
        ax.set_zlim(min_vals[2] - padding[2], max_vals[2] + padding[2])

    ax.view_init(elev=45, azim=-45)
    ax.set_xlabel('X Axis (East-West)', fontsize=10, labelpad=10)
    ax.set_ylabel('Y Axis (North-South)', fontsize=10, labelpad=10)
    ax.set_zlabel('Elevation', fontsize=10, labelpad=10)
    ax.set_title(
        f'Solar Access Map\nAzimuth: {solar_azimuth}°, Zenith: {solar_zenith}°',
        fontsize=12, pad=15
    )

    legend_elements = [
        plt.matplotlib.patches.Patch(facecolor='#ffff00', alpha=0.9, label='Direct Sunlight'),
        plt.matplotlib.patches.Patch(facecolor='#b0b0b0', alpha=0.8, label='Shaded Areas')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()
    ax.xaxis.set_pane_color((0.95, 0.95, 0.95))
    ax.yaxis.set_pane_color((0.95, 0.95, 0.95))
    ax.zaxis.set_pane_color((0.97, 0.97, 0.97))
    ax.grid(False)

    plt.show()


def simulate_period_with_shading(centroid, tilt, azimuth, mesh_triangles, building_triangles, current_idx, timezone_str,
                                 start_time, end_time, num_samples, time_base='hourly'):
    """Simulate solar flux for a mesh triangle considering shading from other meshes and buildings."""
    location = {
        'latitude': centroid[1],
        'longitude': centroid[0],
        'timezone': timezone_str
    }
    site = Location(location['latitude'], location['longitude'], tz=location['timezone'])
    triangle_geometry = mesh_triangles[current_idx]

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

    shaded_mask = np.zeros(len(times), dtype=bool)
    for i, (ts, pos) in enumerate(solar_pos.iterrows()):
        solar_azimuth = pos['azimuth']
        solar_zenith = pos['apparent_zenith']
        solar_dir = solar_vector(solar_azimuth, solar_zenith)

        # Check if the current triangle is shaded at this time
        shaded_mask[i] = is_shaded(triangle_geometry, solar_dir, building_triangles, num_samples)

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
    total_flux[shaded_mask] = 0

    if time_base == 'hourly':
        aggregated = total_flux
    elif time_base == 'daily':
        aggregated = total_flux.resample('D').sum()
    elif time_base == 'weekly':
        aggregated = total_flux.resample('W').sum()

    #print(times)
    #print(poa)
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

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def load_and_process_building(params):
    """Load and process building geometry data"""
    vertices, faces = read_polyshape_3d.read_polyshape(params['input_file'])
    roof = RoofSolarPanel(
        V=vertices,
        F=faces,
        **params['panel_config']
    )
    #roof.display_building_and_rooftops()
    #roof.plot_building_with_mesh_grid()
    #roof.plot_rooftops_with_mesh_points()


    ground_centroid = roof.get_ground_centroid()[:2]

    # Convert building coordinates
    converted_building = convert_coordinate_system_building(
        params, roof.V,
        *ground_centroid,
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
    ground_centroid = roof.get_ground_centroid()[:2]
    """Process and triangulate solar panel meshes"""
    converted_mesh = convert_coordinate_system(
        params, roof.mesh_objects,
        *ground_centroid,
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



# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def generate_sample_points(triangle, num_samples=11):
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

def calculate_shading_status(mesh_triangles, building_triangles, solar_azimuth, solar_zenith, num_samples=1):
    """Simplified shading calculation using enhanced is_shaded"""
    solar_dir = solar_vector(solar_azimuth, solar_zenith)
    shaded = []
    unshaded = []

    for idx, (tri, mesh_idx) in enumerate(mesh_triangles):
        if is_shaded(tri, solar_dir, building_triangles, num_samples):
            shaded.append(tri)
        else:
            unshaded.append(tri)

    return shaded, unshaded

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def process_results(raw_results, coordinate_map):
    """Process and enrich simulation results"""
    averages = calculate_mesh_averages(raw_results)
    return create_comprehensive_results(averages, coordinate_map)

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
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

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
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

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def prepare_mesh_data(mesh_triangles):
    """Prepare mesh data for simulation"""
    return {
        'orientations': [calculate_tilt_azimuth(tri) for tri, _ in mesh_triangles],
        'centroids': [compute_centroid(tri) for tri, _ in mesh_triangles]
        # Removed coordinate map creation from here
    }

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
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
        time_base=config['simulation_params']['resolution'],
        num_samples=config['simulation_params']['shading_samples']
    )

# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def final_results(raw_results, solar_meshes):
    """Process results"""
    # Calculate averages and create comprehensive dataset
    averages = calculate_mesh_averages(raw_results)

    # Use coordinate map from solar_meshes
    comprehensive_results = create_comprehensive_results(averages, solar_meshes['coordinate_map'])

    return comprehensive_results


def create_3d_visualization_with_color_scheme(results_data):
    """Generate 3D visualization of results with fixed color scale (0-160 W/m²)"""
    # Fixed color scale from 0 to 160 W/m²
    norm = Normalize(vmin=0, vmax=160)
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    for mesh_id, mesh in results_data.items():
        coords = np.array(mesh['original_coordinates'])
        polygon = Poly3DCollection([coords], alpha=0.8)

        # Use fixed color scale (0-160 range)
        radiance_value = mesh['average_radiance']
        polygon.set_facecolor(cmap(norm(radiance_value)))
        ax.add_collection3d(polygon)

    ax.set_xlabel('X Axis', fontsize=9)
    ax.set_ylabel('Y Axis', fontsize=9)
    ax.set_zlabel('Elevation', fontsize=9)
    ax.grid(True)

    # Add colorbar with fixed scale (0-160 W/m²)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Average Radiance (W/m²)', fontsize=10)

    cbar.set_ticks([0, 20, 40, 60, 80, 100, 120, 140, 160])

    ax.view_init(elev=45, azim=-45)
    plt.tight_layout()
    plt.show()


# For the clarity of the main. To debug, see try.pvlib_shaded_simulation
def create_3d_visualization(results_data):
    #Generate 3D visualization of results with mesh labels.
    radiances = [mesh['average_radiance'] for mesh in results_data.values()]
    norm = Normalize(vmin=min(radiances), vmax=max(radiances))
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot meshes with labels
    for mesh_id, mesh in results_data.items():
        # Plot the polygon
        coords = np.array(mesh['original_coordinates'])
        polygon = Poly3DCollection([coords], alpha=0.8)
        polygon.set_facecolor(cmap(norm(mesh['average_radiance'])))
        ax.add_collection3d(polygon)

        """
        # Add mesh number label at centroid
        mesh_number = int(mesh_id.split('_')[1])  # Extract number from "Mesh_X"
        centroid = mesh['centroid']
        ax.text(
            centroid[0], centroid[1], centroid[2],
            str(mesh_number),
            color='white',
            fontsize=9,
            ha='center',
            va='center',
            bbox=dict(
                boxstyle="round",
                facecolor='black',
                alpha=0.7,
                edgecolor='none'
            ),
            zorder=4  # Ensure labels stay on top
        )
        """

    # Configure axes and colorbar
    ax.set_xlabel('X Axis', fontsize=9)
    ax.set_ylabel('Y Axis', fontsize=9)
    ax.set_zlabel('Elevation', fontsize=9)
    ax.grid(True)

    # Add colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label('Average Radiance (W/m²)', fontsize=10)

    # Set viewing angle
    ax.view_init(elev=45, azim=-45)
    plt.tight_layout()
    plt.show()

def save_hourly_data_to_txt(simulation_results, output_dir="hourly_results"):
    """Save hourly irradiance data for mesh pairs to text files."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Sort mesh IDs numerically
    mesh_ids = sorted(simulation_results.keys(),
                      key=lambda x: int(x.split('_')[1]))

    groups = []
    for i in range(0, len(mesh_ids), 2):
        pair = mesh_ids[i:i + 2]
        group_name = f"Group_{i // 2 + 1}_Meshes_{'-'.join([m.split('_')[1] for m in pair])}"
        groups.append((group_name, pair))

    for group_name, mesh_pair in groups:
        filename = os.path.join(output_dir, f"{group_name}_hourly.txt")

        # Get common timestamps (assuming all meshes have same timestamps)
        timestamps = list(simulation_results[mesh_pair[0]].keys())

        with open(filename, 'w') as f:
            # Write header
            #f.write("Timestamp,Solar_Trace_kW_per_m2\n")

            for ts in timestamps:
                # Get values for all meshes in pair
                values = []
                for mesh_id in mesh_pair:
                    values.append(simulation_results[mesh_id].get(ts, 0))

                # Calculate average
                avg_irradiance = (np.mean(values) * 0.2 /1000) if values else 0
                #meshes_list = ','.join([m.split('_')[1] for m in mesh_pair])

                # Write formatted line
                iso_time = ts.isoformat()
                #f.write(f"{iso_time},{avg_irradiance:.16f}\n")
                f.write(f"{avg_irradiance:.16f}\n")
        #print(f"Saved {filename} containing: {', '.join(mesh_pair)}")


if __name__ == '__main__':
    """Main execution flow for solar potential analysis"""
    # Load configuration parameters
    CONVERSION_PARAMS = {
    'input_file': "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/single_segment.txt",
    'earth_radius': 6378137.0,
    #'geo_centroid': (52.19850723176565, 0.11353796391182779),
    #'geo_centroid':  (-69.3872203183979, -67.70319583227294),
    'geo_centroid':  (52.1986125198786, 0.11358089726501427),
    #'geo_centroid':(-40, 23.879985430205615),
    #'geo_centroid': (40, 23.879985430205615),
    'unit_scaling': (1.0, 1.0, 1.0),
    'timezone': 'Europe/London',
    'panel_config': {
        'panel_dx': 1.0,
        'panel_dy': 1.0,
        'max_panels': 10,
        'b_scale_x': 0.05,
        'b_scale_y': 0.05,
        'b_scale_z': 0.05,
        'exclude_face_indices': [],
        'grid_size': 1.0            #so now, when I do the simulation, it is 1m^2 for each mesh grid, if input is 1
                                    #the triangular meshgrid is triangulated, so it should be 1/2 3/4... averaged between the 2
    },
    'simulation_params': {
        'start': datetime.datetime(2023, 1, 1, 0, 0),
        'end': datetime.datetime(2023,  12, 31, 23, 0),
        'resolution': 'hourly',
        'shading_samples': 10     # when simulating the ray-tracing algo, the number of samples drawn
    },
    'visualization': {
        'face_color': plt.cm.viridis(0.5),
        'edge_color': 'k',
        'alpha': 0.5,
        'labels': ('Longitude', 'Latitude', 'Elevation (m)')
    }
    }
    """
    # for a specific solar asimuth and zenith, calculate and plot the unshaded and shaded mesh
    roof, converted_building, building_triangles = load_and_process_building(CONVERSION_PARAMS)
    converted_mesh, mesh_triangles = process_solar_meshes(roof, CONVERSION_PARAMS)
    shaded, unshaded = calculate_shading_status(
        mesh_triangles, building_triangles,
        solar_azimuth=60, solar_zenith=45,
        num_samples=CONVERSION_PARAMS['simulation_params']['shading_samples']
    )
    plot_solar_access(
        shaded=shaded,
        unshaded=unshaded,
        solar_azimuth=0,
        solar_zenith=45
    )
    """

    start_time = time.time()

    # Run a complete simulation
    building_data, solar_meshes = initialize_components(CONVERSION_PARAMS)
    simulation_results = run_complete_simulation(building_data, solar_meshes, CONVERSION_PARAMS)

    save_hourly_data_to_txt(simulation_results)

    comprehensive_results = final_results(simulation_results, solar_meshes)
    #create_3d_visualization(comprehensive_results)

    create_3d_visualization(comprehensive_results)
    create_3d_visualization_with_color_scheme(comprehensive_results)
    print(comprehensive_results)

    end_time = time.time()
    print(f"Calculation time: {end_time - start_time:.4f} seconds")

    """
    # To visualize hourly shading area of the building
    building_triangles = building_data['building_triangles']
    mesh_triangles_list = solar_meshes['mesh_triangles']

    roof = building_data['roof']
    ground_centroid = roof.get_ground_centroid()
    latitude, longitude = ground_centroid[1], ground_centroid[0]
    timezone_str = CONVERSION_PARAMS['timezone']
    site = Location(latitude, longitude, tz=timezone_str)

    start = CONVERSION_PARAMS['simulation_params']['start']
    end = CONVERSION_PARAMS['simulation_params']['end']
    times = pd.date_range(start=start, end=end, freq='H', tz=timezone_str)

    for ts in times:
        # Get solar position
        solar_pos = site.get_solarposition(ts)
        solar_azimuth = solar_pos['azimuth'].iloc[0]
        solar_zenith = solar_pos['apparent_zenith'].iloc[0]

        shaded, unshaded = calculate_shading_status(
            mesh_triangles_list,
            building_triangles,
            solar_azimuth,
            solar_zenith,
            num_samples=CONVERSION_PARAMS['simulation_params']['shading_samples']
        )

        plot_solar_access(shaded, unshaded, solar_azimuth, solar_zenith)
        plt.pause(0.1)  # Allows interactive viewing (optional)
        """


