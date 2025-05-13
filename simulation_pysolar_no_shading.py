import pytz
from pysolar import radiation, solar
import read_polyshape_3d
from coplanarity_mesh import RoofSolarPanel
from cartesian_lonlat import convert_coordinate_system
from typing import Dict
import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np


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
    """Compute centroid (average of longitude, latitude, elevation) of each mesh for point simulation."""
    n = len(mesh)
    avg_lon = sum(p[0] for p in mesh) / n
    avg_lat = sum(p[1] for p in mesh) / n
    avg_alt = sum(p[2] for p in mesh) / n
    return (avg_lon, avg_lat, avg_alt)


def simulate_period_for_centroid(centroid, timezone_str, start_time, end_time, time_base='daily'):
    """Simulate solar radiance for a centroid between specified timestamps."""
    tz = pytz.timezone(timezone_str)
    results = {}

    # Timezone and UTC
    if not start_time.tzinfo:
        start_time = tz.localize(start_time)
    if not end_time.tzinfo:
        end_time = tz.localize(end_time)

    utc_start = start_time.astimezone(pytz.utc)
    utc_end = end_time.astimezone(pytz.utc)

    if time_base == 'hourly':
        hourly_totals = {}
        current = start_time.replace(minute=0, second=0, microsecond=0)
        while current <= end_time:
            try:
                localized_time = tz.localize(current.replace(tzinfo=None), is_dst=False)
            except pytz.NonExistentTimeError:
                current += datetime.timedelta(hours=1)
                continue

            utc_time = localized_time.astimezone(pytz.utc)
            if utc_time < utc_start or utc_time > utc_end:
                current += datetime.timedelta(hours=1)
                continue

            altitude = solar.get_altitude(centroid[1], centroid[0], utc_time)
            if altitude > 0:
                direct_rad = radiation.get_radiation_direct(utc_time, altitude)
                hourly_rad = direct_rad
            else:
                hourly_rad = 0.0

            #print(current)
            #print(hourly_rad)
            #print()

            hourly_totals[localized_time] = hourly_rad
            current += datetime.timedelta(hours=1)

        results = hourly_totals

    elif time_base == 'daily':
        daily_totals = {}
        current_date = start_time.date()
        end_date = end_time.date()

        while current_date <= end_date:
            total_rad_day = 0.0

            # First simulate hourly radiance
            for hour in range(24):
                try:
                    naive_time = datetime.datetime.combine(current_date, datetime.time(hour, 0))
                    localized_time = tz.localize(naive_time, is_dst=False)
                except pytz.NonExistentTimeError:
                    continue

                utc_time = localized_time.astimezone(pytz.utc)
                if utc_time < utc_start or utc_time > utc_end:
                    continue

                altitude = solar.get_altitude(centroid[1], centroid[0], utc_time)
                if altitude > 0:
                    direct_rad = radiation.get_radiation_direct(utc_time, altitude)
                    total_rad_day += direct_rad

            daily_totals[current_date] = total_rad_day
            current_date += datetime.timedelta(days=1)

        results = daily_totals

    elif time_base == 'weekly':
        daily_totals = {}
        current_date = start_time.date()
        end_date = end_time.date()

        # First collect daily totals
        while current_date <= end_date:
            total_rad_day = 0.0
            for hour in range(24):
                try:
                    naive_time = datetime.datetime.combine(current_date, datetime.time(hour, 0))
                    localized_time = tz.localize(naive_time, is_dst=False)
                except pytz.NonExistentTimeError:
                    continue

                utc_time = localized_time.astimezone(pytz.utc)
                if utc_time < utc_start or utc_time > utc_end:
                    continue

                altitude = solar.get_altitude(centroid[1], centroid[0], utc_time)
                if altitude > 0:
                    direct_rad = radiation.get_radiation_direct(utc_time, altitude)
                    total_rad_day += direct_rad

            daily_totals[current_date] = total_rad_day
            current_date += datetime.timedelta(days=1)

        # Group into weeks starting from first day of simulation
        weekly_totals = {}
        days = sorted(daily_totals.keys())
        for i in range(0, len(days), 7):
            week_days = days[i:i + 7]
            week_start = week_days[0]
            week_total = sum(daily_totals[day] for day in week_days if day in daily_totals)
            weekly_totals[week_start] = week_total

        results = weekly_totals

    else:
        raise ValueError(f"Invalid time_base: {time_base}")

    #print(results)
    return results

def calculate_mesh_averages(input_data: Dict[str, Dict[datetime.datetime, float]]) -> Dict[str, float]:
    """
    Calculate average value for each mesh grid across all time entries.

    Args:
        input_data: Dictionary mapping mesh IDs to datetime-value pairs

    Returns:
        Dictionary mapping mesh IDs to their average values
    """
    mesh_averages = {}

    for mesh_id, time_series in input_data.items():
        # Extract all numerical values from the time series
        values = list(time_series.values())

        if not values:  # Handle empty time series
            mesh_averages[mesh_id] = 0.0
            continue

        # Calculate average and round for readability (optional)
        average = sum(values) / len(values)
        mesh_averages[mesh_id] = average

    return mesh_averages

# Add these helper functions
def create_mesh_coordinate_map(meshes):
    """Create mapping from mesh ID to original coordinates and centroid"""
    return {
        f"Mesh_{idx+1}": {
            'original_coordinates': mesh,
            'centroid': compute_centroid(mesh)
        }
        for idx, mesh in enumerate(meshes)
    }

def create_comprehensive_results(averages, coordinate_map):
    """Combine averages with coordinate information"""
    return {
        mesh_id: {
            'average_radiance': avg,
            'original_coordinates': coordinate_map[mesh_id]['original_coordinates'],
            'centroid': coordinate_map[mesh_id]['centroid']
        }
        for mesh_id, avg in averages.items()
    }


if __name__ == '__main__':
    # Configuration
    # Scale factors assume decimeter
    CONVERSION_PARAMS = {
        'input_file': "C:/Users/Sharon/Desktop/SGA21_roofOptimization-main/SGA21_roofOptimization-main/RoofGraphDataset/res_building/BJ39_500_098051_0020.polyshape",
        'earth_radius': 6378137.0,  # WGS84 in meters
        'local_centroid': (12.516, 21.043),  # (x, y) in meters
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

    converted_mesh = convert_coordinate_system(CONVERSION_PARAMS,
        roof.mesh_objects,
        *CONVERSION_PARAMS['local_centroid'],
        *CONVERSION_PARAMS['geo_centroid'],
        *CONVERSION_PARAMS['unit_scaling']
    )

    # print(converted_mesh)
    mesh_input = converted_mesh


    # Maintain four example configurations (though only location_config is used)
    location_confi = {
        'timezone': 'America/New_York',
        'start_time': datetime.datetime(2023, 5, 10, 10, 30),
        'end_time': datetime.datetime(2023, 5, 10, 12, 45),
        'time_base': 'weekly'
    }

    location_config = {
        'timezone': 'Europe/London',
        'start_time': datetime.datetime(2023, 1, 1, 0, 0),
        'end_time': datetime.datetime(2023, 12, 31, 23, 0),
        'time_base': 'hourly'
    }

    location_confi = {
        'timezone': 'Asia/Tokyo',
        'start_time': datetime.datetime(2023, 1, 1),
        'end_time': datetime.datetime(2023, 1, 3),
        'time_base': 'daily'
    }

    location_confi = {
        'timezone': 'Europe/London',
        'start_time': datetime.datetime(2023, 12, 31),
        'end_time': datetime.datetime(2024, 1, 1),
        'time_base': 'hourly'
    }

    meshes = extract_meshes(mesh_input)
    # Create coordinate mapping first
    mesh_coordinate_map = create_mesh_coordinate_map(meshes)

    # Then process centroids (existing code)
    centroids = [compute_centroid(mesh) for mesh in meshes]

    # Simulate for each centroid
    results = {}
    for idx, centroid in enumerate(centroids):
        print(f"Processing Mesh {idx + 1}...")
        results[f"Mesh_{idx + 1}"] = simulate_period_for_centroid(
            centroid,
            location_config['timezone'],
            location_config['start_time'],
            location_config['end_time'],
            location_config.get('time_base', 'daily')
        )

    # Example Output
    #print("\nSample Output:")
    time_base = location_config['time_base']
    #print(len(results))
    #print(type(results))
    #print(results)

    example_key = next(iter(results['Mesh_1']))  # Get first key for demonstration

    if time_base == 'hourly':
        formatted_time = example_key.strftime("%Y-%m-%d %H:%M:%S %Z")
        radiance = results['Mesh_1'][example_key]
        print(f"{formatted_time}: {radiance:.2f} W/m²")
    elif time_base == 'daily':
        formatted_date = example_key.strftime("%Y-%m-%d")
        radiance = results['Mesh_1'][example_key]
        print(f"{formatted_date}: {radiance:.2f} W/m²")
    elif time_base == 'weekly':
        formatted_date = example_key.strftime("%Y-%m-%d")
        radiance = results['Mesh_1'][example_key]
        print(f"Week starting {formatted_date}: {radiance:.2f} W/m²")

    averages = calculate_mesh_averages(results)
    #print(averages)

    # After calculating averages
    comprehensive_results = create_comprehensive_results(averages, mesh_coordinate_map)
    print(comprehensive_results)
    # comprehensive_results

    # Extract all radiance values for normalization
    radiances = [mesh['average_radiance'] for mesh in comprehensive_results.values()]
    norm = Normalize(vmin=min(radiances), vmax=max(radiances))
    cmap = plt.get_cmap('viridis')  # Choose a colormap

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each mesh as a 3D polygon
    for name, mesh in comprehensive_results.items():
        coords = np.array(mesh['original_coordinates'])
        polygon = Poly3DCollection([coords], alpha=0.8)
        color = cmap(norm(mesh['average_radiance']))
        polygon.set_facecolor(color)
        ax.add_collection3d(polygon)

    # Configure axes and labels
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title('3D Mesh Grids Colored by Average Radiance')

    # Add a colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Average Radiance')

    # Adjust view and display
    ax.autoscale_view()
    plt.show()


    '''
    # Example usage of the comprehensive results
    sample_mesh = 'Mesh_38'
    print(f"\nDetailed results for {sample_mesh}:")
    print(f"Average radiance: {comprehensive_results[sample_mesh]['average_radiance']} W/m²")
    print("Original coordinates:")
    for coord in comprehensive_results[sample_mesh]['original_coordinates']:
        print(f"  {coord}")
    print(f"Centroid: {comprehensive_results[sample_mesh]['centroid']}")
    '''
