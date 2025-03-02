import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import datetime
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance
import pandas as pd

# Building geometry data (from your input)
vertices = np.array([
    [164.50558044, 329.01000118, 75.0], [20.40044234, 156.96361342, 75.0],
    [169.70321062, 324.65800463, 75.0], [172.85640652, 322.01782181, 75.0],
    [205.0326525, 295.07652461, 75.0], [207.14948088, 293.30409597, 75.0],
    [153.81296376, 166.39000878, 144.97271159], [179.47298813, 196.96694547, 145.0],
    [188.94729299, 197.99500568, 139.75138436], [289.36409936, 224.46547338, 75.0],
    [300.80422428, 238.26216257, 75.0], [194.36185086, 78.08146699, 75.0],
    [161.42359129, 38.90001885, 75.0], [317.86245609, 220.72886882, 91.87541569],
    [295.90895169, 177.52917059, 99.28181494], [261.04806716, 137.13388901, 140.0388996],
    [386.29267304, 268.65153886, 75.0], [337.74201912, 208.24415734, 75.0],
    [332.26798555, 188.10063676, 75.0], [332.36566724, 188.46008769, 75.0],
    [371.17038459, 155.86651339, 75.0], [432.915604, 229.61910651, 75.0],
    [379.54312919, 148.95821993, 75.0], [267.83846225, 15.53723991, 75.0],
    [164.50558044, 329.01000118, 0.0], [20.40044234, 156.96361342, 0.0],
    [161.42359129, 38.90001885, 0.0], [194.36185086, 78.08146699, 0.0],
    [267.83846225, 15.53723991, 0.0], [379.54312919, 148.95821993, 0.0],
    [332.36566724, 188.46008769, 0.0], [332.26798555, 188.10063676, 0.0],
    [371.17038459, 155.86651339, 0.0], [432.915604, 229.61910651, 0.0],
    [386.29267304, 268.65153886, 0.0], [337.74201912, 208.24415734, 0.0],
    [300.80422428, 238.26216257, 0.0], [289.36409936, 224.46547338, 0.0],
    [205.0326525, 295.07652461, 0.0], [207.14948088, 293.30409597, 0.0],
    [169.70321062, 324.65800463, 0.0], [172.85640652, 322.01782181, 0.0]
])

rooftop_faces = [
    [1, 12, 6], [23, 22, 15], [11, 23, 15, 8], [12, 11, 8, 7, 6],
    [7, 0, 3, 2, 5, 4, 9, 14, 19, 22, 15, 8], [14, 9, 10, 13],
    [14, 13, 17, 18, 19], [18, 20, 21, 16, 17], [1, 0, 7, 6]
]


def calculate_face_normal(face_vertices):
    """Calculate normal vector for a face using Newell's method"""
    normal = np.zeros(3)
    for i in range(len(face_vertices)):
        current = face_vertices[i]
        next_v = face_vertices[(i + 1) % len(face_vertices)]
        normal[0] += (current[1] - next_v[1]) * (current[2] + next_v[2])
        normal[1] += (current[2] - next_v[2]) * (current[0] + next_v[0])
        normal[2] += (current[0] - next_v[0]) * (current[1] + next_v[1])
    return normal / np.linalg.norm(normal)


def calculate_orientation(normal):
    """Convert normal vector to tilt and azimuth angles"""
    tilt = np.degrees(np.arccos(normal[2]))  # Angle from vertical
    azimuth = np.degrees(np.arctan2(normal[0], normal[1])) % 360
    return tilt, azimuth


def calculate_solar_flux(location, timestamp, tilt, azimuth):
    """Modified solar flux calculator for 3D visualization"""
    site = Location(
        latitude=location['latitude'],
        longitude=location['longitude'],
        tz=location['timezone']
    )

    time_index = pd.DatetimeIndex([pd.to_datetime(timestamp)]).tz_localize(location['timezone'])
    solar_pos = site.get_solarposition(time_index)
    clearsky = site.get_clearsky(time_index, model='ineichen')

    poa_irradiance = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_pos['apparent_zenith'],
        solar_azimuth=solar_pos['azimuth']
    )

    return (poa_irradiance['poa_direct'] + poa_irradiance['poa_diffuse']).clip(lower=0).iloc[0]


def plot_3d_radiation(vertices, faces, flux_values):
    """3D visualization of solar radiation on roof faces"""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Normalize flux values for coloring
    norm = plt.Normalize(min(flux_values), max(flux_values))
    cmap = plt.get_cmap('viridis')

    for i, face in enumerate(faces):
        face_verts = vertices[face]
        poly = Poly3DCollection([face_verts[:, :3]], alpha=0.8)
        poly.set_color(cmap(norm(flux_values[i])))
        poly.set_edgecolor('k')
        ax.add_collection3d(poly)

    # Set plot limits and labels
    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Height (m)')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Solar Flux (W/m²)')

    plt.title('3D Solar Radiation Distribution on Rooftop')
    plt.show()


if __name__ == "__main__":
    # Location configuration (New York City)
    location = {
        'latitude': 40.7128,
        'longitude': -74.0060,
        'timezone': 'America/New_York'
    }

    # Simulation time (March 26, 2025 at 3:30 PM local time)
    timestamp = datetime(2025, 3, 26, 15, 30)

    # Calculate solar flux for each rooftop face
    flux_values = []
    for face in rooftop_faces:
        # Get face vertices
        face_vertices = vertices[face]

        # Calculate face orientation
        normal = calculate_face_normal(face_vertices[:, :3])
        tilt, azimuth = calculate_orientation(normal)

        # Calculate solar flux
        flux = calculate_solar_flux(location, timestamp, tilt, azimuth)
        flux_values.append(flux)
        print(f"Face {rooftop_faces.index(face) + 1}: Tilt {tilt:.1f}°, Azimuth {azimuth:.1f}°, Flux {flux:.2f} W/m²")

    # Visualize in 3D
    plot_3d_radiation(vertices, rooftop_faces, flux_values)