import pvlib
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance
import pandas as pd
from datetime import datetime, timedelta


def calculate_solar_flux(latitude, longitude, timezone, timestamp, tilt=0, azimuth=180):
    """
    Calculate solar flux (W/m²) on a surface with specified tilt and azimuth.

    Parameters:
    latitude (float): Latitude of the location in degrees.
    longitude (float): Longitude of the location in degrees.
    timezone (str): Timezone of the location (e.g., 'America/New_York').
    timestamp (datetime or str): Timestamp for the calculation.
    tilt (float): Tilt angle of the surface in degrees (0 = horizontal).
    azimuth (float): Azimuth angle of the surface (180° = south-facing).

    Returns:
    float: Total solar flux (W/m²) on the surface.
    """
    # Create a Location object for the given coordinates
    site = Location(latitude, longitude, tz=timezone)

    # Convert timestamp to a pandas DatetimeIndex
    if not isinstance(timestamp, pd.DatetimeIndex):
        timestamp = pd.DatetimeIndex([pd.to_datetime(timestamp)]).tz_localize(timezone)

    # Calculate solar position (zenith and azimuth)
    solar_pos = site.get_solarposition(timestamp)

    # Get clear-sky irradiance (DNI, GHI, DHI) using the Ineichen model
    clearsky = site.get_clearsky(timestamp, model='ineichen')

    # Calculate in-plane irradiance components (direct, diffuse, albedo)
    poa_irradiance = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_pos['apparent_zenith'],
        solar_azimuth=solar_pos['azimuth']
    )

    # Sum direct and diffuse radiation (ignore albedo)
    total_flux = poa_irradiance['poa_direct'] + poa_irradiance['poa_diffuse']

    # Ensure non-negative values (clip to 0)
    total_flux = total_flux.clip(lower=0)

    return total_flux.iloc[0]

# Example usage
if __name__ == "__main__":
    # Input parameters (New York City example)
    latitude = 40.7128
    longitude = -74.0060
    timezone = 'America/New_York'
    timestamp = datetime(2025, 3, 26, 15, 30)  # Noon on July 1st, 2023

    # Calculate solar flux for a flat roof (tilt=0)
    flux = calculate_solar_flux(latitude, longitude, timezone, timestamp)
    print(f"Solar Flux: {flux:.2f} W/m²")

    def yearly_radiation_simulation():
        latitude = 40.7128  # New York City latitude
        longitude = -74.0060  # New York City longitude
        timezone_str = 'America/New_York'  # Timezone information

        results = []
        n = 0
        for day in range(1, 366):  # Loop through all days of the year
            dt = datetime(2023, 1, 1, 18, 45) + timedelta(days=day - 1)
            flux = calculate_solar_flux(latitude, longitude, timezone_str, dt)
            results.append((dt, flux))
            print(f"{dt}: Solar Flux: {flux:.2f} W/m²")
            n+=flux

        return n


    # Run the yearly simulation
    yearly_radiation_data = yearly_radiation_simulation()
    print(yearly_radiation_data)