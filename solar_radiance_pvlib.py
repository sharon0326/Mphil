import time

import pandas as pd
from datetime import datetime
from pvlib.location import Location
from pvlib.irradiance import get_total_irradiance

def calculate_solar_flux(
        location: dict,
        timestamp: datetime,
        tilt: float = 0,
        azimuth: float = 180,
        model: str = 'ineichen',
        linke_turbidity: float = 3.0
) -> float:
    """
    Calculate solar flux (W/m²) on a surface with specified orientation

    Parameters:
    location (dict): Location parameters containing:
        - latitude: float
        - longitude: float
        - timezone: str
    timestamp (datetime/str): Timestamp for calculation
    tilt (float): Surface tilt angle (0-90°)
    azimuth (float): Surface azimuth angle (0-360°)
    model (str): Clear sky model (default: 'ineichen')
    linke_turbidity (float): Atmospheric turbidity (average default: 3.0)

    Returns:
    float: Total solar flux (W/m²)
    """
    required_keys = ['latitude', 'longitude', 'timezone']
    if not all(key in location for key in required_keys):
        missing = [key for key in required_keys if key not in location]
        raise ValueError(f"Missing location parameters: {missing}")

    site = Location(
        latitude=location['latitude'],
        longitude=location['longitude'],
        tz=location['timezone']
    )

    # Convert to pandas DatetimeIndex
    if not isinstance(timestamp, pd.DatetimeIndex):
        time_index = pd.DatetimeIndex([pd.to_datetime(timestamp)])
        if time_index.tz is None:
            time_index = time_index.tz_localize(location['timezone'])
    else:
        time_index = timestamp

    # Get solar position and clear sky data
    solar_pos = site.get_solarposition(time_index)
    try:
        clearsky = site.get_clearsky(time_index, model=model)
    except KeyError:
        clearsky = site.get_clearsky(time_index, model=model, linke_turbidity=linke_turbidity)

    # Calculate plane of array irradiance
    poa_irradiance = get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=clearsky['dni'],
        ghi=clearsky['ghi'],
        dhi=clearsky['dhi'],
        solar_zenith=solar_pos['apparent_zenith'],
        solar_azimuth=solar_pos['azimuth']
    )

    # Calculate total flux with non-negative constraint
    total_flux = (poa_irradiance['poa_direct'] + poa_irradiance['poa_diffuse']).clip(lower=0)

    print(f"Pvlib")
    print(f"Date: {timestamp}")
    print(f"Direct Radiation: {total_flux.iloc[0]:.2f} W/m²")

    return total_flux.iloc[0]


if __name__ == "__main__":
    # Configuration
    LOCATION = {
        'latitude': 40.7128,
        'longitude': -74.0060,
        'timezone': 'America/New_York'
    }

    start_time = time.time()
    # Single time calculation
    flux = calculate_solar_flux(
        location=LOCATION,
        timestamp=datetime(2025, 3, 26, 15, 30),
        tilt=0,
        azimuth=180
    )
    end_time = time.time()
    print(f"Solar flux: {flux} W/m²")
    print(f"Calculation time: {end_time - start_time:.4f} seconds")