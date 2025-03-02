import datetime
from pysolar import radiation
from pysolar.solar import get_altitude, get_azimuth
from solar_radiance_pysolar import find_sun_direction_vector

def calculate_direct_radiation(
        location: dict,
        timestamp: datetime.datetime,
        use_utc: bool = True
) -> float:
    """
    Calculate direct solar radiation (W/m²) using Pysolar library

    Parameters:
    location (dict): Location parameters containing:
        - latitude: float
        - longitude: float
        - timezone: str
    timestamp (datetime): Timestamp for calculation
    use_utc (bool): Convert timestamp to UTC (default: True)

    Returns:
    float: Direct solar radiation (W/m²)
    """

    # Validate location parameters
    required_keys = ['latitude', 'longitude', 'timezone']
    if not all(key in location for key in required_keys):
        missing = [key for key in required_keys if key not in location]
        raise ValueError(f"Missing location parameters: {missing}")

    # Convert to UTC
    if use_utc:
        if timestamp.tzinfo is None:
            localized_time = timestamp.replace(tzinfo=datetime.timezone.utc)
        else:
            localized_time = timestamp.astimezone(datetime.timezone.utc)
    else:
        localized_time = timestamp

    # Solar position
    try:
        altitude = get_altitude(
            location['latitude'],
            location['longitude'],
            localized_time
        )

    except Exception as e:
        raise RuntimeError(f"Solar position calculation failed: {str(e)}")

    direct_flux = radiation.get_radiation_direct(localized_time, altitude)

    print(f"Pysolar")
    print(f"Date: {timestamp}")
    print(f"Direct Radiation: {max(direct_flux, 0.0):.2f} W/m²")

    return max(direct_flux, 0.0)

def get_sun_position(
        location: dict,
        timestamp: datetime.datetime
) -> tuple:
    """
    Get sun position (altitude and azimuth) using Pysolar

    Parameters:
    location (dict): Location parameters (same as calculate_direct_radiation)
    timestamp (datetime): Calculation timestamp

    Returns:
    tuple: (altitude_deg, azimuth_deg)
    """
    utc_time = timestamp.astimezone(datetime.timezone.utc)
    altitude = get_altitude(location['latitude'], location['longitude'], utc_time)
    azimuth = get_azimuth(location['latitude'], location['longitude'], utc_time)
    return (altitude, azimuth)

if __name__ == "__main__":
    LOCATION = {
        'latitude': 40.7128,
        'longitude': -74.0060,
        'timezone': 'America/New_York'
    }

    # Single time calculation
    sample_time = datetime.datetime(2025, 3, 26, 15, 30, tzinfo=datetime.timezone.utc)
    direct_flux = calculate_direct_radiation(LOCATION, sample_time)