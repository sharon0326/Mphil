import math
import datetime
import time

from pysolar import radiation, solar

def find_sun_direction_vector(gamma_s_deg, psi_s_deg):
    """
    Converts solar altitude and azimuth angles into a Cartesian direction vector.

    Parameters:
    gamma_s_deg (float): Solar altitude angle in degrees.
    psi_s_deg (float): Solar azimuth angle in degrees (measured from north, clockwise).

    Returns:
    tuple: (x, y, z) direction vector in the defined coordinate system.
    """
    gamma_s = math.radians(gamma_s_deg)
    psi_s = math.radians(psi_s_deg)

    theta_ZS = math.pi / 2 - gamma_s
    phi = math.pi / 2 - psi_s

    x = math.sin(theta_ZS) * math.cos(phi)
    y = math.sin(theta_ZS) * math.sin(phi)
    z = math.cos(theta_ZS)

    return (x, y, z)

# use pysolar to do simulation
def pysolar_simulation(
        location: dict,
        specific_datetime: datetime.datetime,
):
    """
    Calculate solar position and radiation using Pysolar

    Parameters:
    location (dict): Dictionary containing location parameters
        - latitude: float
        - longitude: float
        - timezone: str
        - altitude: float
    specific_datetime (datetime): Datetime for single calculation
    daily_time (datetime.time): Time to use for daily calculations
    yearly_year (int): Optional override for simulation year
    """
    # Single time calculation
    altitude = solar.get_altitude(location['latitude'], location['longitude'], specific_datetime)
    azimuth = solar.get_azimuth(location['latitude'], location['longitude'], specific_datetime)
    direct_rad = radiation.get_radiation_direct(specific_datetime, altitude)

    print(f"PySolar")
    print(f"Date: {specific_datetime}")
    print(f"Direct Radiation: {direct_rad:.2f} W/m²")

    return direct_rad

if __name__ == "__main__":
    location_config = {
        'latitude': 0.11362452473497865,  # New York City
        'longitude': 52.19844407134989,
        'timezone': 'America/New_York',
        'altitude': 10,
    }

    start_time = time.time()
    # Create datetime objects for inputs
    sample_datetime = datetime.datetime(2023, 3, 26, 15, 30, tzinfo=datetime.timezone.utc)
    noon_time = datetime.time(12, 0)

    # Run simulation
    flux =  pysolar_simulation(
        location=location_config,
        specific_datetime=sample_datetime,
    )
    end_time = time.time()
    print(f"Solar flux: {flux} W/m²")
    print(f"Calculation time: {end_time - start_time:.4f} seconds")