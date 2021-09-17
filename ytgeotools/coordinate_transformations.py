import numpy as np

from ytgeotools.typing import all_numbers


def sphere2cart(
    phi: all_numbers, theta: all_numbers, radius: all_numbers
) -> all_numbers:
    """
    seis_model.sphere2cart(phi,theta,radius)

    transformation from yt spherical coordinates to cartesian

    Parameters
    ----------
    phi : ndarray or scalar float/ing
        angle from north in radians (0 = north pole)
    theta : ndarray or scalar float/ing
        longitudinal angle in radians
    radius : ndarray or scalar float/ing
        radius in any units

    all arrays must be the same size (or 2 of 3 can be scalars)

    Returns
    -------
    (x,y,z) : tuple of cartesian x,y,z in same units as radius
    """
    x = radius * np.sin(phi) * np.sin(theta)
    y = radius * np.sin(phi) * np.cos(theta)
    z = radius * np.cos(phi)
    return (x, y, z)


def cart2sphere(
    x: all_numbers, y: all_numbers, z: all_numbers, geo: bool = True
) -> all_numbers:
    """
    seis_model.cart2sphere(x,y,z,geo=True)

    transformation from cartesian to spherical coordinates

    Parameters
    ----------
    x, y, z   cartesian coordinate arrays
    geo       boolean, if True then latitude is 0 at equator, otherwise 0 at
              the north pole.

    all arrays must be the same size (or 2 of 3 can be scalars)

    Returns
    -------
    (R,lat,lon) : tuple of cartesian radius, lat and lon (lat,lon in degrees)

    """

    xy = x ** 2 + y ** 2
    R = np.sqrt(xy + z ** 2)
    lat = np.arctan2(np.sqrt(xy), z) * 180.0 / np.pi
    lon = np.arctan2(y, x) * 180.0 / np.pi
    if geo:
        lat = lat - 90.0  # equator is at 0, +90 is N pole

    return (R, lat, lon)


def geosphere2cart(
    lat: all_numbers, lon: all_numbers, radius: all_numbers
) -> all_numbers:
    """
    transformation from latitude, longitude to to cartesian

    Parameters
    ----------
    lat : ndarray or scalar float/int
        latitude, -180 to 180 or 0 to 360
    lon : ndarray or scalar float/int
        longitude, -90 to 90
    radius : ndarray or scalar float/int
        radius in any units

    all arrays must be the same size (or 2 of 3 can be scalars)

    Returns
    -------
    (x,y,z) : tuple of cartesian x,y,z in same units as radius
    """

    phi = (90.0 - lat) * np.pi / 180.0  # lat is now deg from North

    if isinstance(lon, np.ndarray):
        lon[lon < 0.0] = lon[lon < 0.0] + 360.0
    elif lon < 0.0:
        lon = lon + 360.0
    theta = lon * np.pi / 180.0

    return sphere2cart(phi, theta, radius)
