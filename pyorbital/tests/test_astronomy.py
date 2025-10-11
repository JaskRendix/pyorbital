#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2013, 2014, 2022, 2024 Pytroll Community

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Unit testing the Astronomy methods and functions."""


import datetime as dt

import dask.array as da
import numpy as np
import numpy.typing as npt
import pytest

import pyorbital.astronomy as astr

try:
    from xarray import DataArray
except ImportError:
    DataArray = None


def _create_dask_array(input_list: list, dtype: npt.DTypeLike) -> da.Array:
    """Create a dummy dask array for testing."""
    np_arr = np.array(input_list, dtype=dtype)
    return da.from_array(np_arr)


def _create_xarray_numpy(input_list: list, dtype: npt.DTypeLike) -> DataArray:
    """Create a dummy xarray DataArray for testing."""
    np_arr = np.array(input_list, dtype=dtype)
    return DataArray(np_arr)


def _create_xarray_dask(input_list: list, dtype: npt.DTypeLike) -> DataArray:
    """Create a dummy daskified xarray DataArray for testing."""
    dask_arr = _create_dask_array(input_list, dtype)
    return DataArray(dask_arr)


class TestAstronomy:
    """Testing the Astronomy class."""

    @pytest.mark.parametrize(
        ("dat", "exp_jdays", "exp_j2000"),
        [
            (dt.datetime(2000, 1, 1, 12, 0), 2451545.0, 0),
            (dt.datetime(2009, 10, 8, 14, 30), 2455113.1041666665, 3568.1041666666665),
        ],
    )
    def test_jdays(self, dat, exp_jdays, exp_j2000):
        """Test julian day functions."""
        assert astr.jdays(dat) == exp_jdays
        assert astr.jdays2000(dat) == exp_j2000

    @pytest.mark.parametrize(
        ("lon", "lat", "exp_theta"),
        [
            # Norrkoping
            (16.1833, 58.6167, 60.371433482557833),
            (0.0, 0.0, 1.8751916863323426),
        ],
    )
    @pytest.mark.parametrize(
        ("dtype", "array_construct"),
        [
            (None, None),
            (np.float32, np.array),
            (np.float64, np.array),
            (np.float32, _create_dask_array),
            (np.float64, _create_dask_array),
            (np.float32, _create_xarray_numpy),
            (np.float64, _create_xarray_numpy),
            (np.float32, _create_xarray_dask),
            (np.float64, _create_xarray_dask),
        ],
    )
    def test_sunangles(self, lon, lat, exp_theta, dtype, array_construct):
        """Test the sun-angle calculations."""
        if array_construct is None and dtype is not None:
            pytest.skip(reason="Xarray dependency unavailable")

        time_slot = dt.datetime(2011, 9, 23, 12, 0)
        abs_tolerance = 1e-8
        if dtype is not None:
            lon = array_construct([lon], dtype=dtype)
            lat = array_construct([lat], dtype=dtype)
            if np.dtype(dtype).itemsize < 8:
                abs_tolerance = 1e-4

        sun_theta = astr.sun_zenith_angle(time_slot, lon, lat)
        if dtype is None:
            assert sun_theta == pytest.approx(exp_theta, abs=abs_tolerance)
            assert isinstance(sun_theta, float)
        else:
            assert sun_theta.dtype == dtype
            np.testing.assert_allclose(sun_theta, exp_theta, atol=abs_tolerance)
            assert isinstance(sun_theta, type(lon))

    def test_sun_earth_distance_correction(self):
        """Test the sun-earth distance correction."""
        utc_time = dt.datetime(2022, 6, 15, 12, 0, 0)
        corr = astr.sun_earth_distance_correction(utc_time)
        corr_exp = 1.0156952156742332
        assert corr == pytest.approx(corr_exp, abs=1e-8)


def test_gmst():
    """Test Greenwich Mean Sidereal Time."""
    utc_time = dt.datetime(2000, 1, 1, 12, 0)
    gmst_val = astr.gmst(utc_time)
    assert isinstance(gmst_val, float)
    assert 0 <= gmst_val <= 2 * np.pi


def test_lmst():
    """Test Local Mean Sidereal Time."""
    utc_time = dt.datetime(2000, 1, 1, 12, 0)
    longitude = np.deg2rad(10.0)
    lmst_val = astr._lmst(utc_time, longitude)
    assert isinstance(lmst_val, float)
    assert 0 <= lmst_val <= 2 * np.pi


def test_sun_ecliptic_longitude():
    """Test sun ecliptic longitude."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)
    eclon = astr.sun_ecliptic_longitude(utc_time)
    assert isinstance(eclon, float)
    # Normalize to [0, 2π] for assertion
    eclon_norm = eclon % (2 * np.pi)
    assert 0 <= eclon_norm <= 2 * np.pi


def test_sun_ra_dec():
    """Test sun right ascension and declination."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)
    ra, dec = astr.sun_ra_dec(utc_time)
    assert isinstance(ra, float)
    assert isinstance(dec, float)
    # Right ascension can be negative; normalize for assertion
    ra_norm = ra % (2 * np.pi)
    assert 0 <= ra_norm <= 2 * np.pi
    assert -np.pi / 2 <= dec <= np.pi / 2


def test_local_hour_angle():
    """Test local hour angle."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)
    lon = np.deg2rad(10.0)
    ra, _ = astr.sun_ra_dec(utc_time)
    ha = astr._local_hour_angle(utc_time, lon, ra)
    assert isinstance(ha, float)
    # Hour angle can exceed 2π; just check it's a valid float
    assert np.isfinite(ha)


def test_get_alt_az():
    """Test altitude and azimuth calculation."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)
    lon = 10.0
    lat = 50.0
    alt, az = astr.get_alt_az(utc_time, lon, lat)
    assert isinstance(alt, float)
    assert isinstance(az, float)
    assert -np.pi / 2 <= alt <= np.pi / 2
    assert 0.0 <= az < 2 * np.pi


def test_azimuth_normalization():
    """Verify that azimuth is normalized to the range [0, 2π) radians."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)
    lon = 10.0
    lat = 50.0
    _, az = astr.get_alt_az(utc_time, lon, lat)
    assert 0.0 <= az < 2 * np.pi, f"Azimuth out of range: {az}"


def test_cos_zen():
    """Test cosine of zenith angle."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)
    lon = 10.0
    lat = 50.0
    csza = astr.cos_zen(utc_time, lon, lat)
    assert isinstance(csza, float)
    assert -1.0 <= csza <= 1.0


def test_observer_position():
    """Test observer ECI position and velocity."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)
    lon = 10.0
    lat = 50.0
    alt = 0.2  # km
    pos, vel = astr.observer_position(utc_time, lon, lat, alt)
    for val in pos + vel:
        assert isinstance(val, float)


def test_observer_position_scalar_and_array():
    """Test observer_position with scalar and array inputs."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)
    lon = 10.0
    lat = 50.0
    alt = 0.2

    # Scalar test
    pos, vel = astr.observer_position(utc_time, lon, lat, alt)
    for val in pos + vel:
        assert isinstance(val, float)

    # Array test
    lon_arr = np.array([10.0, 20.0], dtype=np.float32)
    lat_arr = np.array([50.0, 60.0], dtype=np.float32)
    alt_arr = np.array([0.2, 0.3], dtype=np.float32)
    pos_arr, vel_arr = astr.observer_position(utc_time, lon_arr, lat_arr, alt_arr)
    for arr in pos_arr + vel_arr:
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.float32


@pytest.mark.parametrize(
    ("lon", "lat", "alt", "dtype", "constructor"),
    [
        # Scalars
        (10.0, 50.0, 0.2, None, None),
        # NumPy arrays
        ([10.0], [50.0], [0.2], np.float32, np.array),
        ([10.0], [50.0], [0.2], np.float64, np.array),
        # Dask arrays
        ([10.0], [50.0], [0.2], np.float32, _create_dask_array),
        ([10.0], [50.0], [0.2], np.float64, _create_dask_array),
        # xarray DataArrays
        ([10.0], [50.0], [0.2], np.float32, _create_xarray_numpy),
        ([10.0], [50.0], [0.2], np.float64, _create_xarray_numpy),
        ([10.0], [50.0], [0.2], np.float32, _create_xarray_dask),
        ([10.0], [50.0], [0.2], np.float64, _create_xarray_dask),
    ],
)
def test_observer_position_parametrized(lon, lat, alt, dtype, constructor):
    """Test observer_position with various input types and dtypes."""
    utc_time = dt.datetime(2022, 3, 20, 12, 0)

    if constructor is None:
        pos, vel = astr.observer_position(utc_time, lon, lat, alt)
        for val in pos + vel:
            assert isinstance(val, float)
    else:
        lon_arr = constructor(lon, dtype)
        lat_arr = constructor(lat, dtype)
        alt_arr = constructor(alt, dtype)
        pos, vel = astr.observer_position(utc_time, lon_arr, lat_arr, alt_arr)
        for arr in pos + vel:
            assert isinstance(arr, type(lon_arr))
            assert arr.dtype == dtype


@pytest.mark.parametrize(
    ("utc_time", "lon", "lat", "expected_range"),
    [
        (np.datetime64("2023-06-21T12:00"), 0.0, 0.0, (1000, 1400)),  # Equator at noon
        (np.datetime64("2023-06-21T00:00"), 0.0, 0.0, (0, 10)),  # Nighttime
        (np.datetime64("2023-12-21T12:00"), 10.0, 50.0, (200, 1000)),  # Mid-latitude winter
        (np.datetime64("2023-12-21T00:00"), 10.0, 50.0, (0, 10)),  # Nighttime in winter
    ],
)
def test_estimate_solar_irradiance(utc_time, lon, lat, expected_range):
    irradiance = astr.estimate_solar_irradiance(utc_time, lon, lat)
    assert isinstance(irradiance, float)
    assert expected_range[0] <= irradiance <= expected_range[1]
