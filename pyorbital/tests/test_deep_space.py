#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2012-2024 Pytroll Community

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

"""Test the deep space."""

from datetime import datetime, timezone

import numpy as np
import pytest

from pyorbital.deep_space import DeepSpace
from pyorbital.orbital import ECC_EPS, ECC_LIMIT_HIGH, SGDP4_DEEP_NORM, _Keplerians


class DummyParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@pytest.fixture
def default_orbit_elements():
    return DummyParams(
        eo=0.0001,
        xincl=np.radians(0.1),
        xnodp=1.0027,
        aodp=6.6,
        xmo=0.0,
        omegao=0.0,
        xnodeo=0.0,
        xmdot=0.0,
        omgdot=0.0,
        xnodot=0.0,
        xlcof=0.0,
        aycof=0.0,
        sinIO=np.sin(np.radians(0.1)),
        cosIO=np.cos(np.radians(0.1)),
        x3thm1=0.0,
        x1mth2=0.0,
        x7thm1=0.0,
        t_0=np.datetime64("2025-10-06T08:01:00"),
        sinXMO=0.0,
        eta=0.0,
        delmo=1.0,
        c1=0.0,
        c4=0.0,
        c5=0.0,
        d2=0.0,
        d3=0.0,
        d4=0.0,
        t2cof=0.0,
        t3cof=0.0,
        t4cof=0.0,
        t5cof=0.0,
        xmcof=0.0,
        xnodcf=0.0,
        mode=SGDP4_DEEP_NORM,
    )


@pytest.mark.parametrize(
    "ecc,incl,mmotion",
    [
        (0.0001, 0.1, 1.0027),
        (0.7, 63.4, 2.0),
        (0.0, 0.0, 1.0),
    ],
)
def test_radius_positive(default_orbit_elements, ecc, incl, mmotion):
    params = default_orbit_elements
    params.eo = ecc
    params.xincl = np.radians(incl)
    params.xnodp = mmotion
    params.aodp = 6.6

    kep = _Keplerians(params)
    utc_time = params.t_0.astype("O").replace(tzinfo=timezone.utc)
    result = kep.calculate(utc_time)
    assert result["radius"] > 0


def test_velocity_vector_shape(default_orbit_elements):
    kep = _Keplerians(default_orbit_elements)
    utc_time = default_orbit_elements.t_0.astype("O").replace(tzinfo=timezone.utc)
    result = kep.calculate(utc_time)
    velocity = result["velocity"]
    assert isinstance(velocity, np.ndarray)
    assert velocity.shape == (3,)
    assert np.linalg.norm(velocity) > 0


@pytest.mark.parametrize("angle", [0, 2 * np.pi, 4 * np.pi, -2 * np.pi])
def test_angle_normalization(default_orbit_elements, angle):
    params = default_orbit_elements
    params.xmo = angle
    params.omegao = angle
    params.xnodeo = angle
    params.xincl = angle

    ds = DeepSpace(params)
    ds._normalize_angles()

    assert 0 <= params.xmo < 2 * np.pi
    assert 0 <= params.omegao < 2 * np.pi
    assert 0 <= params.xnodeo < 2 * np.pi
    assert 0 <= params.xincl < 2 * np.pi


@pytest.mark.parametrize(
    "mmotion,expected_resonance,expected_sync",
    [
        (1.0027, True, True),
        (0.004, True, False),
        (0.01, False, False),
    ],
)
def test_resonance_flags(default_orbit_elements, mmotion, expected_resonance, expected_sync):
    params = default_orbit_elements
    params.xnodp = mmotion
    ds = DeepSpace(params)
    assert ds.resonance is expected_resonance
    assert ds.synchronous is expected_sync


def test_crash_detection(default_orbit_elements):
    params = default_orbit_elements
    params.aodp = 0.9  # below threshold
    kep = _Keplerians(params)
    utc_time = params.t_0.astype("O").replace(tzinfo=timezone.utc)
    with pytest.raises(Exception, match="Satellite crashed"):
        kep.calculate(utc_time)


@pytest.mark.parametrize("ecc", [-0.01, 0.999999, 1.1])
def test_eccentricity_clamping(default_orbit_elements, ecc):
    params = default_orbit_elements
    params.eo = ecc
    ds = DeepSpace(params)
    ds.update_periodic(0)
    assert ECC_EPS <= params.eo <= ECC_LIMIT_HIGH


def test_periodic_update_stability(default_orbit_elements):
    ds = DeepSpace(default_orbit_elements)
    ds.update_periodic(tsince=1440)  # 1 day
    assert ECC_EPS <= default_orbit_elements.eo <= ECC_LIMIT_HIGH
