#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2023 Pyorbital developers


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

"""Functionality to support deep space."""
import numpy as np

ECC_EPS = 1.0e-6  # Too low for computing further drops.
ECC_LIMIT_LOW = -1.0e-3
ECC_LIMIT_HIGH = 1.0 - ECC_EPS  # Too close to 1
ECC_ALL = 1.0e-4

EPS_COS = 1.5e-12

NR_EPS = 1.0e-12

CK2 = 5.413080e-4
CK4 = 0.62098875e-6
E6A = 1.0e-6
QOMS2T = 1.88027916e-9
S = 1.01222928
S0 = 78.0
XJ3 = -0.253881e-5
XKE = 0.743669161e-1
XKMPER = 6378.135
XMNPDA = 1440.0
# MFACTOR = 7.292115E-5
AE = 1.0
SECDAY = 8.6400e4

F = 1 / 298.257223563  # Earth flattening WGS-84
A = 6378.137  # WGS84 Equatorial radius


SGDP4_ZERO_ECC = 0
SGDP4_DEEP_NORM = 1
SGDP4_NEAR_SIMP = 2
SGDP4_NEAR_NORM = 3

KS = AE * (1.0 + S0 / XKMPER)
A3OVK2 = (-XJ3 / CK2) * AE**3


class DeepSpace:
    """Handles deep-space initialization and updates for SGDP4."""

    def __init__(self, params):
        """Initializes the DeepSpace model with orbital parameters."""
        self._params = params
        self._initialize_deep_space()

    def _initialize_deep_space(self):
        """Sets up resonance conditions and periodic perturbation coefficients."""
        self.zns = 1.19459e-5
        self.zes = 0.01675
        self.znl = 1.5835218e-4
        self.zel = 0.05490
        self.thdt = 4.37526908801129966e-3
        self.step2 = 259200.0

        # Initialize resonance terms to default values
        self.del1 = 0.0
        self.del2 = 0.0
        self.del3 = 0.0
        self.d2201 = 0.0
        self.d2211 = 0.0
        self.d3210 = 0.0
        self.d3222 = 0.0
        self.d4410 = 0.0
        self.d4422 = 0.0
        self.fasx2 = 0.0
        self.fasx4 = 0.0
        self.fasx6 = 0.0

        self.resonance = False
        self.synchronous = False

        mm = self._params.xnodp
        if 0.0034906585 < mm < 0.0052359877:
            self.resonance = True
        elif abs(mm - 1.0027) < 0.0003:
            self.resonance = True
            self.synchronous = True

        # Initialize resonance terms if synchronous
        if self.synchronous:
            self.fasx2 = 0.13130908
            self.fasx4 = 2.8843198
            self.fasx6 = 0.37448087
            self.del1 = 3.0 * mm**2 * self.zns * (self._params.aodp**2)
            self.del2 = 2.0 * self.del1 * self._params.aodp
            self.del3 = 3.0 * self.del1 * self._params.aodp**2 * self.zns
            self.d2201 = self.del1 * self.fasx2
            self.d2211 = self.del2 * self.fasx2
            self.d3210 = self.del2 * self.fasx4
            self.d3222 = self.del3 * self.fasx4
            self.d4410 = self.del2 * self.fasx6
            self.d4422 = self.del3 * self.fasx6

        # 🌙 Periodic coefficients for solar and lunar perturbations
        sinio = np.sin(self._params.xincl)
        cosio = np.cos(self._params.xincl)

        self._params.se = self.zes * 0.5 * sinio
        self._params.si = self.zes * -0.5 * cosio
        self._params.sl = self.zes * 0.5 * sinio
        self._params.sgh = self.zes * -0.5 * cosio
        self._params.sh = self.zes * -0.5 * sinio

        self._params.se2 = self.zel * 0.5 * sinio
        self._params.si2 = self.zel * -0.5 * cosio
        self._params.sl2 = self.zel * 0.5 * sinio
        self._params.sgh2 = self.zel * -0.5 * cosio
        self._params.sh2 = self.zel * -0.5 * sinio

    def update_secular(self, tsince):
        """Applies secular (long-term) orbital updates including resonance effects."""
        self._params.xmo += self._params.xmdot * tsince
        self._params.omegao += self._params.omgdot * tsince
        self._params.xnodeo += self._params.xnodot * tsince

        if self.resonance:
            xll = self._params.xmo + self._params.omegao + self._params.xnodeo
            xll += self.del1 * np.sin(self.fasx2 * tsince)
            xll += self.del2 * np.sin(self.fasx4 * tsince)
            xll += self.del3 * np.sin(self.fasx6 * tsince)
            self._params.xmo = xll - self._params.omegao - self._params.xnodeo

        self._normalize_angles()

    def update_periodic(self, tsince):
        """Applies periodic corrections due to solar and lunar gravitational perturbations."""
        day_fraction = (
            self._params.t_0.astype("datetime64[D]").astype(float) + tsince / 1440.0
        )

        # Solar perturbation angle
        zm_s = self.zns * day_fraction
        zf_s = zm_s + 2.0 * self.zes * np.sin(zm_s)
        sinzf_s = np.sin(zf_s)
        coszf_s = np.cos(zf_s)

        # Lunar perturbation angle
        zm_l = self.znl * day_fraction
        zf_l = zm_l + 2.0 * self.zel * np.sin(zm_l)
        sinzf_l = np.sin(zf_l)
        coszf_l = np.cos(zf_l)

        # Apply periodic corrections
        self._params.eo += self._params.se * sinzf_s + self._params.se2 * sinzf_l
        self._params.xincl += self._params.si * sinzf_s + self._params.si2 * sinzf_l
        self._params.omegao += self._params.sgh * sinzf_s + self._params.sgh2 * sinzf_l
        self._params.xnodeo += self._params.sh * sinzf_s + self._params.sh2 * sinzf_l

        # Clamp eccentricity and normalize angles
        self._params.eo = np.clip(self._params.eo, ECC_EPS, ECC_LIMIT_HIGH)
        self._normalize_angles()

    def _normalize_angles(self):
        """Normalizes angular orbital parameters to the [0, 2π] range."""
        self._params.xmo = np.mod(self._params.xmo, 2 * np.pi)
        self._params.omegao = np.mod(self._params.omegao, 2 * np.pi)
        self._params.xnodeo = np.mod(self._params.xnodeo, 2 * np.pi)
        self._params.xincl = np.mod(self._params.xincl, 2 * np.pi)
