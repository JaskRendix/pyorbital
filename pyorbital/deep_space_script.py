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

"""Functionality to support deep space comparison between SGDP4 and Skyfield.

python deep_space_script.py --compare all --plot
python deep_space_script.py --compare radius
python deep_space_script.py --compare velocity --plot"""

import argparse
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from skyfield.api import EarthSatellite, load, utc

from pyorbital.orbital import _SGDP4

tle_lines = [
    "GOES 16",
    "1 41866U 16071A   25279.33402778  .00000045  00000-0  00000+0 0  9990",
    "2 41866   0.0890  75.2000 0001000  90.0000 270.0000  1.00270000    00",
]


class OrbitElements:
    def __init__(self):
        self.excentricity = 0.0001000
        self.inclination = np.radians(0.0890)
        self.original_mean_motion = 1.00270000 * 2 * np.pi / 1440.0  # rad/min
        self.mean_motion = self.original_mean_motion
        self.bstar = 0.00000045
        self.arg_perigee = np.radians(90.0000)
        self.mean_anomaly = np.radians(270.0000)
        self.right_ascension = np.radians(75.2000)
        self.epoch = np.datetime64("2025-10-06T08:01:00")  # from TLE epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare SGDP4 vs Skyfield propagation."
    )
    parser.add_argument(
        "--hours", type=int, default=72, help="Total hours to propagate"
    )
    parser.add_argument("--step", type=int, default=6, help="Step size in hours")
    parser.add_argument(
        "--compare",
        choices=["radius", "velocity", "elements", "all"],
        default="all",
        help="Which comparison to run",
    )
    parser.add_argument("--plot", action="store_true", help="Plot results")
    return parser.parse_args()


def main():
    args = parse_args()

    ts = load.timescale()
    sat = EarthSatellite(tle_lines[1], tle_lines[2], tle_lines[0], ts)

    orbit_elements = OrbitElements()
    model = _SGDP4(orbit_elements)

    times = []
    sgdp4_radii = []
    skyfield_radii = []
    sgdp4_speeds = []
    skyfield_speeds = []
    inclinations = []
    raan_values = []
    arg_perigees = []

    print("Time (UTC)       | SGDP4 Radius (km) | Skyfield Radius (km) | Δ Radius (km)")
    print("-" * 70)

    for hours in range(0, args.hours + 1, args.step):
        utc_time = orbit_elements.epoch + np.timedelta64(hours, "h")
        sgdp4_result = model.propagate(utc_time)
        sgdp4_radius = sgdp4_result["radius"]
        sgdp4_velocity = sgdp4_result["velocity"]

        skyfield_time = ts.utc(utc_time.astype(datetime).replace(tzinfo=utc))
        skyfield_position = sat.at(skyfield_time).position.km
        skyfield_velocity = sat.at(skyfield_time).velocity.km_per_s
        skyfield_radius = np.linalg.norm(skyfield_position)

        times.append(utc_time.astype(datetime))
        sgdp4_radii.append(sgdp4_radius)
        skyfield_radii.append(skyfield_radius)
        sgdp4_speeds.append(np.linalg.norm(sgdp4_velocity))
        skyfield_speeds.append(np.linalg.norm(skyfield_velocity))
        inclinations.append(np.degrees(model._params.xincl))
        raan_values.append(np.degrees(model._params.xnodeo))
        arg_perigees.append(np.degrees(model._params.omegao))

        if args.compare in ["radius", "all"]:
            delta = abs(sgdp4_radius - skyfield_radius)
            print(
                f"{utc_time} | {sgdp4_radius:17.3f} | {skyfield_radius:18.3f} | {delta:12.3f}"
            )

    if args.compare in ["velocity", "all"]:
        print("\nVelocity Comparison (km/s):")
        for t, v1, v2 in zip(times, sgdp4_speeds, skyfield_speeds):
            print(f"{t} | SGDP4: {v1:.5f} | Skyfield: {v2:.5f} | Δ: {abs(v1 - v2):.5f}")

    if args.compare in ["elements", "all"]:
        print("\nOrbital Elements Comparison (degrees):")
        for t, inc, raan, argp in zip(times, inclinations, raan_values, arg_perigees):
            print(f"{t} | Incl: {inc:.4f} | RAAN: {raan:.4f} | ArgPer: {argp:.4f}")

    if args.plot:
        plt.figure(figsize=(10, 6))
        if args.compare in ["radius", "all"]:
            plt.plot(times, sgdp4_radii, label="SGDP4 Radius", marker="o")
            plt.plot(times, skyfield_radii, label="Skyfield Radius", marker="x")
            plt.ylabel("Radius (km)")
            plt.title("Satellite Radius Over Time")
        elif args.compare == "velocity":
            plt.plot(times, sgdp4_speeds, label="SGDP4 Speed", marker="o")
            plt.plot(times, skyfield_speeds, label="Skyfield Speed", marker="x")
            plt.ylabel("Speed (km/s)")
            plt.title("Satellite Speed Over Time")
        elif args.compare == "elements":
            plt.plot(times, inclinations, label="Inclination")
            plt.plot(times, raan_values, label="RAAN")
            plt.plot(times, arg_perigees, label="Arg of Perigee")
            plt.ylabel("Degrees")
            plt.title("Orbital Elements Over Time")

        plt.xlabel("UTC Time")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("orbit_comparison.png")


if __name__ == "__main__":
    main()
