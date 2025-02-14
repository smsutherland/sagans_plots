#!/bin/env python

from pathlib import Path
from typing import Literal, Optional

import astropy.units as u
import numpy as np
import yaml
from astropy.cosmology import z_at_value
from astropy import cosmology
from astropy.table import QTable
from matplotlib.axes import Axes


def read_info_swift(path: Path, box_size):
    sfr = QTable.read(
        path,
        format="ascii.basic",
        names=[
            "step",
            "Time",
            "a",
            "z",
            "total M_star",
            "SFR (active)",
            "SFR*dt (active)",
            "SFR (total)",
        ],
        units=[
            1.0,
            9.778131e5 * u.Myr,
            1.0,
            1.0,
            1e10 * u.Msun,
            1.02269e-2 * u.Msun / u.yr,
            1e10 * u.Msun,
            1.02269e-2 * u.Msun / u.yr,
        ],
    )
    with open(path.parent / "params.yml", "r") as f:
        params = yaml.safe_load(f)
    h = params["Cosmology"]["h"]
    O_cdm = params["Cosmology"]["Omega_cdm"]
    O_l = params["Cosmology"]["Omega_lambda"]
    O_b = params["Cosmology"]["Omega_b"]
    O_m = O_b + O_cdm
    name = params["MetaData"]["run_name"]

    # TODO: read in the box size from somewhere. It's not in the parameter file
    volume = box_size**3
    sfr["SFR density"] = sfr["SFR (total)"] / volume
    sfr["SFR density"] = sfr["SFR density"].to(u.Msun / u.yr / u.Mpc**3)

    sfr.meta["O_cdm"] = O_cdm
    sfr.meta["O_l"] = O_l
    sfr.meta["O_b"] = O_b
    sfr.meta["O_m"] = O_m
    sfr.meta["name"] = name
    sfr.meta["h"] = h
    return sfr


def read_info_simba(path: Path, box_size):
    scale, sfr = np.loadtxt(path, usecols=(0, 2), unpack=True)
    sfr = sfr << u.Msun / u.yr
    h = 0.6711
    z = 1.0 / scale - 1.0
    volume = box_size**3
    sfr_density = sfr / volume

    tab = QTable()
    tab["a"] = scale
    tab["z"] = z
    tab["SFR"] = sfr
    tab["SFR density"] = sfr_density.to(u.Msun / u.yr / u.Mpc**3)
    tab.meta["h"] = h
    return tab


def read_info_astrid(path: Path, box_size):
    scale, sfr = np.loadtxt(path, usecols=(0, 2), unpack=True)
    sfr = sfr << u.Msun / u.yr
    h = 0.6711
    z = 1.0 / scale - 1.0
    volume = box_size**3
    sfr_density = sfr / volume

    tab = QTable()
    tab["a"] = scale
    tab["z"] = z
    tab["SFR"] = sfr
    tab["SFR density"] = sfr_density.to(u.Msun / u.yr / u.Mpc**3)
    tab.meta["h"] = h
    return tab


def madau_dickinson(z):
    pow1 = 2.7
    pow2 = 5.6
    scale1 = 0.015
    scale2 = 2.9
    z1 = 1 + z
    return (
        scale1
        * np.power(z1, pow1)
        / (1 + np.power(z1 / scale2, pow2))
        * u.Msun
        / u.yr
        / u.Mpc**3
    )


def smooth(data, z_min, z_max, n=10_000):
    z = data["z"]
    indices = np.argsort(z)
    z = z[indices]
    SFR = data["SFR density"][indices]
    bins = np.linspace(z_min, z_max, n)
    smoothed = np.interp(bins, z, SFR)
    return bins, smoothed


def plot_sfrd(
    ax: Axes,
    file: str,
    kind: Literal["SWIFT", "SIMBA", "ASTRID", "auto"] = "auto",
    xscale: Literal["redshift", "time"] = "redshift",
    time: Literal["forward", "lookback"] = "forward",
    reversed: bool = False,
    nbins: int = 10_000,
    cosmo: Optional[cosmology.Cosmology] = None,
    zrange=(0, 10),
    top_ticks=None,
    mk_twin=True,
    box_size: float = 25,
    **kwargs,
):
    box_size = box_size / 0.6711 * u.Mpc
    if mk_twin and cosmo is None:
        raise ValueError("mk_twin was set, but cosmo was None")
    if xscale == "time" and cosmo is None:
        raise ValueError(
            "xscale is 'time' and needs cosmo to perform that conversion, but cosmo was None"
        )
    path = Path(file)
    if kind == "auto":
        if path.name == "SFR.txt":
            kind = "SWIFT"
        elif path.name == "sfr.txt":
            kind = "SIMBA"
        else:
            raise ValueError(f"Could not identify simulation type for {file}")
    if kind == "SWIFT":
        sfrd = read_info_swift(path, box_size)
    elif kind == "SIMBA":
        sfrd = read_info_simba(path, box_size)
    elif kind == "ASTRID":
        sfrd = read_info_astrid(path, box_size)
    else:
        raise ValueError(f"Unknown kind: `{kind}`")

    if top_ticks is None:
        if xscale == "redshift":
            if time == "forward":
                top_ticks = [1, 2, 3, 5, 7, 13] * u.Gyr
            elif time == "lookback":
                top_ticks = [1, 7, 9, 11, 12, 13] * u.Gyr
        elif xscale == "time":
            top_ticks = [0, 1, 2, 4, 6]

    if xscale == "redshift":
        ax.set_xlim(zrange[0], zrange[1])
    elif xscale == "time":
        tmin = cosmo.age(zrange[1]).value
        tmax = cosmo.age(zrange[0]).value
        ax.set_xlim(tmin, tmax)

    ax.set_yscale("log")
    ax.set_ylabel("Star formation rate density [$M_\\odot/yr/cMpc$]")
    if xscale == "redshift":
        bins, smoothed = smooth(sfrd, zrange[0], zrange[1], nbins)
        ax.plot(bins, smoothed, **kwargs)
        if mk_twin:
            top_ax = add_time_axis(ax, cosmo, top_ticks, time)
            if reversed:
                top_ax.invert_xaxis()
        ax.set_xlabel("Redshift")
    elif xscale == "time":
        bins, smoothed = smooth(sfrd, zrange[0], zrange[1], nbins)
        if time == "forward":
            time_bins = cosmo.age(bins)
            ax.set_xlabel("Cosmic time [Gyr]")
        elif time == "lookback":
            time_bins = cosmo.lookback_time(bins)
            ax.set_xlabel("Lookback time [Gyr]")
        ax.plot(time_bins, smoothed, **kwargs)
        if mk_twin:
            top_ax = add_redshift_axis(ax, cosmo, top_ticks, time)
            if reversed:
                top_ax.invert_xaxis()

    if reversed:
        ax.invert_xaxis()
    if mk_twin:
        return top_ax


def add_time_axis(ax, cosmo, ages, time: Literal["forward", "lookback"] = "forward"):
    top_ax = ax.twiny()
    if time == "forward":
        top_ax.set_xlabel("Cosmic Time [Gyr]")
        f = cosmo.age
    elif time == "lookback":
        top_ax.set_xlabel("Lookback Time [Gyr]")
        f = cosmo.lookback_time
    ageticks = z_at_value(f, ages)
    top_ax.set_xticks(ageticks)
    top_ax.set_xticklabels([f"{a:g}" for a in ages.value])
    xlim = ax.get_xlim()
    top_ax.set_xlim(*xlim)
    return top_ax


def add_redshift_axis(ax, cosmo, zs, time: Literal["forward", "lookback"] = "forward"):
    top_ax = ax.twiny()
    if time == "forward":
        zticks = cosmo.age(zs).value
    elif time == "lookback":
        zticks = cosmo.lookback_time(zs).value
    top_ax.set_xticks(zticks)
    top_ax.set_xticklabels([f"{z:g}" for z in zs])
    xlim = ax.get_xlim()
    top_ax.set_xlim(*xlim)
    top_ax.set_xlabel("Redshift")
    return top_ax
