#!/bin/env python
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Container, List, Literal, Tuple, Union

import camels_library as cl
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from . import sfrd

StrPath = Union[str, bytes, os.PathLike]

nbins = 20


def put_number(ax: Axes, num: str, loc: Literal["lower"] | Literal["upper"]):
    params = {
        "x": 0.05,
        "y": None,
        "s": num,
        "transform": ax.transAxes,
        "fontsize": 20,
        "fontstyle": "italic",
        "fontfamily": "serif",
        "va": None,
        "ha": "left",
    }
    if loc == "lower":
        params["y"] = 0.1
        params["va"] = "top"
    elif loc == "upper":
        params["y"] = 0.9
        params["va"] = "bottom"
    ax.text(**params)
    return ax


@lru_cache
def pK_cache(snapshot: StrPath, grid, MAS, threads, ptype):
    return cl.compute_Pk(str(snapshot), grid, MAS, threads, ptype)


def plot_matter_power_spectrum(ax: Axes, snap: StrPath, **kwargs):
    power = pK_cache(snap, 512, "CIC", 40, range(6))
    assert not isinstance(power, int), f"Error with `{snap}`"

    ax.plot(power[0], power[1], **kwargs)
    return ax


def plot_gas_power_spectrum(ax: Axes, snap: StrPath, **kwargs):
    power = pK_cache(snap, 512, "CIC", 40, (0,))
    assert not isinstance(power, int), f"Error with `{snap}"

    ax.plot(power[0], power[1], **kwargs)
    return ax


def plot_halo_mass_function(ax: Axes, fof: StrPath, snap: StrPath, **kwargs):
    hmf = cl.halo_mass_function(
        1e10,
        1e14,
        20,
        fof,
        snap,
    )
    assert hmf is not None, f"Error with `{snap}`"
    ax.plot(hmf[1], hmf[2], **kwargs)
    return ax


def plot_SFRD(ax: Axes, sfr, **kwargs):
    bins, smoothed = sfrd.smooth(sfr, 0, 10)
    ax.plot(bins, smoothed, rasterized=True, **kwargs)
    return ax


def plot_mass_function(
    ax: Axes, fof: StrPath, ptypes: List[int], lower=1e9, upper=5e11, **kwargs
):
    f = h5py.File(fof)
    size = f["Header"].attrs["BoxSize"] / 1e3  # Mpc / h
    masses = np.sum(f["Subhalo/SubhaloMassType"][:, ptypes], axis=1) * 1e10  # Msun / h

    bins = np.geomspace(lower, upper, nbins + 1)  # Msun / h
    hist = np.histogram(masses, bins=bins)[0]
    widths = bins[1:] - bins[:-1]
    hist = hist / widths / size**3
    xs = (bins[1:] + bins[:-1]) / 2
    ax.plot(xs, hist, **kwargs)


def plot_baryon_fraction(ax: Axes, fof: StrPath, low_thresh=-1, **kwargs):
    RMmin = 1e10
    RMmax = 1e14

    with h5py.File(fof) as f:
        Om: float = f["Header"].attrs["Omega0"]
        halo_mass = f["Group/GroupMass"][:] * 1e10
        halo_mass_type = f["Group/GroupMassType"][:] * 1e10
        halo_part_type = f["Group/GroupLenType"][:]  # number of particles in each halo

    # take only halos with more than 50 CDM particles
    indexes = np.where(halo_part_type[:, 1] > 50)[0]
    halo_mass = halo_mass[indexes]
    halo_mass_type = halo_mass_type[indexes]

    # define the bins with the number of CDM particles and the intervals mean
    RM_bins = np.logspace(np.log10(RMmin), np.log10(RMmax), nbins + 1)
    RM_mean = 10 ** (0.5 * (np.log10(RM_bins[1:]) + np.log10(RM_bins[:-1])))

    # compute baryon fraction in units of cosmic fraction
    fraction = (
        (halo_mass_type[:, 0] + halo_mass_type[:, 4] + halo_mass_type[:, 5]) / halo_mass
    ) / (0.049 / Om)

    # take bins in halo mass / Omega_m. Compute average baryon fraction
    mean_fraction = np.histogram(halo_mass / Om, RM_bins, weights=fraction)[0]
    Number = np.histogram(halo_mass / Om, RM_bins)[0]
    Number[np.where(Number == 0)] = 1.0
    mean_fraction = mean_fraction / Number

    ax.plot(RM_mean, mean_fraction, **kwargs)
    if low_thresh > -1:
        low_bins = np.nonzero(Number <= low_thresh)[0]
        low = RM_bins[low_bins]
        high = RM_bins[low_bins + 1]

        masks = (low < halo_mass[:, None] / Om) & (halo_mass[:, None] / Om < high)
        mask = np.any(masks, axis=1)
        ax.scatter(halo_mass[mask] / Om, fraction[mask], **kwargs)

    return ax


def plot_halo_temp(ax: Axes, snap: StrPath, **kwargs) -> Axes:
    f = h5py.File(snap)
    if "Temperatures" in f["PartType0"]:
        T = f["PartType0/Temperatures"][:]
    else:
        ne = f["PartType0/ElectronAbundance"][:]
        energy = f["PartType0/InternalEnergy"][:]  # (km/s)^2

        yhelium = 0.0789
        T = energy * (1.0 + 4.0 * yhelium) / (1.0 + yhelium + ne) * 1e10 * (2.0 / 3.0)
        BOLTZMANN = 1.38065e-16  # erg/K - NIST 2010
        PROTONMASS = 1.67262178e-24  # gram  - NIST 2010
        T *= PROTONMASS / BOLTZMANN
    v = f["PartType0/SmoothingLength"][:] ** 3 * 4 / 3 * np.pi  # kpc^3 / h^3
    m = f["PartType0/Masses"][:] * 1e10  # Msun / h
    # p = m / v  # Msun h^2 / kpc^3
    bins = np.geomspace(1e2, 5e7, nbins + 1)
    hist, edges = np.histogram(T, bins, weights=m, density=True)
    xs = (edges[1:] + edges[:-1]) / 2
    ax.plot(xs, hist, **kwargs)
    return ax


def plot_galaxy_radius(ax: Axes, fof: StrPath, **kwargs):
    with h5py.File(fof) as f:
        radii = f["Subhalo/SubhaloHalfmassRad"][:]  # kpc / h
        mass = f["Subhalo/SubhaloMassType"][:, 4] * 1e10  # Msun / h

    bins = np.geomspace(1e9, 5e11, nbins + 1)  # Msun / h
    means, edges, _ = scipy.stats.binned_statistic(mass, radii, "mean", bins)
    xs = (edges[1:] + edges[:-1]) / 2
    ax.plot(xs, means, **kwargs)


def plot_BH_mass(ax: Axes, fof: StrPath, **kwargs):
    with h5py.File(fof) as f:
        bh_mass = f["Subhalo/SubhaloMassType"][:, 5] * 1e10  # Msun / h
        stellar_mass = f["Subhalo/SubhaloMassType"][:, 4] * 1e10  # Msun / h

    bins = np.geomspace(1e9, 5e11, nbins + 1)  # Msun / h
    means, edges, _ = scipy.stats.binned_statistic(stellar_mass, bh_mass, "mean", bins)
    xs = (edges[1:] + edges[:-1]) / 2
    ax.plot(xs, means, **kwargs)


def plot_v_circ(ax: Axes, fof: str, **kwargs):
    with h5py.File(fof) as f:
        velocity = f["Subhalo/SubhaloVmax"][:]  # km / s
        stellar_mass = f["Subhalo/SubhaloMassType"][:, 4] * 1e10  # Msun / h

    bins = np.geomspace(1e9, 5e11, nbins + 1)  # Msun / h
    means, edges, _ = scipy.stats.binned_statistic(stellar_mass, velocity, "mean", bins)
    xs = (edges[1:] + edges[:-1]) / 2
    ax.plot(xs, means, **kwargs)


def plot_gal_sfr(ax: Axes, fof: StrPath, **kwargs):
    with h5py.File(fof) as f:
        sfr = f["Subhalo/SubhaloSFR"][:]  # Msun / yr
        stellar_mass = f["Subhalo/SubhaloMassType"][:, 4] * 1e10  # Msun / h

    mask = sfr > 0
    sfr = sfr[mask]
    stellar_mass = stellar_mass[mask]

    bins = np.geomspace(1e9, 5e11, nbins + 1)  # Msun / h
    means, edges, _ = scipy.stats.binned_statistic(stellar_mass, sfr, "mean", bins)
    xs = (edges[1:] + edges[:-1]) / 2
    ax.plot(xs, means, **kwargs)


def sfr_snap(snap: StrPath, fix_unit: bool = False):
    with h5py.File(snap) as f:
        sfr = f["PartType0/StarFormationRate"][:]  # Msun / yr
        box_size = f["Header"].attrs["BoxSize"]  # kpc / h
        h = f["Header"].attrs["HubbleParam"]
        z = f["Header"].attrs["Redshift"]
        if isinstance(box_size, np.ndarray):
            box_size = box_size[0]
        if isinstance(h, np.ndarray):
            h = h[0]
        if isinstance(z, np.ndarray):
            z = z[0]
        if fix_unit:
            sfr *= 10.227

    tot_sfr = np.sum(sfr)
    sfr_density = tot_sfr / (box_size / h / 1000) ** 3  # Msun / yr / Mpc
    return sfr_density, z


def power_ratio(ax: Axes, snap: StrPath, **kwargs) -> Axes:
    p_hydro = pK_cache(snap, 512, "CIC", 40, range(6))
    assert not isinstance(p_hydro, int), f"Error with {snap}"
    # FIXME:
    print("Warning: needs to find matching DM run")
    p_nbody = pK_cache(
        "/mnt/ceph/users/camels/Sims/SIMBA_DM/1P/1P_p1_0/snapshot_090.hdf5",
        512,
        "CIC",
        40,
        range(6),
    )
    assert not isinstance(p_nbody, int), f"Error with {snap}"

    ratio = p_hydro[1] / p_nbody[1]

    left, right = ax.get_xlim()
    filter = (p_hydro[0] > left) & (p_hydro[0] < right)

    ax.plot(p_hydro[0][filter], ratio[filter], **kwargs)
    return ax


def mk_plot(
    sims: List[Tuple[StrPath, StrPath, StrPath, dict[str, Any]]],
    panels: Container[int] = range(1, 13),
    every_legend: bool = False,
    numbers: bool = True,
):
    fig = plt.figure(figsize=(20, 20))

    # Matter Power Spectrum
    if 1 in panels:
        ax = fig.add_subplot(
            4,
            3,
            1,
            xscale="log",
            yscale="log",
            xlim=(0.3, 35),
            ylim=(1e-1, 1e3),
            xlabel=r"$k \left[h \text{Mpc}^{-1}\right]$",
            ylabel=r"$P_m(k)\left[h^{-3}\text{Mpc}^3\right]$",
        )
        for snap, _, _, kwargs in sims:
            plot_matter_power_spectrum(ax, snap, **kwargs)
        if numbers:
            put_number(ax, "I", loc="lower")

    # Gas Power Spectrum
    if 2 in panels:
        ax = fig.add_subplot(
            4,
            3,
            2,
            xscale="log",
            yscale="log",
            xlim=(0.3, 35),
            ylim=(1e-2, 1e3),
            xlabel=r"$k \left[h \text{Mpc}^{-1}\right]$",
            ylabel=r"$P_g(k)\left[h^{-3}\text{Mpc}^3\right]$",
        )
        for snap, _, _, kwargs in sims:
            plot_gas_power_spectrum(ax, snap, **kwargs)
        if numbers:
            put_number(ax, "II", loc="lower")

    # Matter Power Spectrum Hydro/Nbody Ratio
    if 3 in panels:
        ax = fig.add_subplot(
            4,
            3,
            3,
            xlim=(0.3, 35),
            # ylim=(0.5, 1.05),
            xscale="log",
            xlabel=r"$k \left[h \text{Mpc}^{-1}\right]$",
            ylabel="$P_{hydro}(k)/P_{Nbody}(k)$",
        )
        for snap, _, _, kwargs in sims:
            power_ratio(ax, snap, **kwargs)
        if numbers:
            put_number(ax, "III", loc="lower")

    # Halo Mass Function
    if 4 in panels:
        ax = fig.add_subplot(
            4,
            3,
            4,
            yscale="log",
            xscale="log",
            xlabel=r"$M_\text{halo} / \Omega_m \left[h^{-1} M_\odot\right]$",
            ylabel=r"HMF $[h^4 \text{Mpc}^{-3} M_\odot^{-1}]$",
        )
        for snap, fof, _, kwargs in sims:
            plot_halo_mass_function(ax, fof, snap, **kwargs)
        if numbers:
            put_number(ax, "IV", loc="lower")

    # Star Formation Rate Density
    if 5 in panels:
        ax = fig.add_subplot(
            4,
            3,
            5,
            xlim=(0, 7),
            ylim=(5e-4, 2e-1),
            yscale="log",
            xlabel="redshift (z)",
            ylabel=r"SFRD [$M_\odot yr^{-1} Mpc^{-3}$]",
        )
        for _, _, base, kwargs in sims:
            if "SWIMBA" in base:
                sfrd_data = sfrd.read_info_swift(Path(base) / "SFR.txt")
            elif "SIMBA" in base:
                sfrd_data = sfrd.read_info_simba(Path(base) / "extra_files/sfr.txt")
            else:
                continue
            plot_SFRD(ax, sfrd_data, **kwargs)
        if numbers:
            put_number(ax, "V", loc="lower")

    # Stellar Mass Function
    if 6 in panels:
        ax = fig.add_subplot(
            4,
            3,
            6,
            xscale="log",
            yscale="log",
            xlabel=r"$M_*$ [$h^{-1} M_\odot$]",
            ylabel=r"SMF [$h^4 Mpc^{-3} M_\odot^{-1}$]",
        )
        for _, fof, _, kwargs in sims:
            plot_mass_function(ax, fof, [4], **kwargs)
        if numbers:
            put_number(ax, "VI", loc="lower")

    # Baryon Fraction
    if 7 in panels:
        ax = fig.add_subplot(
            4,
            3,
            7,
            xscale="log",
            xlabel=r"$M_\text{halo} / \Omega_m \left[h^{-1} M_\odot\right]$",
            ylabel=r"$M_b / M_{halo} / (\Omega_b/\Omega_m)$",
        )
        for _, fof, _, kwargs in sims:
            plot_baryon_fraction(ax, fof, 5, **kwargs)
        if numbers:
            put_number(ax, "VII", loc="upper")

    # Halo Temperature
    if 8 in panels:
        ax = fig.add_subplot(
            4,
            3,
            8,
            xscale="log",
            yscale="log",
            xlabel=r"Temperature [K]",
            ylabel=r"PDF [$K^{-1}$]",
        )
        for snap, _, _, kwargs in sims:
            plot_halo_temp(ax, snap, **kwargs)
        if numbers:
            put_number(ax, "VIII", loc="upper")

    # Galaxy Radius
    if 9 in panels:
        ax = fig.add_subplot(
            4,
            3,
            9,
            xscale="log",
            yscale="log",
            xlabel=r"$M_*$ [$h^{-1} M_\odot$]",
            ylabel=r"R$_{1/2}$ [$h^{-1} kpc$]",
        )
        for _, fof, _, kwargs in sims:
            plot_galaxy_radius(ax, fof, **kwargs)
        if numbers:
            put_number(ax, "IX", loc="upper")

    # Black Hole Mass
    if 10 in panels:
        ax = fig.add_subplot(
            4,
            3,
            10,
            xscale="log",
            yscale="log",
            xlabel=r"$M_*$ [$h^{-1} M_\odot$]",
            ylabel=r"M$_{BH}$ [$h^{-1} M_\odot$]",
        )
        for _, fof, _, kwargs in sims:
            plot_BH_mass(ax, fof, **kwargs)
        if numbers:
            put_number(ax, "X", loc="upper")

    # Max Circular Velocity
    if 11 in panels:
        ax = fig.add_subplot(
            4,
            3,
            11,
            xscale="log",
            xlabel=r"$M_*$ [$h^{-1} M_\odot$]",
            ylabel=r"max($\sqrt{GM/R}$) [km/s]",
        )
        for _, fof, _, kwargs in sims:
            plot_v_circ(ax, fof, **kwargs)
        if numbers:
            put_number(ax, "XI", loc="upper")

    # Galaxy SFR
    if 12 in panels:
        ax = fig.add_subplot(
            4,
            3,
            12,
            xscale="log",
            yscale="log",
            xlabel=r"$M_*$ [$h^{-1} M_\odot$]",
            ylabel=r"SFR [$M_\odot yr^{-1}$]",
        )
        for _, fof, _, kwargs in sims:
            plot_gal_sfr(ax, fof, **kwargs)
        if numbers:
            put_number(ax, "XII", loc="upper")

    handles = [Line2D([0], [0], **kwargs) for _, _, _, kwargs in sims]
    labels = [kwargs["label"] for _, _, _, kwargs in sims]
    if every_legend:
        for ax in fig.axes:
            ax.legend(handles, labels)
    else:
        ax = fig.axes[0]
        ax.legend(handles, labels)

    return fig
