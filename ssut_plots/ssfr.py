#!/bin/env python

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

nbins = 20


def plot_gal_ssfr(ax: Axes, fof: str, mmin: float, mmax: float, **kwargs):
    with h5py.File(fof) as f:
        sfr: np.ndarray = f["Subhalo/SubhaloSFR"][:]  # Msun / yr
        stellar_mass: np.ndarray = f["Subhalo/SubhaloMassType"][:, 4] * 1e10  # Msun / h

    mask = (stellar_mass > mmin) & (stellar_mass < mmax)
    sfr = sfr[mask]
    stellar_mass = stellar_mass[mask]
    ssfr = sfr / stellar_mass * 1e9 * 0.6711  # 1 / Gyr

    bins = np.logspace(-3, 1, nbins, endpoint=True)
    bins[0] = 0
    hist, edges = np.histogram(ssfr, bins)
    hist = hist / ssfr.size
    xs = edges[1:]
    ax.plot(xs, hist, **kwargs)
    return ssfr.size
