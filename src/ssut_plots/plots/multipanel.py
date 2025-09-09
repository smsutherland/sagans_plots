import typing as T
import warnings
from functools import lru_cache
from pathlib import Path

import MAS_library
import numpy as np
import Pk_library
import scipy.stats
import unyt as u
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from yt.data_objects.selection_objects.region import YTRegion

from ssut_plots.data.loaders import load_snapshot
from ssut_plots.data.run import Run
from ssut_plots.data.snapshot import Snapshot


class MultipanelFigure(Figure):
    _panels: T.Iterable[int]
    _multipanel_axes: list[T.Optional[Axes]]

    def __init__(
        self,
        *args,
        panels: int | T.Iterable[int] = range(1, 13),
        numbers=True,
        **kwargs,
    ) -> None:
        if kwargs["figsize"] is None:
            kwargs["figsize"] = [20, 20]
        super().__init__(*args, **kwargs)

        if isinstance(panels, int):
            self._panels = (panels,)
        else:
            self._panels = panels

        self._multipanel_axes = [None] * 12
        for p in self._panels:
            ax = self._multipanel_axes[p - 1] = self.add_subplot(
                4, 3, p, **self._panel_kwargs[p - 1]
            )
            if numbers:
                if p >= 6:
                    loc = "upper"
                else:
                    loc = "lower"
                roman = [
                    "I",
                    "II",
                    "III",
                    "IV",
                    "V",
                    "VI",
                    "VII",
                    "VIII",
                    "IX",
                    "X",
                    "XI",
                    "XII",
                ]
                put_number(ax, roman[p - 1], loc)

    def multipanel_get_axes(self, number: int) -> T.Optional[Axes]:
        if number not in range(1, 13):
            raise ValueError(
                "Use the number of the axes as they appear on the figure [1-12]"
            )
        return self._multipanel_axes[number - 1]

    def plot_sim(self, sim: Run, **kwargs):
        z0 = sim.snapshots.get_redshift(0)

        # matter power spectrum
        if (ax := self._multipanel_axes[0]) is not None:
            k, Pk = compute_Pk(z0, threads=40, ptypes=range(6))
            lims = ax.get_xlim()
            filter = (k >= lims[0]) & (k <= lims[1])
            ax.plot(k[filter], Pk[filter], **kwargs)

        # gas power spectrum
        if (ax := self._multipanel_axes[1]) is not None:
            k, Pk = compute_Pk(z0, threads=40, ptypes=0)
            lims = ax.get_xlim()
            filter = (k >= lims[0]) & (k <= lims[1])
            ax.plot(k[filter], Pk[filter], **kwargs)

        # hydro / nbody power spectrum
        if (ax := self._multipanel_axes[2]) is not None:
            # TODO: actually make hydro-nbody matching robust
            snap_path = Path(z0.snap.filename).resolve()
            snap_path_parts = snap_path.parts

            index = None
            for s in ("CV", "1P", "LH"):
                try:
                    index = snap_path_parts.index(s)
                    break
                except ValueError:
                    continue
            else:
                warnings.warn("Could not find matching DM run. Skipping panel 3")
            if index is not None:
                snapshot_name = snap_path_parts[-1]
                snapshot_number = int(snapshot_name.split("_")[1].split(".")[0])

                dm_snap = load_snapshot(
                    f"/mnt/ceph/users/camels/PUBLIC_RELEASE/Sims/SIMBA_DM/{snap_path_parts[index]}/{snap_path_parts[index + 1]}/snapshot_{snapshot_number:03d}.hdf5",
                )

                k_hydro, Pk_hydro = compute_Pk(z0, threads=40, ptypes=range(6))
                k_nbody, Pk_nbody = compute_Pk(dm_snap, threads=40, ptypes=range(6))

                np.testing.assert_allclose(k_hydro, k_nbody, rtol=1e-5)

                Pk_ratio = Pk_hydro / Pk_nbody
                lims = ax.get_xlim()
                filter = (k_hydro >= lims[0]) & (k_hydro <= lims[1])
                ax.plot(k_hydro[filter], Pk_ratio[filter], **kwargs)
                del dm_snap

        # halo mass function
        if (ax := self._multipanel_axes[3]) is not None:
            bins = np.geomspace(5e10, 5e14, 21)
            hmf = halo_mass_function(z0, bins)[0]
            xs = np.sqrt(bins[1:] * bins[:-1])
            ax.plot(xs, hmf, **kwargs)

        # star formation rate density
        if (ax := self._multipanel_axes[4]) is not None:
            if sim.sfr_data is None:
                raise ValueError("Run has no sfr data")
            volume = np.prod(sim.box_size)
            sfr: u.unyt_array = sim.sfr_data.sfr.to_value("Msun/yr")  # type: ignore
            z = sim.sfr_data.z
            max = np.max(z)
            min = np.min(z)
            z_plot = np.linspace(max, min, 2048)
            sfr_plot = np.interp(
                z_plot, z[::-1], sfr[::-1]
            )  # interp requires input to be increasing, but z counts backwards
            sfr_dens = sfr_plot / volume
            ax.plot(z_plot, sfr_dens, **kwargs)

        # stellar mass function
        if (ax := self._multipanel_axes[5]) is not None:
            bins = np.logspace(10, 14, 21)
            shmf = subhalo_mass_function(z0, bins)[0]
            xs = np.sqrt(bins[1:] * bins[:-1])
            ax.plot(xs, shmf, **kwargs)

        # baryon fraction
        if (ax := self._multipanel_axes[6]) is not None:
            if z0.subfind is None:
                raise ValueError("Cannot find baryon fraction without subfind")
            bins = np.logspace(10, 14, 21)
            ad = z0.subfind.all_data()
            group_mass = ad["Group", "GroupMass"].to("Msun")
            omega_m = z0.subfind.cosmology.omega_matter
            baryon_fraction = (
                (
                    ad["Group", "GroupMassType_0"]
                    + ad["Group", "GroupMassType_4"]
                    + ad["Group", "GroupMassType_5"]
                )
                / group_mass
                / (z0.omega_b / omega_m)
            )

            means, _, _ = scipy.stats.binned_statistic(
                group_mass,
                baryon_fraction,
                "mean",
                bins=bins,  # type: ignore the type of bins is valid
            )
            xs = np.sqrt(bins[1:] * bins[:-1])
            ax.plot(xs, means, **kwargs)

        # temperature distribtuion
        if (ax := self._multipanel_axes[7]) is not None:
            ad = z0.snap.all_data()
            temps = ad["PartType0", "temperature"].to_value("K")
            bins = np.geomspace(1e2, 5e7, 21)
            m = ad["PartType0", "particle_mass"].to_value("Msun")
            ax.hist(
                temps,
                bins,  # type: ignore the type of bins is valid
                histtype="step",
                density=True,
                weights=m,
                **kwargs,
            )

        # galaxy radius
        if (ax := self._multipanel_axes[8]) is not None:
            if z0.subfind is None:
                raise ValueError("Cannot find galaxy radius without subfind")
            ad = z0.subfind.all_data()
            bins = np.geomspace(1e9, 5e11, 21)
            radii = ad["Subhalo", "SubhaloHalfmassRadType_4"].to_value("kpc")
            mass = ad["Subhalo", "SubhaloMassType_4"].to_value("Msun")
            means, _, _ = scipy.stats.binned_statistic(
                mass,
                radii,
                "mean",
                bins,  # type: ignore the type of bins is valid
            )
            xs = np.sqrt(bins[:1] * bins[:-1])
            ax.plot(xs, means, **kwargs)

        # black hole mass
        if (ax := self._multipanel_axes[9]) is not None:
            if z0.subfind is None:
                raise ValueError("Cannot find galaxy bh mass without subfind")
            ad = z0.subfind.all_data()
            bh_mass = ad["Subhalo", "SubhaloBHMass"].to_value("Msun")
            gal_mass = ad["Subhalo", "SubhaloMassType_4"].to_value("Msun")
            bins = np.geomspace(1e9, 5e11, 21)
            means, _, _ = scipy.stats.binned_statistic(
                gal_mass,
                bh_mass,
                "median",
                bins,  # type: ignore the type of bins is valid
            )
            xs = np.sqrt(bins[1:] * bins[:-1])
            ax.plot(xs, means, **kwargs)

        # max velocity
        if (ax := self._multipanel_axes[10]) is not None:
            if z0.subfind is None:
                raise ValueError("Cannot find galaxy v_max without subfind")
            ad = z0.subfind.all_data()
            vmax = ad["Subhalo", "SubhaloVmax"].to_value("km/s")
            gal_mass = ad["Subhalo", "SubhaloMassType_4"].to_value("Msun")
            bins = np.geomspace(1e9, 5e11, 21)
            means, _, _ = scipy.stats.binned_statistic(
                gal_mass,
                vmax,
                "mean",
                bins,  # type: ignore the type of bins is valid
            )
            xs = np.sqrt(bins[1:] * bins[:-1])
            ax.plot(xs, means, **kwargs)

        # galaxy sfr
        if (ax := self._multipanel_axes[11]) is not None:
            if z0.subfind is None:
                raise ValueError("Cannot find galaxy sfr without subfind")
            ad = z0.subfind.all_data()
            sfr = ad["Subhalo", "SubhaloSFR"].to_value("Msun/yr")
            gal_mass = ad["Subhalo", "SubhaloMassType_4"].to_value("Msun")
            bins = np.geomspace(1e9, 5e11, 21)
            mask = sfr > 0
            means, _, _ = scipy.stats.binned_statistic(
                gal_mass[mask],
                sfr[mask],
                "mean",
                bins,  # type: ignore the type of bins is valid
            )
            xs = np.sqrt(bins[1:] * bins[:-1])
            ax.plot(xs, means, **kwargs)

    _panel_kwargs: list[dict[str, T.Any]] = [
        {  # 1: matter power spectrum
            "xscale": "log",
            "yscale": "log",
            "xlim": (0.3, 30),
            "ylim": (1e-1, 1e3),
            "xlabel": r"k $\left[\mathrm{Mpc^{-1}}\right]$",
            "ylabel": r"$\mathrm{P_m(k)}\,\left[\mathrm{Mpc^3}\right]$",
        },
        {  # 2: gas power spectrum
            "xscale": "log",
            "yscale": "log",
            "xlim": (0.3, 30),
            "ylim": (1e-2, 1e3),
            "xlabel": r"k $\left[\mathrm{Mpc^{-1}}\right]$",
            "ylabel": r"$\mathrm{P_g(k)}\,\left[\mathrm{Mpc^3}\right]$",
        },
        {  # 3: hydro / nbody matter power spectrum ratio
            "xlim": (0.3, 30),
            "xscale": "log",
            "xlabel": r"k $\left[\mathrm{Mpc^{-1}}\right]$",
            "ylabel": r"$\mathrm{P_{hydro}(k)/P_{Nbody}}(k)$",
        },
        {  # 4: halo mass function
            "yscale": "log",
            "xscale": "log",
            "xlabel": r"$\mathrm{M_{halo}} / \Omega_\mathrm{m} \left[\mathrm{M_\odot}\right]$",
            "ylabel": r"HMF $\left[\mathrm{Mpc^{-3}\,M_\odot^{-1}}\right]$",
        },
        {  # 5: star formation rate density
            "xlim": (0, 7),
            "ylim": (5e-4, 2e-1),
            "yscale": "log",
            "xlabel": "redshift (z)",
            "ylabel": r"SFRD $\left[\mathrm{M_\odot\,yr^{-1}\,cMpc^{-3}}\right]$",
            # We don't set the projection to SfrdProjection here because of the differing cosmologies in camels.
            # SfrdProjection primarilly works for comparing runs of a single cosmology
            # For this, we only plot redshift.
        },
        {  # 6: stellar mass function
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"$\mathrm{M}_*$ $\left[\mathrm{M_\odot}\right]$",
            "ylabel": r"SMF $\left[\mathrm{Mpc^{-3}\,M_\odot^{-1}}\right]$",
        },
        {  # 7: baryon fraction
            "xscale": "log",
            "xlabel": r"$\mathrm{M_{halo}} / \Omega_\mathrm{m} \left[\mathrm{M_\odot}\right]$",
            "ylabel": r"$\mathrm{M_b / M_{halo} / (\Omega_b/\Omega_m)}$",
        },
        {  # 8: temperature distribution
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"Temperature [K]",
            "ylabel": r"PDF $\left[\mathrm{K}^{-1}\right]$",
        },
        {  # 9: galaxy radius
            "xscale": "log",
            # "yscale": "log",
            "xlabel": r"$\mathrm{M}_*$ $\left[\mathrm{M_\odot}\right]$",
            "ylabel": r"$\mathrm{R}_{1/2}$ $\left[\mathrm{kpc}\right]$",
        },
        {  # 10: black hole mass
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"$\mathrm{M}_*$ $\left[\mathrm{M_\odot}\right]$",
            "ylabel": r"$\mathrm{M_{BH}}$ $\left[\mathrm{M_\odot}\right]$",
        },
        {  # 11: max circular velocity
            "xscale": "log",
            "xlabel": r"$\mathrm{M}_*$ $\left[\mathrm{M_\odot}\right]$",
            "ylabel": r"max($\sqrt{\mathrm{GM/R}}$) $\left[\mathrm{km\,s^{-1}}\right]$",
            "ylim": (0, 400),
        },
        {  # 12: galaxy sfr
            "xscale": "log",
            "yscale": "log",
            "xlabel": r"$\mathrm{M}_*$ $\left[\mathrm{M_\odot}\right]$",
            "ylabel": r"SFR $\left[\mathrm{M_\odot\,yr^{-1}}\right]$",
        },
    ]


def put_number(ax: Axes, num: str, loc: T.Literal["lower", "upper"]):
    params = {
        "transform": ax.transAxes,
        "fontsize": 20,
        "fontstyle": "italic",
        "fontfamily": "serif",
        "va": None,
        "ha": "left",
    }
    if loc == "lower":
        y = 0.1
        params["va"] = "top"
    elif loc == "upper":
        y = 0.9
        params["va"] = "bottom"
    ax.text(x=0.05, y=y, s=num, **params)


@lru_cache
def compute_Pk(
    snap: Snapshot,
    grid: T.Optional[int] = None,
    MAS="CIC",
    threads: int = 1,
    ptypes: int | list[int] = list(range(6)),
) -> np.ndarray:
    if isinstance(ptypes, int):
        ptypes = [ptypes]

    if grid is None:
        counts = snap.snap.particle_type_counts
        total_dm = counts["PartType1"]
        grid2: int = round(total_dm ** (1 / 3)) * 2
        grid = grid2  # This exists to make pyright be quiet. Otherwise it'll complain that the `: int` obscures the parameter declaration

    Ntot = sum(snap.snap.particle_type_counts[f"PartType{ty}"] for ty in ptypes)

    coords = np.empty((Ntot, 3), dtype=np.float32)
    masses = np.empty(Ntot, dtype=np.float32)

    ad: YTRegion = snap.snap.all_data()

    offset = 0
    for ty in ptypes:
        length = snap.snap.particle_type_counts[f"PartType{ty}"]
        if length == 0:
            continue
        coords[offset : offset + length, :] = ad[
            f"PartType{ty}", "Coordinates"
        ].to_value("Mpccm")  # type: ignore to_value exists
        masses[offset : offset + length] = ad[
            f"PartType{ty}", "particle_mass"
        ].to_value(  # type: ignore to_value exists
            "Msun"
        )  # Units will be divided out later. This is just in case not all masses are given in the same units
        offset += length
    assert offset == Ntot

    box_size = snap.snap.domain_width.to_value("Mpccm").mean()  # type: ignore to_value exists

    density = np.empty((grid,) * 3, dtype=np.float32)
    MAS_library.MA(coords, density, box_size, MAS, W=masses)  # type: ignore Type checkers don't like cython
    density /= np.mean(density, dtype=np.float64)
    density -= 1.0

    axis = 0
    Pk = Pk_library.Pk(density, box_size, axis, MAS, threads)  # type: ignore Type checkers don't like cython
    return np.array([Pk.k3D, Pk.Pk[:, 0]])


def halo_mass_function(
    snap: Snapshot,
    bins: int | str | np.ndarray = "auto",
    reduce: bool = True,
    ptypes: T.Literal["total"] | int | T.Iterable[int] = "total",
) -> T.Tuple[np.ndarray, np.ndarray]:
    if snap.subfind is None:
        raise ValueError("Cannot find halo mass function without subfind catalog")

    halo_data = snap.subfind.all_data()
    if ptypes == "total":
        halo_masses = halo_data["Group", "GroupMass"].to_value("Msun")
    elif isinstance(ptypes, int):
        halo_masses = halo_data["Group", f"GroupMassType_{ptypes}"].to_value("Msun")
    else:
        halo_masses: np.ndarray = sum(  # type: ignore sum can return more than just an int
            halo_data["Group", f"GroupMassType_{i}"].to_value("Msun") for i in ptypes
        )

    n_dm = halo_data["Group", "GroupLenType_1"]
    halo_masses = halo_masses[
        n_dm > 50
    ]  # Only look at halos with at least 50 dm particles
    box_size = snap.subfind.domain_width.to_value("Mpccm").mean()  # type: ignore to_value exists

    if reduce:
        Om = snap.subfind.cosmology.omega_matter
    else:
        Om = 1

    hmf, edges = np.histogram(halo_masses / Om, bins)
    dM = edges[1:] - edges[:-1]
    hmf = hmf / (box_size**3 * dM * Om)
    return hmf, edges


def subhalo_mass_function(
    snap: Snapshot,
    bins: int | str | np.ndarray = "auto",
    reduce: bool = True,
    ptypes: T.Literal["total"] | int | T.Iterable[int] = "total",
) -> T.Tuple[np.ndarray, np.ndarray]:
    if snap.subfind is None:
        raise ValueError("Cannot find subhalo mass function without subfind catalog")

    subhalo_data = snap.subfind.all_data()
    if ptypes == "total":
        subhalo_masses = subhalo_data["Subhalo", "SubhaloMass"].to_value("Msun")
    elif isinstance(ptypes, int):
        subhalo_masses = subhalo_data["Subhalo", f"SubhaloMassType_{ptypes}"].to_value(
            "Msun"
        )
    else:
        subhalo_masses: np.ndarray = sum(  # type: ignore sum can return more than just an int
            subhalo_data["Subhalo", f"SubhaloMassType_{i}"].to_value("Msun")
            for i in ptypes
        )

    box_size = snap.subfind.domain_width.to_value("Mpccm").mean()  # type: ignore to_value exists

    if reduce:
        Om = snap.subfind.cosmology.omega_matter
    else:
        Om = 1

    shmf, edges = np.histogram(subhalo_masses / Om, bins)
    dM = edges[1:] - edges[:-1]
    shmf = shmf / (box_size**3 * dM * Om)
    return shmf, edges
