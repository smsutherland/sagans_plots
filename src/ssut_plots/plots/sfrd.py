import typing as T

import numpy as np
from matplotlib.projections import register_projection
from yt.utilities.cosmology import Cosmology

from ssut_plots.data.run import Run

from .cosmo import AxisType, AxisTypeLong, Cosmo


class SfrPlot(Cosmo):
    name = "sfrd"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_ylabel(
            r"Star Formation Density $\left[\mathrm{ M_\odot \, yr^{-1} \, Mpc^3 }\right]$"
        )

    def plot_sfrd(self, run: Run, resolution: int = 500):
        if run.sfr_data is None:
            raise ValueError("Run has no sfr data")
        volume = run.box_size**3
        sfr = run.sfr_data.sfr.to_value("Msun/yr")
        z = run.sfr_data.z
        max = np.max(z)
        min = np.min(z)
        z_plot = np.linspace(max, min, resolution)
        sfr_plot = np.interp(z_plot, z, sfr)
        sfr_dens = sfr_plot / volume

        match self._primary_axis:
            case "t":
                x = self._t_from_z(z_plot)
            case "-t":
                x = self._lt_from_z(z_plot)
            case "z":
                x = z_plot
            case "a":
                x = 1 / (z_plot + 1)

        self.plot(x, sfr_dens)


class SfrdProjection:
    def __init__(
        self,
        cosmology: Cosmology = Cosmology(),
        primary_axis: AxisType | AxisTypeLong = "t",
        secondary_axis: AxisType | AxisTypeLong = "t",
    ):
        self.kwargs = {
            "cosmology": cosmology,
            "primary_axis": primary_axis,
            "secondary_axis": secondary_axis,
        }
        pass

    def _as_mpl_axes(self) -> tuple[T.Type[SfrPlot], dict[str, T.Any]]:
        return SfrPlot, self.kwargs


register_projection(SfrPlot)
