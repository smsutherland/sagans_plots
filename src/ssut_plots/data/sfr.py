import typing as T
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import unyt as u

from ssut_plots.types import SimulationType


class SFRData:
    file: str | Path
    _data: T.Optional[pd.DataFrame] = None
    hint: SimulationType

    def __init__(self, file: str | Path, hint: SimulationType) -> None:
        self.file = file
        self.hint = hint

    @property
    def data(self) -> pd.DataFrame:
        if self._data is None:
            match self.hint:
                case "SIMBA":
                    # TODO: SFR*dt (active) and total M_stars are in code units
                    # Get the code units from a snapshot and use them to convert to Msun and Msun/yr
                    self._data = pd.read_table(
                        self.file,
                        sep=" ",
                        names=[
                            "a",
                            "SFR*dt (active)",
                            "SFR (total)",
                            "SFR (active)",
                            "total M_stars",
                        ],
                    )
                    self._data["z"] = 1 / self._data["a"] - 1
                    self._data.index.name = "Step"
                case "SWIFT":
                    self._data = pd.read_table(
                        self.file,
                        comment="#",
                        sep=r"\s+",
                        index_col=0,
                        names=[
                            "Step",
                            "Time",
                            "a",
                            "z",
                            "total M_stars",
                            "SFR (active)",
                            "SFR*dt (active)",
                            "SFR (total)",
                        ],
                    )
                    units = self._read_swift_units()
                    self._data["Time"] *= units[0]
                    self._data["total M_stars"] *= units[1]
                    self._data["SFR (active)"] *= units[2]
                    self._data["SFR*dt (active)"] *= units[3]
                    self._data["SFR (total)"] *= units[4]
        return self._data

    @property
    def z(self) -> np.ndarray[tuple[T.Any], np.dtype[np.float64]]:
        return self.data["z"].to_numpy(copy=False)

    @property
    def a(self) -> np.ndarray[tuple[T.Any], np.dtype[np.float64]]:
        return self.data["a"].to_numpy(copy=False)

    @property
    def sfr(self) -> u.unyt_array:
        return u.unyt_array(self.data["SFR (total)"].to_numpy(copy=False), "Msun/yr")  # type: ignore unyt_array DOES return itself

    def _read_swift_units(self) -> list[float]:
        if self.hint != "SWIFT":
            warnings.warn(
                "Trying to read SWIFT units for a non-SWIFT SFR. Returning ones."
            )
            return [1] * 5
        with open(self.file) as f:
            for _ in range(7):
                next(f)
            time_s = next(f)
            for _ in range(7):
                next(f)
            mass_g = next(f)
            for _ in range(2):
                next(f)
            active_sfr_gps = next(f)
            for _ in range(2):
                next(f)
            sfr_dt_g = next(f)
            for _ in range(2):
                next(f)
            total_sfr_gps = next(f)
        units = [time_s, mass_g, active_sfr_gps, sfr_dt_g, total_sfr_gps]
        units_floats = [float(unit.split()[3]) for unit in units]
        units_floats[0] *= (1 * u.s).to(u.Gyr).value  # type: ignore units do not appear to type checkers
        units_floats[1] *= (1 * u.g).to(u.Msun).value  # type: ignore units do not appear to type checkers
        units_floats[2] *= (1 * u.g / u.s).to(u.Msun / u.yr).value  # type: ignore units do not appear to type checkers
        units_floats[3] *= (1 * u.g).to(u.Msun).value  # type: ignore units do not appear to type checkers
        units_floats[4] *= (1 * u.g / u.s).to(u.Msun / u.yr).value  # type: ignore units do not appear to type checkers
        return units_floats
