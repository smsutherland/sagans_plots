import typing as T

import numpy as np
import unyt as u
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.projections import register_projection
from numpy.typing import ArrayLike
from yt.utilities.cosmology import Cosmology

AxisType = T.Literal["t", "-t", "z", "a"]
AxisTypeLong = T.Literal["time", "lookback", "redshift", "scale"]
GyrToS = 3.15576e16


def filter_empty(f):
    def filtered(self, x: ArrayLike) -> ArrayLike:
        x = np.array(x)
        if len(x) == 0:
            return x
        else:
            return f(self, x)

    return filtered


def filter_zero(f):
    def filtered(self, x: ArrayLike) -> ArrayLike:
        x = np.array(x)
        x[x == 0] = np.nan
        return f(self, x)

    return filtered


class Cosmo(Axes):
    name = "cosmo"
    _cosmology: Cosmology
    _primary_axis: AxisType
    _secondary_axis: AxisType

    def __init__(
        self,
        *args,
        cosmology: Cosmology = Cosmology(),
        primary_axis: AxisType | AxisTypeLong = "t",
        secondary_axis: AxisType | AxisTypeLong = "t",
        **kwargs,
    ):
        self._cosmology = cosmology
        self._primary_axis = _normalize_axis_name(primary_axis)
        self._secondary_axis = _normalize_axis_name(secondary_axis)
        super().__init__(*args, **kwargs)

        match self._primary_axis:
            case "t":
                self.set_xlabel("Cosmic Time [Gyr]")
            case "-t":
                self.set_xlabel("Lookback Time [Gyr]")
            case "z":
                self.set_xlabel("Redshift")
            case "a":
                self.set_xlabel("Scale-Factor")

        # There's definitely some weird liveness issues with some of these being lambdas and some not.
        match self._primary_axis, self._secondary_axis:
            case "t", "t":
                functions = (self._t_from_t, self._t_from_t)
            case "t", "-t":
                functions = (self._lt_from_t, self._t_from_lt)
            case "t", "z":
                functions = (self._z_from_t, self._t_from_z)
            case "t", "a":
                functions = (self._a_from_t, self._t_from_a)
            case "-t", "t":
                functions = (self._t_from_lt, self._lt_from_t)
            case "-t", "-t":
                functions = (self._lt_from_lt, self._lt_from_lt)
            case "-t", "z":
                functions = (self._z_from_lt, self._lt_from_z)
            case "-t", "a":
                functions = (self._a_from_lt, self._lt_from_a)
            case "z", "t":
                functions = (self._t_from_z, self._z_from_t)
            case "z", "-t":
                functions = (self._lt_from_z, self._z_from_lt)
            case "z", "z":
                functions = (self._z_from_z, self._z_from_z)
            case "z", "a":
                functions = (self._a_from_z, self._a_from_z)
            case "a", "t":
                functions = (self._t_from_a, self._a_from_t)
            case "a", "-t":
                functions = (self._lt_from_a, self._a_from_lt)
            case "a", "z":
                functions = (self._z_from_a, self._a_from_z)
            case "a", "a":
                functions = (self._a_from_a, self._a_from_a)

        second_ax = self.secondary_xaxis("top", functions=functions)
        match self._secondary_axis:
            case "t":
                second_ax.set_xlabel("Cosmic Time [Gyr]")
            case "-t":
                second_ax.set_xlabel("Lookback Time [Gyr]")
            case "z":
                second_ax.set_xlabel("Redshift")
            case "a":
                second_ax.set_xlabel("Scale-Factor")

    def plot(
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        fmt: str = "",
        scalex: bool = True,
        scaley: bool = True,
        data=None,
        **kwargs,
    ) -> list[Line2D]:
        if isinstance(x, u.unyt_array):
            x = x.to_value("Gyr") # type: ignore we just checked this is a unyt_array, not a float

        return super().plot(
            x, y, fmt, scalex=scalex, scaley=scaley, data=data, **kwargs
        )

    @filter_empty
    def _t_from_t(self, t: ArrayLike) -> ArrayLike:
        return t

    @filter_empty
    def _t_from_lt(self, lt: ArrayLike) -> ArrayLike:
        total_age = self._cosmology.t_from_z(0).to_value("Gyr")
        return total_age - lt

    @filter_empty
    def _t_from_z(self, z: ArrayLike) -> ArrayLike:
        return self._cosmology.t_from_z(z).to_value("Gyr")

    @filter_empty
    def _t_from_a(self, a: ArrayLike) -> ArrayLike:
        return self._cosmology.t_from_a(a).to_value("Gyr")

    @filter_empty
    @filter_zero
    def _lt_from_t(self, t: ArrayLike) -> ArrayLike:
        total_age = self._cosmology.t_from_z(0).to_value("Gyr")
        return total_age - t

    @filter_empty
    def _lt_from_lt(self, lt: ArrayLike) -> ArrayLike:
        return lt

    @filter_empty
    def _lt_from_z(self, z: ArrayLike) -> ArrayLike:
        total_age = self._cosmology.t_from_z(0).to_value("Gyr")
        return total_age - self._t_from_z(z)

    @filter_empty
    @filter_zero
    def _lt_from_a(self, a: ArrayLike) -> ArrayLike:
        total_age = self._cosmology.t_from_z(0).to_value("Gyr")
        return total_age - self._t_from_a(a)

    @filter_empty
    @filter_zero
    def _z_from_t(self, t: ArrayLike) -> ArrayLike:
        t = np.array(t)
        return self._cosmology.z_from_t(t * GyrToS)

    @filter_empty
    def _z_from_lt(self, lt: ArrayLike) -> ArrayLike:
        return self._z_from_t(self._t_from_lt(lt))

    @filter_empty
    def _z_from_z(self, z: ArrayLike) -> ArrayLike:
        return z

    @filter_empty
    @filter_zero
    def _z_from_a(self, a: ArrayLike) -> ArrayLike:
        a = np.array(a)
        return 1 / a - 1

    @filter_empty
    @filter_zero
    def _a_from_t(self, t: ArrayLike) -> ArrayLike:
        t = np.array(t)
        return self._cosmology.a_from_t(t * GyrToS)

    @filter_empty
    def _a_from_lt(self, lt: ArrayLike) -> ArrayLike:
        return self._a_from_t(self._t_from_lt(lt))

    @filter_empty
    def _a_from_z(self, z: ArrayLike) -> ArrayLike:
        z = np.array(z)
        return 1 / (z + 1)

    @filter_empty
    def _a_from_a(self, a: ArrayLike) -> ArrayLike:
        return a


def _normalize_axis_name(name: AxisType | AxisTypeLong) -> AxisType:
    match name:
        case "time" | "t":
            return "t"
        case "lookback" | "-t":
            return "-t"
        case "redshift" | "z":
            return "z"
        case "scale" | "a":
            return "a"


register_projection(Cosmo)
