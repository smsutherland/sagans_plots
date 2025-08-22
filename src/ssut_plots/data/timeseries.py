import typing as T

import numpy as np
import unyt as u
from yt.utilities.cosmology import Cosmology

from .snapshot import Snapshot


class Timeseries:
    snapshots_list: T.List[Snapshot]
    redshift_list: np.ndarray[tuple[T.Any], np.dtype[np.float64]]
    cosmology: Cosmology

    @T.overload
    def __init__(self, snapshots: Snapshot | T.Iterable[Snapshot], /) -> None: ...
    @T.overload
    def __init__(self, *snapshots: Snapshot) -> None: ...
    def __init__(self, *snapshots) -> None:
        self.snapshots_list = []
        for snap in snapshots:
            if isinstance(snap, Snapshot):
                self.snapshots_list.append(snap)
            else:
                self.snapshots_list.extend(snap)
        if len(self.snapshots_list) == 0:
            raise ValueError("Cannot create a Timeseries of 0 snapshots")
        self.cosmology = self.snapshots_list[0].snap.cosmology
        redshift_list = [
            getattr(snapshot.snap, "current_redshift", None)
            for snapshot in self.snapshots_list
        ]
        snapshots_without_redshift = list(
            map(
                lambda x: x[0],
                filter(
                    lambda x: x[1],
                    enumerate(redshift is None for redshift in redshift_list),
                ),
            )
        )
        if len(snapshots_without_redshift) > 0:
            error_str = "Cannot create a timeseries with snapshots missing redshifts. Missing redshifts:"
            for i in snapshots_without_redshift:
                error_str = error_str + "\n\t" + self.snapshots_list[i].snap.filename

            raise ValueError(error_str)

        self.redshift_list = redshift_list  # type: ignore We've checked above that there are no `None`s left

    def get_redshift(self, redshift: float) -> Snapshot:
        index = np.abs(self.redshift_list - redshift).argmin()
        return self.snapshots_list[index]

    def get_time(self, time: u.unyt_quantity | float) -> Snapshot:
        if isinstance(time, (float, int)):
            time *= u.Gyr  # type: ignore unyts do not appear in type checking
        z = self.cosmology.z_from_t(time)
        return self.get_redshift(z)
