import typing as T

import unyt as u
from yt.utilities.cosmology import Cosmology

from .sfr import SFRData
from .timeseries import Timeseries


class Run:
    snapshots: Timeseries
    sfr_data: T.Optional[SFRData] = None
    cosmology: Cosmology
    box_size: u.unyt_array

    def __init__(
        self, timeseries: Timeseries, sfr_data: T.Optional[SFRData] = None
    ) -> None:
        self.snapshots = timeseries
        self.sfr_data = sfr_data
        self.cosmology = timeseries.cosmology
        self.box_size = self.snapshots.snapshots_list[0].snap.domain_width.to("Mpccm")  # type: ignore this is a unyt_array
