import typing as T

from yt.utilities.cosmology import Cosmology

from .sfr import SFRData
from .timeseries import Timeseries


class Run:
    snapshots: Timeseries
    sfr_data: T.Optional[SFRData]
    cosmology: Cosmology

    def __init__(
        self, timeseries: Timeseries, sfr_data: T.Optional[SFRData] = None
    ) -> None:
        self.snapshots = timeseries
        self.sfr_data = sfr_data
        self.cosmology = timeseries.cosmology
