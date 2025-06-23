import typing as T
import warnings
from pathlib import Path

import yt
from yt.data_objects.static_output import Dataset
from yt.frontends.gadget_fof.data_structures import GadgetFOFDataset


class Snapshot:
    def __init__(
        self, snap: Dataset, subfind: T.Optional[GadgetFOFDataset] = None
    ) -> None:
        self.snap = snap
        self.subfind = subfind

    def with_subfind(self, fname: str | Path, force: bool = False):
        if self.subfind is not None and not force:
            warnings.warn(
                "Snapshot already has an attached Subfind catalog.\nUse force=True to overwrite."
            )
            return
        self.subfind = yt.load(  # type: ignore yt.load DOES exist please trust me
            fname,
            hint="GadgetFOF",
        )
