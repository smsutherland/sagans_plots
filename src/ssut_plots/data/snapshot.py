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
        self.fix_subfind()

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
        self.fix_subfind()

    def fix_subfind(self):
        """
        YT adds ".%(num)i" to the filename template so it can turn `fof_subhalo_tab_090.hdf5` to `fof_subhalo_tab_090.0.hdf5`.
        This is not necessary for our purposes, so we take it out.
        This could maybe turn into a PR to YT.
        """
        if self.subfind is None:
            return
        if ".%(num)i" not in self.subfind.filename_template:
            return
        self.subfind.filename_template = "".join(self.subfind.filename_template.split(".%(num)i"))
