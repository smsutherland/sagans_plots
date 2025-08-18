import typing as T
from glob import glob
from pathlib import Path

import yt

from .snapshot import Snapshot
from .timeseries import Timeseries
from .types import (
    CV,
    LH,
    CamelsSimulationType,
    OneP,
    SimulationType,
    Volume,
)


def load_snapshot(
    fname: str | Path,
    subfind: T.Optional[str | Path] = None,
    kind: T.Optional[SimulationType] = None,
) -> Snapshot:
    match kind:
        case None:
            hint = None
        case "SIMBA":
            hint = "Gizmo"
        case "SWIMBA":
            hint = "SWIFT"

    snap = yt.load(fname, hint=hint)  # type: ignore yt.load DOES exist please trust me
    if subfind is not None:
        fof = yt.load(subfind, hint="GadgetFOF")  # type: ignore yt.load DOES exist please trust me
    else:
        fof = None

    return Snapshot(snap, fof)


@T.overload
def load_series(
    *snapshots: str | Path,
    expand_glob: T.Literal[False],
    kind: T.Optional[SimulationType] = None,
) -> Timeseries: ...
@T.overload
def load_series(
    snapshot: str,
    /,
    *,
    expand_glob: T.Literal[True],
    kind: T.Optional[SimulationType] = None,
) -> Timeseries: ...
def load_series(*snapshots, expand_glob, kind=None) -> Timeseries:
    if expand_glob:
        assert isinstance(snapshots[0], str)
        snapshot_list = glob(snapshots[0])
    else:
        snapshot_list = list(snapshots)
    return Timeseries(*(load_snapshot(fname, kind) for fname in snapshot_list))


def load_camels(
    simulation: CamelsSimulationType,
    run: OneP | CV | LH,
    number: int,
    volume: Volume = 25,
    subfind: bool = False,
):
    path = Path("/mnt/ceph/users/camels/PUBLIC_RELEASE/Sims")
    path /= simulation
    match volume:
        case 25:
            volume_str = "L25n256"
        case 50:
            volume_str = "L50n512"
    path /= volume_str
    match run:
        case OneP():
            path = path / "1P" / str(run)
        case CV():
            path = path / "CV" / str(run)
        case LH():
            path = path / "LH" / str(run)

    match simulation:
        case "SIMBA" | "IllustrisTNG":
            # fmt:off
            allowed_numbers = [
                14, 18, 24, 28, 32, 34, 36, 38, 40,
                42, 44, 46, 48, 50, 52, 54, 56, 58,
                60, 62, 64, 66, 68, 70, 72, 74, 76,
                78, 80, 82, 84, 86, 88, 90,
            ]
            hint = "Gizmo"
        case "Astrid":
            allowed_numbers = range(0, 91, 2)
            hint = "Gizmo"
        case "Swift-EAGLE":
            allowed_numbers = range(0, 91)
            hint = "SWIFT"
    if number not in allowed_numbers:
        raise ValueError(
            f"Snapshot {number} does not exist for Simulation {simulation}\nExisting snapshots: {allowed_numbers}"
        )
    path /= f"snapshot_{number:03}.hdf5"

    snap = yt.load(path, hint=hint)  # type: ignore yt.load DOES exist please trust me
    if subfind:
        fof = yt.load(path.parent / f"groups_{number:03}.hdf5", hint="GadgetFOF")  # type: ignore yt.load DOES exist please trust me
    else:
        fof = None
    return Snapshot(snap, fof)
