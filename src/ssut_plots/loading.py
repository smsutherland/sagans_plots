import typing as T
from pathlib import Path

import yt
import yt.data_objects.static_output

from .types import (
    CamelsSimulationType,
    SimulationType,
    Volume,
    OneP,
    CV,
    LH,
)


def load_snapshot(
    fname: str | Path, kind: T.Optional[SimulationType]
) -> yt.data_objects.static_output.Dataset:
    match kind:
        case None:
            hint = None
        case "SIMBA":
            hint = "Gizmo"
        case "SWIMBA":
            hint = "SWIFT"

    return yt.load(fname, hint=hint)  # type: ignore yt.load DOES exist please trust me


def load_series(
    *snapshots,
    redshifts: T.Optional[T.List[float]] = None,
    snapshot_nums: T.Optional[T.List[int]] = None,
):
    raise RuntimeError("TODO: load series of snapshots")


def load_camels(
    simulation: CamelsSimulationType,
    run: OneP | CV | LH,
    number: int,
    volume: Volume = 25,
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
    return yt.load(path, hint=hint)  # type: ignore yt.load DOES exist please trust me
