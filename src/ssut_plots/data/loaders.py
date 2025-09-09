import typing as T
from glob import glob
from pathlib import Path

import yt
import yt.loaders

from ssut_plots.types import (
    CV,
    LH,
    CamelsSimulationType,
    OneP,
    SimulationType,
    Volume,
)

from .run import Run
from .sfr import SFRData
from .snapshot import Snapshot
from .timeseries import Timeseries


def load_snapshot(
    fname: str | Path,
    subfind: T.Optional[str | Path] = None,
    kind: T.Optional[SimulationType] = None,
) -> Snapshot:
    """
    Load a snapshot at the given path.

    Parameters
    ----------
    fname
        Path to the snapshot file.
    subfind
        If provided, loads the subfind catalog at the specified path alongside the snapshot.
        This is required for use in some plots, but not all.
    kind
        If provided, gives a hint to yt for what kind of snapshot is being loaded.

    Returns
    -------
    Snapshot
        Snapshot that has been loaded.
        If subfind is provided, the Snapshot will have a loaded subfind catalog as well.
    """
    match kind:
        case None:
            hint = None
        case "SIMBA":
            hint = "Gizmo"
        case "SWIFT":
            hint = "SWIFT"

    snap = yt.loaders.load(fname, hint=hint)
    if subfind is not None:
        fof = yt.loaders.load(
            subfind,
            hint="GadgetFOF",
            unit_base={
                "length": (1, "kpccm/h"),
            },
        )
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
def load_series(*snapshots, expand_glob=False, kind=None) -> Timeseries:
    """
    Load a collection of snapshots into a Timeseries.
    No subfind catalogs will be loaded with the snapshots (for now).

    Parameters
    ----------
    *snapshots
        Arguments of type `str` or Path.
        Each one will be loaded as a Snapshot and combined into a Timeseries.
        If expand_glob is True, only one snapshot can be provided, and must be of type `str`.
    expand_glob
        If True, interprets the provided string as a glob pattern.
        If False, the provided strings are taken as paths to the snapshots to be loaded.
        See documentation for glob.glob for more information.
    hint
        If provided, gives a hint to yt for what kind of snapshots are being loaded.

    Returns
    -------
    Timeseries
        Timeseries that has been loaded.
    """
    if expand_glob:
        assert isinstance(snapshots[0], str)
        snapshot_list = glob(snapshots[0])
    else:
        snapshot_list = list(snapshots)
    return Timeseries(*(load_snapshot(fname, kind) for fname in snapshot_list))


def _camels_path(
    simulation: CamelsSimulationType,
    run: OneP | CV | LH,
    volume: Volume,
) -> Path:
    """
    Resolve the path for a CAMELS simulation.

    Parameters
    ----------
    simulation
        Which simulation kind.
        Allowed values: 'SIMBA', 'Swift-EAGLE', 'IllustrisTNG', 'Astrid'
    run
        Which series of runs.
        Allowed values: `OneP`, `CV`, `LH`
    volume
        What size run.
        Allowed values: 25, 50
    """
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
    return path


def load_camels(
    simulation: CamelsSimulationType,
    run: OneP | CV | LH,
    number: int,
    volume: Volume = 25,
    subfind: bool = False,
) -> Snapshot:
    """
    Load a snapshot from CAMELS.

    Parameters
    ----------
    simulation
        Which simulation kind.
        Allowed values: 'SIMBA', 'Swift-EAGLE', 'IllustrisTNG', 'Astrid'
    run
        Which series of runs.
        Allowed values: `OneP`, `CV`, `LH`
    number
        Which snapshot number to load.
    volume
        What size run.
        Allowed values: 25, 50
    subfind
        If True, load the corresponding subfind catalog for the snapshot.
    """
    path = _camels_path(simulation, run, volume)
    allowed_numbers = _allowed_snapshot_numbers(simulation)
    match simulation:
        case "SIMBA" | "IllustrisTNG" | "Astrid":
            hint = "Gizmo"
        case "Swift-EAGLE":
            hint = "SWIFT"
    if number not in allowed_numbers:
        raise ValueError(
            f"Snapshot {number} does not exist for Simulation {simulation}\nExisting snapshots: {allowed_numbers}"
        )
    path /= f"snapshot_{number:03}.hdf5"

    snap = yt.loaders.load(path, hint=hint)
    if subfind:
        fof = yt.loaders.load(
            path.parent / f"groups_{number:03}.hdf5",
            hint="GadgetFOF",
            unit_base={
                "length": (1, "kpccm/h"),
            },
        )
    else:
        fof = None
    return Snapshot(snap, fof)


def load_camels_run(
    simulation: CamelsSimulationType,
    run: OneP | CV | LH,
    volume: Volume = 25,
    subfind: bool = False,
    snapshots: int | T.Iterable[int] | T.Literal["all"] = "all",
) -> Run:
    path = _camels_path(simulation, run, volume)

    if snapshots == "all":
        numbers = _allowed_snapshot_numbers(simulation)
    elif isinstance(snapshots, int):
        numbers = [snapshots]
    else:
        numbers = snapshots
    numbers = sorted(numbers)

    match simulation:
        case "SIMBA" | "IllustrisTNG" | "Astrid":
            hint = "Gizmo"
        case "Swift-EAGLE":
            hint = "SWIFT"

    snapshot_ds = (
        yt.loaders.load(f"{path}/snapshot_{n:03d}.hdf5", hint=hint) for n in numbers
    )
    if subfind:
        subfind_ds = (
            yt.loaders.load(
                f"{path}/groups_{n:03d}.hdf5",
                hint="GadgetFOF",
                unit_base={
                    "length": (1, "kpccm/h"),
                },
            )
            for n in numbers
        )
    else:
        subfind_ds = (None for _ in numbers)

    snaps = [Snapshot(snapshot, fof) for snapshot, fof in zip(snapshot_ds, subfind_ds)]
    timeseries = Timeseries(snaps)

    match simulation:
        case "SIMBA":
            sfr = SFRData(path / "extra_files" / "sfr.txt", hint="SIMBA")
        case "Swift-EAGLE":
            sfr = SFRData(path / "extra_files" / "SFR.txt", hint="SWIFT")
        case _:
            sfr = None

    return Run(timeseries, sfr)


def load_swimba_run(
    path: str | Path,
    subfind: bool = False,
    snapshots: int | T.Iterable[int] | T.Literal["all"] = "all",
) -> Run:
    path = Path(path)

    if snapshots == "all":
        numbers = range(0, 91)
    elif isinstance(snapshots, int):
        numbers = [snapshots]
    else:
        numbers = snapshots
    numbers = sorted(numbers)

    snapshot_ds = (
        yt.loaders.load(f"{path}/snaps/snapshot_{n:04d}.hdf5", hint="SWIFT")
        for n in numbers
    )
    if subfind:
        subfind_ds = (
            yt.loaders.load(
                fname,
                hint="GadgetFOF",
                unit_base={
                    "length": (1, "kpccm/h"),
                },
            )
            if (fname := path / f"snaps/subs/fof_subhalo_tab_{n:03d}.hdf5").exists()
            else None
            for n in numbers
        )
    else:
        subfind_ds = (None for _ in numbers)

    timeseries = Timeseries(
        Snapshot(
            snap,
            fof,
        )
        for snap, fof in zip(snapshot_ds, subfind_ds)
    )
    return Run(timeseries, SFRData(path / "SFR.txt", hint="SWIFT"))


def _allowed_snapshot_numbers(simulation: CamelsSimulationType) -> T.Iterable[int]:
    match simulation:
        case "SIMBA" | "IllustrisTNG":
            # fmt:off
            allowed =  [
                14, 18, 24, 28, 32, 34, 36, 38, 40,
                42, 44, 46, 48, 50, 52, 54, 56, 58,
                60, 62, 64, 66, 68, 70, 72, 74, 76,
                78, 80, 82, 84, 86, 88, 90,
            ]
        case "Astrid":
            allowed = range(0, 91, 2)
        case "Swift-EAGLE":
            allowed = range(0, 91)
    return allowed
