import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path
from shutil import which
from sys import stderr

import numpy as np
import yt
from matplotlib.colors import LogNorm
from matplotlib.pyplot import imsave
from tqdm import tqdm


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def movie(snaps, output, framerate=8):
    if which("ffmpeg") is None:
        print("Error: ffmpeg not found in PATH", file=stderr)
        return

    images = []
    with Pool(64) as p:
        for i, im in tqdm(
            enumerate(
                p.imap(
                    mk_image,
                    snaps,
                )
            ),
            total=len(snaps),
        ):
            tqdm.write(f"{bcolors.OKGREEN}Imaging{bcolors.ENDC} {snaps[i]}")
            images.append(im)

    images = np.array(images)
    mk_movie(images, output, framerate=framerate)


def mk_image(snap: Path):
    data = yt.load(snap)
    plot = yt.ProjectionPlot(
        data, "z", ("gas", "temperature"), weight_field=("gas", "density")
    )
    return plot.plots[("gas", "temperature")].image.get_array()


def mk_movie(images, output, framerate):
    tempdir = tempfile.mkdtemp()

    vmax = images.max()
    vmin = images.min()
    if vmin == 0:
        vmin = vmax / 1e6
    norm = LogNorm(vmax=vmax, vmin=vmin)
    for i, im in tqdm(enumerate(images), total=images.shape[0]):
        tqdm.write(f"{bcolors.OKGREEN}Saving{bcolors.ENDC}  {tempdir}/out_{i}.png")
        imsave(
            f"{tempdir}/out_{i}.png",
            norm(im),
            cmap="viridis",
        )
    command = [
        "ffmpeg",
        "-framerate",
        str(framerate),
        "-i",
        f"{tempdir}/out_%d.png",
        str(output),
        "-y",
    ]
    print(" ".join(command))
    subprocess.run(command)
