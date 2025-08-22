from ssut_plots.data.loaders import load_swimba_run
from ssut_plots.plots.sfrd import SfrPlot, SfrdProjection
import matplotlib.pyplot as plt

data = load_swimba_run("path/to/some/run")

fig = plt.figure()
ax: SfrPlot = fig.add_subplot( # type: ignore type checkers think this always returns a plain Axes
    111,
    projection=SfrdProjection(
        cosmology=data.cosmology, primary_axis="z", secondary_axis="lookback"
    ),
)

ax.plot_sfrd(data)
