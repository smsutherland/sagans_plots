from ssut_plots.data.loaders import load_swimba_run, load_camels_run
from ssut_plots.plots.sfrd import SfrPlot, SfrdProjection
import matplotlib.pyplot as plt

from ssut_plots.types import OneP

data = load_swimba_run("/mnt/ceph/users/ssutherland/SWIMBA/final_stretch/post_final/new_master/18.sf_norm_more_supp_no_ent_floor/")
data2 = load_camels_run("SIMBA", OneP(1, 0))

fig = plt.figure()
ax: SfrPlot = fig.add_subplot(  # type: ignore type checkers think this always returns a plain Axes
    111,
    projection=SfrdProjection(
        cosmology=data.cosmology, primary_axis="z", secondary_axis="lookback"
    ),
)

ax.plot_sfrd(data)
ax.plot_sfrd(data2)
ax.set_ylim(3.1e-3, None)

fig.savefig("a.pdf")
