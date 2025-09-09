import matplotlib.pyplot as plt

from ssut_plots.data.loaders import load_camels_run, load_swimba_run
from ssut_plots.plots.multipanel import MultipanelFigure
from ssut_plots.types import OneP


data = load_swimba_run(
    "/mnt/ceph/users/ssutherland/SWIMBA/SWIMBA/1P/1P_p1_0", subfind=True, snapshots=90
)
data2 = load_camels_run("SIMBA", OneP(1, 0), subfind=True, snapshots=90)

plt.rc("font", size=18)

fig: MultipanelFigure = plt.figure(FigureClass=MultipanelFigure)  # type: ignore type checkers always think this returns a plain Figure

fig.plot_sim(data, label="SWIMBA", color="r")
fig.plot_sim(data2, label="SIMBA", color="b")
if (ax1 := fig.multipanel_get_axes(1)) is not None:
    ax1.legend()
fig.tight_layout()

fig.savefig("multipanel.pdf")
