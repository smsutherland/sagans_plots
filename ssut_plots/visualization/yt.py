import yt


def projection(snap, out):
    ds = yt.load(snap)
    yt.ProjectionPlot(ds, "z", ("gas", "density")).save(out)
