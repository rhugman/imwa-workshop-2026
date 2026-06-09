import os
import shutil
import platform
import flopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from flopy.utils.gridintersect import GridIntersect
from shapely.geometry import Polygon

delr = 20

def prep_bins(dest_path, src_path=os.path.join('bin'), get_only=[]):
    """copy the executables from the bin folder to the destination folder
    Parameters
    ----------
    dest_path : str
        path to the destination folder
    src_path : str
        path to the source folder
    get_only : list
        list of executables to copy
    Returns
        -------
        None"""

    if "linux" in platform.platform().lower():
        bin_path = os.path.join(src_path, "linux")
    elif "darwin" in platform.platform().lower() or "macos" in platform.platform().lower():
        bin_path = os.path.join(src_path, "mac")
    else:
        bin_path = os.path.join(src_path, "win")
    files = os.listdir(bin_path)
    if len(get_only) > 0:
        files = [f for f in files if f.split(".")[0] in get_only]
    for f in files:
        if os.path.exists(os.path.join(dest_path, f)):
            try:
                os.remove(os.path.join(dest_path, f))
            except:
                continue
        shutil.copy2(os.path.join(bin_path, f), os.path.join(dest_path, f))
    return


def plot_setup(gwf, pitcells, kper=1, show_hk=False, gw_level_locs=None, tsf_cells=None):

    width_max = 174 / 25.4  # mm to inch
    height_max = 234 / 25.4  # mm to inch
    fig = plt.figure(figsize=(width_max * .75, height_max / 2),
                     constrained_layout=True)
    ax = fig.add_subplot()
    ax.set_aspect("equal")
    handles = []
    mapview = flopy.plot.PlotMapView(model=gwf, layer=0)
    linecollection = mapview.plot_grid(alpha=0.1)

    if show_hk:
        a = gwf.npf.k.array[0, :, :]
        quadmesh = mapview.plot_array(np.log10(a), alpha=0.4)
        cb = plt.colorbar(quadmesh, shrink=0.45, orientation="horizontal",
                          label="log Hydraulic conductivity (m/day)")

    for pck, c, nme in zip(
        ['drn-gde', 'ghb-inflow', 'wel-dewater', 'wel-mar'],
        ["cyan", "b", "r", "k"],
        ['Groundwater-\ndependent\necosystem (DRN)', 'Inflow (GHB)',
         'Dewatering\nwells (WEL)', 'Reinjection\nwells (WEL)'],
    ):
        try:
            mapview.plot_bc(pck, kper=kper, color=c)
            handles.append(plt.Line2D([0], [0], marker='s', color='w',
                                      markerfacecolor=c, markersize=10, label=nme))
        except:
            pass

    mg = gwf.modelgrid
    centx = mg.xcellcenters
    centy = mg.ycellcenters
    pitcells_x = [centx[i, j] for (i, j) in pitcells]
    pitcells_y = [centy[i, j] for (i, j) in pitcells]
    xmin, xmax = min(pitcells_x), max(pitcells_x)
    ymin, ymax = min(pitcells_y), max(pitcells_y)
    rect = patches.Rectangle(
        xy=(xmin - delr / 2, ymin - delr / 2),
        width=xmax - xmin + delr, height=ymax - ymin + delr,
        lw=1, edgecolor='k', facecolor='none', label="Pit",
    )
    plt.gca().add_patch(rect)
    handles.append(plt.Line2D([0], [0], marker='s', color='w',
                               markerfacecolor="w", markeredgecolor='k',
                               markersize=15, label="Pit"))

    if tsf_cells is not None:
        tsf_arr = np.array([c[0] for c in tsf_cells])   # (layer, row, col) tuples
        tsf_x = [centx[r, c] for (_, r, c) in tsf_arr]
        tsf_y = [centy[r, c] for (_, r, c) in tsf_arr]
        tsf_xmin, tsf_xmax = min(tsf_x), max(tsf_x)
        tsf_ymin, tsf_ymax = min(tsf_y), max(tsf_y)
        tsf_rect = patches.Rectangle(
            xy=(tsf_xmin - delr / 2, tsf_ymin - delr / 2),
            width=tsf_xmax - tsf_xmin + delr, height=tsf_ymax - tsf_ymin + delr,
            lw=1.5, edgecolor='navy', facecolor='steelblue', alpha=0.5,
        )
        plt.gca().add_patch(tsf_rect)
        handles.append(plt.Line2D([0], [0], marker='s', color='w',
                                   markerfacecolor='steelblue', markeredgecolor='navy',
                                   markersize=10, label='Tailings (TSF)'))

    if gw_level_locs is not None:
        ax.scatter(gw_level_locs.x.values, gw_level_locs.y.values,
                   marker="^", facecolor='w', edgecolor='k', s=15, linewidths=0.4)
        handles.append(plt.Line2D([0], [0], marker='^', color='w',
                                   markerfacecolor="w", markeredgecolor='k',
                                   markersize=10, label="Groundwater level\nobservations"))

    scale = 250  # m
    ax.set_xticks(np.arange(ax.get_xlim()[0], ax.get_xlim()[1], scale))
    ax.set_xticks(np.arange(ax.get_ylim()[0], ax.get_ylim()[1], scale))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis='x', which='both', top=True, direction='out')
    ax.tick_params(axis='y', which='both', right=True, direction='out')

    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    scalebar = AnchoredSizeBar(ax.transData, scale, f"{scale} m", 'lower right',
                               pad=0.1, color='white', frameon=False, size_vertical=1)
    plt.gca().add_artist(scalebar)

    mapview.ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    # plt.savefig("model_setup.pdf")
    # plt.close(fig)


def specify_pit_cells(gwf, pit_side_length):
    """Return (row, col) array of cells inside the pit polygon."""
    mg = gwf.modelgrid
    ix = GridIntersect(mg)
    xmin_g, xmax_g, ymin_g, ymax_g = mg.extent
    xmid, ymid = 0.5 * (xmin_g + xmax_g), 0.5 * (ymin_g + ymax_g)
    half = 0.5 * pit_side_length
    poly = Polygon([
        (xmid - half, ymid - half),
        (xmid - half, ymid + half),
        (xmid + half, ymid + half),
        (xmid + half, ymid - half),
    ])
    cells = np.array(ix.intersect(poly, 'polygon').cellids.tolist(), dtype=int)
    idomain = gwf.dis.idomain.array
    for i in cells:
        idomain[:, i[0], i[1]] = 2
    gwf.dis.idomain.set_data(idomain.tolist())
    return cells


def build_wells_dewater(sim, pitcells, rate):
    """Build WEL package for dewatering wells ringing the pit."""
    assert rate <= 0, "dewatering rate must be negative"
    gwf = sim.get_model()
    mg = gwf.modelgrid
    nodes = mg.get_node([(0, *i) for i in pitcells])
    neighbours = mg.neighbors()
    neighbour_nodes = list(set(n for nd in nodes for n in neighbours[nd]) - set(nodes))
    neighbour_lrc = mg.get_lrc(neighbour_nodes)

    tsnames = [f"dewater{i}" for i in range(len(neighbour_lrc))]
    tsmethods = ['stepwise'] * len(neighbour_lrc)
    tsdata = [(0.0, 0.0), (1.0, rate), (3650, 0.0), (1e30, 0.0)]

    spd = {1: [(cell, nm, nm) for cell, nm in zip(neighbour_lrc, tsnames)]}

    centx, centy = mg.xcellcenters, mg.ycellcenters
    df = pd.DataFrame(spd[1]).iloc[:, :-1].rename(columns={1: "name"})
    df[["k", "i", "j"]] = df.iloc[:, 0].apply(lambda x: pd.Series(x))
    df["x"] = [centx[i, j] for i, j in zip(df.i, df.j)]
    df["y"] = [centy[i, j] for i, j in zip(df.i, df.j)]
    # df.iloc[:, 1:].to_csv("location_dewater_wells.csv", index=False)

    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=spd,
                                   pname='wel-dewater', filename='dewater.wel',
                                   boundnames=True, auto_flow_reduce=0.1, print_input=True)
    wel.ts.initialize(filename='dewater0.ts', timeseries=tsdata,
                      time_series_namerecord=[tsnames[0]],
                      interpolation_methodrecord=[tsmethods[0]])
    for i in range(1, len(neighbour_lrc)):
        wel.ts.append_package(filename=f'dewater{i}.ts', timeseries=tsdata,
                              time_series_namerecord=[tsnames[i]],
                              interpolation_methodrecord=[tsmethods[i]])
    wel.ts.set_all_data_external()
    wel.set_all_data_external()
    return wel


def build_wells_mar(sim, pitcells, rate):
    """Build WEL package for managed aquifer recharge wells."""
    assert rate >= 0, "MAR rate must be positive"
    gwf = sim.get_model()
    mg = gwf.modelgrid
    mar_j = int(np.min(pitcells.T[1]) / 2)
    mar_skip = 10
    mar_cells = [(0, i, mar_j)
                 for i in range(mar_skip // 2, gwf.dis.nrow.data, mar_skip)]

    tsnames = [f"mar{i}" for i in range(len(mar_cells))]
    tsmethods = ['stepwise'] * len(mar_cells)
    tsdata = [(0.0, 0.0), (1.0, rate), (3650, 0.0), (1e30, 0.0)]

    spd = {1: [(cell, nm, nm) for cell, nm in zip(mar_cells, tsnames)]}

    centx, centy = mg.xcellcenters, mg.ycellcenters
    df = pd.DataFrame(spd[1]).iloc[:, :-1].rename(columns={1: "name"})
    df[["k", "i", "j"]] = df.iloc[:, 0].apply(lambda x: pd.Series(x))
    df["x"] = [centx[i, j] for i, j in zip(df.i, df.j)]
    df["y"] = [centy[i, j] for i, j in zip(df.i, df.j)]
    # df.iloc[:, 1:].to_csv("location_mar_wells.csv", index=False)

    wel = flopy.mf6.ModflowGwfwel(gwf, stress_period_data=spd,
                                   pname='wel-mar', filename='mar.wel',
                                   boundnames=True, print_input=True)
    wel.ts.initialize(filename='mar0.ts', timeseries=tsdata,
                      time_series_namerecord=[tsnames[0]],
                      interpolation_methodrecord=[tsmethods[0]])
    for i in range(1, len(mar_cells)):
        wel.ts.append_package(filename=f'mar{i}.ts', timeseries=tsdata,
                              time_series_namerecord=[tsnames[i]],
                              interpolation_methodrecord=[tsmethods[i]])
    wel.ts.set_all_data_external()
    wel.set_all_data_external()
    return wel


def specify_tsf_cells(col_center=65, row_center=30, half_ncol=2, half_nrow=1, conc=1.0):
    """Return CNC-format list [((layer, row, col), conc), ...] for the TSF footprint.

    Default location is northeast of the pit — upgradient so AMD migrates westward
    through the dewatering zone toward the GDE.
    """
    cells = []
    for row in range(row_center - half_nrow, row_center + half_nrow + 1):
        for col in range(col_center - half_ncol, col_center + half_ncol + 1):
            cells.append(((0, row, col), conc))
    return cells


def sout_to_array(df, col, idom):
    """Pivot a sout.csv time-slice into a 2-D array ready for PlotMapView.plot_array().

    Parameters
    ----------
    df   : DataFrame slice filtered to a single time (e.g. sout[sout['time_d'] == t])
    col  : column name from sout.csv (e.g. 'solution_ph')
    idom : 2-D idomain array (nrow, ncol) — inactive cells become NaN

    Returns
    -------
    arr  : (nrow, ncol) float64 array
    """
    arr = df.pivot(index='row', columns='col', values=col).values.astype(float)
    arr[idom <= 0] = np.nan
    return arr


def build_utlobs(gwf, pitcells):
    """Build OBS utility package for head monitoring at pit and MAR wells."""
    obs_layer = 0
    obs_pit = [(f"hds_pit_i:{r}_j:{c}", "HEAD", (obs_layer, r, c))
               for r, c in pitcells]
    obs_mar = [(f"hds_mar_i:{cell[1]}_j:{cell[2]}", "HEAD", (obs_layer, cell[1], cell[2]))
               for cell in gwf.get_package("wel-mar").stress_period_data.data[1].cellid]
    continuous = {
        f'{gwf.name}.obs.head.pit.csv': obs_pit,
        f'{gwf.name}.obs.head.wel-mar.csv': obs_mar,
    }
    return flopy.mf6.ModflowUtlobs(gwf, digits=10,
                                    filename=f"{gwf.name}.obs",
                                    pname="obs-head",
                                    continuous=continuous)
