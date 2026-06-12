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
from pathlib import Path

delr = 20

def tidy_array(fpath):
    # read unordered txt file
    with open(fpath, 'r') as f:
        data = f.read().split()
    data = [float(x) for x in data]
    arr = np.array(data)
    arr = arr.flatten()
    #arr = arr.reshape(sr.ncpl)
    np.savetxt(fpath, arr, fmt='%1.6e')
    return

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

    nper = sim.tdis.nper.get_data()
    names = [f"dewater{i}" for i in range(len(neighbour_lrc))]

    # SP 1: dewatering active; SP 2+ wells off
    spd = {1: [(tuple(cell), rate, 0.0, nm) for cell, nm in zip(neighbour_lrc, names)]}
    for kper in range(2, nper):
        spd[kper] = [(tuple(cell), 0.0, 0.0, nm) for cell, nm in zip(neighbour_lrc, names)]

    # print_input must stay OFF: PRINT_INPUT + external list files triggers heap
    # corruption in libmf6's table writer (bndext write_list) and random aborts
    wel = flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=spd,
        pname='wel-dewater', filename='dewater.wel',
        auxiliary='tracer', boundnames=True,
        auto_flow_reduce=0.1,
        print_input=True,
        afrcsv_filerecord='dewater.autoreduce.csv'
    )
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

    nper = sim.tdis.nper.get_data()
    names = [f"mar{i}" for i in range(len(mar_cells))]

    # SP 1: MAR active; SP 2+ wells off
    spd = {1: [(cell, rate, 0.0, nm) for cell, nm in zip(mar_cells, names)]}
    for kper in range(2, nper):
        spd[kper] = [(cell, 0.0, 0.0, nm) for cell, nm in zip(mar_cells, names)]

    # print_input must stay OFF — see note in build_wells_dewater
    wel = flopy.mf6.ModflowGwfwel(
        gwf, stress_period_data=spd,
        pname='wel-mar', filename='mar.wel',
        print_input=True,
        auxiliary='tracer', boundnames=True,
    )
    wel.set_all_data_external()
    return wel


def specify_tsf_cells(col_center=20, row_center=45, half_ncol=2, half_nrow=3, conc=1.0):
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
    arr[arr >= 1e29] = np.nan   # PhreeqcRM inactive-cell fill (e.g. no-react pit cells)
    return arr


def proc_ph_at_gde(wd='.', drn_pname='drn-gde', fname='gde_ph.csv'):
    """Minimum pH across the GDE drain footprint per sout output time.

    The management criterion for the tutorials is 'pH at the GDE stays
    above the threshold', which is equivalent to min(pH) over all drain
    cells staying above it. Writes the result to `ws/gde_ph.csv` so it
    can serve as a PEST observation file (one row per output time).

    Parameters
    ----------
    ws   : str
        model folder containing sout.csv; the output CSV is written here
    gwf  : flopy GWF model holding the DRN package
    drn_pname : str
        name of the GDE drain package
    fname : str
        name of the output CSV written into ws

    Returns
    -------
    DataFrame indexed by time_d with columns:
        ph_min      — minimum pH over the drain cells
        row, col    — location of the minimum (the 'worst' cell)
    """
    sim = flopy.mf6.MFSimulation.load(sim_ws=wd, verbosity_level=0)
    # load flow model
    gwf = sim.get_model('gwf')
    sout  = pd.read_csv(os.path.join(wd, 'sout.csv'))
    spd   = gwf.get_package(drn_pname).stress_period_data.get_data()[0]
    # sout.csv row/col are 1-based; flopy cellids are 0-based
    cells = pd.DataFrame([(r + 1, c + 1) for (_, r, c) in spd['cellid']],
                         columns=['row', 'col'])
    gde   = sout.merge(cells, on=['row', 'col'])
    idx   = gde.groupby('time_d')['solution_ph'].idxmin()
    out   = (gde.loc[idx, ['time_d', 'solution_ph']]
                .rename(columns={'solution_ph': 'ph_min',
                                 'time_d': 'time'
                                 })
                .set_index('time'))
    out.to_csv(os.path.join(wd, fname))
    return fname, out


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

def get_input_filenames(tag, template_ws=os.path.join('pest','pst_template'), extension='.txt', startswith = False):
    """
    Get the input filenames from the template workspace

    Parameters:
        tag: str, tag to search for
        template_ws: str, template workspace
    """
    if startswith:
        files = [
            f for f in os.listdir(template_ws)
            if f.lower().startswith(tag) and f.endswith(extension)
            ]   
    
    else:
        files = [
            f for f in os.listdir(template_ws)
            if tag in f.lower() and f.endswith(extension)
            ]
    files = sorted(files, key=extract_layer_number)
    return files 

def extract_layer_number(filename):
    import re
    match = re.search(r'layer(\d+)', filename)
    return int(match.group(1)) if match else float('inf')

def copy_parameterized_transport_files(ws=".",
                                parameterized_species="h",
                                dsp_par =  [], 
                                mst_par = ['porosity']
                                 ):

    def flatten(xss):
        return [x for xs in xss for x in xs]
    
    sim = flopy.mf6.MFSimulation.load(sim_ws=ws, verbosity_level=0)
    species = sim.model_names[1:]
    species.remove(parameterized_species)

    tag = []

    for e, par in enumerate(dsp_par):
        tag.append(f"dsp_{par}_")

    for  e, par in enumerate(mst_par):
         tag.append(f"mst_{par}_")
         
    fnames_to_copy = [get_input_filenames(f"{parameterized_species}.{t}", template_ws=ws, startswith=True) for t in tag]
    fnames_to_copy = flatten(fnames_to_copy)

    print(
        f"Warning: copying files with tag: {', '.join(tag)} from species: {parameterized_species.upper()} to the following species: "
        f"{', '.join(sp.capitalize() for sp in species if sp != parameterized_species)}"
    )

    for sp in species:
        fnames_to_replace = [get_input_filenames(f"{sp}.{t}", template_ws=ws, startswith=True) for t in tag]
        fnames_to_replace = flatten(fnames_to_replace)
        assert sorted([x.split('.')[1] for x in fnames_to_copy]) == sorted([x.split('.')[1] for x in fnames_to_replace]), f'list of files to replace and to copy does not contain the same files names '
        # sort fnames_to_copy and fnames_to_replace according to the assert above
        fnames_to_copy = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_copy], fnames_to_copy))]
        fnames_to_replace = [x for _, x in sorted(zip([x.split('.')[1] for x in fnames_to_replace], fnames_to_replace))]
        fileszipped = list(zip(fnames_to_copy, fnames_to_replace))
        [shutil.copyfile(Path(ws, f[0]), Path(ws, f[1])) for f in fileszipped];
    return fileszipped

def extract_hds_arrays_and_list_dfs():
    """
    Extract head data arrays and list budget dataframes from MODFLOW output files.
    """

    hds = flopy.utils.HeadFile("gwf.hds")
    for it,t in enumerate(hds.get_times()):
        d = hds.get_data(totim=t)
        for k,dlay in enumerate(d):
            np.savetxt("hdslay{0}_t{1}.txt".format(k+1,it+1),d[k,:,:],fmt="%15.6E")

    lst = flopy.utils.Mf6ListBudget("gwf.lst")
    inc,cum = lst.get_dataframes(diff=True,start_datetime=None)
    inc.columns = inc.columns.map(lambda x: x.lower().replace("_","-"))
    cum.columns = cum.columns.map(lambda x: x.lower().replace("_", "-"))
    inc.index.name = "totim"
    cum.index.name = "totim"
    #one lil trick to help with opt later
    rd = pd.read_csv("dewater.autoreduce.csv")
    summed = rd.groupby("time").sum()["wel-reduction"].to_dict()
    inc["abstotwel"] = np.abs(inc["wel"]) + np.abs(inc["wel2"])
    cum["abstotwel"] = np.abs(cum["wel"]) + np.abs(cum["wel2"])
    inc["totwel"] = inc["wel"] + inc["wel2"]
    cum["totwel"] = cum["wel"] + cum["wel2"]
    inc["wel-reduction"] = 0.0
    for t,v in summed.items():
        inc.loc[t,"wel-reduction"] = v
    inc.to_csv("inc.csv")
    cum.to_csv("cum.csv")
    return

def test_extract_hds_arrays(d):
    """
    Test the extraction of head data arrays.
    """
    cwd = os.getcwd()
    os.chdir(d)
    extract_hds_arrays_and_list_dfs()
    os.chdir(cwd)
