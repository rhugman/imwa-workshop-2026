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

def prep_bins(dest_path, src_path=None, get_only=[]):
    """copy the executables from the bin folder to the destination folder
    Parameters
    ----------
    dest_path : str
        path to the destination folder
    src_path : str
        path to the source folder. If None (default), resolve 'bin' relative to
        this module's location so the call works regardless of the caller's cwd.
    get_only : list
        list of executables to copy
    Returns
        -------
        None"""

    # Resolve the bin path relative to herebedragons.py (not the caller's cwd) so
    # batch scripts run from a repo root still find the committed executables.
    if src_path is None:
        src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bin')

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
    return fig, ax


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
        # print_input=True,
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
        # print_input=True,
        auxiliary='tracer', boundnames=True,
    )
    wel.set_all_data_external()
    return wel


def specify_tsf_cells(col_center=40, row_center=30, half_ncol=3, half_nrow=4, conc=1.0):
    """Return CNC-format list [((layer, row, col), conc), ...] for the TSF footprint.

    Default location is northwest of the pit — downgradient (between the pit and the GDE),
    so AMD migrates westward toward the GDE under natural flow; the pit's cone of depression
    can pull it back east during dewatering.
    """
    cells = []
    for row in range(row_center - half_nrow, row_center + half_nrow ):
        for col in range(col_center - half_ncol, col_center + half_ncol ):
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

    #save abs min pH over sim time
    np.savetxt(os.path.join(wd, "_min_ph.txt"), out.min())
    return fname, out

def proc_chem_at_wells(wd='.', wel_pnames=('wel-dewater', 'wel-mar'), prefix='chemwell'):
    """Per-species well-cell water chemistry, one wide CSV per quantity.

    Writes `<prefix>_<species>.csv` with columns named `<species>_i:<i>_j:<j>`
    (one per well cell, 0-based) and time as the index; returns {species: DataFrame}.

    The dissolved-species molalities span many orders of magnitude, so they are written
    as log10 of the value (floored at the 1e-12 mol truncation) -- history matching then
    conditions in log space. pH is already a log quantity and is written untransformed.
    """
    sim  = flopy.mf6.MFSimulation.load(sim_ws=wd, verbosity_level=0)
    gwf  = sim.get_model('gwf')
    sout = pd.read_csv(os.path.join(wd, 'sout.csv'))

    frames = []

    for p in wel_pnames:
        spd = gwf.get_package(p).stress_period_data.get_data()[1]
        ids = pd.DataFrame({
            'ij':  [f"i:{r}_j:{c}" for (_, r, c) in spd['cellid']],
            'row': [r + 1 for (_, r, c) in spd['cellid']],
            'col': [c + 1 for (_, r, c) in spd['cellid']],
        })
        frames.append(sout.merge(ids, on=['row', 'col']))
    well = pd.concat(frames, ignore_index=True)

    locators     = {'time_d', 'cell', 'layer', 'row', 'col', 'ij'}
    species_cols = [c for c in well.columns if c not in locators]
    def short(col):
        for pre in ('solution_total_molality_', 'solution_', 'equilibrium_phases_'):
            if col.startswith(pre):
                col = col[len(pre):]
                break
        return col.lower().replace('(', '').replace(')', '')

    sel_cols = [c for c in species_cols if short(c) in ['ph', 'ca', 's6', 'k', 'fe2', 'fe3',
                                                 'cl', 'ca', 'c4', 'al', 'mg', 'na']]

    out = {}
    for col in sel_cols:
        name = short(col)
        df = well.pivot_table(index='time_d', columns='ij', values=col)
        df = df.mask(df >= 1e29)
        # molalities span orders of magnitude, so condition on log10(value): a relative error
        # in native space becomes a constant, symmetric one in log space. Floor at the 1e-12 mol
        # PHREEQC truncation so absent species can't drive the log to -inf. pH is already log.
        if name != 'ph':
            df = np.log10(df.clip(lower=1e-12))
        df.columns = [f"{name}_{ij}" for ij in df.columns]
        df.index.name = 'time'
        df.to_csv(os.path.join(wd, f'{prefix}_{name}.csv'))
        out[name] = df
    return out


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
                                dsp_par =  ['alh'], 
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


def apply_mar_treatment(ws='.', treat_file='treat.dat',
                        wel_file='mar.wel_stress_period_data_2.txt'):
    """Forward-run pre-processor: rewrite the MAR injection chemistry by the `treat` decision.

    The injected per-component totals on every MAR well cell are set to a linear blend of two
    end-members, `(1 - treat) * untreated + treat * treated`, where `treat` (in [0, 1]) is read
    from `treat_file`: 0 = inject the untreated (acidic) make-up water, 1 = inject the fully
    treated (buffered) water. The cell ids, the PEST-parameterised injection rate, and the
    boundnames are left untouched. mf6rtm re-equilibrates through PHREEQC every step, so blending
    the injected totals and letting PHREEQC speciate is the lever that buffers (or not) the
    reinjected plume — the same principle as the background-chemistry priors. Registered with
    `is_pre_cmd=True`, it runs after the rate is written and before `mf6rtm`.

    The two end-members are per-component totals (mol/m3) in the order of the WEL `auxiliary`
    block (H2O H O Charge Al C Ca Cl Fe K Mg Na S). UNTREATED is solution 3 in
    02-reactive-transport.ipynb (the acidic make-up water FloPy writes into the MAR WEL file).
    TREATED is the commented limestone/lime-treated end-member from the same `solutionsdf`
    (pH 8.2, ~6 meq/L bicarbonate alkalinity, lime Ca, sulfate/Fe/Al stripped), equilibrated
    through PhreeqcRM with the model's own database and settings (componenth2o=True,
    add_charge_flag='Ca', charge_offset=1e-12); that derivation reproduces the untreated totals to
    ~1e-9, so both end-members are on the same footing. Self-contained so `add_py_function` can
    lift it whole into forward_run.py.
    """
    import shlex
    naux = 13   # H2O H O Charge Al C Ca Cl Fe K Mg Na S
    untreated = np.array([55342.0856, 2.39952776, 166.405099, 8.82387613e-13, 3.37000539,
                          2.04393818, 10.2250192, 0.352953227, 24.0539618, 0.634119356,
                          0.993054841, 1.38588979, 39.9814248])
    treated   = np.array([5.5342085481e+04, 5.7541156389e+00, 1.8674203633e+01, 1.9693670085e-09,
                          9.9704301202e-06, 5.9822580722e+00, 2.4117137617e+00, 9.9704301202e-01,
                          1.0967473132e-04, 4.9852150601e-02, 2.9911290465e-01, 1.9940860240e+00,
                          1.9940860242e-01])

    # treat fraction (default 0 = untreated if the file is absent, so this is a safe no-op on
    # templates without a treatment parameter). PEST may write the direct value as 'treat,<v>'
    # or quote-padded ('treat "   <v>"'), so tokenise quote-aware and take the last number.
    treat = 0.0
    tpath = Path(ws, treat_file)
    if tpath.exists():
        for tok in shlex.split(tpath.read_text().replace(',', ' ')):
            try:
                treat = float(tok)
            except ValueError:
                continue
    treat = float(min(1.0, max(0.0, treat)))
    blend = (1.0 - treat) * untreated + treat * treated

    fpath = Path(ws, wel_file)
    out_lines = []
    for line in fpath.read_text().splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        # row layout: layer row col rate <13 aux concentrations> boundname. The rate is a direct
        # PEST parameter that can arrive quoted with internal padding (e.g. '"      7.6E+02"'), so
        # tokenise with shlex (quote-aware) and strip the rate -- a bare split() would shatter the
        # quoted field, corrupt the row, and break the MAR WEL package.
        toks = shlex.split(line.replace(',', ' '))
        if len(toks) < 4 + naux + 1:
            out_lines.append(line)             # unexpected layout -> leave untouched
            continue
        head = toks[:3] + [toks[3].strip()]    # layer, row, col, rate (de-padded, unquoted)
        name = toks[4 + naux:]                 # boundname(s), kept as-is
        aux  = [f'{v:.8e}' for v in blend]
        out_lines.append('  ' + '  '.join(head + aux + name))
    fpath.write_text('\n'.join(out_lines) + '\n')
    print(f'apply_mar_treatment: treat={treat:.4f} -> blended MAR injection chemistry '
          f'(Fe {blend[8]:.3g}, S {blend[12]:.3g}, C {blend[5]:.3g} mol/m3)')
    return treat


def clean_model_outputs(ws='.'):
    """Forward-run pre-processor: delete the previous run's model outputs before `mf6rtm`.

    PANTHER workers are reused dir-by-dir, so a worker keeps the *previous* run's output files.
    If a forward run's `mf6rtm` fails (or never runs), the post-processors and PEST instruction
    files would otherwise read those STALE outputs and report a bogus 'success' — silently
    poisoning the response matrix / ensemble with the previous run's values (this is what made
    failed optimisation runs look like ~1 s 'completed' runs). Wiping the outputs up front means
    a failed run has nothing to read and so fails loudly. Registered `is_pre_cmd=True`, it runs
    last among the pre-commands, just before `mf6rtm`. Only model *outputs* are removed (raw MF6
    output, PHREEQC selected output, and the post-processor CSVs PEST reads) — never inputs.
    Self-contained so `add_py_function` can lift it whole into forward_run.py.
    """
    import glob
    patterns = [
        'sout.csv',                                  # PHREEQC selected output (pH/chem post-procs)
        'gwf.hds', 'gwf.lst', 'gwf.cbc', 'gwf.cbb',  # MF6 GWF binary/list output
        'mfsim.lst', '*.ucn',                        # sim listing + GWT concentration binaries
        'gwf.obs*.csv',                              # MF6 OBS output CSVs read directly by PEST
        'dewater.autoreduce.csv',                    # AUTO_FLOW_REDUCE output
        'gde_ph.csv', '_min_ph.txt',                 # proc_ph_at_gde outputs
        'chemwell_*.csv',                            # proc_chem_at_wells outputs
        'hdslay*.txt', 'inc.csv', 'cum.csv',         # extract_hds_arrays_and_list_dfs outputs
        'mar_throughput.dat',                        # proc_mar_throughput output (07-mou)
        'gde_ph_min.dat',                            # proc_gde_ph_min output (07-mou)
    ]
    removed = 0
    for pat in patterns:
        for f in glob.glob(os.path.join(ws, pat)):
            try:
                os.remove(f)
                removed += 1
            except OSError:
                pass
    print(f'clean_model_outputs: removed {removed} stale output file(s)')
    return removed


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


def proc_mar_throughput(wd='.', treat_file='treat.dat', inc_file='inc.csv',
                        totim=3651, fname='mar_throughput.dat'):
    """Forward-run post-processor: treated-MAR throughput = treat * total MAR injection.

    This is the quantity 07-mou prices as its 'treatment' objective. `treat` (in [0, 1]) is the
    same global treatment fraction `apply_mar_treatment` consumes; we read it from `treat.dat` the
    identical quote-aware way (PEST may write it padded/quoted). The total MAR injection is the
    realised `wel2` listing-budget term in `inc.csv` at end-of-mining (`totim` 3651) — the second
    WEL package is the MAR injectors, so its budget term is `wel2`. We deliberately use the realised
    budget (after any head-dependent clipping / auto-flow-reduce) rather than the requested
    `marwelgr` rate sum, because `wel2` is what the model actually injected. `wel2` is positive
    (injection), so throughput >= 0 and the objective is a straight minimisation, no sign flip.

    Writes one bare number to `mar_throughput.dat` so it can be a single PEST observation
    (`mar_throughput`). MUST run AFTER `extract_hds_arrays_and_list_dfs` (which writes `inc.csv`).
    Self-contained so `add_py_function` can lift it whole into forward_run.py.
    """
    import shlex
    # treat fraction (default 0 = untreated if the file is absent -> safe no-op on a template
    # without a treatment parameter). Tokenise quote-aware and take the last parseable number,
    # exactly as apply_mar_treatment does, then clamp to [0, 1].
    treat = 0.0
    tpath = os.path.join(wd, treat_file)
    if os.path.exists(tpath):
        with open(tpath) as f:
            txt = f.read().replace(',', ' ')
        for tok in shlex.split(txt):
            try:
                treat = float(tok)
            except ValueError:
                continue
    treat = float(min(1.0, max(0.0, treat)))

    inc = pd.read_csv(os.path.join(wd, inc_file))
    mar_flux = float(inc.loc[inc['totim'].astype(float) == float(totim), 'wel2'].values[0])
    throughput = treat * mar_flux

    with open(os.path.join(wd, fname), 'w') as f:
        f.write(f'{throughput:.8e}\n')
    print(f'proc_mar_throughput: treat={treat:.4f} * wel2={mar_flux:.3g} -> '
          f'throughput={throughput:.6g}')
    return throughput


def proc_gde_ph_min(wd='.', gde_file='gde_ph.csv',
                    times=(1, 3651, 6571, 9491, 12411, 15331, 18251),
                    fname='gde_ph_min.dat'):
    """Forward-run post-processor: worst-case GDE pH = min(ph_min) over the output times.

    07-mou treats GDE pH as an objective to MAXIMISE rather than a hard constraint, so it needs a
    single scalar summarising the whole simulation: the lowest pH the GDE drain footprint ever sees.
    `proc_ph_at_gde` already wrote `gde_ph.csv` with one row per sout output time (columns `time`,
    `ph_min`, where `ph_min` is the minimum pH across the GDE drain cells at that time). Here we take
    the MIN of `ph_min` over the requested output `times` (PH_TIMES) — the single most-acidic
    moment-and-place over the run. Raising this number is the management goal, so as an objective it
    is a straight maximisation, no sign flip.

    Writes one bare number to `gde_ph_min.dat` so it can be a single PEST observation
    (`gde_ph_min`). MUST run AFTER `proc_ph_at_gde` (which writes `gde_ph.csv`). If the input file is
    absent (e.g. a failed run after clean_model_outputs wiped it) we write a safe default of 0.0 — a
    failed-low pH that loses on a maximisation objective rather than silently passing. Self-contained
    so `add_py_function` can lift it whole into forward_run.py.
    """
    gde_path = os.path.join(wd, gde_file)
    if not os.path.exists(gde_path):
        ph_min = 0.0
        with open(os.path.join(wd, fname), 'w') as f:
            f.write(f'{ph_min:.8e}\n')
        print(f'proc_gde_ph_min: {gde_file} absent -> wrote safe default {ph_min:.6g}')
        return ph_min

    gde = pd.read_csv(gde_path)
    # keep only the requested output times, then take the overall worst (lowest) pH
    times = [float(t) for t in times]
    sel = gde.loc[gde['time'].astype(float).isin(times), 'ph_min']
    ph_min = float(sel.min())

    with open(os.path.join(wd, fname), 'w') as f:
        f.write(f'{ph_min:.8e}\n')
    print(f'proc_gde_ph_min: min(ph_min) over {len(sel)} time(s) -> ph_min={ph_min:.6g}')
    return ph_min


def test_extract_hds_arrays(d):
    """
    Test the extraction of head data arrays.
    """
    cwd = os.getcwd()
    os.chdir(d)
    extract_hds_arrays_and_list_dfs()
    os.chdir(cwd)
