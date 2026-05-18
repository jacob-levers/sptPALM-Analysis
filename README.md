# FIREFLY

**Fluorescence Inference & Reconstruction Engine — Framework for Localization Yields**

A single-particle tracking PALM / dSTORM analysis pipeline for `.czi` (Zeiss)
and `.tif` / `.tiff` image stacks. Localisation, linking, MSD / diffusion /
motion-class analysis, JDD, dwell-time, MSS, DBSCAN clustering, redundant-
cross-correlation drift correction, turning-angle and radial-distribution
analysis, plus a multi-group comparison mode with statistical tests and a
multi-page PDF report.

Built with Python + PySide6 + napari. Localisation runs on the GPU via
PyTorch (Apple MPS or NVIDIA CUDA) and falls back to trackpy on CPU.

By Jacob Levers · macOS and Windows

---

## Table of contents

1. [Installation](#installation)
   - [Standalone app (recommended)](#standalone-app-recommended)
   - [From source (advanced)](#from-source-advanced)
2. [Quick tour](#quick-tour)
3. [Features](#features)
4. [Workflow](#workflow)
   - [Analyse a sample](#analyse-a-sample)
   - [Batch a folder](#batch-a-folder)
   - [Compare groups](#compare-groups)
   - [Visualise tracks](#visualise-tracks)
5. [Outputs](#outputs)
6. [Performance notes](#performance-notes)
7. [Troubleshooting](#troubleshooting)
8. [Acknowledgements](#acknowledgements)

---

## Installation

### Standalone app (recommended)

No Python installation required. Pre-built binaries are attached to each
release on the [Releases page](https://github.com/jacob-levers/FIREFLY/releases).

**macOS**

1. Download `FIREFLY-macOS.dmg` from the latest release.
2. Double-click the `.dmg` to mount it.
3. Drag `FIREFLY.app` into `/Applications`.
4. First launch: right-click → **Open** if macOS warns that the developer
   can't be verified, then **Open anyway**.

**Windows**

1. Download `FIREFLY-Windows.exe` from the latest release.
2. Double-click to launch. First launch unpacks bundled libraries to
   `%TEMP%` (~30 s); subsequent launches are instant.

> **GPU acceleration on Windows:** the bundled `FIREFLY-Windows.exe`
> ships **CPU-only** PyTorch. The CUDA-enabled torch wheel is ~2.5 GB
> on its own and pushes the .exe past GitHub Releases' 2 GiB asset
> cap, so we can't bundle it. If you have an NVIDIA GPU and want to
> run the localiser on it, follow **"From source (advanced)"** below
> and add the CUDA install step shown there. macOS Apple-Silicon
> users already get MPS acceleration from the bundled torch — no
> extra setup needed.

### From source (advanced)

Python 3.10 or newer. Clone the repo and run the launcher for your OS — it
sets up a virtual environment, installs dependencies, and starts the app.

```bash
git clone https://github.com/jacob-levers/FIREFLY.git
cd FIREFLY
```

**macOS** — double-click `Launch_FIREFLY.app`.

**Windows** — double-click `Launch_FIREFLY.bat`.

First launch opens a terminal showing pip installing PySide6, napari,
PyTorch, scipy and friends (~3–8 minutes). The GUI starts automatically
when the install finishes; subsequent launches skip the install and open
immediately.

**Enabling CUDA (Windows + NVIDIA GPU)**

The default `pip install torch` on Windows pulls the CPU-only wheel.
To get a CUDA-enabled torch, run *after* the first-launch install
finishes (still inside the project's virtual environment):

```powershell
.\venv\Scripts\activate
pip install --upgrade --index-url https://download.pytorch.org/whl/cu124 "torch>=2.3,<3"
```

Restart FIREFLY. The Analysis tab's backend dropdown will pick up
CUDA automatically; `Backend: torch (device: cuda)` should appear in
the log when a run starts. cu124 needs an NVIDIA driver ≥ R535
(Aug 2023); for older drivers swap `cu124` for `cu121` or `cu118`.

---

## Quick tour

The app opens on a welcome page with four action cards:

| Card | Workflow |
|---|---|
| **Analyse a sample** | Run the full pipeline on one `.czi` / `.tif` file |
| **Batch a folder** | Process every file in a folder, sequentially |
| **Compare groups** | Overlay 2–6 analysis-output folders into one figure |
| **Visualise tracks** | Open a previous run in an embedded napari viewer |

Once you pick a card the welcome page is replaced by the workflow tabs
(**Import / Analysis / Figures / Compare / Visualise**) plus a sidebar of
analysis parameters. The landing page is shown only at launch, not on
every tab switch.

---

## Features

### Detection & tracking

- **Trackpy or PyTorch backend**, auto-selected per platform — Apple MPS,
  NVIDIA CUDA, or CPU-trackpy. The auto-resolver prefers the GPU but
  drops back cleanly when it's unavailable.
- **Streaming chunked localisation** so large stacks (10⁴+ frames) don't
  need to live in RAM all at once. Each chunk's mass values stream into a
  live histogram on the Analysis tab so a bad threshold is obvious within
  seconds.
- **Live detection preview** during analysis — every preprocessed frame
  flows through a 60 FPS canvas with detected spots overlaid, so you can
  *watch* the pipeline at work.
- **Live preview viewer on the Import tab** — embedded napari viewer that
  auto-loads on file selection. Scrub frames, see detection circles
  colour-coded by integrated mass (turbo on log scale), toggle a
  bandpass-filtered view that shows what the detector actually sees, and
  overlay the auto/manual-threshold ROI mask in real time as you tweak it.
- **Per-file polygon ROIs** drawn directly in the preview viewer, saved
  automatically per file.

### Diffusion & motion analysis

- Per-track MSD with linear-LSQ fits for D and α.
- Motion classification (Immobile / Confined / Brownian / Directed) with
  configurable α thresholds.
- Jump-distance distribution with 1, 2, or 3 mobility populations.
- Mean-squared displacement scaling spectrum (MSS).
- Dwell-time survival curves with exponential τ fit.
- Turning-angle and signed-angle radial distributions.
- DBSCAN clustering of localisations with per-cluster area / density.

### Drift correction

- Redundant cross-correlation (RCC) drift correction (Wang et al. 2014).
- Solves the over-determined `drift[j] − drift[i] = Δᵢⱼ` system across
  every pair of time segments — robust to bad segments, redundancy
  averages out cross-correlation noise.

### ROI handling

Four modes:

- **None** — analyse the whole frame.
- **Auto threshold** — Li / Otsu / Triangle on the normalised mean
  projection.
- **Manual threshold** — drag a slider; the green mask overlay redraws
  live in the preview viewer.
- **Manual polygon** — draw a freehand polygon per file in the preview
  viewer; per-file polygons are remembered.

### Figures

- Single-sample combined figure with 15 panels (A–O): max projection,
  trajectories, MSD curves, log₁₀(D) distribution, motion-class
  breakdown, anomalous-exponent distribution, position density, mobile
  fraction over time, JDD with multi-population fit, cluster map,
  dwell-time histogram, MSS slope, radial distribution, and more.
- Multi-group comparison figure (10 panels: ensemble MSD, log₁₀(D)
  distribution, mobile fraction, motion-class fractions, track-length
  CDF, JDD overlay, dwell-time CDF, turning-angle distribution, radial
  distribution, MSD-AUC bar chart) with automatic statistical-test
  selection (Welch's t / Mann-Whitney U / one-way ANOVA / Kruskal-Wallis)
  and Bonferroni correction.
- Theme picker (**Dark / Light / Publication**) and projection-colormap
  picker (Inferno / Hot / Viridis / Plasma / Greys), with a side-by-side
  live preview that renders synthetic sample / comparison figures at
  440 DPI as you change settings.
- PNG + optional vector PDF + per-panel PNG exports.

### Compare mode

- 2–6 groups of analysis-output folders.
- Auto-selects t-test / Mann-Whitney / ANOVA / Kruskal-Wallis based on
  Shapiro-Wilk normality screening.
- Per-replicate scatter dots overlaid on bar charts (when n ≥ 2).
- Significance brackets with stars + numeric p-values.
- Multi-page PDF report: figure, parameter cover, per-replicate scalar
  table, full statistics table.

### Workflow conveniences

- **Reproducibility manifests** — every run writes a self-contained
  `<stem>_run_manifest.json` with FIREFLY version, git SHA, input file
  SHA-256, all parameters, host info. "Load run manifest…" on the Import
  tab replays a run exactly.
- **Parameter presets** — save / load named bundles of sidebar settings
  in `~/.firefly/presets/`. Two ship by default (PC12 Cells, Drosophila
  Neurons) to give new users a sensible starting point.
- **Per-series batch tree** — multi-file series (`name.tif`, `name(1).tif`,
  `name(2).tif`, …) are grouped under one parent node; expand to
  individually deselect sister files within a series. Loader concatenates
  exactly the checked subset.
- **Quality-control panel** — link ratio, locs / frame, median track
  length, gap fraction, stuck-track fraction, with colour-coded warnings
  for runs that look off (e.g. <10 % linked, >800 locs/frame).
- **Resource monitor** — 1 Hz CPU / RAM / GPU / VRAM strip on the
  Analysis tab. On Apple Silicon, GPU% is read live via `ioreg`. Catches
  silent CPU fallbacks instantly.
- **Time-elapsed counter** ticking at 1 Hz during runs (`MM:SS` /
  `H:MM:SS`).
- **Interactive track inspector** on the Visualise tab — click a track
  in the napari Tracks layer to see its particle ID, length, frame span,
  D, α, motion class, displacement, path length, straightness, mean mass.
- **Auto-update check** at launch — non-blocking GitHub Releases ping;
  shows a pill in the header when a newer version is available.
- **Crash reporter** — every uncaught exception writes a detailed report
  (parameters, hardware, pipeline state, traceback) to
  `~/Library/Logs/FIREFLY/crash_reports/` (macOS) or
  `%LOCALAPPDATA%/FIREFLY/crash_reports/` (Windows).

### Memory safety

- Multi-file TIF series → memmap-on-disk when the combined stack would
  exceed available RAM.
- A **4 GB (or 20 % of total RAM)** reserve is held back for the OS and
  the user's other apps so a parallel Safari tab won't push the machine
  into swap mid-analysis. Override with `FIREFLY_USER_RAM_RESERVE_GB=<n>`.
- Live-preview emit is auto-throttled and dropped when free memory falls
  below 1.5 GB.
- Bounded inter-process queue (`maxsize=2000`) so a stalled GUI can't
  back-pressure the worker into swap.

---

## Workflow

### Analyse a sample

1. **Import** tab — pick a `.czi` / `.tif` input file. The preview
   viewer below auto-loads with 30 sampled frames; scrub through them and
   tune the **Diameter**, **Threshold** and **Background radius** in the
   sidebar.
2. Detection circles are coloured by mass (blue = dim, red = bright);
   when you raise threshold the dim ones vanish first.
3. If you want a custom ROI: set **ROI Mode = Manual polygon** in the
   sidebar and draw it on the viewer; it persists per file.
4. Click **Start**. Switch to the **Analysis** tab to watch the live
   detection cockpit (frame + spots) and mass histogram.
5. When done: figure renders to `<output_folder>/figures/`, manifest +
   CSVs to `<output_folder>/`.

### Batch a folder

1. **Import** tab → **Batch (folder)** mode.
2. Pick a folder. The tree groups files into series; expand a series to
   deselect individual sister files.
3. Click **Open in viewer** to preview any series before starting
   (the heavy file load only fires here, never on checkbox toggles —
   selecting / deselecting is always instant).
4. Click **Start**. The Analysis cockpit resets between series; each
   gets its own subfolder under `<input>/batch_results/<stem>/`.

### Compare groups

1. **Compare** tab — drag analysis-output folders into the group cards,
   one card per condition (e.g. "Pre", "Post" / "WT", "KO", "Rescue").
2. Style on the **Figures** tab: theme, which of the 10 comparison
   panels to include, whether to also emit the multi-page PDF report.
3. **Generate comparison** — figure + summary CSV + stats CSV + PDF
   report land in the chosen output folder.

### Visualise tracks

1. **Visualise** tab — click **Load analysis run…** and pick an output
   folder.
2. The original stack loads as an Image layer; trajectories as a Tracks
   layer auto-coloured by motion class.
3. Click any track to populate the **Track inspector** panel on the
   right with that particle's stats.

---

## Outputs

Each run produces three subfolders inside the output folder plus a
manifest at the root:

```
<output_folder>/
├── <stem>_run_manifest.json       # full provenance — replay with "Load run manifest…"
├── data/                          # PALM-Tracer-compatible CSVs
│   ├── <stem>_localisations.csv
│   ├── <stem>_trajectories.csv
│   ├── <stem>_palm_tracer.csv
│   └── ...
├── firefly_extras/                # everything not in PALM-Tracer format
│   ├── <stem>_diffusion_summary.csv   # per-track D, α, motion class, ...
│   ├── <stem>_ensemble_msd.csv
│   ├── <stem>_cluster_stats.csv       # one row per DBSCAN cluster
│   ├── <stem>_jdd.json                # JDD fit (D values + fractions)
│   ├── <stem>_dwell_times.csv
│   ├── <stem>_turning_angles.csv
│   ├── <stem>_mobile_fraction.csv     # sliding-window mobile fraction
│   ├── <stem>_drift.csv               # (if drift correction enabled)
│   ├── <stem>_params.json             # parameter snapshot for Compare
│   └── <stem>_roi_mask.png            # ROI preview (if enabled)
└── figures/
    ├── <stem>_sptpalm_figure.png      # combined 15-panel figure
    ├── <stem>_sptpalm_figure.pdf      # vector copy (optional)
    └── panels/                        # per-panel exports (optional)
        ├── <stem>_panel_A.png
        ├── <stem>_panel_B.png
        └── ...
```

**Batch mode** wraps the above per-series under
`<input_folder>/batch_results/<stem>/` and adds a `batch_summary.csv`
with one row per series.

**Compare mode** writes:

```
<output_folder>/
├── compare_<labels>.png            # multi-panel comparison figure
├── compare_<labels>.pdf            # vector copy
├── compare_<labels>_summary.csv    # per-replicate scalar metrics
├── compare_<labels>_stats.csv      # statistical tests (pairwise)
└── compare_<labels>_report.pdf     # 4-page PDF: figure + cover + tables
```

---

## Performance notes

- Use the **uniform-filter** background method unless you have a reason
  to need rolling-ball (uniform is ~1700× faster with comparable
  results for PALM data).
- The **PyTorch** backend on a recent Apple-Silicon / NVIDIA GPU is
  typically 5–20× faster than trackpy on the same machine.
- Chunk size 500 (default) is a good balance. Larger needs more RAM;
  smaller wastes per-chunk overhead.
- DBSCAN is capped at 250 k localisations to keep clustering tractable;
  larger inputs are randomly subsampled before clustering. Spatial
  pattern is preserved.

---

## Troubleshooting

**macOS: "FIREFLY can't be opened because the developer cannot be
verified"**
→ Right-click the app → **Open** → **Open anyway** the first time.

**No particles found / very few trajectories**
→ Lower the PSF diameter by 2 px. Disable **Auto-detect** for threshold
and try smaller values. Check the **channel** index for CZI files.

**Too many trajectories / noise being tracked**
→ Raise threshold. Raise background radius. Enable ROI masking.

**Pixel size or frame interval shows as a warning**
→ Couldn't read the metadata. Tick **Override** and enter the right
value from your acquisition.

**Out of memory during localisation**
→ Reduce chunk size in the Performance section. The loader will
automatically switch to memmap-on-disk when the combined stack exceeds
available RAM minus the user-reserve.

**Run feels slow even with GPU set**
→ Check the resource monitor on the Analysis tab. If GPU% sits at 0,
the backend fell back to CPU — look at the log for the resolver's
verdict. On macOS, `Torch — Apple MPS` requires PyTorch ≥ 2.0 and a
recent macOS; on older systems the resolver auto-falls back to trackpy.

**Compare panels show "no data" placeholders**
→ Older analysis folders (pre-v1.0.55) don't have every per-run JSON /
CSV the Compare tab needs. Re-run the affected experiments to regenerate
the full set.

**Hard freeze during analysis**
→ Almost always memory pressure or an MPS driver hang. Close other
apps, lower chunk size, and check
`~/Library/Logs/DiagnosticReports/` for a panic log.

If you hit a crash, FIREFLY writes a full report to
`~/Library/Logs/FIREFLY/crash_reports/` (macOS) or
`%LOCALAPPDATA%/FIREFLY/crash_reports/` (Windows). Attach the report
when reporting the issue.

---

## Acknowledgements

Built on the shoulders of:

- [trackpy](http://soft-matter.github.io/trackpy/) (Allan et al.) —
  Crocker-Grier localisation, linking
- [scikit-image](https://scikit-image.org/) — preprocessing,
  thresholding, morphology
- [scipy](https://scipy.org/), [scikit-learn](https://scikit-learn.org/) —
  statistics, DBSCAN
- [matplotlib](https://matplotlib.org/) — figure rendering
- [napari](https://napari.org/) +
  [vispy](https://vispy.org/) — embedded image viewer
- [PySide6 / Qt 6](https://www.qt.io/) — GUI
- [PyTorch](https://pytorch.org/) — GPU localisation
- [tifffile](https://github.com/cgohlke/tifffile),
  [aicspylibczi](https://github.com/AllenCellModeling/aicspylibczi) —
  format-specific loaders

Algorithm references:

- Crocker & Grier (1996) — feature detection
- Wang et al. (2014, *Nature Methods*) — RCC drift correction
- Saxton (1997), Yu et al. (2014) — jump-distance distribution
- Ferrari et al. (2001) — moment-scaling spectrum
- Thompson, Larson & Webb (2002) — localisation precision
- Otsu (1979), Li & Lee (1993), Zack-Rogers-Latt (1977) — auto-thresholds
- Ester et al. (1996) — DBSCAN

Developed with AI assistance.
