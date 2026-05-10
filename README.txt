sptPALM Analysis Pipeline — By Jacob Levers
============================================
Single-particle tracking PALM analysis for Zeiss Elyra CZI and TIFF files.
Compatible with PC12 cells, Drosophila neurons, and other cell types.

The app has two top-level modes:
  • Analyse Data  — run the full localisation / tracking / MSD pipeline
                    on a single .czi/.tif (or a folder of them) and save
                    per-experiment results to disk.
  • Compare Data  — overlay 2-6 groups of analysis output folders into
                    multi-panel comparative figures with statistical tests
                    and a combined PDF report.

You pick a mode from the welcome screen on first launch; thereafter, switch
freely between modes via the tab strip at the top of the window.


================================================================
REQUIREMENTS
================================================================

- Windows 10/11  OR  macOS 11 (Big Sur) or newer
- 8 GB RAM minimum (16-32 GB recommended for large datasets)
- A .czi or multi-page .tif file from a PALM/STORM acquisition


================================================================
INSTALLATION — STANDALONE APP (recommended)
================================================================

No Python installation required. Everything is bundled.

macOS
-----
1. Download sptPALM-macOS.dmg from the Releases page on GitHub
2. Double-click the .dmg to mount it
3. Drag sptPALM.app into your Applications folder
4. Double-click sptPALM.app to launch
   (First launch: right-click → Open if macOS asks for confirmation)

Windows
-------
1. Download sptPALM-Windows.exe from the Releases page on GitHub
2. Save it somewhere convenient (Desktop is fine)
3. Double-click to launch
   First launch is slower (~30 s) while the bootloader unpacks
   bundled libraries to %TEMP%; subsequent launches are quick.


================================================================
INSTALLATION — RUN FROM SOURCE (advanced)
================================================================

If you prefer to run the Python source directly, Python 3.10 or
newer is required.

macOS
-----
Install Python from https://www.python.org/downloads/ or via
Homebrew (brew install python), then double-click:

   Launch_sptPALM.app

On first launch the app detects missing libraries and installs them
automatically. This takes 3-5 minutes. Subsequent launches are instant.

Windows
-------
Install Python from https://www.python.org/downloads/
IMPORTANT: Tick "Add Python to PATH" during installation.

Then double-click:

   Launch_sptPALM.bat

On first launch the app detects missing libraries and installs them
automatically. This takes 3-5 minutes. Subsequent launches are instant.


================================================================
USING THE APP — OVERVIEW
================================================================

When you open the app you see a welcome card with two prominent buttons:

   ▶ Analyse Data       Single file or batch run from a .czi/.tif image
   ⇆ Compare Data       Overlay results from N folders vs M folders

Pick one to enter that mode.  A persistent tab strip just below the
header lets you flick between modes any time without losing state.


================================================================
ANALYSE DATA TAB
================================================================

Single file
-----------
1. Click "Browse" next to Input file and select your .czi or .tif file
2. Set an output folder (defaults to same folder as input)
3. Review the settings panel on the left — hover over any ⓘ icon
   for a description of that parameter and suggested values
4. Click "▶ Run Analysis"
5. Watch the live preview and progress log on the right
6. When done, click "Open Output Folder" to see results

Batch processing (multiple files)
----------------------------------
1. Set your analysis parameters as normal in the settings panel
2. Click "Batch" (next to the input file field)
3. Select a folder containing your .czi or .tif files
4. Confirm the file list and click OK
5. Results for each file are saved to a batch_results/ sub-folder
   inside the selected folder, plus a batch_summary.csv summary table

Key settings to check before your first run:
  - Pixel size (µm/px)     Auto-read from CZI. Verify it is correct.
  - Frame interval (s)     Auto-read from CZI. Verify it is correct.
  - PSF diameter (px)      Must be odd. Typically 7 px for 561 nm on Elyra.
  - Search range (px)      Set to ~2-3× expected single-step displacement.
  - Min track length       At least 2-3× the "Fit first N lag points" value.


================================================================
COMPARE DATA TAB
================================================================

Compare overlays results from previously-completed analyses.  No raw
images are reanalysed — the comparison reads the CSVs/JSONs already
saved inside each output folder, so it runs in seconds instead of
minutes.

Workflow
--------
1. Switch to the Compare Data tab.
2. Add folders to each group:
     • + Add        — pick one analysis output folder
     • + Add many   — pick a parent folder; every valid analysis
                      subfolder inside it is auto-imported
     • Drag and drop folders directly onto the listbox (Mac/Windows)
3. Type a label for each group (e.g. "Pre"/"Post", "WT"/"KO"/"Rescue").
4. Click the small coloured swatch beside the label to change that
   group's colour.
5. Add more groups with "+ Add group" (up to 6).  Click "× Remove"
   on a group's card to delete it.
6. Pick the figure theme (Dark / Light / Publication — same set as
   the Analyse-mode figures).
7. Toggle which panels you want via the checkboxes; toggle the
   PDF report on or off.
8. Pick an output folder and click "▶ Generate Comparison".

The comparison scales gracefully:
  • 1 vs 1     two MSD curves overlaid, two bars (no scatter dots, no
               significance stars — t-test needs n≥2 each side)
  • N vs M     full lab-style figure with mean ± SEM bands, scatter
               dots per replicate, t-test stars on bar charts
               (Welch's t-test or Mann-Whitney by default)
  • 3+ groups  ANOVA / Kruskal-Wallis omnibus p-value annotation on
               every panel + full pairwise comparisons in the stats CSV

Available panels (any subset can be selected):
  1. MSD curve overlay (mean ± SEM band)
  2. AUC of MSD bar chart (per-replicate scatter, t-test stars)
  3. LogD frequency distribution (mobile/immobile threshold line)
  4. Mobile/Immobile ratio bar chart
  5. Motion class fractions (Immobile / Confined / Brownian / Directed)
  6. Track length CDF (x-axis clipped at 99th percentile)
  7. Jump distance distribution (per-population D, marker size ∝ fraction)
  8. Dwell time survival curves
  9. Turning angle distribution (degrees, 0-180°)

Outputs (next to the figure)
----------------------------
   compare_<labels>.png            Multi-panel comparison figure
   compare_<labels>.pdf            Same, vector-friendly
   compare_<labels>_summary.csv    Per-replicate scalar metrics
                                   (one row per folder × group)
   compare_<labels>_stats.csv      Statistical tests
                                   (one row per metric × pair, with
                                    test name, p-value, stars, n,
                                    mean and SEM per group)
   compare_<labels>_report.pdf     Combined report (4 pages):
                                     1. The figure
                                     2. Cover with theme + groups +
                                        folder lists
                                     3. Per-replicate scalar table
                                     4. Statistics table

Note on existing analysis folders
---------------------------------
Compare reads files saved by the Analyse pipeline (v1.0.55+).  Older
output folders will still produce the MSD overlay, AUC, LogD, mobile/
immobile ratio, motion classes, and track length panels — anything
derivable from `_diffusion_summary.csv`, `_trajectories.csv` and
`_ensemble_msd.csv`.  The JDD, dwell-time and turning-angle panels
need the newer per-run files (`_jdd.json`, `_dwell_times.csv`,
`_turning_angles.csv`) and will show a "no data" placeholder for
folders that don't have them.  Re-running an old experiment regenerates
the full set.


================================================================
OUTPUT FILES (per analysis run)
================================================================

After a run completes, the chosen output folder contains two
subdirectories:

   figures/   PNG / PDF figures
   data/      CSV / JSON files

In figures/:
   <stem>_sptpalm_figure.png   Combined results figure (see below)
   <stem>_sptpalm_figure.pdf   Same, vector format
   panels/                     Optional: each panel exported separately
                               (toggle in Settings → Export)

In data/:
   <stem>_localisations.csv     Raw molecule positions per frame
   <stem>_trajectories.csv      Full x/y trajectory data
   <stem>_diffusion_summary.csv D, alpha, motion class, confinement
                                radius, localisation precision per track
   <stem>_ensemble_msd.csv      Ensemble-averaged MSD curve
   <stem>_drift.csv             Per-frame drift estimate (if drift
                                correction enabled)
   <stem>_cluster_stats.csv     One row per detected cluster (DBSCAN)
   <stem>_jdd.json              JDD fit results (D values + fractions)
   <stem>_dwell_times.csv       Per-track dwell time (s) for confined/
                                immobile tracks
   <stem>_turning_angles.csv    Step-to-step turning angles (degrees)
   <stem>_mobile_fraction.csv   Mobile fraction over time (sliding window)
   <stem>_params.json           Snapshot of analysis parameters
                                (px, fi, diameter, etc.) — used by the
                                Compare tab to ensure consistent units
   <stem>_roi_mask.png          Preview of the ROI mask if enabled

Figure contents
---------------
The combined figure organises panels into a grid with letter labels
(A, B, C, ...).  Typical contents:

  A — Track overlay on mean projection
  B — MSD curve (ensemble)
  C — Diffusion coefficient histogram
  D — Alpha (anomalous exponent) histogram
  E — Motion-type pie chart
  F — Track length histogram
  G — Localisation density heatmap
  H — Jump Distance Distribution with population fits
  I — Cluster overlay (DBSCAN, if enabled)
  J — Mobile fraction over time
  K — Dwell time histogram for confined/immobile tracks
  L — Turning angle distribution

Panels not available for your dataset (e.g. clusters when none
detected) are skipped automatically.

Batch output
------------
Inside batch_results/:
   <filename>/            Per-file sub-folder with the same files as above
   batch_summary.csv      One row per file — n_tracks, mean D, mobile
                          fraction, mean confinement radius, JDD populations


================================================================
FIGURE THEMES
================================================================

Three styles are available in both Analyse and Compare modes
(Settings → Figure Style → Theme):

  Dark           Dark grey background, light text — best for
                 presentations and live-preview viewing.
  Light          White background with light panels — good for
                 reports, slides and review documents.
  Publication    Pure white background, serif font, minimal
                 gridlines — suitable for journal figures.

The Compare tab defaults to whatever the Analyse tab is using, so
both pipelines stay visually consistent unless you override.


================================================================
JUMP DISTANCE DISTRIBUTION (JDD)
================================================================

JDD analysis fits the cumulative distribution of single-step displacements
to a multi-population diffusion model. It is more robust than MSD-based
methods for short tracks and gives the diffusion coefficient and fractional
population of each mobility state.

Set the number of JDD populations (1, 2, or 3) in Settings → MSD & JDD.
Two populations (mobile + immobile) is recommended for most sptPALM datasets.

The JDD panel in the figure shows the step-size histogram with the fitted
PDF overlaid. Population D values and fractions are shown in the results
panel and saved to <stem>_jdd.json.


================================================================
D-VALUE FILTER
================================================================

Enable "Filter by D" in Settings → MSD & JDD to restrict JDD fitting and
downstream statistics to tracks within a specific diffusion coefficient range.
Useful for isolating a mobility population of interest (e.g. only mobile
molecules, or only those above a confinement threshold).

The filter does not alter MSD fitting — all tracks are fitted first, then
only tracks with D inside [D min, D max] are passed to JDD and summary stats.


================================================================
CLUSTER ANALYSIS  (DBSCAN)
================================================================

DBSCAN groups raw localisations into spatial clusters (e.g. receptor
nanodomains). Results are shown in figure panel L and saved to
<stem>_cluster_stats.csv (one row per cluster — n_locs, area µm²,
density, centroid position).

Settings in Settings → Cluster Analysis (DBSCAN):
  - DBSCAN radius (nm)      Search radius. Typical values: 30-80 nm.
                            Smaller = tighter clusters, larger = merges
                            nearby clusters together.
  - Min localisations       Minimum points to form a cluster. Increase
                            to reject noise, decrease to find small clusters.

Performance note: DBSCAN is capped at 250,000 localisations. On larger
datasets a random subsample is used, which reliably reproduces the spatial
cluster pattern with much faster processing. If your dataset is sparse and
you are concerned about missing small clusters, reduce the acquisition
density or run dedicated cluster analysis software (e.g. SR-Tesseler).


================================================================
DRIFT CORRECTION
================================================================

Enable in Settings → Drift Correction → Enable.

Uses reference-free redundant cross-correlation (RCC, Wang et al. 2014).
No fiducial markers required. Applied before trajectory linking so
corrected positions improve track quality.

Segment size: aim for >200 localisations per segment.
Typical values: 200-500 frames for dense labelling,
400-1000 frames for sparse labelling.


================================================================
ROI MASKING
================================================================

Restrict analysis to the cell region, discarding localisations that
fall outside. Four modes are available in Settings → ROI Masking:

Disabled
   Full frame is analysed (default).

Auto threshold
   Thresholds the mean projection automatically. Three algorithms
   are available — Triangle (default, best for sptPALM), Li (good
   for sparse neurites), and Otsu (general purpose).

Manual threshold
   Set the threshold yourself on a 0-1 scale relative to the
   brightest region in the mean projection.
   Start at 0.10-0.15 for PC12 cell bodies.
   Start at 0.05-0.08 for thin Drosophila neurites.

Draw ROI
   Click "Open ROI Editor" to draw a freehand polygon on the mean
   projection. Left-click to add vertices, double-click to close,
   right-click to undo. Click Apply to save.

Always check the *_roi_mask.png preview before trusting filtered results.


================================================================
COMING FROM PALMTRACER (FIJI)
================================================================

sptPALM Analysis uses the same localisation algorithm (Crocker & Grier)
and the same nearest-neighbour linker as PALMTracer, so results should
be comparable with equivalent settings.

Key differences to be aware of:

- PALMTracer applies a hard MSD cutoff that removes slow/immobile molecules.
  This pipeline retains them, so your immobile fraction may appear higher.
  Those molecules are real — they were not artefacts, just filtered out.

- JDD analysis here uses CDF fitting, which is more robust for short tracks
  than the histogram fitting used in some PALMTracer versions.

- Confinement radius is reported per track in <stem>_diffusion_summary.csv.
  This is the mean distance of all positions from the track centroid (µm),
  equivalent to the confinement radius from PALMTracer's confined motion mode.

- Localisation precision (nm) is reported per track if your acquisition
  software wrote it into the CZI metadata (trackpy ep column).


================================================================
TROUBLESHOOTING
================================================================

App won't open on macOS — "cannot be opened because the developer
cannot be verified"
   Right-click the app → Open → Open anyway.
   This is a macOS Gatekeeper warning for unsigned apps.

App takes a long time to start on Windows
   First launch of the onefile build extracts ~200 MB of bundled
   libraries to %TEMP% — this can take 10-30 s on slow disks.
   Subsequent launches are quick because Python reuses the cached
   extraction.

No particles found / very few trajectories
   Lower the PSF diameter by 2 px and re-run.
   Disable auto-detect minmass and manually set a lower value (0.3-1.0).
   Check that the correct channel index is selected.

Too many trajectories / noise being tracked
   Increase minmass (try 1.0-3.0).
   Increase background radius.
   Enable ROI masking to exclude out-of-cell noise.

Pixel size or frame interval shows as WARNING
   The CZI metadata could not be read. Tick "Set manually" and
   enter the correct values from your acquisition settings.

JDD fit does not converge / populations look wrong
   Try reducing the number of JDD populations to 1 or 2.
   Ensure min track length is at least 3 frames (more steps = better JDD).
   If using the D filter, widen the D range and re-run.

Out of memory during localisation
   Reduce Chunk size in Settings → Performance (try 200-300 frames).

Analysis is slow
   Increase CPU workers in Settings → Performance.
   Use "Uniform Filter" background method (much faster than Rolling Ball).

Batch run stops partway through
   Check the log panel — a specific file likely failed. That file's
   sub-folder in batch_results/ will be incomplete or absent. Fix the
   file or exclude it and re-run batch on the remaining files.

Compare tab — JDD / dwell / turning-angle panels show "no data"
   Those panels rely on per-run files (<stem>_jdd.json,
   <stem>_dwell_times.csv, <stem>_turning_angles.csv) introduced in
   v1.0.55+.  Older folders won't have them.  Re-run the analysis
   on the same .czi to regenerate the full set; the rest of the
   Compare panels work on older folders unchanged.

Compare tab — drag-and-drop does nothing
   Drag-and-drop relies on the optional `tkinterdnd2` package.  If it
   failed to load, the subheader text will not include the "Tip: drag
   folders directly onto a group's list" hint and the + Add / + Add many
   buttons remain the only way to populate a group.


================================================================
ACKNOWLEDGEMENTS
================================================================

Uses trackpy (Allan et al.), scikit-image, and scipy.
Localisation algorithm based on Crocker & Grier (1996).
Auto-thresholding uses Otsu (1979), Li & Lee (1993), and
Zack, Rogers & Latt (1977) algorithms via scikit-image.
Drift correction based on Wang et al. (2014) RCC method.
Statistical tests via scipy.stats (Welch's t-test, Mann-Whitney U,
one-way ANOVA, Kruskal-Wallis, Shapiro-Wilk).
Drag-and-drop via tkinterdnd2.
Developed with AI assistance (Anthropic Claude).


================================================================
CONTACT / SUPPORT
================================================================

If you encounter an error not listed above, copy the full error
message from the log panel and seek support from your facility's
microscopy or bioinformatics team.
