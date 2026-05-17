FIREFLY — Fluorescence Inference & Reconstruction Engine
=========================================================
Framework for Localization Yields  —  By Jacob Levers
Version 2.2.0

A single-particle tracking PALM pipeline for .czi (Zeiss) and .tif/.tiff
image stacks. Localisation, tracking, MSD / diffusion / motion-class
analysis, JDD, dwell-time, MSS, DBSCAN clusters, drift correction (full
RCC), turning-angle and radial-distribution analysis, plus a Compare
mode that overlays N analysis-output folders with statistical tests and
a combined PDF report.

The app opens on a welcome landing page with four action cards:
  • Analyse a sample   — run the pipeline on one .czi/.tif file
  • Batch a folder     — process every file in a folder, sequentially
  • Compare groups     — overlay 2-6 analysis-output folders into a
                          comparison figure with stats + PDF report
  • Visualise tracks   — open a previous run in an embedded napari
                          viewer (with the interactive track inspector)

Clicking a card switches to the relevant tab.  Once in the workflow
UI, the landing page is out of the way — the tab bar at the top is
how you move between Import / Analysis / Figures / Compare / Visualise.


================================================================
WHAT'S NEW IN 2.2
================================================================

  • Welcome landing page — one-way gateway with four action cards;
    no more burying the workflow under tabs from the start.
  • Live preview viewer on the Import tab — embedded napari viewer
    that auto-loads when you pick a file.  Detection circles colour-
    coded by mass, ROI threshold mask overlay, "Filtered view"
    toggle showing the bandpass-preprocessed image, and a min-mass
    slider that updates the spot count live.
  • Figures tab — style and output settings (theme, projection
    colormap, DPI, vector PDF, per-panel exports) for both single-
    sample and comparison figures.  Live preview renders synthetic
    examples in your chosen style so you can iterate without re-
    running anything.
  • Real-time mass histogram during a run — chunks stream into a
    histogram on the Analysis tab so you can spot a bad minmass
    before the figure is even rendered.
  • Quality-control panel — link ratio, locs/frame, median track
    length, gap rate, stuck-track fraction.  Colour-coded warnings
    surface dud runs at a glance.
  • Reproducibility manifests — every run writes a
    <stem>_run_manifest.json with FIREFLY version, git SHA, input
    file SHA-256, full parameters, and a widget-state snapshot.
    A "Load run manifest…" button on the Import tab repopulates
    every sidebar control from a saved manifest so you can replay
    a run exactly.
  • Interactive track inspector — click a track in the Visualise-
    tab napari viewer to see its D, α, motion class, length,
    displacement, mean mass and more in a side panel.  Tracks are
    coloured by motion class via napari features.


================================================================
REQUIREMENTS
================================================================

- Windows 10/11  OR  macOS 11 (Big Sur) or newer
- 8 GB RAM minimum (16-32 GB recommended for large datasets)
- A .czi or multi-page .tif file from a PALM / STORM acquisition


================================================================
INSTALLATION — STANDALONE APP (recommended)
================================================================

No Python installation required. Everything is bundled.

macOS
-----
1. Download FIREFLY-macOS.dmg from the Releases page on GitHub
2. Double-click the .dmg to mount it
3. Drag FIREFLY.app into your Applications folder
4. Double-click FIREFLY.app to launch
   (First launch: right-click → Open if macOS asks for confirmation)

Windows
-------
1. Download FIREFLY-Windows.exe from the Releases page on GitHub
2. Save it somewhere convenient (Desktop is fine)
3. Double-click to launch
   First launch is slower (~30 s) while the bootloader unpacks
   bundled libraries to %TEMP%; subsequent launches are quick.


================================================================
INSTALLATION — RUN FROM SOURCE (advanced)
================================================================

Python 3.10 or newer is required.

macOS
-----
Install Python from https://www.python.org/downloads/ or via
Homebrew (brew install python), then double-click:

   Launch_FIREFLY.app

On first launch a Terminal window opens showing pip installing the
required libraries (PySide6, napari, PyTorch, scipy, etc.).  Takes
3-8 minutes depending on network speed.  FIREFLY launches automatically
when the install finishes.  Subsequent launches are instant — no
Terminal window appears.

Windows
-------
Install Python from https://www.python.org/downloads/
IMPORTANT: Tick "Add Python to PATH" during installation.

Then double-click:

   Launch_FIREFLY.bat

On first launch a cmd window opens showing pip installing the
required libraries (PySide6, napari, PyTorch, scipy, etc.).  Takes
3-8 minutes depending on network speed.  FIREFLY launches automatically
when the install finishes.  Subsequent launches are instant.


================================================================
USING THE APP — OVERVIEW
================================================================

When you open the app you see a welcome landing page with four big
action cards:

   ▶ Analyse a sample   Single .czi/.tif run with the full pipeline
   ⊞ Batch a folder     Process every file in a folder sequentially
   ⇄ Compare groups     Overlay results from 2-6 folder groups
   ◉ Visualise tracks   Open a prior run in the napari viewer

Click any card to enter the workflow.  From then on the tab strip
(Import → Analysis → Figures → Compare → Visualise) is how you move
between sections without losing state.  The landing page is shown
only on launch, not on every tab switch.


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

Multi-file series
-----------------
Some acquisitions split long recordings into companion files
(experiment.czi + experiment(1).czi + experiment(2).czi …).  Just
pick the first file — the loader auto-detects the series, stitches
the stacks in order, and runs a single analysis on the combined
recording.  The same auto-detection applies to .tif series with
matching naming.

Batch processing (multiple files / multiple series)
---------------------------------------------------
1. Set your analysis parameters as normal in the settings panel
2. Click "Batch" (next to the input file field)
3. Select a folder containing your .czi or .tif files
4. The dialog tells you how many separate analyses will run.  Each
   multi-file series (experiment.czi + experiment(N).czi…) is
   detected and counted as ONE analysis, not one per file.
5. Confirm and the runs proceed sequentially.
6. Results for each experiment are saved to a sub-folder inside the
   selected parent, plus a batch_summary.csv summary table.

Live preview viewer (Import tab)
--------------------------------
The Import tab embeds a napari viewer that auto-loads as soon as the
input file path settles.  It serves five purposes simultaneously:

  • Frame scrubbing      — preview ~30 sampled frames of the stack.
  • Detection overlay    — every spot found by tp.locate on the
                            current frame is drawn as a circle; the
                            circle's colour encodes the spot's mass
                            on a turbo log-scale (low = blue, high
                            = red).  A small turbo legend strip in
                            the header reminds you of the mapping.
                            The status bar shows live counts and
                            mass quantiles:
                              "15 spots (d=7, minmass=1.42) —
                               mass 222 / med 628 / 6458"
                            Drag the min-mass slider (sidebar →
                            Detection) and watch dim spots vanish.
  • ROI threshold mask   — when ROI Mode is "Auto threshold" or
                            "Manual threshold", a translucent green
                            mask overlays the pixels that pass.
                            Drag the manual-threshold slider and the
                            mask redraws live.
  • ROI polygon drawing  — when ROI Mode is "Manual polygon", draw
                            the polygon directly.  Per-file
                            polygons are auto-saved and reused on
                            the next run for that file.
  • Filtered view toggle — the "Filtered view" checkbox in the
                            header swaps the background image for
                            its bandpass-filtered version (what the
                            detector actually sees).  Real PSFs
                            light up, flat noise drops away.  Spot
                            counts still come from raw frames so the
                            numbers match the eventual run.

Min-mass scale note: after FIREFLY's per-frame normalisation, useful
minmass values are typically 0.5–50.  The slider uses a square-law
mapping so the low end (where most real adjustments happen) gets
fine control.

Key settings to check before your first run:
  - Pixel size (µm/px)     Auto-read from file metadata. Verify it.
  - Frame interval (s)     Auto-read from file metadata. Verify it.
  - PSF diameter (px)      Must be odd. Typically 7 px at ~0.1 µm/px.
  - Search range (px)      Set to ~2-3× expected single-step displacement.
  - Min track length       At least 2-3× the "Fit first N lag points".
  - α immobile / confined / directed cut-offs  (configurable since
    v1.0.61) — alpha thresholds used to classify motion type. Defaults
    are the conventional sptPALM values 0.5 / 0.9 / 1.1; some labs use
    different values (e.g. 0.4 / 0.8 / 1.2).
  - Mobile D threshold (µm²/s)  — diffusion cut-off separating
    Immobile from Mobile (default 0.05 µm²/s; used by the
    Mobile/Immobile ratio and LogD threshold line).


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
   group's colour (the swatch itself is clickable — no separate "Pick"
   button needed).
5. Add more groups with "+ Add group" (up to 6).  Click "× Remove"
   on a group's card to delete it (disabled when only 2 remain).
6. Output folder, output filename stem.
7. Style settings — theme, which panels are included, and whether
   to emit the multi-page PDF report — live on the FIGURES tab now
   (see below).
8. Click "▶ Generate Comparison".

The comparison scales gracefully:
  • 1 vs 1     two MSD curves overlaid, two bars (no scatter dots, no
               significance stars — t-test needs n≥2 each side)
  • N vs M     full lab-style figure with mean ± SEM bands, scatter
               dots per replicate, t-test stars + numeric p-values
               on the bar charts
  • 3+ groups  ANOVA / Kruskal-Wallis omnibus p-value annotation on
               every bar panel + pairwise comparisons (Welch's t /
               Mann-Whitney) with Bonferroni correction in the
               stats CSV

Available panels (any subset can be selected):
   1. MSD curve overlay (mean ± SEM band)
   2. AUC of MSD bar chart (per-replicate scatter, p-value + stars)
   3. LogD frequency distribution (with mobile/immobile threshold line)
   4. Mobile/Immobile ratio bar chart
   5. Motion class fractions (Immobile / Confined / Brownian / Directed)
   6. Track length CDF (x-axis clipped at 99th percentile)
   7. Jump distance distribution (per-population D, marker size ∝ fraction)
   8. Dwell time survival curves
   9. Turning angle distribution (|θ| line plot, 0°-180°, normalised
      per group)
  10. Radial distribution (polar bar chart of signed turning angles,
      side-by-side bars per group, each group normalised to its own
      total so distribution SHAPE is compared)

Outputs (next to the figure)
----------------------------
   compare_<labels>.png            Multi-panel comparison figure
   compare_<labels>.pdf            Same, vector-friendly
   compare_<labels>_summary.csv    Per-replicate scalar metrics
                                   (one row per folder × group)
   compare_<labels>_stats.csv      Statistical tests (one row per
                                   metric × pair):
                                     metric, comparison, test,
                                     p_value, stars,
                                     p_value_bonferroni,
                                     stars_bonferroni,
                                     n_a, n_b, mean_a, mean_b,
                                     sem_a, sem_b, label_a, label_b
   compare_<labels>_report.pdf     Combined report (4 pages):
                                     1. The figure
                                     2. Cover with theme + groups +
                                        folder lists
                                     3. Per-replicate scalar table
                                     4. Statistics table

Statistical tests
-----------------
Test selection is automatic:

  - Normality screening: Shapiro-Wilk on each group (p<0.05 → non-normal)
  - 2 groups, all normal:     Welch's t-test (unequal-variance two-sample)
  - 2 groups, any non-normal: Mann-Whitney U
  - 3+ groups, all normal:     one-way ANOVA  (omnibus)
  - 3+ groups, any non-normal: Kruskal-Wallis (omnibus)
  - Pairwise post-hoc (3+ groups): same Welch's t / Mann-Whitney rule
    per pair, Bonferroni-corrected

Bar charts annotate the chosen test name + numeric p-value + stars:
   "p = 0.003  **"        for 2-group panels (bracket above bars)
   "Welch's t-test
    p = 0.012  *"         for omnibus annotation on 3+ group panels

Stars convention: * p<0.05, ** p<0.01, *** p<0.001, ns otherwise.

Note on existing analysis folders
---------------------------------
Compare reads files saved by the Analyse pipeline.  Older runs
(pre-v1.0.55) may not have all the per-run JSON / CSV files needed by
every panel — those panels show "no data" placeholders for missing
folders.  Re-running an old experiment regenerates the full set.


================================================================
FIGURES TAB
================================================================

Centralised style + output customisation for both the single-sample
figure (from Analyse) and the comparison figure (from Compare).

Two side-by-side previews on the right of the tab render synthetic
sample/comparison figures in the chosen styles, updated live.

Single-sample figure
  • Theme              Dark / Light / Publication
  • Projection cmap    Inferno / Hot / Viridis / Plasma / Greys
  • PNG DPI            72-600 (150 default)
  • Also save PDF      Vector copy alongside the PNG
  • Per-panel exports  Toggle, plus a 2-column checkbox grid that
                       picks which of the 15 panels (A-O) are
                       written into figures/panels/.

Comparison figure
  • Theme              Independent from single-sample so you can
                       mix styles per output type
  • PDF report         Multi-page (figure + cover + per-replicate
                       table + stats table)
  • Panel selector     2-column checkbox grid (10 panels) — same
                       choices as before, just moved here from the
                       Compare tab.


================================================================
ANALYSIS TAB — LIVE FEEDBACK + QUALITY CONTROL
================================================================

While a run is in progress, the Analysis tab shows:

  • Stage label + overall progress bar
  • Elapsed-time counter ticking at 1 Hz (MM:SS or H:MM:SS)
  • Live mass histogram — every chunk's mass values stream into a
    40-bin histogram with a dashed orange line at your current
    minmass.  If the histogram peak is far from the dashed line,
    abort the run and re-tune minmass instead of waiting an hour.

When a run completes, the results panel adds a Quality-control
section with:

  • Localisations linked       fraction that ended up in tracks
                                (< 10% → red warning;
                                 10-25% → orange; otherwise green)
  • Locs / frame (avg)         flagged orange above 800
  • Median track length        flagged orange below 6 frames
  • Tracks with gaps           any track whose frame span exceeds
                                its length
  • Stuck tracks  (D < 1e-3)   flagged orange above 30%

Followed by one-line flag messages (red ⚠ for warnings,
muted ℹ for informational notes about gappy blinking probes etc.).


================================================================
REPRODUCIBILITY — RUN MANIFESTS
================================================================

Every run writes a self-contained JSON next to the outputs:

   <out_dir>/<stem>_run_manifest.json

It records, in one file:

  • schema_version, FIREFLY version, git SHA (if running from a
    checkout)
  • host info (hostname, platform, Python version)
  • created_at timestamp
  • input.path + input.sha256 (SHA-256 of the input file's bytes)
  • output_dir, stem
  • parameters  — full kwargs the worker received
  • widget_state — every sidebar control's value, keyed by setting
    path (e.g. "analysis/diameter", "figures/theme")

To replay a run, open the Import tab and click "Load run manifest…"
next to the file/output pickers.  Picking a manifest applies the
saved widget_state to every sidebar control and repopulates the
input/output paths.  Hit Start and you'll reproduce the original
run exactly (same backend, same minmass, same ROI mode, same
figure theme, …).


================================================================
VISUALISE TAB — TRACK INSPECTOR
================================================================

The Visualise tab is split: napari viewer on the left, "Track
inspector" panel on the right.

  • "Load image stack…"  open a .czi / .tif as an Image layer
  • "Load tracks…"       open a *_trajectories.csv (auto-detects a
                          sibling *_diffusion_summary.csv so it can
                          colour by motion class and populate the
                          inspector)
  • "Load analysis run…" pick a run output folder; auto-loads the
                          original stack + tracks + diffusion-summary
  • "Auto-load after analysis"  flips on the inspector right after
                                 a successful Analyse run

Click a track point in the viewer (within ~8 px of any point on its
trajectory at the current frame) and the inspector populates:

   Particle ID
   Track length
   Frame span
   Motion class           (colour-coded — red Immobile, orange
                            Confined, blue Brownian, green Directed)
   Diffusion D            µm²/s
   α (anomalous exponent)
   Net displacement       nm  (start → end Euclidean)
   Total path             nm  (sum of single-step lengths)
   Straightness           net / path
   Mean mass              average integrated mass over the track


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
   <stem>_diffusion_summary.csv Per-track:
                                  particle, D, alpha, motion class,
                                  confinement_radius_um (legacy alias
                                    for mean_radial_displacement_um),
                                  mean_radial_displacement_um =
                                    ⟨|r-r̄|⟩  (first absolute moment),
                                  radius_of_gyration_um =
                                    √⟨|r-r̄|²⟩  (RMS — standard Rg),
                                  loc_precision_nm,
                                  mss_slope (if MSS computed)
   <stem>_ensemble_msd.csv      Ensemble-averaged MSD curve
   <stem>_drift.csv             Per-frame drift estimate (if enabled)
   <stem>_cluster_stats.csv     One row per detected DBSCAN cluster
   <stem>_jdd.json              JDD fit (D values + fractions)
   <stem>_dwell_times.csv       Three columns per confined/immobile track:
                                  dwell_time_total_s   (canonical;
                                    last_frame − first_frame + 1) × Δt,
                                  dwell_time_observed_s
                                    (n_observations × Δt),
                                  dwell_time_s          (alias for total)
   <stem>_turning_angles.csv    Column `turning_angle_deg` —
                                signed degrees in (-180°, +180°]
                                (rotational direction is preserved).
                                Older runs may use `turning_angle_rad`
                                with unsigned [0°, 180°] values; both
                                are read by the Compare tab.
   <stem>_mobile_fraction.csv   Sliding-window mobile fraction over time
   <stem>_params.json           Snapshot of analysis parameters
                                (pixel size, frame interval, diameter,
                                alpha thresholds, mobile D threshold,
                                etc.).  Read by the Compare tab to
                                ensure consistent units across folders.
   <stem>_roi_mask.png          Preview of the ROI mask (if enabled)

At the root of the output folder:

   <stem>_run_manifest.json     Reproducibility manifest — FIREFLY
                                version, git SHA, input file path +
                                SHA-256, full parameters, and a
                                widget-state snapshot.  Load via
                                Import tab → "Load run manifest…"
                                to replay the run exactly.

Figure contents
---------------
Analyse-mode combined figure (15 panels A-O on a 6-row × 3-column
grid; some rows span columns):

  A — Track overlay on mean projection
  B — Localisation density heatmap
  C — Trajectories coloured by D
  D — MSD curves (per-track + ensemble + linear fit overlay)
  E — Diffusion coefficient (log10 D) histogram, stacked by motion class
  F — Motion-type pie chart
  G — Anomalous exponent α distribution
  H — Position density (alt)
  I — Turning Angle Distribution
        |θ| line plot, 0°–180°, relative frequency, with a uniform
        reference line at 1/N_bins (signal of perfectly random motion).
  J — Mobile fraction over time
  K — Jump Distance Distribution with multi-population fits
  L — Cluster overlay (DBSCAN, if any)
  M — Dwell-time histogram + exponential τ fit
  N — Moment Scaling Spectrum (MSS slope)
  O — Radial Distribution (polar histogram of SIGNED turning angles;
        0° at top = straight ahead, right hemisphere = positive turns,
        left = negative; back-tracking peaks at ±180°)

Panels not applicable (e.g. clusters when none detected) are skipped
automatically.

Batch output
------------
Inside batch_results/:
   <filename>/            Per-file (or per-series) sub-folder with the
                          same files as above.
   batch_summary.csv      One row per experiment — n_tracks, median D,
                          mobile fraction, JDD populations, etc.


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

JDD analysis fits the cumulative distribution of single-step
displacements to a multi-population diffusion model.  Robust for short
tracks; gives D and fractional population of each mobility state.

Set the number of populations (1, 2 or 3) in Settings → MSD & JDD.
Two populations (mobile + immobile) is typical for sptPALM.

The JDD panel in the figure shows the step-size histogram with the
fitted PDF overlaid; D values and fractions are saved to <stem>_jdd.json.


================================================================
D-VALUE FILTER
================================================================

Enable "Filter by D" in Settings → MSD & JDD to restrict JDD fitting
and downstream statistics to tracks within a specific diffusion
coefficient range.  Useful for isolating a mobility population of
interest (e.g. mobile only, or above a confinement threshold).

The filter does not alter MSD fitting — all tracks are fitted first,
then only those with D inside [D min, D max] are passed to JDD and
summary stats.


================================================================
CLUSTER ANALYSIS  (DBSCAN)
================================================================

DBSCAN groups raw localisations into spatial clusters (receptor
nanodomains etc.).  Results in <stem>_cluster_stats.csv:
n_locs, area_µm² (convex hull), density, centroid x/y.

Settings:
  - DBSCAN radius (nm)      Search radius (typical 30-80 nm).
  - Min localisations       Minimum points to form a cluster.

Performance note: DBSCAN is capped at 250,000 localisations.  On
larger datasets a random subsample is used — preserves cluster spatial
pattern with much faster processing.


================================================================
DRIFT CORRECTION
================================================================

Enable in Settings → Drift Correction → Enable.

Uses full redundant cross-correlation (RCC, Wang et al. 2014):
  • The acquisition is divided into N segments (default ~200 frames).
  • An upsampled localisation density map is built for each segment.
  • EVERY pair (i, j) with i<j is cross-correlated.
  • The over-determined linear system Δ_{ij} = drift[j] − drift[i]
    is solved by least-squares with drift[0] gauge-fixed.

Robust to bad segments (one segment with too few localisations no
longer corrupts the cumulative drift), and the redundancy averages
out cross-correlation noise that a consecutive-only chain accumulates.
Per-frame drift is interpolated and subtracted before linking.

Segment size: aim for >200 localisations per segment.
  Typical values: 200-500 frames for dense labelling,
                  400-1000 frames for sparse labelling.


================================================================
ROI MASKING
================================================================

Restrict analysis to the cell region.  Four modes in Settings → ROI:

Disabled       Full frame is analysed.
Auto threshold Mean projection thresholded automatically.  Three
               algorithms (Triangle, Li, Otsu).
Manual         Set the threshold on a 0–1 scale yourself.
Draw ROI       Click "Open ROI Editor" to draw a freehand polygon.

Always check the *_roi_mask.png preview before trusting filtered results.


================================================================
TROUBLESHOOTING
================================================================

App won't open on macOS — "cannot be opened because the developer
cannot be verified"
   Right-click the app → Open → Open anyway.

App takes a long time to start on Windows
   First launch of the onefile build extracts ~200 MB of bundled
   libraries to %TEMP% — 10-30 s on slow disks.  Subsequent launches
   are quick because Python reuses the cached extraction.

No particles found / very few trajectories
   Lower the PSF diameter by 2 px and re-run.
   Disable auto-detect minmass and manually set a lower value.
   Check that the correct channel index is selected.

Too many trajectories / noise being tracked
   Increase minmass (1.0–3.0).
   Increase background radius.
   Enable ROI masking to exclude out-of-cell noise.

Pixel size or frame interval shows as WARNING
   The CZI metadata could not be read.  Tick "Set manually" and
   enter the correct values from your acquisition settings.

JDD fit does not converge / populations look wrong
   Reduce the number of JDD populations to 1 or 2.
   Ensure min track length is at least 3 frames.
   If using the D filter, widen the D range and re-run.

Out of memory during localisation
   Reduce Chunk size in Settings → Performance (200–300 frames).

Analysis is slow
   Increase CPU workers in Settings → Performance.
   Use "Uniform Filter" background method (much faster than rolling
   ball).

Batch run stops partway through
   Check the log panel — a specific file likely failed.  That file's
   sub-folder will be incomplete or absent.  Fix the file or exclude
   it and re-run batch.

Compare panels show "no data" placeholders
   Some panels rely on per-run JSON / CSV files (jdd.json,
   dwell_times.csv, turning_angles.csv, mobile_fraction.csv,
   params.json) introduced in later versions of the pipeline.  Older
   folders won't have them; re-run the analysis to regenerate.

Compare drag-and-drop does nothing
   Drag-and-drop uses Qt's native QMimeData handler in FIREFLY 2.0+.
   If it fails (rare), the + Add buttons on each group card remain
   the only way to populate a group.


================================================================
ACKNOWLEDGEMENTS
================================================================

Uses trackpy (Allan et al.), scikit-image, scipy, scikit-learn, and
matplotlib.
Localisation algorithm based on Crocker & Grier (1996).
Auto-thresholding via Otsu (1979), Li & Lee (1993), and
Zack-Rogers-Latt (1977) from scikit-image.
Drift correction based on Wang et al. (2014) RCC method (full pairwise
least-squares variant).
Jump distance distribution based on Saxton (1997) and Yu et al. (2014).
Moment scaling spectrum based on Ferrari et al. (2001).
Localisation precision after Thompson, Larson & Webb (2002).
DBSCAN from Ester et al. (1996).
Statistical tests via scipy.stats (Welch's t-test, Mann-Whitney U,
one-way ANOVA, Kruskal-Wallis, Shapiro-Wilk).
GUI built on PySide6 (Qt6).  Embedded image viewer via napari + vispy.
Developed with AI assistance (Anthropic Claude).


================================================================
CONTACT / SUPPORT
================================================================

If you encounter an error not listed above, copy the full error
message from the log panel and seek support from your facility's
microscopy or bioinformatics team.
