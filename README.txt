sptPALM Analysis Pipeline — By Jacob Levers
============================================
Single-particle tracking PALM analysis for Zeiss Elyra CZI and TIFF files.
Compatible with PC12 cells, Drosophila neurons, and other cell types.


================================================================
REQUIREMENTS
================================================================

- Windows 10/11  OR  macOS 11 (Big Sur) or newer
- 8 GB RAM minimum (16–32 GB recommended for large datasets)
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
1. Download sptPALM-Windows.zip from the Releases page on GitHub
2. Right-click the .zip → Extract All → choose a folder
3. Open the extracted folder and double-click sptPALM.exe


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
automatically. This takes 3–5 minutes. Subsequent launches are instant.

Windows
-------
Install Python from https://www.python.org/downloads/
IMPORTANT: Tick "Add Python to PATH" during installation.

Then double-click:

   Launch_sptPALM.bat

On first launch the app detects missing libraries and installs them
automatically. This takes 3–5 minutes. Subsequent launches are instant.


================================================================
USING THE APP
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
  - Frame interval (s)      Auto-read from CZI. Verify it is correct.
  - PSF diameter (px)       Must be odd. Typically 7 px for 561 nm on Elyra.
  - Search range (px)       Set to ~2–3× expected single-step displacement.
  - Min track length        At least 2–3× the "Fit first N lag points" value.


================================================================
OUTPUT FILES
================================================================

After the run completes, a results folder appears containing:

   *_sptpalm_figure.png      Results figure (see Figure Contents below).
   *_diffusion_summary.csv   D coefficient, alpha, motion type, confinement
                             radius, and localisation precision per track.
                             Open in Excel.
   *_trajectories.csv        Full x/y trajectory data for every particle.
   *_localisations.csv       Raw molecule positions for every frame.
   *_ensemble_msd.csv        Ensemble-averaged MSD curve data.
   *_drift.csv               Per-frame drift estimates (if drift correction
                             was enabled).
   *_roi_mask.png            Preview of the ROI mask (if ROI masking was
                             enabled).

Figure contents
---------------
Row 1:  A — Track overlay on mean projection
        B — MSD curve (ensemble)
        C — Diffusion coefficient histogram
Row 2:  D — Alpha (anomalous exponent) histogram
        E — Motion-type pie chart
        F — Track length histogram
Row 3:  G — Localisation density heatmap (all track positions in µm)
        H — Jump Distance Distribution (JDD) with population fits

Batch output
------------
Inside batch_results/:
   <filename>/            Per-file sub-folder with the same files as above
   batch_summary.csv      One row per file — n_tracks, mean D, mobile
                          fraction, mean confinement radius, JDD populations


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
panel and saved to *_diffusion_summary.csv.


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
DRIFT CORRECTION
================================================================

Enable in Settings → Drift Correction → Enable.

Uses reference-free redundant cross-correlation (RCC, Wang et al. 2014).
No fiducial markers required. Applied before trajectory linking so
corrected positions improve track quality.

Segment size: aim for >200 localisations per segment.
Typical values: 200–500 frames for dense labelling,
400–1000 frames for sparse labelling.


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
   Set the threshold yourself on a 0–1 scale relative to the
   brightest region in the mean projection.
   Start at 0.10–0.15 for PC12 cell bodies.
   Start at 0.05–0.08 for thin Drosophila neurites.

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

- Confinement radius is reported per track in *_diffusion_summary.csv.
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

No particles found / very few trajectories
   Lower the PSF diameter by 2 px and re-run.
   Disable auto-detect minmass and manually set a lower value (0.3–1.0).
   Check that the correct channel index is selected.

Too many trajectories / noise being tracked
   Increase minmass (try 1.0–3.0).
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
   Reduce Chunk size in Settings → Performance (try 200–300 frames).

Analysis is slow
   Increase CPU workers in Settings → Performance.
   Use "Uniform Filter" background method (much faster than Rolling Ball).

Batch run stops partway through
   Check the log panel — a specific file likely failed. That file's
   sub-folder in batch_results/ will be incomplete or absent. Fix the
   file or exclude it and re-run batch on the remaining files.


================================================================
ACKNOWLEDGEMENTS
================================================================

Uses trackpy (Allan et al.), scikit-image, and scipy.
Localisation algorithm based on Crocker & Grier (1996).
Auto-thresholding uses Otsu (1979), Li & Lee (1993), and
Zack, Rogers & Latt (1977) algorithms via scikit-image.
Drift correction based on Wang et al. (2014) RCC method.
Developed with AI assistance (Anthropic Claude).


================================================================
CONTACT / SUPPORT
================================================================

If you encounter an error not listed above, copy the full error
message from the log panel and seek support from your facility's
microscopy or bioinformatics team.
