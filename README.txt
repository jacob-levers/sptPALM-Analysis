sptPALM Analysis Pipeline — By Jacob Levers
============================================
Single-particle tracking PALM analysis for Zeiss Elyra CZI and TIFF files.
Compatible with PC12 cells, Drosophila neurons, and other cell types.
Developed at the University of Queensland, 2026.


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

1. Click "Browse" and select your .czi or .tif file
2. Set an output folder (defaults to same folder as input)
3. Review the settings panel on the left — hover over any ⓘ icon
   for a description of that parameter and suggested values
4. Click "▶ Run Analysis"
5. Watch the live preview and progress log on the right
6. When done, click "Open Output Folder" to see results

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

   *_sptpalm_figure.png      6-panel results figure. Open this first.
   *_diffusion_summary.csv   D coefficient, alpha, and motion type per
                             trajectory. Open in Excel.
   *_trajectories.csv        Full x/y trajectory data for every particle.
   *_localisations.csv       Raw molecule positions for every frame.
   *_ensemble_msd.csv        Ensemble-averaged MSD curve data.
   *_drift.csv               Per-frame drift estimates (if drift correction
                             was enabled).
   *_roi_mask.png            Preview of the ROI mask (if ROI masking was
                             enabled).


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

Results look different from PALMTracer
   Different localisation algorithms and filtering produce different
   results. PALMTracer applies a hard MSD cutoff that removes slow
   molecules. This pipeline retains them, which is why the immobile
   fraction may appear higher — those molecules are real, not artefacts.

Out of memory during localisation
   Reduce Chunk size in Settings → Performance (try 200–300 frames).

Analysis is slow
   Increase CPU workers in Settings → Performance.
   Use "Uniform Filter" background method (much faster than Rolling Ball).


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
