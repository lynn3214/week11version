# Weekly Progress Report - Data Pipeline Reconstruction

## 1. Overview

This week's work focused on rebuilding and stabilizing the entire data pipeline after discovering inconsistencies among multiple historical versions of the codebase. Most tasks were engineering-heavy but essential for ensuring a reliable training workflow.

### Main Goals
- Unify intermediate data formats
- Remove duplicate/contradicting normalization steps
- Improve automatic click extraction quality
- Redesign noise dataset construction
- Fix misalignment issues triggered after switching to `.npy` format
- Ensure all modules interoperate correctly

Although model training has not started yet, this week laid the necessary foundation for reproducible and scalable experiments.

---

## 2. Major Work Completed

### 2.1 Dataset Restructuring (Based on Last Meeting's Decisions)

Following the recommendations from the email last week:

- **Removed**: The Dataport dataset has been completely removed
- **Postponed**: The Dolphin Clicks dataset is temporarily not used (will be introduced later only if needed for performance improvement)

**Current working datasets include**:
- High-SNR click data (06Nov21_left / 06Sept15_left)
- Low-SNR click data from Singapore Waters
- Five pure noise pickle files (all converted to NPY this week)

This cleanup helps avoid domain mismatch and simplifies subsequent data handling.

### 2.2 Unifying All Intermediate Results to `.npy` Format

Based on the email:
> "Avoid saving intermediate results as WAV, because WAV introduces clipping. Use NPY for all internal steps."

**This week's implementation**:
- Completely migrated all intermediate outputs from `.wav` → `.npy`
- Updated all code paths to read/write NPY correctly:
  - batch-detect
  - manual labeler
  - dataset builder
  - noise segmentation
- Added a new utility script: `export_npy_to_wav.py` to optionally convert selected NPY clips back to WAV for manual inspection

**Benefits**: This significantly reduces the risk of amplitude clipping and ensures reproducibility.

### 2.3 Full Audit and Cleanup of Normalization Logic

A comprehensive review of the entire project revealed multiple duplicated and inconsistent normalization steps, accumulated from older code versions:

**Problems identified**:
- Different normalization methods used at different stages:
  - max-normalization
  - RMS normalization
  - zero-mean / unit-variance
  - median vs MAD
- Many samples were normalized multiple times, unintentionally distorting temporal/spectral structures
- Inconsistent application across click/noise pipelines

These inconsistencies can change click shape, spectral centroid, transient sharpness, or energy distribution—ultimately harming CNN training.

**Fix implemented this week**:
- Removed all duplicated normalizations across the codebase
- The **only remaining normalization** happens in `build-dataset`, right before mixing signal and noise:
  - For click signals → `median(abs(x))` normalization
  - For noise → same strategy applied during dataset build
- No normalization in batch-detect; no normalization when splitting noise; no normalization during augmentation

This ensures the entire pipeline is clean, unified, and mathematically consistent.

**Future plan**:
- Later explore RMS normalization vs median normalization effects
- Possibly introduce per-stage curriculum normalization for low-SNR clicks

### 2.4 Updating All Related Modules & Ensuring End-to-End Compatibility

After unifying formats and normalization, several modules required code edits and bug fixes:
- `main.py` (batch-detect, build-dataset)
- dataset builder (500 ms unified window)
- manual labeler
- noise segmentation & manifest creation
- file naming scheme (now includes source filename)
- path consistency & sanity checks

Regression testing confirms that the revised codebase runs end-to-end without critical errors.

### 2.5 Improved Click-Extraction Thresholds to Capture Weak Clicks

The previous detector settings were overly strict, producing only very "clean" and obvious clicks. This introduces a risk: CNN may only learn strong clicks and fail on low-SNR real-world data.

**Threshold adjustments**:
- `refractory_ms`: 10 → 1.5
- `min_dolphin_likelihood`: 0.5 → 0.3
- Relaxed frequency-domain thresholds

**After adjustment**:
- Total extracted clips: **2681**
  - High-confidence: **1305**
  - Low-confidence ("uncertain"): **1376**
  - Hard negatives: **0**

**Plan**:
- Use HQ samples directly for Stage 1 CNN training
- If needed, manually curate the uncertain group to improve sample diversity
- This enables a curriculum-learning strategy (details below)

---

## 3. Key Issues Encountered and Solutions

### 3.1 NPY-Based Extraction Revealed Serious Center-Alignment and Padding Bugs

After switching from WAV to NPY intermediate files, new problems emerged:
- Extracted clips had blank regions at the beginning
- Some clips looked like concatenated blocks
- Click not centered
- Reflect padding introduced unnatural mirrored noise

**Why these problems didn't appear last week**:
- Last week only processed 120 ms segments
- This week's 500 ms window makes boundary issues unavoidable
- WAV may implicitly rescale or align data when saving
- NPY exposes the raw indexing bugs directly

**Root cause**: Faulty center-window extraction logic

```python
start_idx = max(0, peak_idx - half_window)
end_idx = min(len(audio), peak_idx + half_window)
segment = audio[start_idx:end_idx]

if len(segment) < segment_samples:
    segment = np.pad(segment, reflect)
```

**Problems**:
1. **Boundary truncation**: When `peak_idx` is near the left edge, `start_idx = 0`, shifting the click to the right
2. **Reflect padding corrupts the spectral shape**: Mirroring near-silence or noise produces unnatural patterns
3. **Inconsistent temporal context**: Some clicks centered, some shifted—bad for CNN

**Solution (adopted this week)**:
- Skip all boundary clicks and only keep fully valid 500 ms windows

**Benefits**:
- Ensures identical temporal alignment
- Eliminates reflect-padding artifacts
- Improves CNN stability
- Simplifies future work
- Weak clicks near boundaries can re-enter later in Stage 2 curriculum training

---

## 4. Additional Technical Thoughts

### 4.1 Risk of "Too-Clean" Training Samples

Automatic rule-based detection tends to extract "easy" high-SNR clicks.

**Concern**: Will the CNN fail to generalize if trained only on clean clicks?

**Solution**: ✔ Adopt a **Curriculum Learning strategy**
- **Stage 1 (now)**: Train on high-quality clicks
- **Stage 2 (later)**: Gradually include weak clicks or uncertain candidates
- **Stage 3 (optional)**: Adversarial mixing / synthetic degradation

This allows stable convergence while preserving robustness.

### 4.2 Normalization Choice: MAD vs Median-Abs

**Option A**: `signal / median(abs(signal))`
- ✔ Keeps the waveform shape, maintains transient structure
- ✔ Best choice for click detection

**Option B**: MAD z-score
```python
median = np.median(signal)
mad = np.median(np.abs(signal - median))
normalized = (signal - median) / (1.4826 * mad)
```
- ✘ Removes DC offset, scales by robust variance
- ✘ Distorts transient spikes
- ✘ Not suitable for dolphin clicks

**Final choice**: Use `median(abs(x))` for both signal and noise.

### 4.3 No Training Results Yet

**Reasons**:
- Exam preparation
- Large pipeline refactoring
- Need for code consistency validation

Model training, validation, and final test evaluation will resume after exams.

---

## 5. Summary

This week's work focused on core pipeline engineering and consistency fixes, including:
- ✅ Full dataset restructuring
- ✅ Migration to NPY pipeline
- ✅ Elimination of redundant normalization
- ✅ Robust click extraction improvements
- ✅ Noise segmentation and manifest generation
- ✅ Debugging alignment issues introduced by NPY windows
- ✅ Preparing for future curriculum learning

These steps are critical foundations and represent substantial and meaningful progress toward a stable and scalable experimental workflow.