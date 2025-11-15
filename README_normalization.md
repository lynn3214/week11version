---

````markdown
# Preprocessing Pipeline Documentation

## Overview
This document describes the **full preprocessing workflow** used for model training and evaluation.  
The goal is to ensure that both training and testing data undergo identical signal processing steps, avoiding distribution mismatch.

---

## Complete Preprocessing Pipeline

### Step 1: Resampling and Filtering
```bash
python resample_and_filter.py \
  --input your_data \
  --output data_resampled \
  --sr_target 44100 \
  --hp_cutoff 1000 \
  --verbose
````

**Parameters**:

* `sr_target`: Target sampling rate 44.1 kHz
* `hp_cutoff`: High-pass filter cutoff frequency 1000 Hz
* Output: Mono-channel float32 WAV files

**What this step does**:

* Resamples all audio to 44.1 kHz
* Applies a 4th-order Butterworth high-pass filter at 1 kHz
* Converts stereo to mono (if applicable)
* Ensures consistent `float32` format

> The output of this step serves as the input to Step 2 (Normalization).

---

### Step 2: Normalization

#### For Click Segments (Positive Samples):

```bash
python normalize_segments.py \
  --input data_resampled/clicks \
  --output data_normalized/clicks \
  --method peak \
  --target 0.95 \
  --recursive
```

**Normalization Formula**:

```
x_normalized = x / max(|x|) * 0.95
```

**Why peak normalization for clicks?**

* Preserves waveform shape characteristics
* Peak amplitude is a key distinguishing feature of dolphin clicks
* Simple, interpretable, and effective
* Prevents clipping (5% headroom)

---

#### For Noise Segments (Negative Samples, if needed):

```bash
python normalize_segments.py \
  --input data_resampled/noise \
  --output data_normalized/noise \
  --method rms \
  --target-rms 0.1 \
  --peak-limit 0.95 \
  --recursive
```

**Normalization Formula**:

```
Step 1: x_rms = x * (0.1 / RMS(x))
Step 2: if max(|x_rms|) > 0.95:
            x_normalized = x_rms / max(|x_rms|) * 0.95
        else:
            x_normalized = x_rms
```

**Why RMS normalization for noise?**

* Controls noise energy level (RMS reflects signal power)
* Enables precise SNR-based mixing during dataset generation
* Aligns with common underwater acoustic signal processing practices

---

## Data Characteristics

### Training Sample Specifications

* **Sampling rate**: 44100 Hz
* **Segment length**: 22050 samples (500 ms)
* **Amplitude range**: [-1.0, 1.0]
* **Typical peak**: ~0.95
* **Click RMS**: Variable (depends on original amplitude)
* **Noise RMS**: ~0.1

### SNR Mixing (Training Set Only)

* **SNR calculation**: Based on RMS, not peak amplitude
* **SNR range**: [-5, 15] dB
* **Mixing formula**:

```
SNR_linear = 10^(SNR_dB / 20)
noise_scaled = noise * (RMS_signal / (SNR_linear * RMS_noise))
mixed = clip(signal + noise_scaled, -1.0, 1.0)
```

---

## Verify Your Preprocessing

Run this Python script to check if your data is correctly preprocessed:

```python
import soundfile as sf
import numpy as np
from pathlib import Path

def check_preprocessing(data_dir):
    """Verify preprocessing correctness"""
    wav_files = list(Path(data_dir).rglob('*.wav'))
    
    print(f"Checking {len(wav_files)} files...\n")
    
    for wav in wav_files[:10]:  # Check first 10 files
        audio, sr = sf.read(wav)
        
        print(f"{wav.name}:")
        print(f"  Sample rate: {sr} Hz")
        print(f"  Length: {len(audio)} samples ({len(audio)/sr*1000:.1f} ms)")
        print(f"  Range: [{audio.min():.3f}, {audio.max():.3f}]")
        print(f"  Peak: {np.max(np.abs(audio)):.3f}")
        print(f"  RMS: {np.sqrt(np.mean(audio**2)):.3f}")
        print()

check_preprocessing('data_normalized/clicks')
```

**Expected Output**:

```
Sample rate: 44100 Hz
Length: 22050 samples (500.0 ms)
Range: [-0.95, 0.95] (or close to this)
Peak: ~0.95
```

**Red Flags**:

* ❌ Sample rate ≠ 44100 Hz → resampling step skipped
* ❌ Length ≠ 22050 samples → incorrect segment extraction
* ❌ Peak > 1.0 or Peak < 0.5 → indicates normalization or DC offset issue
* ❌ Extreme RMS values (< 0.001 or > 0.5) → inconsistent power scaling

---

## Quick Reference: Command Examples

### Processing click segments from manual labeling

```bash
# Step 1: Resample and filter
python resample_and_filter.py \
  --input raw_clicks/ \
  --output resampled_clicks/ \
  --sr_target 44100 \
  --hp_cutoff 1000 \
  --verbose

# Step 2: Normalize
python normalize_segments.py \
  --input resampled_clicks/ \
  --output normalized_clicks/ \
  --method peak \
  --target 0.95 \
  --recursive
```

### Processing noise segments

```bash
# Step 1: Resample and filter
python resample_and_filter.py \
  --input raw_noise/ \
  --output resampled_noise/ \
  --sr_target 44100 \
  --hp_cutoff 1000 \
  --verbose

# Step 2: Normalize
python normalize_segments.py \
  --input resampled_noise/ \
  --output normalized_noise/ \
  --method rms \
  --target-rms 0.1 \
  --peak-limit 0.95 \
  --recursive
```

---

## Configuration Files

All preprocessing parameters are stored in configuration files.

**`configs/detection_enhanced.yaml`** (for batch-detect)

```yaml
sample_rate: 44100

filter:
  low_freq: 2000
  high_freq: 20000
  order: 4
```

**`configs/training.yaml`** (for dataset building)

```yaml
sample_rate: 44100

dataset:
  window_ms: 120.0      # Kept for backward compatibility (not used)
  unified_length_ms: 500.0
  
augmentation:
  snr_range: [-5, 15]
```

> Note: `window_ms` is retained only for backward compatibility.
> The current pipeline uses 500 ms segments for both training and evaluation.

---

## File Structure for Testing

Recommended directory structure:

```
your_test_data/
├── raw/                          # Original recordings
│   ├── file1.wav
│   └── file2.mat
├── resampled/                    # After Step 1
│   ├── file1.wav
│   └── file2.wav
└── normalized/                   # After Step 2 (ready for model)
    ├── file1.wav
    └── file2.wav
```

> Files in the `normalized/` directory are ready for direct model inference or evaluation.

---

## Summary

Following this pipeline ensures that all datasets—training, validation, and testing—share the same preprocessing configuration,
which is critical for reproducible and reliable dolphin click detection results.

```

