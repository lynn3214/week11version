### 1. Review of Previous Meeting & Identified Issues  

**Problem 1 — Fragmentation strategy (120 ms → 500 ms)**  
- Destroyed temporal pattern: Randomly combined clicks lost the true ICI (Inter-Click Interval) structure.  
- Inconsistent noise background: Each click came from a different file, causing noise mismatch.  
- Lost contextual cues: Local background and pre-/post-click acoustic context disappeared.  
- Not realistic for deployment: In real detection, the model receives continuous 500 ms audio segments, not manually concatenated sequences.

**Problem 2 — CNN architecture design flaw**  
- Dropout was incorrectly placed at the end instead of between convolutional layers.  
- Only one convolutional layer, insufficient for hierarchical feature extraction.

---

### 2. Major Modifications  

#### (1) Data construction pipeline  
Directly segment 500 ms audio clips around detected peaks  
```python
start_idx = peak_idx - 250ms
end_idx   = peak_idx + 250ms
````

Add 500 ms noise segments from same-source noise files (0 dB SNR).
Simplified dataset builder:

```python
def cmd_build_dataset_simplified(args):
    """
    Simplified dataset builder:
    1. Load 500 ms click segments (already contextual)
    2. Load 500 ms noise segments
    3. Optional augmentation
    4. Save to output directory
    """
    positive_samples = load_click_segments(events_dir)
    negative_samples = load_noise_segments(noise_dir)

    if augment:
        positive_samples = augment_samples(positive_samples)

    save_dataset(positive_samples, negative_samples, output_dir)
```

Simplified YAML config:

* `model.input_length = 22050` (500 ms @ 44.1 kHz)
* `num_blocks = 3`, `base_channels = 16`, `dropout = 0.3`
* Removed deprecated `click_train` block
* Kept `balance_ratio = 1.0`, `val_split = 0.2`

---

#### (2) CNN Architecture Revision  

The current CNN model adopts the **OptimizedClickClassifier** structure, redesigned for 500 ms input windows (22 050 samples @ 44.1 kHz).  

**Key characteristics:**  
- **Depth:** 6 convolutional layers + 1 residual block (simplified)  
- **Width:** Maximum 32 channels (vs 128 in the previous version)  
- **Temporal resolution:** Final layer receptive field ≈ 8 ms  
- **Dropout:** Applied *only* to the final fully-connected layer (for small dataset stability)  
- **Parameter count:** ≈ 5 k–8 k parameters (≈ 75 % reduction compared to the original model)  

**Detailed layer structure:**  
| Stage | Operation | Channels | Kernel / Stride / Padding | Notes |
|--------|------------|-----------|---------------------------|-------|
| Input | 1-D Conv + BN + ReLU | 8 | 3 / 1 / 1 | Initial feature extraction |
| Residual Block 1 | Conv → BN → ReLU → Conv → BN + Shortcut | 8 → 8 | 3 / 1 / 1 | Preserves transient envelope |
| Conv2 | 8 → 8 | 5 / 2 / 2 | Down-sampling (≈ 4× reduction) |
| Conv3 | 8 → 16 | 3 / 2 / 1 | Mid-level features |
| Conv4 | 16 → 16 | 5 / 2 / 2 | High-frequency context |
| Conv5 | 16 → 32 | 3 / 2 / 1 | Deep representation, ≈ 8 ms resolution |
| Global Pool + FC | Adaptive AvgPool → Dropout(0.3) → Linear(32 → 2) | – | – | Output logits |

**Design rationale:**  
- **Residual connection** stabilizes gradient flow in early layers.  
- **Progressive down-sampling** increases receptive field while keeping time precision sufficient for click transients.  
- **Narrow channels** prevent overfitting given limited positive samples.  
- **Single dropout layer** provides lightweight regularization without harming convergence.  

This structure replaces the previous single-layer CNN and misplaced dropout design, achieving clearer feature hierarchies and faster convergence while maintaining robustness on limited data.


---

#### (3) Dataset grouping & SNR control

* Grouped noise segments by original file source to maintain consistent background.
* Dataset split:

  * Train groups: group 1–3 (≈ 6000 files)
  * Validation group: group 4 (≈ 2000 files)
  * Test group: group 5 (≈ 2000 files)
* Test split rule:

  * Sentosa → apply added noise (controlled SNR)
  * Singapore coastal → use as-is (already noisy)

---

#### (4) Manual click verification

* Implemented new script `manual_click_labeler.py` for manual review of detected click segments.
* Allows removing false positives and tagging low-SNR or ambiguous clicks.

---

### 3. Current Progress

| Item                     | Status                              | Notes                                             |
| ------------------------ | ----------------------------------- | ------------------------------------------------- |
| Code refactoring         | Completed                           | All modules updated and verified to run           |
| Click segment extraction | 4000+ segments                      | Manual verification ongoing                       |
| Manual labeling          | In progress                         | High-quality subset selected for initial training |
| Noise augmentation       | 0 dB same-source mixing             | Works as intended                                 |
| Training pipeline        | Functional                          | Early test shows good performance                 |
| Detection config         | Updated (`detection_enhanced.yaml`) | Refined transient thresholds                      |

---

### 4. Preliminary Findings

* Initial training with a few hundred labeled 500 ms clips already produces very high accuracy, suggesting the revised segmentation and architecture are effective.
* Almost no misclassified shrimp noise after enhancement.
* Some extremely low-SNR or sparse clicks are harder to label and will be considered in secondary training.

---

### 5. Next Steps

1. Finish manual labeling of remaining click segments.
2. Retrain CNN with full verified dataset.
3. Evaluate separately on:

   * Sentosa (added noise)
   * Singapore coastal (natural noise)
4. Analyze feature maps and misclassified samples to confirm focus on transient features.
5. Optional experiments:

   * Multi-SNR augmentation (0 → –5 dB)
   * Compare context length (500 ms vs 1 s)

```
```

### 6. Workflow Summary (Current Implementation)

The current data processing and training workflow is as follows:

1. **Organize raw data** (separate training / test sets)

   python prepare_data.py \
     --raw_dir data/raw \
     --output_dir data \
     --verbose

2. **Resample and high-pass filter** all datasets to 44.1 kHz, cutoff 1 kHz


   # Training data
   python preprocessing/resample_and_filter.py \
     --input data/raw/training_sources \
     --output data/training_resampled \
     --sr_target 44100 \
     --hp_cutoff 1000 \
     --verbose

   # Test data
   python preprocessing/resample_and_filter.py \
     --input data/test_raw \
     --output data/test_resampled \
     --verbose

   # Noise data
   python preprocessing/resample_and_filter.py \
     --input data/raw/noise \
     --output data/noise_resampled \
     --verbose


3. **Split noise into 1-second segments and generate manifest**


   python scripts/split_noise_manifest.py \
     --input-dir data/noise_resampled \
     --output manifests/noise_manifest.csv


   Output summary:

   * 10,000 noise files → 20,000 half-segments
   * Train: 12,000, Val: 4,000, Test: 4,000 (each = 1 s × 2000 files)

4. **Detect and extract 500 ms click segments**


   python main.py batch-detect \
     --input-dir data/training_resampled \
     --output-dir data/detection_results \
     --config configs/detection_enhanced.yaml \
     --save-audio \
     --segment-ms 500 \
     --recursive \
     --verbose
   

   *Observed some misclassifications — plan to refine rule-based detector thresholds before next round.*

5. **Manual verification of detected clicks**


   python scripts/manual_click_labeler.py \
     --input data/detection_results/audio \
     --output data/manual_labelled \
     --csv data/detection_results/all_events.csv


   Output directory structure:


   data/manual_labelled/
   ├── Positive_HQ/       ← high-quality clicks (used for training)
   ├── Negative_Hard/     ← false detections
   └── Quarantine/        ← uncertain cases


   *The high-quality clicks in `Positive_HQ/` are used directly for dataset construction.*

6. **Build training dataset (clicks + noise overlay at 0 dB SNR)**


   python main.py build-dataset \
     --events-dir data/manual_labelled/Positive_HQ \
     --noise-manifest manifests/noise_manifest.csv \
     --split train \
     --output-dir data/training_dataset \
     --config configs/training.yaml \
     --save-wav \


7. **Train CNN model**
   python main.py train \
       --dataset-dir data/training_dataset \
       --output-dir checkpoints/v1.1 \
       --config configs/training.yaml \
       --verbose
   

This pipeline now replaces the previous `collect-clicks` and `click-train` based approach.
It ensures that every training sample originates from **verified click events** and **consistent background noise sources**, significantly improving dataset reliability and realism.



去掉dataport数据集
dolphin clicks下面的数据集先不用，之后想要improve的时候再用
修改归一化代码，不要使用多种归一化和Max归一化，考虑使用median(abs(x)) normalization
避免中间结果被保存成wav格式，因为会被clipped，这个可能才是被clipped的原因。中间结果使用npy格式保存

尝试另一种方法，乘1.349左右的数字。这个我没太懂艾q儿这样的发音。高斯假设
robust因为不用考虑越界值
1.349
提取的时候如果click在开头或者结尾的时候，不需要去掉，直接往前或者往后取值，而不是位于中间就可以


直接提取一秒左右长度的片段，然后在这个一秒范围内取多个片段，这样就可以实现augmentation，最多一个片段取三个子片段
提升effective number of samples，因为如果overlap太多的话实际上就是在对非常类似的片段进行训练和识别
提升ESS