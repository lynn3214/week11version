# week13
# 步骤1.1: 组织原始数据（分离训练/测试集）
python prepare_data.py \
  --raw_dir data/raw \
  --output_dir data \
  --verbose

# 步骤2.1: 统一处理训练数据 (WAV + MAT)
python preprocessing/resample_and_filter.py \
  --input data/raw/training_sources \
  --output data/training_resampled \
  --sr_target 44100 \
  --hp_cutoff 1000 \
  --verbose

# 步骤2.2: 处理测试数据
python preprocessing/resample_and_filter.py \
  --input data/test_raw \
  --output data/test_resampled \
  --verbose

# 步骤2.3: 处理噪音数据
python preprocessing/resample_and_filter.py \
  --input data/raw/noise \
  --output data/noise_resampled \
  --verbose

# 步骤3: 噪音集处理（片段划分，数据集组建）
python scripts/segment_and_split_noise.py \
  --input-dir data/noise_resampled \
  --output-dir data/noise_segments \
  --manifest-output manifests/noise_manifest.csv \
  --train-ratio 0.6 \
  --val-ratio 0.1 \
  --test-ratio 0.3 \
  --seed 42 \
  --verbose

Found 10000 noise files

✅ Generated manifest: manifests/noise_manifest.csv
Total half-segments: 20000 (10000 × 1s files)

Split distribution:
  test: 4000 half-segments (2000 1s files)
  train: 12000 half-segments (6000 1s files)
  val: 4000 half-segments (2000 1s files)

# 步骤4.1：检测并提取click片段(500ms)
python main.py batch-detect \
  --input-dir data/training_resampled \
  --output-dir data/detection_results \
  --config configs/detection_enhanced.yaml \
  --save-audio \
  --segment-ms 500 \
  --recursive 

# 步骤4.2：自动筛选分类
python scripts/auto_filter_uncertain.py \
  --events-csv data/detection_results/all_events.csv \
  --audio-dir data/detection_results/audio \
  --output data/filtered

##### 到这一步了，但是用audacity看波形图，发现应该有误判。明天计划是检查rule based detector的规则、逻辑，减少一些误判再进行之后的步骤
## 步骤4.3 手动筛选
python scripts/manual_click_labeler.py \
  --input data/detection_results/audio \
  --output data/manual_labelled \
  --csv data/detection_results/all_events.csv

## 可以调用分批标注功能
# 第一批：标注100个左右
## 步骤4.3 手动筛选，随时中断，继续的时候脚本会记住已标注的。全部运行完之后会更行all_events.csv文件
python scripts/manual_click_labeler.py \
  --input data/detection_results/audio \
  --output data/manual_labelled \
  --shuffle

# 差不多了就退出

# 第二批：继续标注（脚本会记住已标注的）
python scripts/manual_click_labeler.py \
  --input data/detection_results/audio \
  --output data/manual_labelled

# 运行完成之后会更新all_events.csv文件，接下来正常进行之后的步骤就可以
# 输出：
# data/manual_labelled/
# ├── Positive_HQ/             ← ✅ 高质量click
# ├── Negative_Hard/           ← ❌ 明确误判
# └── Quarantine/              ← ⚠️ 不确定
#
# data/detection_results/all_events.csv  ← ✅ 已更新（只保留Positive_HQ）

```
不再需要这一步了。collect click的功能是将原本data/detection_result/audio目录下面，每个音频文件夹下面的click片段整理到同一个目录中
以方便后续处理
顺便检查是否存在命名冲突
现在由于进行了手动标注，所以高质量样本、直接用于训练集正样本的click片段转移到了data/manual_labelled/Positive_HQ/ 这个目录下
因此直接build dataset组建训练集和验证集就可以了
# 步骤4.2: 收集click片段
python scripts/collect_clicks.py \
  --input data/detection_results/audio \
  --output data/training_clicks \
  --verbose
```


# ========== 步骤5: 直接使用高质量片段叠加噪音构建训练集 ==========
python main.py build-dataset \
  --events-dir detection_results/audio \
  --noise-manifest manifests/noise_manifest.csv \
  --split train \
  --output-dir dataset \
  --verbose

# ========== 步骤6: 训练模型 ==========
python main.py train \
    --dataset-dir data/training_dataset \
    --output-dir checkpoints/v1.1 \
    --config configs/training.yaml \
    --verbose
    
python main.py train \
  --dataset-dir data/training_dataset \
  --output-dir models/checkpoints \
  --verbose
