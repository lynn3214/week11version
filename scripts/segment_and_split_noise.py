#!/usr/bin/env python3
"""
噪音切割与划分脚本 (新策略)

策略:
1. 统一噪音池: 所有 purenoise_1-5 混合
2. 全局随机打散
3. 片段级严格分离: train/val/test 无重叠
4. 比例: 60% train / 10% val / 30% test
5. 输出 manifest CSV 供后续使用
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def load_all_noise_segments(
    noise_dir: Path,
    segment_length_ms: float = 500.0,
    sample_rate: int = 44100,
    verbose: bool = False
) -> list:
    """
    加载所有噪音 NPY 文件,切割成固定长度片段
    
    Returns:
        List[dict]: 每个元素包含 {audio, parent_id, segment_id}
    """
    noise_dir = Path(noise_dir)
    
    # 查找所有 purenoise_*.npy 文件
    npy_files = sorted(noise_dir.glob("purenoise_*.npy"))
    
    if not npy_files:
        raise FileNotFoundError(f"未找到任何 purenoise_*.npy 文件: {noise_dir}")
    
    logging.info(f"找到 {len(npy_files)} 个噪音文件")
    
    segment_samples = int(segment_length_ms * sample_rate / 1000)
    all_segments = []
    
    for npy_file in tqdm(npy_files, desc="加载噪音文件", disable=not verbose):
        # 从文件名提取 parent_id
        # 例如: purenoise_3_01234.npy → parent_id=3
        file_stem = npy_file.stem  # purenoise_3_01234
        parts = file_stem.split('_')
        
        if len(parts) >= 2:
            parent_id = int(parts[1])  # 3
        else:
            logging.warning(f"无法解析文件名: {npy_file.name}, 跳过")
            continue
        
        try:
            # 加载音频
            audio = np.load(npy_file)
            
            # 转单声道
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 检查长度
            if len(audio) < segment_samples:
                logging.debug(f"片段太短,跳过: {npy_file.name} ({len(audio)} samples)")
                continue
            
            # 如果恰好是 500ms,直接使用
            if len(audio) == segment_samples:
                all_segments.append({
                    'audio': audio,
                    'parent_id': parent_id,
                    'original_file': npy_file.name,
                    'start': 0,
                    'end': len(audio)
                })
            else:
                # 如果更长,切成多个片段 (无重叠)
                for start_idx in range(0, len(audio) - segment_samples + 1, segment_samples):
                    segment = audio[start_idx:start_idx + segment_samples]
                    
                    all_segments.append({
                        'audio': segment,
                        'parent_id': parent_id,
                        'original_file': npy_file.name,
                        'start': start_idx,
                        'end': start_idx + segment_samples
                    })
        
        except Exception as e:
            logging.error(f"加载失败 {npy_file}: {e}")
            continue
    
    logging.info(f"总共收集 {len(all_segments)} 个 500ms 噪音片段")
    
    # 统计每个 parent_id 的片段数
    from collections import Counter
    parent_counts = Counter([seg['parent_id'] for seg in all_segments])
    logging.info(f"片段分布:")
    for pid in sorted(parent_counts.keys()):
        logging.info(f"  purenoise_{pid}: {parent_counts[pid]} 片段")
    
    return all_segments


def split_segments(
    segments: list,
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
    test_ratio: float = 0.3,
    seed: int = 42
) -> tuple:
    """
    随机划分片段到 train/val/test
    
    Returns:
        (train_segments, val_segments, test_segments)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例之和必须为1.0: {train_ratio + val_ratio + test_ratio}")
    
    # 全局随机打散
    random.seed(seed)
    shuffled = segments.copy()
    random.shuffle(shuffled)
    
    # 计算划分点
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_segments = shuffled[:train_end]
    val_segments = shuffled[train_end:val_end]
    test_segments = shuffled[val_end:]
    
    logging.info(f"\n划分结果:")
    logging.info(f"  Train: {len(train_segments)} 片段 ({len(train_segments)/n*100:.1f}%)")
    logging.info(f"  Val:   {len(val_segments)} 片段 ({len(val_segments)/n*100:.1f}%)")
    logging.info(f"  Test:  {len(test_segments)} 片段 ({len(test_segments)/n*100:.1f}%)")
    
    return train_segments, val_segments, test_segments


def save_segments_as_npy(
    segments: list,
    output_dir: Path,
    split_name: str,
    verbose: bool = False
) -> list:
    """
    保存片段为 NPY 文件
    
    Returns:
        List[dict]: manifest 记录
    """
    output_dir = Path(output_dir) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_records = []
    
    for i, seg in enumerate(tqdm(segments, desc=f"保存 {split_name}", disable=not verbose)):
        # 文件名: noise_{split}_{idx:06d}.npy
        filename = f"noise_{split_name}_{i:06d}.npy"
        filepath = output_dir / filename
        
        # 保存音频
        np.save(filepath, seg['audio'])
        
        # 记录到 manifest
        manifest_records.append({
            'split': split_name,
            'segment_id': i,
            'filename': filename,
            'path': str(filepath),
            'parent_id': seg['parent_id'],
            'original_file': seg['original_file'],
            'start': seg['start'],
            'end': seg['end'],
            'duration_ms': 500.0,
            'samples': len(seg['audio'])
        })
    
    return manifest_records


def verify_no_overlap(train_records, val_records, test_records):
    """验证 train/val/test 无重叠片段"""
    
    def make_key(record):
        # 使用 (parent_id, original_file, start) 作为唯一标识
        return (record['parent_id'], record['original_file'], record['start'])
    
    train_keys = {make_key(r) for r in train_records}
    val_keys = {make_key(r) for r in val_records}
    test_keys = {make_key(r) for r in test_records}
    
    # 检查交集
    train_val_overlap = train_keys & val_keys
    train_test_overlap = train_keys & test_keys
    val_test_overlap = val_keys & test_keys
    
    if train_val_overlap or train_test_overlap or val_test_overlap:
        logging.error(f"❌ 发现重叠片段!")
        logging.error(f"  Train-Val: {len(train_val_overlap)}")
        logging.error(f"  Train-Test: {len(train_test_overlap)}")
        logging.error(f"  Val-Test: {len(val_test_overlap)}")
        return False
    else:
        logging.info(f"✅ 验证通过: Train/Val/Test 无重叠片段")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='噪音切割与划分 (新策略: 全局混洗 + 片段级分离)'
    )
    parser.add_argument(
        '--input-dir', type=str, required=True,
        help='输入目录 (包含 purenoise_*.npy 文件)'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='输出目录 (保存划分后的片段)'
    )
    parser.add_argument(
        '--manifest-output', type=str, default='manifests/noise_manifest.csv',
        help='Manifest CSV 输出路径'
    )
    parser.add_argument(
        '--train-ratio', type=float, default=0.6,
        help='训练集比例 (默认: 0.6)'
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.1,
        help='验证集比例 (默认: 0.1)'
    )
    parser.add_argument(
        '--test-ratio', type=float, default=0.3,
        help='测试集比例 (默认: 0.3)'
    )
    parser.add_argument(
        '--segment-ms', type=float, default=500.0,
        help='片段长度(毫秒), 默认500'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=44100,
        help='采样率, 默认44100'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='随机种子, 默认42'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='详细日志'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logging.info("=" * 70)
    logging.info("噪音切割与划分 (新策略)")
    logging.info("=" * 70)
    logging.info(f"输入目录: {args.input_dir}")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"划分比例: Train {args.train_ratio:.0%} / Val {args.val_ratio:.0%} / Test {args.test_ratio:.0%}")
    logging.info(f"片段长度: {args.segment_ms}ms")
    logging.info(f"随机种子: {args.seed}")
    
    # 步骤1: 加载所有噪音片段
    logging.info("\n步骤1: 加载所有噪音片段...")
    all_segments = load_all_noise_segments(
        noise_dir=Path(args.input_dir),
        segment_length_ms=args.segment_ms,
        sample_rate=args.sample_rate,
        verbose=args.verbose
    )
    
    if len(all_segments) == 0:
        logging.error("❌ 未找到有效的噪音片段!")
        return 1
    
    # 步骤2: 随机划分
    logging.info("\n步骤2: 随机划分 Train/Val/Test...")
    train_segments, val_segments, test_segments = split_segments(
        segments=all_segments,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # 步骤3: 保存片段为 NPY
    logging.info("\n步骤3: 保存片段...")
    output_dir = Path(args.output_dir)
    
    train_records = save_segments_as_npy(train_segments, output_dir, 'train', args.verbose)
    val_records = save_segments_as_npy(val_segments, output_dir, 'val', args.verbose)
    test_records = save_segments_as_npy(test_segments, output_dir, 'test', args.verbose)
    
    # 步骤4: 验证无重叠
    logging.info("\n步骤4: 验证片段分离...")
    verify_no_overlap(train_records, val_records, test_records)
    
    # 步骤5: 保存 Manifest CSV
    logging.info("\n步骤5: 保存 Manifest CSV...")
    all_records = train_records + val_records + test_records
    df = pd.DataFrame(all_records)
    
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)
    
    logging.info(f"✅ Manifest 已保存: {manifest_path}")
    
    # 步骤6: 统计信息
    logging.info("\n" + "=" * 70)
    logging.info("完成!")
    logging.info("=" * 70)
    
    # 按 split 统计
    logging.info("\n按 split 统计:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        logging.info(f"  {split.capitalize()}: {len(split_df)} 片段")
        
        # 统计每个 parent_id 的分布
        parent_dist = split_df['parent_id'].value_counts().sort_index()
        for pid, count in parent_dist.items():
            logging.info(f"    purenoise_{pid}: {count} 片段")
    
    # 总体统计
    logging.info(f"\n总片段数: {len(df)}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"  {output_dir}/train/ : {len(train_records)} 个 NPY 文件")
    logging.info(f"  {output_dir}/val/   : {len(val_records)} 个 NPY 文件")
    logging.info(f"  {output_dir}/test/  : {len(test_records)} 个 NPY 文件")
    
    logging.info("\n下一步:")
    logging.info("1. 使用 train/val 片段构建训练数据集 (正样本叠加 + 负样本)")
    logging.info("2. 使用 test 片段构建测试数据集")
    logging.info("3. 确保叠加噪声来源与对应 split 一致")
    
    return 0


if __name__ == '__main__':
    exit(main())