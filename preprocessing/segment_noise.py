#!/usr/bin/env python3
"""
噪音文件切割脚本 (优化版 - 500ms片段)
功能：
1. 将长噪音文件切成固定500ms片段
2. 支持训练/测试集划分（可选）
3. RMS归一化确保噪音水平一致
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm


def segment_noise_file(audio: np.ndarray,
                       sample_rate: int,
                       segment_samples: int,
                       overlap_ratio: float = 0.0,
                       normalize: bool = True) -> list:
    """
    切割单个噪音文件，支持归一化
    
    Args:
        audio: 音频数组
        sample_rate: 采样率
        segment_samples: 片段长度（样本数）
        overlap_ratio: 重叠比例（0-1）
        normalize: 是否RMS归一化
        
    Returns:
        片段列表
    """
    if overlap_ratio < 0 or overlap_ratio >= 1:
        raise ValueError("overlap_ratio 必须在 [0, 1) 范围内")
    
    step_samples = int(segment_samples * (1 - overlap_ratio))
    segments = []
    
    start = 0
    while start + segment_samples <= len(audio):
        segment = audio[start:start + segment_samples].copy()
        
        # RMS归一化（与训练时一致）
        if normalize:
            rms = np.sqrt(np.mean(segment**2))
            if rms > 1e-8:
                target_rms = 0.1  # 目标RMS水平
                segment = segment * (target_rms / rms)
            
            # 峰值裁剪
            peak = np.max(np.abs(segment))
            if peak > 0.95:
                segment = segment / peak * 0.95
        
        segments.append(segment)
        start += step_samples
    
    # 处理剩余部分（如果足够长）
    if start < len(audio) and len(audio) - start >= segment_samples // 2:
        segment = audio[-segment_samples:].copy()
        
        if normalize:
            rms = np.sqrt(np.mean(segment**2))
            if rms > 1e-8:
                target_rms = 0.1
                segment = segment * (target_rms / rms)
            peak = np.max(np.abs(segment))
            if peak > 0.95:
                segment = segment / peak * 0.95
        
        segments.append(segment)
    
    return segments


def segment_noise(input_dir: Path,
                 output_train: Path,
                 output_test: Path = None,
                 segment_ms: float = 500.0,  # ✅ 改为500ms
                 train_ratio: float = 0.8,
                 overlap_ratio: float = 0.0,
                 sample_rate: int = 44100,
                 normalize: bool = True,
                 verbose: bool = False):
    """
    切割噪音文件并划分训练/测试集
    
    Args:
        input_dir: 输入目录
        output_train: 训练集输出目录
        output_test: 测试集输出目录（可选，如为None则不划分）
        segment_ms: 片段长度（毫秒）
        train_ratio: 训练集比例（仅当output_test不为None时生效）
        overlap_ratio: 重叠比例
        sample_rate: 目标采样率
        normalize: 是否RMS归一化
        verbose: 详细日志
    """
    input_dir = Path(input_dir)
    output_train = Path(output_train)
    
    output_train.mkdir(parents=True, exist_ok=True)
    
    split_dataset = output_test is not None
    if split_dataset:
        output_test = Path(output_test)
        output_test.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')
    
    segment_samples = int(segment_ms * sample_rate / 1000)
    logging.info(f"片段长度: {segment_ms}ms ({segment_samples} 样本)")
    if split_dataset:
        logging.info(f"训练集比例: {train_ratio:.1%}")
    logging.info(f"重叠比例: {overlap_ratio:.1%}")
    logging.info(f"RMS归一化: {'是' if normalize else '否'}")
    
    # 查找所有音频文件
    audio_files = []
    for ext in ['.wav', '.flac', '.mp3']:
        audio_files.extend(list(input_dir.rglob(f'*{ext}')))
    
    if not audio_files:
        logging.error(f"未找到音频文件: {input_dir}")
        return
    
    logging.info(f"找到 {len(audio_files)} 个噪音文件")
    
    train_count = 0
    test_count = 0
    
    # 处理每个文件
    for noise_file in tqdm(audio_files, desc="切割噪音", disable=not verbose):
        try:
            # 读取音频
            audio, sr = sf.read(noise_file)
            
            # 转单声道
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 重采样
            if sr != sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            
            # 切割
            segments = segment_noise_file(
                audio, sample_rate, segment_samples, overlap_ratio, normalize
            )
            
            if len(segments) == 0:
                logging.warning(f"文件太短，跳过: {noise_file.name}")
                continue
            
            # 保存片段
            for i, segment in enumerate(segments):
                # 文件名：原文件名_片段序号
                filename = f"{noise_file.stem}_seg{i:04d}.wav"
                
                # 决定保存位置
                if split_dataset:
                    # 随机划分训练/测试
                    if np.random.rand() < train_ratio:
                        out_path = output_train / filename
                        train_count += 1
                    else:
                        out_path = output_test / filename
                        test_count += 1
                else:
                    # 全部保存到训练集
                    out_path = output_train / filename
                    train_count += 1
                
                sf.write(out_path, segment, sample_rate)
            
        except Exception as e:
            logging.error(f"处理 {noise_file} 时出错: {e}")
            continue
    
    logging.info(f"\n✅ 切割完成")
    if split_dataset:
        logging.info(f"训练集片段: {train_count} → {output_train}")
        logging.info(f"测试集片段: {test_count} → {output_test}")
    else:
        logging.info(f"总片段数: {train_count} → {output_train}")


def main():
    parser = argparse.ArgumentParser(
        description='切割噪音文件（500ms片段，可选训练/测试集划分）'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='输入噪音目录')
    parser.add_argument('--output-train', type=str, required=True,
                       help='训练集输出目录')
    parser.add_argument('--output-test', type=str, default=None,
                       help='测试集输出目录（可选）')
    parser.add_argument('--segment-ms', type=float, default=500.0,  # ✅ 默认500ms
                       help='片段长度（毫秒），默认500')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='训练集比例（仅当指定output-test时生效），默认0.8')
    parser.add_argument('--overlap-ratio', type=float, default=0.0,
                       help='重叠比例，默认0（无重叠）')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='目标采样率，默认44100')
    parser.add_argument('--no-normalize', action='store_true',
                       help='禁用RMS归一化')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细日志')
    
    args = parser.parse_args()
    
    segment_noise(
        input_dir=Path(args.input),
        output_train=Path(args.output_train),
        output_test=Path(args.output_test) if args.output_test else None,
        segment_ms=args.segment_ms,
        train_ratio=args.train_ratio,
        overlap_ratio=args.overlap_ratio,
        sample_rate=args.sample_rate,
        normalize=not args.no_normalize,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()