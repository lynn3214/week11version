#!/usr/bin/env python3
"""
Export NPY files to WAV format for Audacity visualization
批量转换 .npy 文件为 .wav 格式,方便 Audacity 查看
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm


def normalize_for_wav(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    归一化音频到 [-1, 1] 范围供 WAV 保存
    
    Args:
        audio: 输入音频数组
        method: 归一化方法 ('peak' 或 'rms')
    
    Returns:
        归一化后的音频
    """
    if len(audio) == 0:
        return audio
    
    if method == 'peak':
        # 峰值归一化
        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            return audio / peak * 0.95  # 留5%余量避免削波
        else:
            return audio
    
    elif method == 'rms':
        # RMS归一化到特定水平
        rms = np.sqrt(np.mean(audio**2))
        if rms > 1e-8:
            target_rms = 0.1
            audio = audio * (target_rms / rms)
            # 再做峰值限制
            peak = np.max(np.abs(audio))
            if peak > 0.95:
                audio = audio / peak * 0.95
            return audio
        else:
            return audio
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def export_npy_to_wav(
    input_dir: Path,
    output_dir: Path,
    sample_rate: int = 44100,
    norm_method: str = 'peak',
    overwrite: bool = False,
    verbose: bool = False
) -> None:
    """
    批量转换 NPY 文件为 WAV
    
    Args:
        input_dir: 输入目录(包含.npy文件)
        output_dir: 输出目录
        sample_rate: 采样率
        norm_method: 归一化方法 ('peak' 或 'rms')
        overwrite: 是否覆盖已存在的文件
        verbose: 详细日志
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # 设置日志
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )
    
    # 递归查找所有 .npy 文件
    npy_files = list(input_dir.rglob('*.npy'))
    
    if not npy_files:
        logging.error(f"未找到任何 .npy 文件: {input_dir}")
        return
    
    logging.info(f"找到 {len(npy_files)} 个 NPY 文件")
    logging.info(f"归一化方法: {norm_method}")
    logging.info(f"采样率: {sample_rate} Hz")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 处理每个文件
    for npy_path in tqdm(npy_files, desc="转换 NPY → WAV"):
        try:
            # 保持相对路径结构
            rel_path = npy_path.relative_to(input_dir)
            wav_path = output_dir / rel_path.with_suffix('.wav')
            
            # 检查是否已存在
            if wav_path.exists() and not overwrite:
                logging.debug(f"跳过已存在: {wav_path.name}")
                skip_count += 1
                continue
            
            # 创建输出子目录
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 加载 NPY 数据
            audio = np.load(npy_path)
            
            # 检查数据
            if audio.ndim != 1:
                logging.warning(f"跳过非1D数组: {npy_path.name} (shape: {audio.shape})")
                error_count += 1
                continue
            
            if len(audio) == 0:
                logging.warning(f"跳过空数组: {npy_path.name}")
                error_count += 1
                continue
            
            # 归一化
            audio_normalized = normalize_for_wav(audio, method=norm_method)
            
            # 保存为 WAV
            sf.write(str(wav_path), audio_normalized, sample_rate)
            
            success_count += 1
            
        except Exception as e:
            logging.error(f"转换失败 {npy_path.name}: {e}")
            error_count += 1
            continue
    
    # 打印摘要
    print("\n" + "=" * 70)
    print("转换完成")
    print("=" * 70)
    print(f"成功转换: {success_count} 个文件")
    print(f"跳过已存在: {skip_count} 个文件")
    print(f"转换失败: {error_count} 个文件")
    print(f"输出目录: {output_dir}")
    
    if success_count > 0:
        print(f"\n✅ WAV 文件已可在 Audacity 中打开查看")
    
    if error_count == 0 and skip_count == 0:
        print("✅ 所有文件转换成功!")
    elif error_count > 0:
        print(f"⚠️  {error_count} 个文件转换失败")


def main():
    parser = argparse.ArgumentParser(
        description='批量转换 NPY 文件为 WAV 格式供 Audacity 查看'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='输入目录(包含.npy文件)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='输出目录(保存.wav文件)'
    )
    parser.add_argument(
        '--sample-rate', type=int, default=44100,
        help='采样率 (默认: 44100)'
    )
    parser.add_argument(
        '--norm-method', type=str, default='peak',
        choices=['peak', 'rms'],
        help='归一化方法: peak(峰值) 或 rms (默认: peak)'
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='覆盖已存在的文件'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='详细日志'
    )
    
    args = parser.parse_args()
    
    export_npy_to_wav(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        sample_rate=args.sample_rate,
        norm_method=args.norm_method,
        overwrite=args.overwrite,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()