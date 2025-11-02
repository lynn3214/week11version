#!/usr/bin/env python3
"""
Resampling and high-pass filtering utilities for dolphin click detection.
统一处理 WAV / MAT / Pickle 三种格式，全部输出为 44.1kHz WAV
支持 MATLAB v7.3 (HDF5) 格式
"""

import argparse
import logging
from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
from scipy.signal import resample_poly, butter, sosfilt
import soundfile as sf
from tqdm import tqdm
from datetime import datetime

# 全局错误收集列表
error_logs: List[str] = []


def resample_and_hpf(
    x: np.ndarray,
    sr_orig: int,
    sr_target: int = 44100,
    hp_cutoff: int = 1000,
    hp_order: int = 4
) -> np.ndarray:
    """
    Resample signal and apply high-pass filter.
    
    Args:
        x: Input signal array
        sr_orig: Original sampling rate
        sr_target: Target sampling rate (default 44100)
        hp_cutoff: High-pass cutoff frequency in Hz (default 1000)
        hp_order: Filter order (default 4)
        
    Returns:
        Filtered and resampled signal as float32
    """
    # Handle empty input safely
    if x is None or len(x) == 0:
        return np.array([], dtype=np.float32)

    # Resample
    if sr_orig != sr_target:
        g = np.gcd(sr_orig, sr_target)
        up = sr_target // g
        down = sr_orig // g
        y = resample_poly(x, up, down)
    else:
        y = x.copy()
    
    # High-pass filter
    if hp_cutoff > 0 and hp_cutoff < sr_target // 2 and len(y) > 0:
        sos = butter(hp_order, hp_cutoff, 'highpass', fs=sr_target, output='sos')
        y = sosfilt(sos, y)
    
    return y.astype(np.float32)


def _load_mat_v73_with_h5py(file_path: Path) -> Tuple[np.ndarray, int]:
    """
    使用 h5py 读取 MATLAB v7.3 格式文件
    
    Returns:
        (audio_data, sample_rate)
    """
    import h5py
    
    with h5py.File(file_path, 'r') as f:
        # 尝试常见的数据变量名
        data_keys = ['newFiltDat', 'data', 'y', 'audio', 'signal', 'x']
        audio = None
        
        for key in data_keys:
            if key in f:
                dset = f[key]
                audio = dset[()]
                break
        
        if audio is None:
            available_keys = list(f.keys())
            raise KeyError(
                f"未找到音频数据。可用的键: {available_keys}\n"
                f"尝试的键: {data_keys}"
            )
        
        # 查找采样率
        sr_keys = ['fs', 'sr', 'Fs', 'sampleRate', 'sample_rate']
        sr = None
        
        for key in sr_keys:
            if key in f:
                sr = int(f[key][()])
                break
            # 也检查数据集的属性
            elif key in dset.attrs:
                sr = int(dset.attrs[key])
                break
        
        if sr is None:
            logging.warning(f"未找到采样率，假设为 96000 Hz")
            sr = 96000
    
    return np.asarray(audio), sr


def load_audio_file(
    file_path: Path,
    channel: int = 10
) -> Tuple[np.ndarray, int]:
    """
    Load audio from .wav, .mat, or .pkl file.
    Returns mono waveform (float32) and original sample-rate.
    
    Args:
        file_path: Path to audio file
        channel: For multi-channel MAT files, which channel to extract (default: 10)
    
    Supports:
    - .wav: standard audio files
    - .mat: MATLAB files (both old and v7.3 HDF5 format)
    - .pkl / noise files: pickle format (assumes 96kHz)
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()
    
    # ---------- Pickle / Noise files ----------
    if 'noise' in str(file_path).lower() or suffix == '.pkl':
        import pickle
        try:
            noise = pickle.load(open(file_path, 'rb'))
            
            if not isinstance(noise, np.ndarray):
                noise = np.array(noise)
            
            noise = noise.astype(np.float32)
            return noise, 96000
            
        except Exception as e:
            logging.warning(f"无法作为pickle加载 {file_path}: {e}")
    
    # ---------- WAV files ----------
    if suffix == '.wav':
        audio, sr = sf.read(file_path)
        # 转为单声道
        audio = audio.mean(1) if audio.ndim == 2 else audio
        return audio.astype(np.float32), int(sr)
    
    # ---------- MAT files ----------
    if suffix == '.mat':
        try:
            from scipy.io import loadmat
            
            # 先尝试使用 scipy.io.loadmat (适用于旧格式)
            try:
                mat_data = loadmat(file_path, squeeze_me=True)
                
                # 查找音频数据
                audio = None
                data_keys = ['newFiltDat', 'data', 'y', 'audio', 'signal', 'x']
                
                for key in data_keys:
                    if key in mat_data:
                        audio = mat_data[key]
                        break
                
                if audio is None:
                    available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    raise KeyError(
                        f"未找到音频数据。可用的键: {available_keys}\n"
                        f"尝试的键: {data_keys}"
                    )
                
                # 查找采样率
                sr = None
                sr_keys = ['fs', 'sr', 'Fs', 'sampleRate', 'sample_rate']
                
                for key in sr_keys:
                    if key in mat_data:
                        sr_val = mat_data[key]
                        if isinstance(sr_val, np.ndarray):
                            sr = int(sr_val.flatten()[0])
                        else:
                            sr = int(sr_val)
                        break
                
                if sr is None:
                    logging.warning(f"未找到采样率，假设为 96000 Hz: {file_path}")
                    sr = 96000
                
            except NotImplementedError as e:
                # MATLAB v7.3 格式，使用 h5py
                if "HDF" in str(e) or "v7.3" in str(e):
                    logging.info(f"检测到 MATLAB v7.3 格式，使用 h5py: {file_path.name}")
                    audio, sr = _load_mat_v73_with_h5py(file_path)
                else:
                    raise
            
            # 处理数据维度和通道选择
            audio = np.asarray(audio)
            
            # 如果是2D且行少列多，转置
            if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
                audio = audio.T
                logging.debug(f"转置数据: {audio.shape}")
            
            # 多通道处理：选择指定通道
            if audio.ndim == 2:
                num_channels = audio.shape[1]
                # 确保通道索引有效
                ch_idx = min(channel, num_channels - 1)
                audio = audio[:, ch_idx]
                logging.info(f"选择通道 {ch_idx}/{num_channels-1}: {file_path.name}")
            
            # 确保是1D数组
            audio = np.squeeze(audio).astype(np.float32)
            
            return audio, sr
            
        except Exception as e:
            raise ValueError(f"加载MAT文件失败 {file_path}: {str(e)}")
    
    # ---------- 无法识别的格式 ----------
    raise ValueError(f'不支持的文件格式: {suffix}')


def _process_waveform_and_save(
    wave: np.ndarray,
    sr_orig: int,
    out_path: Path,
    cfg: dict
) -> None:
    """Process single waveform and save to file."""
    # 确保输出文件是 .wav 格式
    out_path = out_path.with_suffix('.wav')
    
    # 重采样和滤波
    y = resample_and_hpf(
        wave, sr_orig,
        cfg['sr_target'],
        cfg['hp_cutoff'],
        cfg['hp_order']
    )
    
    # 创建输出目录
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为WAV文件
    sf.write(str(out_path), y, cfg['sr_target'])
    
    logging.debug(f"已保存: {out_path}")


def process_file(
    input_path: Path,
    output_path: Path,
    sr_target: int = 44100,
    hp_cutoff: int = 1000,
    hp_order: int = 4,
    mat_channel: int = 10
) -> bool:
    """
    Process a single audio file (WAV/MAT/Pickle).
    Always outputs .wav format at target sample rate.
    
    Args:
        input_path: Input file path
        output_path: Output file path
        sr_target: Target sampling rate
        hp_cutoff: High-pass filter cutoff frequency
        hp_order: Filter order
        mat_channel: For multi-channel MAT files, which channel to use (default: 10)
    
    Returns:
        True if processed successfully, False if failed
    """
    try:
        cfg = {
            'sr_target': sr_target,
            'hp_cutoff': hp_cutoff,
            'hp_order': hp_order
        }
        
        # 加载音频文件
        audio, sr_orig = load_audio_file(input_path, channel=mat_channel)
        
        # 处理多维数据（pickle文件可能返回2D数组）
        if audio.ndim == 2:
            # 多行数据：每行保存为独立的WAV文件
            for i, row in enumerate(audio):
                out_file = output_path.with_suffix('').as_posix() + f'_{i:05d}.wav'
                _process_waveform_and_save(row, sr_orig, Path(out_file), cfg)
            
            logging.info(f"已处理 {len(audio)} 个片段从: {input_path.name}")
        else:
            # 单个波形：直接保存
            output_wav = output_path.with_suffix('.wav')
            _process_waveform_and_save(audio, sr_orig, output_wav, cfg)
            logging.info(f"已处理: {input_path.name} -> {output_wav.name}")
        
        return True
            
    except Exception as e:
        error_msg = f"处理失败 {input_path}: {str(e)}"
        error_logs.append(error_msg)
        logging.error(error_msg)
        return False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='统一处理音频文件：重采样和滤波 (支持 WAV/MAT/Pickle，全部输出WAV)'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='输入目录或文件'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='输出目录'
    )
    parser.add_argument(
        '--sr_target', type=int, default=44100,
        help='目标采样率 (默认: 44100)'
    )
    parser.add_argument(
        '--hp_cutoff', type=int, default=1000,
        help='高通滤波截止频率 (默认: 1000 Hz)'
    )
    parser.add_argument(
        '--hp_order', type=int, default=4,
        help='滤波器阶数 (默认: 4)'
    )
    parser.add_argument(
        '--mat_channel', type=int, default=10,
        help='MAT文件多通道时选择的通道索引 (默认: 10)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='详细日志输出'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # 检查依赖
    try:
        import h5py
    except ImportError:
        logging.warning("未安装 h5py，无法处理 MATLAB v7.3 文件")
        logging.warning("安装命令: pip install h5py")
    
    # 处理单个文件
    if input_path.is_file():
        process_file(
            input_path, 
            output_path / input_path.name,
            args.sr_target,
            args.hp_cutoff,
            args.hp_order,
            args.mat_channel
        )
        return
    
    # 处理目录
    if not input_path.is_dir():
        logging.error(f"输入路径不存在: {input_path}")
        return
    
    # 扫描所有支持的文件
    files = list(input_path.rglob('*'))
    
    # 文件筛选：WAV/MAT/PKL + noise目录下的无后缀文件
    files = [f for f in files if (
        f.is_file() and
        f.name != '.DS_Store' and
        (
            f.suffix.lower() in ('.wav', '.mat', '.pkl') or
            'noise' in str(f).lower()
        )
    )]
    
    if not files:
        logging.warning(f"在 {input_path} 中未找到支持的文件")
        return
    
    # 统计信息
    wav_files = [f for f in files if f.suffix.lower() == '.wav']
    mat_files = [f for f in files if f.suffix.lower() == '.mat']
    pkl_files = [f for f in files if f.suffix.lower() == '.pkl' or 'noise' in str(f).lower()]
    
    logging.info(f"\n找到文件:")
    logging.info(f"  WAV文件: {len(wav_files)}")
    logging.info(f"  MAT文件: {len(mat_files)}")
    logging.info(f"  Pickle/Noise文件: {len(pkl_files)}")
    logging.info(f"  总计: {len(files)}")
    
    if mat_files:
        logging.info(f"\nMAT文件将使用通道 {args.mat_channel}")
    
    # 处理所有文件
    processed_count = 0
    failed_count = 0
    
    for file_path in tqdm(files, desc="处理文件"):
        rel_path = file_path.relative_to(input_path)
        out_path = output_path / rel_path
        
        success = process_file(
            file_path,
            out_path,
            args.sr_target,
            args.hp_cutoff,
            args.hp_order,
            args.mat_channel
        )
        
        if success:
            processed_count += 1
        else:
            failed_count += 1
    
    # 保存错误日志
    if error_logs:
        log_file = output_path / f"errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(error_logs))
        
        print(f"\n发现 {len(error_logs)} 个错误。详细信息已保存到: {log_file}")
        print("\n错误摘要:")
        for error in error_logs[:10]:
            print(f"  - {error}")
        if len(error_logs) > 10:
            print(f"  ... 以及其他 {len(error_logs) - 10} 个错误")
    
    # 处理摘要
    print("\n" + "=" * 70)
    print("处理完成")
    print("=" * 70)
    print(f"成功处理: {processed_count} 个文件")
    print(f"处理失败: {failed_count} 个文件")
    print(f"输出目录: {output_path}")
    print(f"\n所有文件已统一转换为 {args.sr_target} Hz WAV 格式")
    
    if failed_count == 0:
        print("\n✅ 所有文件处理成功！")
    else:
        print(f"\n⚠️  {failed_count} 个文件处理失败，请检查错误日志")


if __name__ == '__main__':
    main()