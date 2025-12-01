#!/usr/bin/env python3
"""
数据准备脚本 - 修改版
修改点:
1. Singapore waters 数据分层处理:
   - training_cleaned + validation_cleaned → training sources
   - 其余 SG 文件 → test set
2. 保持其他逻辑不变
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import List, Tuple
import yaml


def load_config(config_path: Path) -> dict:
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def is_high_snr_file(file_path: Path) -> bool:
    """检查文件名是否包含highsnr标记"""
    return 'highsnr' in file_path.name.lower()


def scan_labeled_directories(raw_wav_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    扫描标注数据目录，分离高SNR和低SNR文件
    
    返回:
        (high_snr_files, low_snr_files) 元组
    """
    labeled_dirs = [
        '06Nov21_left',
        '06Sept15_left',
        '9 Oct 06 Splash and Penn'
    ]
    
    high_snr_files = []
    low_snr_files = []
    
    for dir_name in labeled_dirs:
        dir_path = raw_wav_dir / dir_name
        if not dir_path.exists():
            logging.warning(f"标注目录未找到: {dir_path}")
            continue
        
        wav_files = list(dir_path.rglob('*.wav'))
        logging.info(f"在 {dir_name} 中找到 {len(wav_files)} 个WAV文件")
        
        for wav_file in wav_files:
            if is_high_snr_file(wav_file):
                high_snr_files.append(wav_file)
            else:
                low_snr_files.append(wav_file)
    
    return high_snr_files, low_snr_files


def scan_singapore_waters(raw_wav_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    扫描新加坡水域数据，分离训练和测试集
    
    返回:
        (sg_training_files, sg_test_files) 元组
    
    逻辑:
    - training_cleaned/ → 训练集
    - validation_cleaned/ → 训练集 (合并)
    - 其余文件 → 测试集
    """
    singapore_dir = raw_wav_dir / 'Singapore_waters_detected_clicks'
    
    sg_training_files = []
    sg_test_files = []
    
    if not singapore_dir.exists():
        logging.warning(f"新加坡水域目录未找到: {singapore_dir}")
        return sg_training_files, sg_test_files
    
    # 1. 收集 training_cleaned
    training_cleaned_dir = singapore_dir / 'training_cleaned'
    if training_cleaned_dir.exists():
        training_files = list(training_cleaned_dir.rglob('*.wav'))
        sg_training_files.extend(training_files)
        logging.info(f"新加坡水域 - training_cleaned: {len(training_files)} 个文件")
    else:
        logging.warning(f"未找到: {training_cleaned_dir}")
    
    # 2. 收集 validation_cleaned (合并到训练集)
    validation_cleaned_dir = singapore_dir / 'validation_cleaned'
    if validation_cleaned_dir.exists():
        validation_files = list(validation_cleaned_dir.rglob('*.wav'))
        sg_training_files.extend(validation_files)
        logging.info(f"新加坡水域 - validation_cleaned: {len(validation_files)} 个文件 (合并到训练集)")
    else:
        logging.warning(f"未找到: {validation_cleaned_dir}")
    
    # 3. 收集其余文件 (测试集)
    # 排除 training_cleaned 和 validation_cleaned 子目录
    exclude_dirs = {'training_cleaned', 'validation_cleaned'}
    
    for wav_file in singapore_dir.rglob('*.wav'):
        # 检查文件是否在排除的子目录中
        is_in_excluded = any(
            excluded in wav_file.parts 
            for excluded in exclude_dirs
        )
        
        if not is_in_excluded:
            sg_test_files.append(wav_file)
    
    logging.info(f"新加坡水域 - 其余文件(测试集): {len(sg_test_files)} 个文件")
    
    # 打印测试集文件分布
    if sg_test_files:
        # 按父目录分组统计
        from collections import defaultdict
        test_by_parent = defaultdict(int)
        for f in sg_test_files:
            # 获取相对于 singapore_dir 的第一级子目录
            try:
                rel_path = f.relative_to(singapore_dir)
                parent_name = rel_path.parts[0] if len(rel_path.parts) > 1 else 'root'
                test_by_parent[parent_name] += 1
            except ValueError:
                test_by_parent['unknown'] += 1
        
        logging.info(f"  测试集文件分布:")
        for parent, count in sorted(test_by_parent.items()):
            logging.info(f"    {parent}: {count} 个文件")
    
    return sg_training_files, sg_test_files


def organize_training_sources(
    raw_dir: Path,
    training_sources_dir: Path,
    high_snr_files: List[Path],
    sg_training_files: List[Path],
    copy_files: bool = True
) -> None:
    """
    组织训练数据源
    修改: 新增 Singapore waters 训练文件
    """
    training_sources_dir.mkdir(parents=True, exist_ok=True)
    
    raw_wav_dir = raw_dir / 'wav'
    raw_mat_dir = raw_dir / 'mat'
    
    # 1. 组织high-SNR标注文件
    if high_snr_files:
        high_snr_dir = training_sources_dir / 'high_snr_labeled'
        high_snr_dir.mkdir(exist_ok=True)
        
        logging.info(f"组织 {len(high_snr_files)} 个high-SNR文件...")
        for src_file in high_snr_files:
            rel_path = src_file.relative_to(raw_wav_dir)
            dst_file = high_snr_dir / rel_path.parts[0] / src_file.name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            if copy_files:
                shutil.copy2(src_file, dst_file)
            else:
                if dst_file.exists():
                    dst_file.unlink()
                dst_file.symlink_to(src_file.resolve())
        
        logging.info(f"High-SNR文件已组织到: {high_snr_dir}")
    
    # 2. ✅ 新增: 组织新加坡水域训练文件
    if sg_training_files:
        sg_training_dir = training_sources_dir / 'singapore_waters_training'
        sg_training_dir.mkdir(exist_ok=True)
        
        logging.info(f"组织 {len(sg_training_files)} 个新加坡水域训练文件...")
        
        singapore_root = raw_wav_dir / 'Singapore_waters_detected_clicks'
        
        for src_file in sg_training_files:
            # 保持子目录结构 (training_cleaned 或 validation_cleaned)
            rel_path = src_file.relative_to(singapore_root)
            dst_file = sg_training_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            if copy_files:
                shutil.copy2(src_file, dst_file)
            else:
                if dst_file.exists():
                    dst_file.unlink()
                dst_file.symlink_to(src_file.resolve())
        
        logging.info(f"新加坡水域训练文件已组织到: {sg_training_dir}")
    
    # 3. 组织其他训练数据源(排除测试集目录)
    training_source_dirs = [
        'Dataport Dolphins Underwater Sounds Database',
        'Dolphin Clicks',
        'OceanParkdolphinclicks',
        'Watkins dolphin'
    ]
    
    for source_name in training_source_dirs:
        source_dir = raw_wav_dir / source_name
        if source_dir.exists():
            wav_files = list(source_dir.rglob('*.wav'))
            if wav_files:
                logging.info(f"组织 {len(wav_files)} 个文件从 {source_name}...")
                
                dst_dir = training_sources_dir / source_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                
                for src_file in wav_files:
                    rel_path = src_file.relative_to(source_dir)
                    dst_file = dst_dir / rel_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    if copy_files:
                        shutil.copy2(src_file, dst_file)
                    else:
                        if dst_file.exists():
                            dst_file.unlink()
                        dst_file.symlink_to(src_file.resolve())
    
    # 4. 组织mat文件
    if raw_mat_dir.exists():
        mat_files = list(raw_mat_dir.rglob('*.mat'))
        if mat_files:
            logging.info(f"组织 {len(mat_files)} 个MAT文件...")
            
            mat_dst_dir = training_sources_dir / 'mat_files'
            mat_dst_dir.mkdir(parents=True, exist_ok=True)
            
            for src_file in mat_files:
                rel_path = src_file.relative_to(raw_mat_dir)
                dst_file = mat_dst_dir / rel_path
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                
                if copy_files:
                    shutil.copy2(src_file, dst_file)
                else:
                    if dst_file.exists():
                        dst_file.unlink()
                    dst_file.symlink_to(src_file.resolve())
            
            logging.info(f"MAT文件已组织到: {mat_dst_dir}")


def organize_test_data(
    raw_wav_dir: Path,
    test_raw_dir: Path,
    low_snr_files: List[Path],
    sg_test_files: List[Path],
    copy_files: bool = True
) -> None:
    """
    组织测试数据
    修改: 新增 Singapore waters 测试文件
    """
    test_raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 组织低SNR标注文件
    if low_snr_files:
        low_snr_dir = test_raw_dir / 'low_snr_labeled'
        low_snr_dir.mkdir(exist_ok=True)
        
        logging.info(f"组织 {len(low_snr_files)} 个低SNR文件...")
        for src_file in low_snr_files:
            rel_path = src_file.relative_to(raw_wav_dir)
            dst_file = low_snr_dir / rel_path.parts[0] / src_file.name
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            if copy_files:
                shutil.copy2(src_file, dst_file)
            else:
                if dst_file.exists():
                    dst_file.unlink()
                dst_file.symlink_to(src_file.resolve())
        
        logging.info(f"低SNR文件已组织到: {low_snr_dir}")
    
    # 2. ✅ 修改: 组织新加坡水域测试文件 (其余文件)
    if sg_test_files:
        sg_test_dir = test_raw_dir / 'singapore_waters_test'
        sg_test_dir.mkdir(exist_ok=True)
        
        logging.info(f"组织 {len(sg_test_files)} 个新加坡水域测试文件...")
        
        singapore_root = raw_wav_dir / 'Singapore_waters_detected_clicks'
        
        for src_file in sg_test_files:
            # 保持原始目录结构
            rel_path = src_file.relative_to(singapore_root)
            dst_file = sg_test_dir / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            
            if copy_files:
                shutil.copy2(src_file, dst_file)
            else:
                if dst_file.exists():
                    dst_file.unlink()
                dst_file.symlink_to(src_file.resolve())
        
        logging.info(f"新加坡水域测试文件已组织到: {sg_test_dir}")


def main():
    """主数据准备流程"""
    parser = argparse.ArgumentParser(
        description='准备训练和测试数据(新加坡数据分层处理)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='配置文件路径'
    )
    parser.add_argument(
        '--raw_dir',
        type=str,
        default='data/raw',
        help='原始数据目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data',
        help='输出基础目录'
    )
    parser.add_argument(
        '--symlink',
        action='store_true',
        help='创建符号链接而非复制文件(节省空间)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='详细日志'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='仅显示将要执行的操作，不实际执行'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        logging.warning(f"配置文件未找到: {config_path}，使用默认值")
        config = {}
    
    # 设置路径
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    
    raw_wav_dir = raw_dir / 'wav'
    test_raw_dir = output_dir / 'test_raw'
    training_sources_dir = raw_dir / 'training_sources'
    
    if not raw_wav_dir.exists():
        logging.error(f"原始WAV目录未找到: {raw_wav_dir}")
        return 1
    
    logging.info("="*70)
    logging.info("数据准备 - 新加坡数据分层处理")
    logging.info("="*70)
    logging.info(f"原始目录: {raw_dir}")
    logging.info(f"输出目录: {output_dir}")
    logging.info(f"模式: {'符号链接' if args.symlink else '复制'}")
    
    # 扫描标注数据
    logging.info("\n扫描标注数据...")
    high_snr_files, low_snr_files = scan_labeled_directories(raw_wav_dir)
    
    # ✅ 扫描新加坡水域数据
    logging.info("\n扫描新加坡水域数据...")
    sg_training_files, sg_test_files = scan_singapore_waters(raw_wav_dir)
    
    # 打印摘要
    logging.info(f"\n数据摘要:")
    logging.info(f"  High-SNR文件(用于训练): {len(high_snr_files)}")
    logging.info(f"  Low-SNR文件(用于测试): {len(low_snr_files)}")
    logging.info(f"  新加坡水域-训练集: {len(sg_training_files)}")
    logging.info(f"  新加坡水域-测试集: {len(sg_test_files)}")
    
    if args.dry_run:
        logging.info("\n=== 试运行模式 ===")
        
        logging.info("\n将组织到 training_sources/:")
        logging.info(f"  - high-SNR文件: {len(high_snr_files)}")
        for f in high_snr_files[:3]:
            logging.info(f"    {f.relative_to(raw_wav_dir)}")
        if len(high_snr_files) > 3:
            logging.info(f"    ... 以及其他 {len(high_snr_files)-3} 个文件")
        
        logging.info(f"  - 新加坡水域训练文件: {len(sg_training_files)}")
        for f in sg_training_files[:3]:
            logging.info(f"    {f.name}")
        if len(sg_training_files) > 3:
            logging.info(f"    ... 以及其他 {len(sg_training_files)-3} 个文件")
        
        logging.info("\n将组织到 test_raw/:")
        logging.info(f"  - low-SNR文件: {len(low_snr_files)}")
        logging.info(f"  - 新加坡水域测试文件: {len(sg_test_files)}")
        for f in sg_test_files[:3]:
            logging.info(f"    {f.name}")
        if len(sg_test_files) > 3:
            logging.info(f"    ... 以及其他 {len(sg_test_files)-3} 个文件")
        
        return 0
    
    # 组织训练数据
    logging.info("\n组织训练数据...")
    organize_training_sources(
        raw_dir=raw_dir,
        training_sources_dir=training_sources_dir,
        high_snr_files=high_snr_files,
        sg_training_files=sg_training_files,  # ✅ 新增参数
        copy_files=not args.symlink
    )
    
    # 组织测试数据
    logging.info("\n组织测试数据...")
    organize_test_data(
        raw_wav_dir=raw_wav_dir,
        test_raw_dir=test_raw_dir,
        low_snr_files=low_snr_files,
        sg_test_files=sg_test_files,  # ✅ 新增参数
        copy_files=not args.symlink
    )
    
    # 总结
    logging.info("\n" + "="*70)
    logging.info("数据准备完成")
    logging.info("="*70)
    logging.info("\n下一步操作:")
    logging.info("1. 对training_sources/运行resample_and_filter.py")
    logging.info("   python preprocessing/resample_and_filter.py --input data/raw/training_sources --output data/training_resampled --verbose")
    logging.info("2. 对test_raw/运行resample_and_filter.py(单独目录)")
    logging.info("   python preprocessing/resample_and_filter.py --input data/test_raw --output data/test_resampled --verbose")
    logging.info("3. 对noise目录运行resample_and_filter.py")
    logging.info("   python preprocessing/resample_and_filter.py --input data/raw/noise --output data/noise_resampled --verbose")
    logging.info("4. 对重采样的训练数据运行batch-detect")
    logging.info("   python main.py batch-detect --input-dir data/training_resampled --output-dir detection_results --save-audio --recursive")
    
    logging.info("\n创建的目录结构:")
    logging.info(f"  {training_sources_dir}/")
    logging.info(f"    ├── high_snr_labeled/ ({len(high_snr_files)} 文件)")
    logging.info(f"    ├── singapore_waters_training/ ({len(sg_training_files)} 文件)")
    logging.info(f"    ├── [其他训练源目录]")
    logging.info(f"    └── mat_files/")
    logging.info(f"  {test_raw_dir}/")
    logging.info(f"    ├── low_snr_labeled/ ({len(low_snr_files)} 文件)")
    logging.info(f"    └── singapore_waters_test/ ({len(sg_test_files)} 文件)")
    logging.info(f"\n注意: noise文件保持在 data/raw/noise/ 原始位置")
    
    return 0


if __name__ == '__main__':
    exit(main())