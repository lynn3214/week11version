#!/usr/bin/env python3
"""
自动过滤脚本: 筛选出需要人工标注的uncertain样本
支持新的文件命名格式: {source_id}_{click_index}_{timestamp}ms.npy
"""

import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
import json
import re
from typing import List, Dict


class UncertainFilter:
    """自动过滤器: 分离 HQ / Uncertain / Hard_Neg"""
    
    def __init__(self,
                 events_csv: Path,
                 audio_dir: Path,
                 output_base: Path):
        """
        初始化过滤器
        
        Args:
            events_csv: all_events.csv 路径
            audio_dir: detection_results/audio 目录
            output_base: 输出基础目录
        """
        self.events_csv = Path(events_csv)
        self.audio_dir = Path(audio_dir)
        self.output_base = Path(output_base)
        
        # 创建输出目录
        self.hq_dir = self.output_base / 'Positive_HQ'
        self.uncertain_dir = self.output_base / 'Uncertain'
        self.hard_neg_dir = self.output_base / 'Hard_Negative'
        
        for d in [self.hq_dir, self.uncertain_dir, self.hard_neg_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 加载 CSV
        self.df = pd.read_csv(self.events_csv)
        print(f"加载 {len(self.df)} 条事件记录")
    
    def _parse_filename(self, filename: str) -> dict:
        """
        解析新格式的文件名
        
        格式: {source_id}_{click_index:04d}_{timestamp_ms:08d}ms.npy
        例如: 46_highsnr_0001_00000398ms.npy
        
        Returns:
            {'source_id': '46_highsnr', 'click_index': 1, 'timestamp_ms': 398}
        """
        # 匹配模式: 任意字符_数字(4位)_数字(8位)ms.npy
        pattern = r'^(.+?)_(\d{4})_(\d{8})ms\.npy$'
        match = re.match(pattern, filename)
        
        if match:
            return {
                'source_id': match.group(1),
                'click_index': int(match.group(2)),
                'timestamp_ms': int(match.group(3))
            }
        else:
            # 兼容旧格式: click_0001_00000398ms.npy
            old_pattern = r'^click_(\d{4})_(\d{8})ms\.npy$'
            old_match = re.match(old_pattern, filename)
            if old_match:
                return {
                    'source_id': 'unknown',
                    'click_index': int(old_match.group(1)),
                    'timestamp_ms': int(old_match.group(2))
                }
            
            raise ValueError(f"无法解析文件名: {filename}")
    
    def filter_all(self):
        """执行自动过滤"""
        print("\n开始自动过滤...")
        
        hq_count = 0
        uncertain_count = 0
        hard_neg_count = 0
        
        results = []
        
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="过滤中"):
            # 决策
            category, reason = self._classify_event(row)
            
            # ✅ 修改点: 构建新格式的文件名
            file_id = row.get('file_id', '')
            peak_time_s = row.get('peak_time', 0)
            timestamp_ms = int(peak_time_s * 1000)
            
            # 在 audio_dir 中查找匹配的文件
            # 新格式: {file_id}_{click_index}_{timestamp}ms.npy
            matching_files = list(self.audio_dir.glob(f"{file_id}_*_{timestamp_ms:08d}ms.npy"))
            
            if len(matching_files) == 0:
                # 尝试匹配旧格式
                matching_files = list(self.audio_dir.rglob(f"*_{timestamp_ms:08d}ms.npy"))
            
            if len(matching_files) == 0:
                print(f"⚠️  未找到匹配文件: {file_id} @ {timestamp_ms}ms")
                continue
            
            matching_file = matching_files[0]  # 取第一个匹配
            
            if len(matching_files) > 1:
                print(f"⚠️  找到多个匹配: {len(matching_files)} 个，使用第一个")
            
            # 复制到对应目录
            if category == 'HQ':
                dest_dir = self.hq_dir
                hq_count += 1
            elif category == 'Uncertain':
                dest_dir = self.uncertain_dir
                uncertain_count += 1
            else:  # Hard_Neg
                dest_dir = self.hard_neg_dir
                hard_neg_count += 1
            
            dest_path = dest_dir / matching_file.name
            
            # 处理重名
            counter = 1
            while dest_path.exists():
                stem = matching_file.stem
                dest_path = dest_dir / f"{stem}_{counter}.npy"
                counter += 1
            
            shutil.copy2(matching_file, dest_path)
            
            # 记录
            results.append({
                'file_path': str(matching_file),
                'dest_path': str(dest_path),
                'category': category,
                'reason': reason,
                'source_id': file_id,
                'peak_time': peak_time_s,
                'confidence': row.get('confidence_score', 0),
                'dolphin_likelihood': row.get('dolphin_likelihood', 1.0)
            })
        
        # 保存结果
        self._save_results(results, hq_count, uncertain_count, hard_neg_count)
        
        print(f"\n✅ 过滤完成:")
        print(f"  Positive_HQ: {hq_count} 个")
        print(f"  Uncertain: {uncertain_count} 个 ← 需要人工标注")
        print(f"  Hard_Negative: {hard_neg_count} 个")
    
    def _classify_event(self, row: pd.Series) -> tuple:
        """
        分类单个事件
        
        返回: (category, reason)
        category ∈ {'HQ', 'Uncertain', 'Hard_Neg'}
        """
        # === 提取特征 ===
        confidence = row.get('confidence_score', 0)
        dolphin_likelihood = row.get('dolphin_likelihood', 1.0)
        
        # 瞬态特征
        peak_sharpness = row.get('transient_peak_sharpness', 0)
        attack_time_ms = row.get('transient_attack_time_ms', 0)
        energy_conc = row.get('transient_energy_concentration', 0)
        
        # 频谱特征
        spectral_centroid = row.get('spectral_centroid', 0)
        high_low_ratio = row.get('high_low_ratio', 0)
        
        # === 规则1: 明确的 Shrimp 特征 → Hard_Neg ===
        if (peak_sharpness > 20 and 
            attack_time_ms < 0.5 and 
            energy_conc > 0.95):
            return 'Hard_Neg', 'shrimp_signature'
        
        if dolphin_likelihood < 0.2:
            return 'Hard_Neg', 'very_low_dolphin_likelihood'
        
        # === 规则2: 明确的高质量 Click → HQ ===
        if (confidence > 0.7 and 
            dolphin_likelihood > 0.8 and
            spectral_centroid > 25000):
            return 'HQ', 'high_confidence_high_freq'
        
        if (confidence > 0.6 and
            1.0 < attack_time_ms < 5.0 and
            0.75 < energy_conc < 0.90):
            return 'HQ', 'good_transient_features'
        
        # === 规则3: 边界情况 → Uncertain ===
        if 0.3 < confidence < 0.6:
            return 'Uncertain', 'medium_confidence'
        
        if (0.5 < attack_time_ms < 1.0 or
            0.90 < energy_conc < 0.95):
            return 'Uncertain', 'borderline_transient'
        
        if 0.3 < dolphin_likelihood < 0.6:
            return 'Uncertain', 'medium_dolphin_likelihood'
        
        # === 默认: 高置信度 → HQ ===
        if confidence > 0.5:
            return 'HQ', 'default_high_confidence'
        
        # === 其他: 低置信度 → Uncertain ===
        return 'Uncertain', 'default_uncertain'
    
    def _save_results(self, results: List[Dict], hq: int, uncertain: int, hard_neg: int):
        """保存过滤结果"""
        # CSV
        df_results = pd.DataFrame(results)
        csv_path = self.output_base / 'filter_results.csv'
        df_results.to_csv(csv_path, index=False)
        
        # JSON 统计
        stats = {
            'total': len(results),
            'positive_hq': hq,
            'uncertain': uncertain,
            'hard_negative': hard_neg,
            'hq_percentage': hq / len(results) * 100 if results else 0,
            'uncertain_percentage': uncertain / len(results) * 100 if results else 0
        }
        
        json_path = self.output_base / 'filter_stats.json'
        with open(json_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n结果已保存:")
        print(f"  CSV: {csv_path}")
        print(f"  Stats: {json_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='自动过滤低置信度样本供人工标注 (支持新文件命名格式)'
    )
    parser.add_argument('--events-csv', type=str, required=True,
                       help='all_events.csv 路径')
    parser.add_argument('--audio-dir', type=str, required=True,
                       help='detection_results/audio 目录')
    parser.add_argument('--output', type=str, required=True,
                       help='输出基础目录')
    
    args = parser.parse_args()
    
    filter_tool = UncertainFilter(
        events_csv=Path(args.events_csv),
        audio_dir=Path(args.audio_dir),
        output_base=Path(args.output)
    )
    
    filter_tool.filter_all()


if __name__ == '__main__':
    main()