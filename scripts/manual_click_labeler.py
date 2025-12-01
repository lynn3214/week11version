#!/usr/bin/env python3
"""
交互式Click片段手动标注工具 (NPY格式)
支持快捷键操作，实时显示波形+频谱图
支持断点续标：自动跳过已标注文件，支持多轮次标注
"""

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import shutil
import json
from scipy import signal as scipy_signal
from datetime import datetime
import random


class ClickLabeler:
    """交互式Click标注工具 (支持NPY格式 + 断点续标)"""
    
    def __init__(self, 
                 input_dir: Path,
                 output_base: Path,
                 csv_path: Path = None,
                 sample_rate: int = 44100,
                 remove_labeled: bool = True):
        """
        初始化标注工具
        
        Args:
            input_dir: 输入目录（data/filtered/Uncertain）
            output_base: 输出基础目录（data/manual_labelled）
            csv_path: all_events.csv路径（可选）
            sample_rate: 采样率
            remove_labeled: 是否从输入目录移除已标注文件（默认True）
        """
        self.input_dir = Path(input_dir)
        self.output_base = Path(output_base)
        self.csv_path = Path(csv_path) if csv_path else None
        self.sample_rate = sample_rate
        self.remove_labeled = remove_labeled
        
        # 创建输出目录结构
        self.categories = {
            '1': 'Positive_HQ',              # 高质量正样本
            '2': 'Negative_Hard',            # 明确负样本（shrimp/noise）
            '3': 'Quarantine/Sparse_Click',  # 稀疏/不确定
            '4': 'Quarantine/LowSNR',        # 低SNR模糊
            '0': 'Skipped'                   # 跳过（暂不决定）
        }
        
        for cat_path in self.categories.values():
            (self.output_base / cat_path).mkdir(parents=True, exist_ok=True)
        
        # 创建WAV导出目录
        self.wav_export_dir = self.output_base / 'export_to_wav'
        self.wav_export_dir.mkdir(parents=True, exist_ok=True)
        
        # 进度文件路径
        self.progress_file = self.output_base / 'progress.json'
        self.labels_file = self.output_base / 'labeling_record.json'
        
        # 加载已有的标注记录和进度
        self.labels = self._load_labels()
        self.labeled_files = set(self.labels.keys())
        
        # 扫描所有npy文件
        all_npy_files = list(self.input_dir.rglob('*.npy'))
        
        # 过滤掉已标注的文件
        self.npy_files = [
            f for f in all_npy_files 
            if str(f) not in self.labeled_files
        ]
        
        self.total_files = len(self.npy_files)
        self.already_labeled = len(self.labeled_files)
        self.grand_total = len(all_npy_files)
        
        # 加载CSV（如果提供）
        self.df_events = None
        if self.csv_path and self.csv_path.exists():
            self.df_events = pd.read_csv(self.csv_path)
        
        self.current_idx = 0
        
        # 设置matplotlib交互模式
        plt.ion()
        
        print("=" * 60)
        print("Interactive Click Labeling Tool (NPY Format)")
        print("=" * 60)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_base}")
        print(f"Already labeled: {self.already_labeled} segments")
        print(f"Remaining to label: {self.total_files} segments")
        print(f"Grand total: {self.grand_total} segments")
        print(f"WAV export directory: {self.wav_export_dir}")
        print(f"Remove labeled files: {self.remove_labeled}")
        print("=" * 60)
    
    def _load_labels(self):
        """加载已有的标注记录"""
        if self.labels_file.exists():
            with open(self.labels_file, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            print(f"✓ Loaded {len(labels)} existing labels from {self.labels_file}")
            return labels
        return {}
    
    def plot_click(self, npy_path: Path):
        """
        绘制单个click片段的波形+频谱图
        
        Args:
            npy_path: NPY文件路径
        """
        # 读取音频
        audio = np.load(npy_path)
        
        # 转单声道
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        
        # 动态统计信息
        current_position = self.current_idx + 1
        remaining = self.total_files
        completed_this_session = self.current_idx
        total_completed = self.already_labeled + completed_this_session
        
        # 设置窗口标题
        fig.canvas.manager.set_window_title(
            f'[Session: {current_position}/{remaining} | Total: {total_completed}/{self.grand_total}] {npy_path.name}'
        )
        
        fig.suptitle(
            f'Click Segment: {npy_path.name}\n'
            f'[This Session: {current_position}/{remaining} | '
            f'Total Progress: {total_completed}/{self.grand_total} ({total_completed/self.grand_total*100:.1f}%)]',
            fontsize=14, fontweight='bold'
        )
        
        # 1. 波形图
        ax1 = plt.subplot(3, 1, 1)
        time_ms = np.arange(len(audio)) / self.sample_rate * 1000
        ax1.plot(time_ms, audio, linewidth=0.8, color='steelblue')
        ax1.set_xlabel('Time (ms)', fontsize=11)
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.set_title('Waveform', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        # 标注峰值位置
        peak_idx = np.argmax(np.abs(audio))
        peak_time = peak_idx / self.sample_rate * 1000
        ax1.axvline(x=peak_time, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'Peak @ {peak_time:.1f}ms')
        ax1.legend(fontsize=9)
        
        # 2. 频谱图（FFT）
        ax2 = plt.subplot(3, 1, 2)
        
        # 应用窗函数
        window = np.hanning(len(audio))
        fft = np.fft.rfft(audio * window)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
        
        # 转dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        ax2.plot(freqs / 1000, magnitude_db, linewidth=0.8, color='darkorange')
        ax2.set_xlabel('Frequency (kHz)', fontsize=11)
        ax2.set_ylabel('Magnitude (dB)', fontsize=11)
        ax2.set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, self.sample_rate / 2000)
        ax2.grid(True, alpha=0.3)
        
        # 标注峰值频率
        peak_freq_idx = np.argmax(magnitude)
        peak_freq = freqs[peak_freq_idx] / 1000
        ax2.axvline(x=peak_freq, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'Peak @ {peak_freq:.1f}kHz')
        ax2.legend(fontsize=9)
        
        # 3. 时频图（Spectrogram）
        ax3 = plt.subplot(3, 1, 3)
        
        # 计算spectrogram
        nperseg = min(512, len(audio))
        f, t, Sxx = scipy_signal.spectrogram(
            audio, self.sample_rate, 
            nperseg=nperseg,
            noverlap=nperseg//2,
            window='hann'
        )
        
        # 转dB并绘制
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        im = ax3.pcolormesh(t * 1000, f / 1000, Sxx_db, 
                           shading='gouraud', cmap='viridis', 
                           vmin=Sxx_db.max()-60, vmax=Sxx_db.max())
        
        ax3.set_xlabel('Time (ms)', fontsize=11)
        ax3.set_ylabel('Frequency (kHz)', fontsize=11)
        ax3.set_title('Spectrogram', fontsize=12, fontweight='bold')
        ax3.set_ylim(0, self.sample_rate / 2000)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3, label='Power (dB)')
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        plt.draw()
        plt.pause(0.001)
    
    def export_to_wav(self, npy_path: Path):
        """
        导出NPY为WAV格式（供Audacity检查）
        
        Args:
            npy_path: NPY文件路径
        """
        try:
            # 读取音频
            audio = np.load(npy_path)
            
            # 转单声道
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # 构建输出路径
            wav_filename = npy_path.stem + '.wav'
            wav_path = self.wav_export_dir / wav_filename
            
            # 保存为WAV
            sf.write(wav_path, audio, self.sample_rate)
            
            print(f"✓ Exported to WAV: {wav_path}")
            
        except Exception as e:
            print(f"✗ Export failed: {e}")
    
    def label_interactive(self):
        """交互式标注主循环"""
        print("\n" + "="*80)
        print("LABELING TOOL STARTED (Resume Mode)")
        print("="*80)
        print("IMPORTANT: Look at the PLOT WINDOW, but TYPE in THIS TERMINAL!")
        print("Press any key (1/2/3/4/0) to label, Q to quit")
        print("Press [E] to Export current segment to WAV for Audacity inspection")
        print("="*80)
        
        if self.total_files == 0:
            print("\n✓ All files have been labeled! Nothing to do.")
            return
        
        print(f"\nResuming from: {self.already_labeled} already labeled")
        print(f"This session: {self.total_files} files to label\n")
        
        while self.current_idx < self.total_files:
            npy_path = self.npy_files[self.current_idx]
            
            # 清除之前的图形
            plt.clf()
            
            # 绘制当前片段
            self.plot_click(npy_path)
            
            # 等待用户输入
            print("\n" + "="*80)
            print(f">>> Current: [{self.current_idx+1}/{self.total_files}] {npy_path.name}")
            print(f">>> Total Progress: {self.already_labeled + self.current_idx}/{self.grand_total}")
            print("="*80)
            print(">>> INPUT HERE (in Terminal): [1/2/3/4/0/P/Q/E]")
            print(">>> Your choice: ", end='', flush=True)
            
            # 获取键盘输入
            choice = input().strip().lower()
            
            # 处理输入
            if choice == 'q':
                print("\nExiting labeling tool...")
                break
            
            elif choice == 'p':
                # 返回上一个
                if self.current_idx > 0:
                    self.current_idx -= 1
                    print("← Previous segment")
                else:
                    print("Already at first segment!")
                continue
            
            elif choice == 'e':
                # 导出为WAV
                print("► Exporting to WAV for Audacity...")
                self.export_to_wav(npy_path)
                continue
            
            elif choice in ['1', '2', '3', '4', '0']:
                # 记录标注
                category = self.categories[choice]
                self.labels[str(npy_path)] = {
                    'category': category,
                    'choice': choice,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"✓ Labeled as: {category}")
                
                # 立即保存（防止意外中断丢失数据）
                self._save_labels_incremental()
                
                self.current_idx += 1
            
            else:
                print("✗ Invalid input! Please try again")
                continue
        
        plt.close('all')
        
        # 最终保存
        self.save_labels()
        
        # 移动文件
        self.organize_files()
        
        # 更新CSV
        if self.df_events is not None:
            self.update_csv()
        
        # 生成报告
        self.generate_report()
    
    def _save_labels_incremental(self):
        """增量保存标注记录（每标注一个就保存）"""
        try:
            with open(self.labels_file, 'w', encoding='utf-8') as f:
                json.dump(self.labels, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️  Warning: Failed to save labels incrementally: {e}")
    
    def save_labels(self):
        """保存标注记录到JSON（最终保存）"""
        with open(self.labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)
        
        # 保存进度信息
        progress = {
            'total_files': self.grand_total,
            'labeled_files': len(self.labels),
            'last_update': datetime.now().isoformat()
        }
        
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Labeling record saved: {self.labels_file}")
        print(f"✓ Progress saved: {self.progress_file}")
    
    def organize_files(self):
        """根据标注结果整理文件"""
        print("\nOrganizing files into categories...")
        
        moved_count = {cat: 0 for cat in self.categories.values()}
        
        # 只处理本次标注的文件
        new_labels = {
            k: v for k, v in self.labels.items()
            if k not in self.labeled_files  # 排除之前已标注的
        }
        
        for npy_path_str, label_info in new_labels.items():
            npy_path = Path(npy_path_str)
            category = label_info['category']
            
            if category == 'Skipped':
                continue  # 跳过的文件不移动
            
            # 目标路径
            dest_dir = self.output_base / category
            dest_path = dest_dir / npy_path.name
            
            # 处理重名
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{npy_path.stem}_{counter}{npy_path.suffix}"
                counter += 1
            
            # 复制或移动文件
            try:
                if self.remove_labeled:
                    # 移动文件（从Uncertain目录删除）
                    shutil.move(str(npy_path), str(dest_path))
                else:
                    # 仅复制（保留原文件）
                    shutil.copy2(str(npy_path), str(dest_path))
                
                moved_count[category] += 1
            except Exception as e:
                print(f"✗ File operation failed {npy_path.name}: {e}")
        
        print("\nFile organization completed:")
        for category, count in moved_count.items():
            if count > 0:
                action = "moved" if self.remove_labeled else "copied"
                print(f"  {category}: {count} files {action}")
    
    def update_csv(self):
        """更新all_events.csv - 移除被标为负样本的记录"""
        print("\nUpdating all_events.csv...")
        
        # 构建需要移除的文件集合（Negative_Hard 和 LowSNR）
        files_to_remove = set()
        
        for npy_path_str, label_info in self.labels.items():
            category = label_info['category']
            
            # 只移除明确的负样本和低质量样本
            if category in ['Negative_Hard', 'Quarantine/LowSNR']:
                npy_path = Path(npy_path_str)
                files_to_remove.add(npy_path.name)
        
        if len(files_to_remove) == 0:
            print("  No files to remove from CSV")
            return
        
        # 匹配逻辑：基于文件名中的时间戳
        def should_keep_row(row):
            source_file = row.get('source_file', '')
            if not source_file:
                return True
            
            source_path = Path(source_file)
            source_name = source_path.name
            
            # 检查是否在移除列表中
            return source_name not in files_to_remove
        
        # 过滤CSV
        mask = self.df_events.apply(should_keep_row, axis=1)
        df_filtered = self.df_events[mask].copy()
        
        # 保存备份
        backup_path = self.csv_path.parent / f"{self.csv_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.df_events.to_csv(backup_path, index=False)
        
        # 保存更新后的CSV
        df_filtered.to_csv(self.csv_path, index=False)
        
        removed = len(self.df_events) - len(df_filtered)
        print(f"  Original records: {len(self.df_events)}")
        print(f"  Removed records: {removed}")
        print(f"  Remaining records: {len(df_filtered)}")
        print(f"  Backup saved: {backup_path}")
    
    def generate_report(self):
        """生成标注报告"""
        print("\n" + "=" * 60)
        print("LABELING STATISTICS REPORT")
        print("=" * 60)
        
        # 统计各类别数量
        category_counts = {}
        for label_info in self.labels.values():
            cat = label_info['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total_labeled = len(self.labels)
        
        print(f"Total segments in source: {self.grand_total}")
        print(f"Total labeled: {total_labeled} ({total_labeled/self.grand_total*100:.1f}%)")
        print(f"Remaining: {self.grand_total - total_labeled}")
        print(f"\nCategory breakdown:")
        
        for category, count in sorted(category_counts.items()):
            percentage = count / total_labeled * 100 if total_labeled > 0 else 0
            print(f"  {category:30s}: {count:4d} ({percentage:5.1f}%)")
        
        # 保存报告
        report_path = self.output_base / 'labeling_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("LABELING STATISTICS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input directory: {self.input_dir}\n")
            f.write(f"Output directory: {self.output_base}\n")
            f.write(f"Total segments in source: {self.grand_total}\n")
            f.write(f"Total labeled: {total_labeled}\n")
            f.write(f"Remaining: {self.grand_total - total_labeled}\n\n")
            f.write("Category breakdown:\n")
            for category, count in sorted(category_counts.items()):
                percentage = count / total_labeled * 100 if total_labeled > 0 else 0
                f.write(f"  {category}: {count} ({percentage:.1f}%)\n")
        
        print(f"\nReport saved: {report_path}")
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Interactive Click Labeling Tool (NPY Format with Resume Support)'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory (e.g., data/filtered/Uncertain)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output base directory (e.g., data/manual_labelled)')
    parser.add_argument('--csv', type=str, default=None,
                       help='all_events.csv path (optional)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='Sample rate (default: 44100)')
    parser.add_argument('--shuffle', action='store_true',
                       help='Randomly shuffle file order')
    parser.add_argument('--keep-source', action='store_true',
                       help='Keep labeled files in source directory (default: move them)')
    
    args = parser.parse_args()
    
    # 创建标注器
    labeler = ClickLabeler(
        input_dir=Path(args.input),
        output_base=Path(args.output),
        csv_path=Path(args.csv) if args.csv else None,
        sample_rate=args.sample_rate,
        remove_labeled=not args.keep_source
    )
    
    # 随机打乱（如果指定）
    if args.shuffle:
        random.shuffle(labeler.npy_files)
        print("✓ File order shuffled\n")
    
    # 开始标注
    try:
        labeler.label_interactive()
    except KeyboardInterrupt:
        print("\n\n⚠️  Labeling interrupted")
        labeler.save_labels()
        print("Current progress saved")
    
    print("\n✓ Labeling completed!")


if __name__ == '__main__':
    main()