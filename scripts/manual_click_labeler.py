#!/usr/bin/env python3
"""
交互式Click片段手动标注工具
支持快捷键操作，实时显示波形+频谱图
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
    """交互式Click标注工具"""
    
    def __init__(self, 
                 input_dir: Path,
                 output_base: Path,
                 csv_path: Path = None,
                 sample_rate: int = 44100):
        """
        初始化标注工具
        
        Args:
            input_dir: 输入目录（detection_results/audio）
            output_base: 输出基础目录（data/manual_labelled）
            csv_path: all_events.csv路径（可选）
            sample_rate: 采样率
        """
        self.input_dir = Path(input_dir)
        self.output_base = Path(output_base)
        self.csv_path = Path(csv_path) if csv_path else None
        self.sample_rate = sample_rate
        
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
        
        # 扫描所有wav文件
        self.wav_files = list(self.input_dir.rglob('*.wav'))
        self.total_files = len(self.wav_files)
        
        # 加载CSV（如果提供）
        self.df_events = None
        if self.csv_path and self.csv_path.exists():
            self.df_events = pd.read_csv(self.csv_path)
        
        # 标注记录
        self.labels = {}
        self.current_idx = 0
        
        # 设置matplotlib交互模式
        plt.ion()
        
        print("=" * 60)
        print("交互式Click片段标注工具")
        print("=" * 60)
        print(f"输入目录: {self.input_dir}")
        print(f"输出目录: {self.output_base}")
        print(f"总片段数: {self.total_files}")
        print("=" * 60)
    
    def plot_click(self, wav_path: Path):
        """
        绘制单个click片段的波形+频谱图
        
        Args:
            wav_path: WAV文件路径
        """
        # 读取音频
        audio, sr = sf.read(wav_path)
        
        # 转单声道
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        
        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        
        # 设置窗口标题（英文，避免编码问题）
        fig.canvas.manager.set_window_title(
            f'[{self.current_idx+1}/{self.total_files}] {wav_path.name} - INPUT IN TERMINAL!'
        )
        
        fig.suptitle(f'Click Segment: {wav_path.name}\n[{self.current_idx+1}/{self.total_files}]', 
                     fontsize=14, fontweight='bold')
        
        # 1. 波形图
        ax1 = plt.subplot(3, 1, 1)
        time_ms = np.arange(len(audio)) / sr * 1000
        ax1.plot(time_ms, audio, linewidth=0.8, color='steelblue')
        ax1.set_xlabel('Time (ms)', fontsize=11)
        ax1.set_ylabel('Amplitude', fontsize=11)
        ax1.set_title('Waveform', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
        
        # 标注峰值位置
        peak_idx = np.argmax(np.abs(audio))
        peak_time = peak_idx / sr * 1000
        ax1.axvline(x=peak_time, color='red', linestyle='--', 
                   linewidth=1.5, alpha=0.7, label=f'Peak @ {peak_time:.1f}ms')
        ax1.legend(fontsize=9)
        
        # 2. 频谱图（FFT）
        ax2 = plt.subplot(3, 1, 2)
        
        # 应用窗函数
        window = np.hanning(len(audio))
        fft = np.fft.rfft(audio * window)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(audio), 1/sr)
        
        # 转dB
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        ax2.plot(freqs / 1000, magnitude_db, linewidth=0.8, color='darkorange')
        ax2.set_xlabel('Frequency (kHz)', fontsize=11)
        ax2.set_ylabel('Magnitude (dB)', fontsize=11)
        ax2.set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
        ax2.set_xlim(0, sr / 2000)
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
            audio, sr, 
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
        ax3.set_ylim(0, sr / 2000)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax3, label='Power (dB)')
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        
        # 添加操作指南（英文，避免编码问题）
        instructions = (
            "=" * 80 + "\n"
            "KEYBOARD SHORTCUTS (Input in Terminal Window!):\n"
            "  [1] Positive_HQ (high quality click)      [2] Negative_Hard (shrimp/noise)\n"
            "  [3] Sparse/Uncertain                      [4] LowSNR (ambiguous)\n"
            "  [0] Skip (decide later)                   [Q] Quit (save and exit)\n"
            "  [P] Previous (go back)                    [Space] Play Audio\n"
            "=" * 80
        )
        '''
        fig.text(0.5, 0.02, instructions, ha='center', fontsize=9,
                family='monospace', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.3))
        '''
        plt.draw()
        plt.pause(0.001)
    
    def play_audio(self, wav_path: Path):
        """播放音频（需要系统支持）"""
        try:
            import subprocess
            import platform
            
            system = platform.system()
            if system == 'Darwin':  # macOS
                subprocess.run(['afplay', str(wav_path)], check=True)
            elif system == 'Linux':
                subprocess.run(['aplay', str(wav_path)], check=True)
            elif system == 'Windows':
                import winsound
                winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
        except Exception as e:
            print(f"无法播放音频: {e}")
    
    def label_interactive(self):
        """交互式标注主循环"""
        print("\n" + "="*80)
        print("LABELING TOOL STARTED")
        print("="*80)
        print("IMPORTANT: Look at the PLOT WINDOW, but TYPE in THIS TERMINAL!")
        print("Press any key (1/2/3/4/0) to label, Q to quit")
        print("="*80 + "\n")
        
        while self.current_idx < self.total_files:
            wav_path = self.wav_files[self.current_idx]
            
            # 清除之前的图形
            plt.clf()
            
            # 绘制当前片段
            self.plot_click(wav_path)
            
            # 等待用户输入
            print("\n" + "="*80)
            print(f">>> Current: [{self.current_idx+1}/{self.total_files}] {wav_path.name}")
            print("="*80)
            print(">>> INPUT HERE (in Terminal): [1/2/3/4/0/P/Q/Space]")
            print(">>> Your choice: ", end='', flush=True)
            
            # 获取键盘输入
            choice = input().strip().lower()
            
            # 处理输入
            if choice == 'q':
                print("\n退出标注...")
                break
            
            elif choice == 'p':
                # 返回上一个
                if self.current_idx > 0:
                    self.current_idx -= 1
                    print("← 返回上一个")
                else:
                    print("已经是第一个！")
                continue
            
            elif choice == ' ' or choice == 'space':
                # 播放音频
                print("▶ 播放音频...")
                self.play_audio(wav_path)
                continue
            
            elif choice in ['1', '2', '3', '4', '0']:
                # 记录标注
                category = self.categories[choice]
                self.labels[str(wav_path)] = {
                    'category': category,
                    'choice': choice,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"✓ 标注为: {category}")
                self.current_idx += 1
            
            else:
                print("❌ 无效输入！请重新选择")
                continue
        
        plt.close('all')
        
        # 保存标注结果
        self.save_labels()
        
        # 移动文件
        self.organize_files()
        
        # 更新CSV
        if self.df_events is not None:
            self.update_csv()
        
        # 生成报告
        self.generate_report()
    
    def save_labels(self):
        """保存标注记录到JSON"""
        labels_file = self.output_base / 'labeling_record.json'
        
        with open(labels_file, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 标注记录已保存: {labels_file}")
    
    def organize_files(self):
        """根据标注结果整理文件"""
        print("\n整理文件到分类目录...")
        
        moved_count = {cat: 0 for cat in self.categories.values()}
        
        for wav_path_str, label_info in self.labels.items():
            wav_path = Path(wav_path_str)
            category = label_info['category']
            
            if category == 'Skipped':
                continue  # 跳过的文件不移动
            
            # 目标路径
            dest_dir = self.output_base / category
            dest_path = dest_dir / wav_path.name
            
            # 处理重名
            counter = 1
            while dest_path.exists():
                dest_path = dest_dir / f"{wav_path.stem}_{counter}{wav_path.suffix}"
                counter += 1
            
            # 移动文件
            try:
                shutil.move(str(wav_path), str(dest_path))
                moved_count[category] += 1
            except Exception as e:
                print(f"❌ 移动失败 {wav_path.name}: {e}")
        
        print("\n文件整理完成:")
        for category, count in moved_count.items():
            if count > 0:
                print(f"  {category}: {count} 个文件")
    
    def update_csv(self):
        """更新all_events.csv"""
        print("\n更新all_events.csv...")
        
        # 构建文件名到标注的映射
        labeled_files = set()
        for wav_path_str, label_info in self.labels.items():
            wav_path = Path(wav_path_str)
            if label_info['category'] != 'Positive_HQ':
                labeled_files.add(wav_path.name)
        
        # 过滤CSV（保留Positive_HQ和未标注的）
        if 'source_file' in self.df_events.columns:
            mask = ~self.df_events['source_file'].apply(
                lambda x: Path(x).name in labeled_files
            )
        else:
            # 如果没有source_file列，尝试通过file_id匹配
            mask = ~self.df_events['file_id'].apply(
                lambda x: any(x in str(p) for p in labeled_files)
            )
        
        df_filtered = self.df_events[mask].copy()
        
        # 保存更新后的CSV
        backup_path = self.csv_path.parent / f"{self.csv_path.stem}_backup.csv"
        self.df_events.to_csv(backup_path, index=False)
        df_filtered.to_csv(self.csv_path, index=False)
        
        removed = len(self.df_events) - len(df_filtered)
        print(f"  原始记录: {len(self.df_events)}")
        print(f"  移除记录: {removed}")
        print(f"  保留记录: {len(df_filtered)}")
        print(f"  备份文件: {backup_path}")
    
    def generate_report(self):
        """生成标注报告"""
        print("\n" + "=" * 60)
        print("标注统计报告")
        print("=" * 60)
        
        # 统计各类别数量
        category_counts = {}
        for label_info in self.labels.values():
            cat = label_info['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        total_labeled = len(self.labels)
        
        print(f"总片段数: {self.total_files}")
        print(f"已标注数: {total_labeled} ({total_labeled/self.total_files*100:.1f}%)")
        print(f"\n分类统计:")
        
        for category, count in sorted(category_counts.items()):
            percentage = count / total_labeled * 100 if total_labeled > 0 else 0
            print(f"  {category:30s}: {count:4d} ({percentage:5.1f}%)")
        
        # 保存报告
        report_path = self.output_base / 'labeling_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("标注统计报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"输入目录: {self.input_dir}\n")
            f.write(f"输出目录: {self.output_base}\n")
            f.write(f"总片段数: {self.total_files}\n")
            f.write(f"已标注数: {total_labeled}\n\n")
            f.write("分类统计:\n")
            for category, count in sorted(category_counts.items()):
                percentage = count / total_labeled * 100 if total_labeled > 0 else 0
                f.write(f"  {category}: {count} ({percentage:.1f}%)\n")
        
        print(f"\n报告已保存: {report_path}")
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='交互式Click片段手动标注工具'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='输入目录 (detection_results/audio)')
    parser.add_argument('--output', type=str, required=True,
                       help='输出基础目录 (data/manual_labelled)')
    parser.add_argument('--csv', type=str, default=None,
                       help='all_events.csv路径 (可选)')
    parser.add_argument('--sample-rate', type=int, default=44100,
                       help='采样率 (默认: 44100)')
    parser.add_argument('--shuffle', action='store_true',
                       help='随机打乱文件顺序')
    
    args = parser.parse_args()
    
    # 创建标注器
    labeler = ClickLabeler(
        input_dir=Path(args.input),
        output_base=Path(args.output),
        csv_path=Path(args.csv) if args.csv else None,
        sample_rate=args.sample_rate
    )
    
    # 随机打乱（如果指定）
    if args.shuffle:
        random.shuffle(labeler.wav_files)
        print("✓ 文件顺序已随机打乱\n")
    
    # 开始标注
    try:
        labeler.label_interactive()
    except KeyboardInterrupt:
        print("\n\n⚠️  标注被中断")
        labeler.save_labels()
        print("已保存当前进度")
    
    print("\n✓ 标注完成！")


if __name__ == '__main__':
    main()