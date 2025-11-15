#!/usr/bin/env python3
"""
Independent Audio Segment Normalization Script
Ensures consistent preprocessing between training and testing data
"""

import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import logging


def normalize_peak(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    """
    Peak normalization (for click segments)
    
    Args:
        audio: Input audio array
        target: Target peak amplitude (default 0.95)
    
    Returns:
        Normalized audio array
    """
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        return audio / peak * target
    return audio


def normalize_rms(audio: np.ndarray, target_rms: float = 0.1, 
                  peak_limit: float = 0.95) -> np.ndarray:
    """
    RMS normalization (for noise segments)
    
    Args:
        audio: Input audio array
        target_rms: Target RMS level (default 0.1)
        peak_limit: Peak amplitude limit (default 0.95)
    
    Returns:
        Normalized audio array
    """
    rms = np.sqrt(np.mean(audio**2))
    
    # RMS normalization
    if rms > 1e-8:
        audio = audio * (target_rms / rms)
    
    # Peak limiting
    peak = np.max(np.abs(audio))
    if peak > peak_limit:
        audio = audio / peak * peak_limit
    
    return audio


def process_directory(input_dir: Path, output_dir: Path, 
                     method: str, target: float, 
                     target_rms: float, peak_limit: float,
                     recursive: bool, verbose: bool):
    """
    Batch process audio files in directory
    """
    logger = logging.getLogger(__name__)
    
    # Find all wav files
    if recursive:
        wav_files = list(input_dir.rglob('*.wav'))
    else:
        wav_files = list(input_dir.glob('*.wav'))
    
    if not wav_files:
        logger.error(f"No WAV files found in: {input_dir}")
        return
    
    logger.info(f"Found {len(wav_files)} files")
    logger.info(f"Normalization method: {method}")
    
    if method == 'peak':
        logger.info(f"  Target peak: {target}")
    elif method == 'rms':
        logger.info(f"  Target RMS: {target_rms}")
        logger.info(f"  Peak limit: {peak_limit}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each file
    success_count = 0
    failed_count = 0
    
    for wav_file in tqdm(wav_files, desc="Normalizing"):
        try:
            # Read audio
            audio, sr = sf.read(wav_file)
            
            # Convert to mono
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            
            # Apply normalization
            if method == 'peak':
                normalized = normalize_peak(audio, target)
            elif method == 'rms':
                normalized = normalize_rms(audio, target_rms, peak_limit)
            else:
                raise ValueError(f"Unknown normalization method: {method}")
            
            # Preserve relative path structure
            rel_path = wav_file.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save normalized audio
            sf.write(out_path, normalized, sr)
            success_count += 1
            
            if verbose:
                logger.debug(f"✅ {wav_file.name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {wav_file}: {e}")
            failed_count += 1
            continue
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Processing Complete")
    logger.info("=" * 60)
    logger.info(f"Successful: {success_count} files")
    logger.info(f"Failed: {failed_count} files")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Audio segment normalization script (ensures train/test preprocessing consistency)'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing audio files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for normalized files')
    parser.add_argument('--method', type=str, default='peak',
                       choices=['peak', 'rms'],
                       help='Normalization method: peak (for clicks) or rms (for noise)')
    parser.add_argument('--target', type=float, default=0.95,
                       help='Target peak amplitude for peak normalization (default: 0.95)')
    parser.add_argument('--target-rms', type=float, default=0.1,
                       help='Target RMS level for RMS normalization (default: 0.1)')
    parser.add_argument('--peak-limit', type=float, default=0.95,
                       help='Peak amplitude limit after RMS normalization (default: 0.95)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Search subdirectories recursively')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s'
    )
    
    # Execute processing
    process_directory(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        method=args.method,
        target=args.target,
        target_rms=args.target_rms,
        peak_limit=args.peak_limit,
        recursive=args.recursive,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()