#!/usr/bin/env python3
import os
import shutil
import argparse

def copy_npy_files(base_dir):
    numpy_dir = os.path.join(base_dir, 'numpy')
    if not os.path.isdir(numpy_dir):
        raise FileNotFoundError(f"Could not find numpy directory: {numpy_dir}")

    # Build mapping: normalized_name (without first 3 chars) -> full .npy filename
    mapping = {}
    for fname in os.listdir(numpy_dir):
        if fname.lower().endswith('.npy') and len(fname) > 3:
            normalized = fname[3:]  # drop the 3-character prefix
            mapping[normalized] = fname

    subsets = ['training', 'validation']
    categories = ['HC_AH', 'PD_AH']

    for subset in subsets:
        for category in categories:
            wav_dir = os.path.join(base_dir, subset, category)
            if not os.path.isdir(wav_dir):
                print(f"[WARN] Directory not found, skipping: {wav_dir}")
                continue

            for fname in os.listdir(wav_dir):
                if not fname.lower().endswith('.wav'):
                    continue
                base = fname[:-4]               # e.g. 'foo' from 'foo.wav'
                normalized_npy = base + '.npy'  # e.g. 'foo.npy'

                npy_fname = mapping.get(normalized_npy)
                if not npy_fname:
                    print(f"[MISSING] No .npy found for {os.path.join(subset,category,fname)}")
                    continue

                src = os.path.join(numpy_dir, npy_fname)
                dst = os.path.join(wav_dir, normalized_npy)

                if os.path.exists(dst):
                    print(f"[SKIP] Destination exists: {dst}")
                else:
                    shutil.copy2(src, dst)
                    print(f"[COPIED] {src} → {dst}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Copy and rename .npy files to match their .wav counterparts."
    )
    parser.add_argument(
        '--base-dir', '-b',
        default='train-test-split',
        help='Root of the split folders (default: train-test-split)'
    )
    args = parser.parse_args()

    copy_npy_files(args.base_dir)
