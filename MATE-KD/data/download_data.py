#!/usr/bin/env python3
"""
Data Download Script for MATE-KD

Downloads CIFAR-10 and CIFAR-10-C datasets with verification.
"""

import os
import sys
import argparse
import hashlib
import requests
import tarfile
import zipfile
from pathlib import Path
from tqdm import tqdm

# Dataset configurations
DATASETS = {
    'cifar10': {
        'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
        'filename': 'cifar-10-python.tar.gz',
        'extract_to': 'cifar-10-batches-py',
        'md5': 'c58f30108f718f92721af3b95e74349a',
        'size_mb': 163
    },
    'cifar10c': {
        'url': 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar',
        'filename': 'CIFAR-10-C.tar', 
        'extract_to': 'CIFAR-10-C',
        'md5': '56bf5dcef84df0e2308c6dcbcbbd8499',
        'size_mb': 2800
    }
}

def calculate_md5(filepath, chunk_size=8192):
    """Calculate MD5 hash of a file"""
    md5_hash = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()

def download_file(url, filepath, expected_size_mb=None):
    """Download file with progress bar and verification"""
    
    if filepath.exists():
        print(f"File already exists: {filepath}")
        return True
    
    print(f"Downloading: {url}")
    print(f"Destination: {filepath}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        # Create parent directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Verify file size
        actual_size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"Downloaded: {actual_size_mb:.1f} MB")
        
        if expected_size_mb and abs(actual_size_mb - expected_size_mb) > expected_size_mb * 0.1:
            print(f"Warning: File size mismatch. Expected ~{expected_size_mb}MB, got {actual_size_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"Download failed: {e}")
        if filepath.exists():
            filepath.unlink()  # Remove incomplete file
        return False

def verify_checksum(filepath, expected_md5):
    """Verify file integrity using MD5 checksum"""
    if not filepath.exists():
        return False
    
    print(f"Verifying checksum for {filepath.name}...")
    actual_md5 = calculate_md5(filepath)
    
    if actual_md5 == expected_md5:
        print("Checksum verified")
        return True
    else:
        print(f"Checksum mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False

def extract_archive(filepath, extract_to=None):
    """Extract tar.gz or tar files"""
    if extract_to is None:
        extract_to = filepath.parent
    else:
        extract_to = Path(extract_to)
    
    print(f"Extracting {filepath.name}...")
    
    try:
        if filepath.suffix == '.gz':
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(path=extract_to)
        elif filepath.suffix == '.tar':
            with tarfile.open(filepath, 'r') as tar:
                tar.extractall(path=extract_to)
        elif filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {filepath.suffix}")
            return False
        
        print(f"Extracted to: {extract_to}")
        return True
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False

def download_dataset(dataset_name, data_dir='./data', verify_checksum_flag=True, extract=True):
    """Download and setup a specific dataset"""
    
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False
    
    config = DATASETS[dataset_name]
    data_dir = Path(data_dir)
    
    # File paths
    archive_path = data_dir / config['filename']
    extract_path = data_dir / config['extract_to']
    
    print(f"Setting up {dataset_name.upper()} dataset")
    print(f"Data directory: {data_dir.absolute()}")
    
    # Check if already extracted
    if extract_path.exists() and any(extract_path.iterdir()):
        print(f"Dataset already exists at: {extract_path}")
        return True
    
    # Download
    success = download_file(
        config['url'], 
        archive_path, 
        config.get('size_mb')
    )
    
    if not success:
        return False
    
    # Verify checksum
    if verify_checksum_flag:
        if not verify_checksum(archive_path, config['md5']):
            print("Checksum verification failed. File may be corrupted.")
            return False
    
    # Extract
    if extract:
        if not extract_archive(archive_path, data_dir):
            return False
        
        # Verify extraction
        if not extract_path.exists():
            print(f"Extraction verification failed. Expected directory: {extract_path}")
            return False
        
        print(f"Dataset ready at: {extract_path}")
    
    return True

def setup_data_directory(data_dir='./data'):
    """Create and setup data directory structure"""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory: {data_dir.absolute()}")
    return data_dir

def main():
    parser = argparse.ArgumentParser(description='Download datasets for MATE-KD')
    parser.add_argument('--dataset', choices=['cifar10', 'cifar10c', 'all'], 
                       default='all', help='Dataset to download')
    parser.add_argument('--data-dir', default='./data', 
                       help='Directory to store datasets')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip checksum verification')
    parser.add_argument('--no-extract', action='store_true',
                       help='Download only, do not extract')
    
    args = parser.parse_args()
    
    # Setup data directory
    data_dir = setup_data_directory(args.data_dir)
    
    # Determine which datasets to download
    if args.dataset == 'all':
        datasets_to_download = ['cifar10', 'cifar10c']
    else:
        datasets_to_download = [args.dataset]
    
    print(f"Downloading datasets: {', '.join(datasets_to_download)}")
    
    # Download each dataset
    results = {}
    for dataset in datasets_to_download:
        print(f"\n{'='*60}")
        success = download_dataset(
            dataset, 
            data_dir, 
            verify_checksum_flag=not args.no_verify,
            extract=not args.no_extract
        )
        results[dataset] = success
        
        if success:
            print(f"SUCCESS: {dataset}")
        else:
            print(f"FAILED: {dataset}")
    
    # Summary
    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    
    successful = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]
    
    if successful:
        print("Successful downloads:")
        for dataset in successful:
            print(f"  - {dataset}")
    
    if failed:
        print("Failed downloads:")
        for dataset in failed:
            print(f"  - {dataset}")
    
    if len(successful) == len(datasets_to_download):
        print("Dataset download completed successfully!")
        return 0
    else:
        print("Some downloads failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 