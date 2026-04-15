#!/usr/bin/env python3
"""
Script 01: Prepare Dataset
==========================

Download and sample from RAGBench for evaluation.

Usage:
    python scripts/01_prepare_dataset.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader


def main():
    print("="*60)
    print("RAG Meta-Analysis: Dataset Preparation")
    print("="*60)
    
    # Initialize data loader
    loader = DataLoader(config_path='configs/config.yaml')
    
    # Load and sample data
    samples = loader.load_samples(save_path='data/samples_200.json')
    
    # Print statistics
    stats = loader.get_statistics()
    
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    for key, value in stats.items():
        if key == 'domains':
            print(f"\nDomains:")
            for domain, count in value.items():
                print(f"  - {domain}: {count}")
        elif key == 'subsets':
            print(f"\nSubsets:")
            for subset, count in value.items():
                print(f"  - {subset}: {count}")
        else:
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    print("\n" + "="*60)
    print(f"Dataset saved to data/samples_200.json")
    print("="*60)


if __name__ == "__main__":
    main()
