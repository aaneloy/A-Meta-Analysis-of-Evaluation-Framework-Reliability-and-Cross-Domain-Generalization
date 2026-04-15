"""
Data Loading Module
===================

Handles loading and preprocessing of RAGBench dataset.
"""

import json
import random
import os
from typing import Dict, List, Optional
from collections import Counter
from datasets import load_dataset
import yaml


class DataLoader:
    """Load and sample data from RAGBench for evaluation."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.random_seed = self.config['dataset']['random_seed']
        random.seed(self.random_seed)

        self.samples = []
        self.domain_mapping = {
            'hotpotqa': 'General Knowledge',
            'msmarco': 'General Knowledge',
            'hagrid': 'General Knowledge',
            'expertqa': 'General Knowledge',
            'finqa': 'Finance',
            'delucionqa': 'Finance',
            'covidqa': 'Biomedicine',
            'pubmedqa': 'Biomedicine'
        }

    def load_ragbench_subset(self, subset_name: str) -> Optional[Dict]:
        try:
            return load_dataset("rungalileo/ragbench", subset_name)
        except Exception as e:
            print(f"Warning: Could not load {subset_name}: {e}")
            return None

    @staticmethod
    def _coerce_binary_label(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and value in {0, 1}:
            return int(value)
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"faithful", "supported", "entails", "true", "yes", "1"}:
                return 1
            if v in {"unfaithful", "unsupported", "contradicted", "false", "no", "0"}:
                return 0
        return None

    def _extract_human_label(self, example: Dict) -> Optional[int]:
        candidate_keys = [
            'human_label', 'human_faithfulness', 'faithfulness_label', 'is_faithful',
            'binary_label', 'label', 'faithful', 'supported'
        ]

        for key in candidate_keys:
            if key in example:
                label = self._coerce_binary_label(example.get(key))
                if label is not None:
                    return label

        all_labels = example.get('all_labels', {})
        if isinstance(all_labels, dict):
            for key in candidate_keys:
                if key in all_labels:
                    label = self._coerce_binary_label(all_labels.get(key))
                    if label is not None:
                        return label

        return None

    def extract_sample(self, example: Dict, subset_name: str, idx: int) -> Dict:
        context = example.get('documents', example.get('context', ''))
        if isinstance(context, list):
            context = '\n\n---\n\n'.join(context)

        ground_truth = example.get('all_labels', {})
        if not isinstance(ground_truth, dict):
            ground_truth = {'answer': example.get('answer', '')}

        human_label = self._extract_human_label(example)

        return {
            'id': f"{subset_name}_{idx}",
            'domain': self.domain_mapping.get(subset_name, 'Unknown'),
            'subset': subset_name,
            'question': example.get('question', ''),
            'answer': example.get('response', example.get('answer', '')),
            'context': context,
            'ground_truth': ground_truth,
            'human_label': human_label,
            'metadata': {
                'source': subset_name,
                'has_trace_labels': 'all_labels' in example,
                'has_human_label': human_label is not None,
            }
        }

    def _balanced_indices_by_human_label(self, data, n_samples: int) -> List[int]:
        labeled = {0: [], 1: []}
        unlabeled = []

        for idx in range(len(data)):
            label = self._extract_human_label(data[idx])
            if label in {0, 1}:
                labeled[label].append(idx)
            else:
                unlabeled.append(idx)

        half = n_samples // 2
        selected = []

        for label in [0, 1]:
            take = min(half, len(labeled[label]))
            if take > 0:
                selected.extend(random.sample(labeled[label], take))

        remaining = n_samples - len(selected)

        pool = [i for i in unlabeled if i not in selected]
        if remaining > 0:
            label_pool = [i for l in [0, 1] for i in labeled[l] if i not in selected]
            pool = label_pool + pool
            remaining = min(remaining, len(pool))
            if remaining > 0:
                selected.extend(random.sample(pool, remaining))

        return selected

    def sample_from_domain(self, domain_config: Dict, domain_name: str) -> List[Dict]:
        subsets = domain_config['subsets']
        target_samples = domain_config['samples']

        available_subsets = []
        for subset in subsets:
            ds = self.load_ragbench_subset(subset)
            if ds is not None:
                available_subsets.append((subset, ds))

        if not available_subsets:
            print(f"Warning: No subsets available for {domain_name}")
            return []

        samples_per_subset = target_samples // len(available_subsets)
        remainder = target_samples % len(available_subsets)

        domain_samples = []

        for i, (subset_name, ds) in enumerate(available_subsets):
            data = ds['test'] if 'test' in ds else ds['validation'] if 'validation' in ds else ds['train']
            n_samples = min(samples_per_subset + (1 if i < remainder else 0), len(data))

            indices = self._balanced_indices_by_human_label(data, n_samples)

            for idx in indices:
                sample = self.extract_sample(data[idx], subset_name, idx)
                if sample['question'] and sample['answer'] and sample['context']:
                    domain_samples.append(sample)

            label_counts = Counter(s.get('human_label') for s in domain_samples if s['subset'] == subset_name)
            print(f"  Sampled {len(indices)} from {subset_name} (human_label balance: {dict(label_counts)})")

        return domain_samples

    def load_samples(self, save_path: Optional[str] = None) -> List[Dict]:
        print("=" * 60)
        print("Loading RAGBench Dataset")
        print("=" * 60)

        all_samples = []

        for domain_key, domain_config in self.config['dataset']['domains'].items():
            domain_name = domain_key.replace('_', ' ').title()
            print(f"\nLoading {domain_name}...")
            samples = self.sample_from_domain(domain_config, domain_name)
            all_samples.extend(samples)
            print(f"  Total for {domain_name}: {len(samples)}")

        self.samples = all_samples

        print("\n" + "=" * 60)
        print("Final Distribution")
        print("=" * 60)
        domain_counts = Counter(s['domain'] for s in self.samples)
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count}")
        print(f"  TOTAL: {len(self.samples)}")
        human_counts = Counter(s.get('human_label') for s in self.samples)
        print(f"  HUMAN LABELS: {dict(human_counts)}")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.samples, f, indent=2)
            print(f"\nSaved to {save_path}")

        return self.samples

    def load_from_file(self, path: str) -> List[Dict]:
        with open(path, 'r') as f:
            self.samples = json.load(f)
        print(f"Loaded {len(self.samples)} samples from {path}")
        return self.samples

    def get_samples_by_domain(self, domain: str) -> List[Dict]:
        return [s for s in self.samples if s['domain'] == domain]

    def get_statistics(self) -> Dict:
        if not self.samples:
            return {}

        return {
            'total_samples': len(self.samples),
            'domains': Counter(s['domain'] for s in self.samples),
            'subsets': Counter(s['subset'] for s in self.samples),
            'human_labels': Counter(s.get('human_label') for s in self.samples),
            'avg_question_length': sum(len(s['question']) for s in self.samples) / len(self.samples),
            'avg_answer_length': sum(len(s['answer']) for s in self.samples) / len(self.samples),
            'avg_context_length': sum(len(s['context']) for s in self.samples) / len(self.samples),
        }


if __name__ == "__main__":
    loader = DataLoader()
    loader.load_samples(save_path="data/samples_200.json")
    stats = loader.get_statistics()
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
