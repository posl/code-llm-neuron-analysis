#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variable renaming similarity analysis - Optimized version
Analyze representation similarity changes before and after variable name replacement
"""

import os
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import torch
from tqdm import tqdm
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Import shared utilities
from utils import (
    setup_paths,
    read_jsonl_file,
    write_jsonl_file
)

# Import from RQ1 modules
from src.RQ1.utils import setup_logging as rq1_setup_logging, set_seed
from src.RQ1.model_loader import load_model_and_tokenizer

# Set up logger
logger = logging.getLogger(__name__)

# Initialize paths
paths = setup_paths()
ROOT_DIR = paths['ROOT_DIR']

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Variable name replacement similarity analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Similarity metrics:
  cosine - Cosine similarity, measures vector direction similarity
        """
    )
    
    parser.add_argument("--model_path", type=str,
                      required=True,
                      help="Model path or huggingface model name")
    parser.add_argument("--original_dataset_dir", type=str,
                      required=True,
                      help="Original code dataset directory")
    parser.add_argument("--renamed_dataset_dir", type=str,
                      required=True,
                      help="Renamed code dataset directory")
    parser.add_argument("--output_dir", type=str,
                      required=True,
                      help="Output directory")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Log level")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run on")
    parser.add_argument("--device_map", type=str, default=None,
                      help="Device mapping configuration")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--target_languages", type=str, nargs="+",
                      default=["python", "cpp", "java", "go", "js", "rust"],
                      help="Target languages")
    parser.add_argument("--max_samples_per_lang", type=int, default=None,
                      help="Max samples per language")
    parser.add_argument("--similarity_metrics", type=str, nargs="+",
                      default=["cosine"],
                      choices=["cosine"],
                      help="Similarity metrics")
    parser.add_argument("--primary_metric", type=str, default="cosine",
                      choices=["cosine"],
                      help="Primary metric")
    parser.add_argument("--enable_all_visualizations", action="store_true", 
                      default=True,
                      help="Enable visualizations")
    
    return parser.parse_args()

class DataLoader:
    """Handle data loading for similarity analysis"""
    
    @staticmethod
    def load_paired_data(
        original_dir: str,
        renamed_dir: str,
        languages: List[str],
        max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Load paired original and renamed code"""
        paired_data = []
        counts = defaultdict(int)
        
        for lang in languages:
            orig_path = Path(original_dir) / lang / "samples.jsonl"
            renamed_path = Path(renamed_dir) / lang / "samples.jsonl"
            
            if not orig_path.exists() or not renamed_path.exists():
                logger.warning(f"Missing data files for {lang}")
                continue
            
            # Load original samples
            orig_samples = {}
            for sample in read_jsonl_file(str(orig_path)):
                orig_samples[sample['task_id']] = sample
            
            # Load and pair renamed samples
            renamed_count = 0
            for renamed_sample in read_jsonl_file(str(renamed_path)):
                if max_samples and renamed_count >= max_samples:
                    break
                
                task_id = renamed_sample.get('task_id')
                if task_id in orig_samples:
                    orig_sample = orig_samples[task_id]
                    
                    orig_code = orig_sample.get('generation', '')
                    renamed_code = renamed_sample.get('generation', '')
                    
                    if orig_code and renamed_code:
                        paired_data.append({
                            'task_id': task_id,
                            'language': lang,
                            'original_code': orig_code,
                            'renamed_code': renamed_code
                        })
                        renamed_count += 1
                        counts[lang] += 1
        
        logger.info(f"Loaded {len(paired_data)} paired samples")
        for lang, count in counts.items():
            logger.info(f"  {lang}: {count} samples")
        
        return paired_data

class ModelHelper:
    """Model utility functions"""
    
    @staticmethod
    def get_num_layers(model) -> int:
        """Get number of model layers"""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return len(model.transformer.h)
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return len(model.encoder.layer)
        else:
            for attr in ["layers", "h", "layer", "blocks"]:
                if hasattr(model, attr):
                    layers = getattr(model, attr)
                    if hasattr(layers, "__len__"):
                        return len(layers)
            raise ValueError("Cannot determine model layer count")
    
    @staticmethod
    def get_layer(model, layer_idx: int):
        """Get specific layer from model"""
        try:
            if hasattr(model, "model") and hasattr(model.model, "layers"):
                return model.model.layers[layer_idx]
            elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
                return model.transformer.h[layer_idx]
            elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
                return model.encoder.layer[layer_idx]
            elif hasattr(model, "layers"):
                return model.layers[layer_idx]
            
            # Try nested structure
            for model_attr in ["model", "transformer", "encoder"]:
                if hasattr(model, model_attr):
                    sub_model = getattr(model, model_attr)
                    for layer_attr in ["layers", "h", "layer", "blocks"]:
                        if hasattr(sub_model, layer_attr):
                            layers = getattr(sub_model, layer_attr)
                            if isinstance(layers, (list, tuple)) and len(layers) > layer_idx:
                                return layers[layer_idx]
            return None
        except (IndexError, AttributeError):
            return None

class RepresentationExtractor:
    """Extract code representations from model"""
    
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Check for multi-GPU setup
        self.multi_gpu = False
        if hasattr(model, "hf_device_map"):
            self.multi_gpu = len(set(model.hf_device_map.values())) > 1
    
    def extract(
        self, 
        code: str, 
        layer_indices: List[int], 
        max_length: int = 512
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Extract representations for specified layers"""
        inputs = self.tokenizer(
            code, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length,
            padding=False
        )
        
        # Determine target device
        target_device = self._get_target_device()
        input_ids = inputs["input_ids"].to(target_device)
        
        # Set up hooks
        representations = {}
        hooks = []
        captured = {}
        
        def create_hook(name):
            def hook(module, input, output):
                device = output[0].device
                captured[name] = output[0].detach().to(device)
            return hook
        
        # Register hooks
        for idx in layer_indices:
            layer = ModelHelper.get_layer(self.model, idx)
            if layer is not None:
                hook = layer.register_forward_hook(create_hook(f"layer_{idx}"))
                hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(input_ids)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Process captured states
        for idx in layer_indices:
            name = f"layer_{idx}"
            if name in captured:
                hidden = captured[name]
                
                # Clamp extreme values
                if hidden.abs().max() > 100:
                    hidden = torch.clamp(hidden, min=-50, max=50)
                
                # Pool (exclude first token if multiple)
                if hidden.shape[1] > 1:
                    pooled = hidden[:, 1:, :].mean(dim=1)
                else:
                    pooled = hidden.mean(dim=1)
                
                representations[idx] = {
                    "hidden_state": hidden.cpu().numpy(),
                    "pooled": pooled.cpu().numpy()
                }
        
        return representations
    
    def _get_target_device(self):
        """Determine target device for inputs"""
        if not self.multi_gpu:
            return self.device
        
        device_values = list(self.model.hf_device_map.values())
        valid = [d for d in device_values if isinstance(d, str) and d != "cpu"]
        if valid:
            return valid[0]
        
        int_devices = [d for d in device_values if isinstance(d, int)]
        if int_devices:
            return f"cuda:{min(int_devices)}"
        
        return device_values[0] if device_values else self.device

class SimilarityAnalyzer:
    """Compute similarity metrics"""
    
    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or ["cosine"]
        self.funcs = {
            "cosine": self._cosine_similarity
        }
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity"""
        sim = 1 - cosine(v1.flatten(), v2.flatten())
        return float(sim) if not np.isnan(sim) else 0.0
    
    def compute(self, v1: np.ndarray, v2: np.ndarray) -> Dict[str, float]:
        """Compute all specified similarity metrics"""
        results = {}
        for metric in self.metrics:
            if metric in self.funcs:
                results[metric] = self.funcs[metric](v1, v2)
        return results

class SimilarityComputer:
    """Compute similarities for paired data"""
    
    def __init__(
        self,
        extractor: RepresentationExtractor,
        analyzer: SimilarityAnalyzer,
        num_layers: int
    ):
        self.extractor = extractor
        self.analyzer = analyzer
        self.num_layers = num_layers
        
        # Sample layers uniformly
        self.layer_indices = self._get_layer_indices()
    
    def _get_layer_indices(self) -> List[int]:
        """Get indices of layers to analyze"""
        if self.num_layers <= 10:
            return list(range(self.num_layers))
        else:
            step = self.num_layers // 10
            indices = list(range(0, self.num_layers, step))
            if (self.num_layers - 1) not in indices:
                indices.append(self.num_layers - 1)
            return indices
    
    def compute_similarities(
        self,
        paired_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute similarities for all paired data"""
        results = {
            "by_language": defaultdict(lambda: defaultdict(list)),
            "by_layer": defaultdict(lambda: defaultdict(list)),
            "overall": defaultdict(list),
            "details": []
        }
        
        for sample in tqdm(paired_data, desc="Computing similarities"):
            # Extract representations
            orig_reps = self.extractor.extract(
                sample['original_code'],
                self.layer_indices
            )
            renamed_reps = self.extractor.extract(
                sample['renamed_code'],
                self.layer_indices
            )
            
            # Compute similarities
            sample_result = {
                "task_id": sample['task_id'],
                "language": sample['language'],
                "layers": {}
            }
            
            for layer_idx in self.layer_indices:
                if layer_idx in orig_reps and layer_idx in renamed_reps:
                    # Original to renamed similarity
                    orig_renamed_sim = self.analyzer.compute(
                        orig_reps[layer_idx]["pooled"],
                        renamed_reps[layer_idx]["pooled"]
                    )
                    
                    # Store results
                    sample_result["layers"][layer_idx] = orig_renamed_sim
                    
                    # Aggregate results
                    for metric, value in orig_renamed_sim.items():
                        results["by_language"][sample['language']][metric].append(value)
                        results["by_layer"][layer_idx][metric].append(value)
                        results["overall"][metric].append(value)
            
            results["details"].append(sample_result)
        
        # Compute statistics
        results["statistics"] = self._compute_statistics(results)
        
        return results
    
    def _compute_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics for similarity results"""
        stats = {
            "by_language": {},
            "by_layer": {},
            "overall": {}
        }
        
        # Language statistics
        for lang, metrics in results["by_language"].items():
            stats["by_language"][lang] = {}
            for metric, values in metrics.items():
                if values:
                    stats["by_language"][lang][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                        "count": len(values)
                    }
        
        # Layer statistics
        for layer, metrics in results["by_layer"].items():
            stats["by_layer"][layer] = {}
            for metric, values in metrics.items():
                if values:
                    stats["by_layer"][layer][metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values))
                    }
        
        # Overall statistics
        for metric, values in results["overall"].items():
            if values:
                stats["overall"][metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "count": len(values)
                }
        
        return stats

class Visualizer:
    """Create visualizations for similarity analysis"""
    
    def __init__(self, output_dir: str, primary_metric: str = "cosine"):
        self.output_dir = Path(output_dir)
        self.primary_metric = primary_metric
        self.viz_dir = self.output_dir / "visualizations_renaming"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def create_all_visualizations(self, results: Dict[str, Any]):
        """Create all visualizations"""
        self.plot_layer_similarities(results)
        self.plot_language_comparisons(results)
        self.plot_similarity_by_language(results)
        self.plot_overall_distribution(results)
    
    def plot_layer_similarities(self, results: Dict[str, Any]):
        """Plot similarity changes across layers"""
        stats = results["statistics"]["by_layer"]
        
        layers = sorted(stats.keys())
        means = [stats[l][self.primary_metric]["mean"] for l in layers]
        stds = [stats[l][self.primary_metric]["std"] for l in layers]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.errorbar(layers, means, yerr=stds, marker='o', linewidth=2,
                   markersize=8, capsize=5, capthick=2)
        
        ax.set_xlabel("Layer Index", fontsize=12)
        ax.set_ylabel(f"{self.primary_metric.capitalize()} Similarity", fontsize=12)
        ax.set_title("Code Representation Similarity After Variable Renaming Across Layers",
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at y=1 for reference
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5,
                  label='Perfect Similarity')
        
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"layer_similarities_{self.primary_metric}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_language_comparisons(self, results: Dict[str, Any]):
        """Plot similarity comparisons across languages"""
        stats = results["statistics"]["by_language"]
        
        if not stats:
            return
        
        languages = list(stats.keys())
        means = [stats[lang][self.primary_metric]["mean"] for lang in languages]
        stds = [stats[lang][self.primary_metric]["std"] for lang in languages]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.bar(languages, means, yerr=stds, capsize=5,
                     color=sns.color_palette("husl", len(languages)))
        
        ax.set_xlabel("Programming Language", fontsize=12)
        ax.set_ylabel(f"Mean {self.primary_metric.capitalize()} Similarity", fontsize=12)
        ax.set_title("Variable Renaming Impact on Code Representations by Language",
                    fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"all_languages_renaming_similarity_comparison_{self.primary_metric}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_similarity_by_language(self, results: Dict[str, Any]):
        """Plot detailed similarity for each language"""
        for lang, metrics in results["by_language"].items():
            if self.primary_metric not in metrics:
                continue
            
            values = metrics[self.primary_metric]
            if not values:
                continue
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Histogram
            ax1.hist(values, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(np.mean(values), color='red', linestyle='--',
                       label=f'Mean: {np.mean(values):.3f}')
            ax1.axvline(np.median(values), color='green', linestyle='--',
                       label=f'Median: {np.median(values):.3f}')
            ax1.set_xlabel(f"{self.primary_metric.capitalize()} Similarity")
            ax1.set_ylabel("Frequency")
            ax1.set_title(f"Distribution of {self.primary_metric.capitalize()} Similarity for {lang.upper()}")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot by layer
            layer_data = defaultdict(list)
            for detail in results["details"]:
                if detail["language"] == lang:
                    for layer, sim in detail["layers"].items():
                        if self.primary_metric in sim:
                            layer_data[layer].append(sim[self.primary_metric])
            
            if layer_data:
                layers = sorted(layer_data.keys())
                data = [layer_data[l] for l in layers]
                ax2.boxplot(data, labels=layers)
                ax2.set_xlabel("Layer Index")
                ax2.set_ylabel(f"{self.primary_metric.capitalize()} Similarity")
                ax2.set_title(f"Layer-wise Similarity Distribution for {lang.upper()}")
                ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f"Variable Renaming Impact Analysis - {lang.upper()}",
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.viz_dir / f"{lang}_renaming_similarity_by_layer_{self.primary_metric}.png",
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_overall_distribution(self, results: Dict[str, Any]):
        """Plot overall similarity distribution"""
        if self.primary_metric not in results["overall"]:
            return
        
        values = results["overall"][self.primary_metric]
        if not values:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(values, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {np.mean(values):.3f}')
        ax.axvline(np.median(values), color='green', linestyle='--', linewidth=2,
                  label=f'Median: {np.median(values):.3f}')
        
        ax.set_xlabel(f"{self.primary_metric.capitalize()} Similarity", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Overall Distribution of Code Similarity After Variable Renaming",
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f"N = {len(values)}\nStd = {np.std(values):.3f}\nMin = {np.min(values):.3f}\nMax = {np.max(values):.3f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f"overall_similarity_distribution_{self.primary_metric}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function"""
    args = parse_args()
    
    # Set up logging
    rq1_setup_logging(args.log_level)
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        device=args.device,
        device_map=args.device_map
    )
    
    # Get number of layers
    num_layers = ModelHelper.get_num_layers(model)
    logger.info(f"Model has {num_layers} layers")
    
    # Load paired data
    logger.info("Loading paired data")
    paired_data = DataLoader.load_paired_data(
        args.original_dataset_dir,
        args.renamed_dataset_dir,
        args.target_languages,
        args.max_samples_per_lang
    )
    
    if not paired_data:
        logger.error("No paired data found")
        return
    
    # Initialize components
    extractor = RepresentationExtractor(model, tokenizer, args.device)
    analyzer = SimilarityAnalyzer(args.similarity_metrics)
    computer = SimilarityComputer(extractor, analyzer, num_layers)
    
    # Compute similarities
    logger.info("Computing similarities")
    results = computer.compute_similarities(paired_data)
    
    # Save results
    results_file = output_dir / "renaming_similarities.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")
    
    # Create visualizations
    if args.enable_all_visualizations:
        logger.info("Creating visualizations")
        visualizer = Visualizer(args.output_dir, args.primary_metric)
        visualizer.create_all_visualizations(results)
        logger.info(f"Visualizations saved to {visualizer.viz_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("Variable Renaming Similarity Analysis Summary")
    print("="*50)
    
    overall_stats = results["statistics"]["overall"]
    if args.primary_metric in overall_stats:
        stats = overall_stats[args.primary_metric]
        print(f"\nOverall {args.primary_metric.capitalize()} Similarity:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std:  {stats['std']:.4f}")
        print(f"  Min:  {stats['min']:.4f}")
        print(f"  Max:  {stats['max']:.4f}")
        print(f"  Samples: {stats['count']}")
    
    print("\nBy Language:")
    for lang, lang_stats in results["statistics"]["by_language"].items():
        if args.primary_metric in lang_stats:
            stats = lang_stats[args.primary_metric]
            print(f"  {lang}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, n={stats['count']}")
    
    print("\nResults saved to:", args.output_dir)

if __name__ == "__main__":
    main()