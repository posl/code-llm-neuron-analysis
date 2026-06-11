#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_memory_info():
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        cached = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        
        info.append(f"GPU {i}: {props.name}, Total: {total:.1f}GB, "
                   f"Allocated: {allocated:.1f}GB, Cached: {cached:.1f}GB")
    
    return "\n".join(info)

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def save_json(data: Any, file_path: Path, ensure_ascii: bool = False):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=ensure_ascii, default=str)

def load_json(file_path: Path) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'total_size_mb': total_params * 4 / 1024 / 1024,
        'trainable_size_mb': trainable_params * 4 / 1024 / 1024
    }

def plot_training_curves(history: Dict[str, List], save_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    if 'train_loss' in history:
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    if 'learning_rate' in history:
        axes[0, 1].plot(history['learning_rate'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].grid(True)
    
    if 'epoch_times' in history:
        axes[1, 0].plot(history['epoch_times'])
        axes[1, 0].set_title('Epoch Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (s)')
        axes[1, 0].grid(True)
    
    axes[1, 1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_metrics_comparison_plot(results_dict: Dict[str, Dict], save_path: Path):
    metrics = ['bleu', 'rouge1_f', 'rouge2_f', 'rougeL_f', 'meteor']
    metric_names = ['BLEU', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR']
    
    models = list(results_dict.keys())
    data = []
    
    for model in models:
        if results_dict[model] and 'main_metrics' in results_dict[model]:
            metrics_data = results_dict[model]['main_metrics']
            data.append([
                metrics_data.get('bleu', 0),
                metrics_data.get('rouge1_f', 0),
                metrics_data.get('rouge2_f', 0),
                metrics_data.get('rougeL_f', 0),
                metrics_data.get('meteor', 0)
            ])
    
    if not data:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    x = np.arange(len(metric_names))
    width = 0.25
    
    for i, model in enumerate(models):
        if i < len(data):
            ax1.bar(x + i * width, data[i], width, label=model, alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metric_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if len(data) > 1:
        heatmap_data = np.array(data)
        sns.heatmap(heatmap_data, 
                   xticklabels=metric_names,
                   yticklabels=models,
                   annot=True, 
                   fmt='.4f',
                   cmap='YlOrRd',
                   ax=ax2)
        ax2.set_title('Performance Heatmap')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_experiment_summary(config: Dict[str, Any]):
    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)
    print(f"Concept layers: {config.get('concept_layers', 'N/A')}")
    print(f"High-resource languages: {config.get('high_resource_languages', 'N/A')}")
    print(f"Low-resource language: {config.get('low_resource_language', 'N/A')}")
    print(f"Training config:")
    
    training_config = config.get('training', {})
    for key, value in training_config.items():
        print(f"  {key}: {value}")
    
    print("=" * 60)

def validate_paths(paths: Dict[str, Path]) -> bool:
    missing_paths = []
    
    for name, path in paths.items():
        if not Path(path).exists():
            missing_paths.append(f"{name}: {path}")
    
    if missing_paths:
        print("Missing paths:")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    
    return True

def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class Timer:
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        print(f"Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        print(f"{self.name} completed in: {format_time(elapsed)}")

def check_dependencies():
    required_packages = [
        'torch', 'transformers', 'numpy', 'matplotlib', 'seaborn', 'datasets',
        'tqdm', 'nltk', 'evaluate'
    ]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Warning: Missing dependencies:")
        print(", ".join(missing))
        print("Please run 'pip install -r requirements.txt'")

def create_structured_prompt(item: Dict[str, Any], for_inference: bool = False) -> str:
    code = item.get('code', '')
    language = item.get('language', 'unknown')

    if for_inference:
        prompt = (
            f"Write a brief, single-sentence description of what this {language} function does. "
            f"Just describe the main action in simple terms and end with a period.\n\n"
            f"### Code:\n{code}\n\n"
            f"### Summary:"
        )
    else:
        summary = item.get('summary', '')
        prompt = (
            f"Write a brief, single-sentence description of what this {language} function does. "
            f"Just describe the main action in simple terms and end with a period.\n\n"
            f"### Code:\n{code}\n\n"
            f"### Summary: {summary}"
        )

    return prompt

def get_stopping_criteria(tokenizer, stop_sequences):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class StopOnSequences(StoppingCriteria):
        def __init__(self, tokenizer, stop_sequences):
            self.tokenizer = tokenizer
            default_stops = ['\n\n', '###']
            self.stop_sequences = (stop_sequences or []) + default_stops

        def __call__(self, input_ids, scores, **kwargs):
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

            if '### Summary:' in generated_text:
                summary_part = generated_text.split('### Summary:')[-1]
                if '###' in summary_part:
                    return True

            for stop_seq in self.stop_sequences:
                if stop_seq in generated_text:
                    return True

            doc_indicators = [
                'Parameters:', 'Args:', 'Returns:', 'Example:', 'Note:', 'Usage:', 'Description:',
                ':param', ':return:', 'Raises:', ':type:', ':rtype:',
                'param ', 'return ', 'type ', 'rtype ',
                '\n-', '\n*', '\n1.', '\n2.', '\n•',
                'Parameter', 'Argument', 'Return'
            ]
            for indicator in doc_indicators:
                if indicator in generated_text:
                    return True

            if '### Summary:' in generated_text:
                summary_only = generated_text.split('### Summary:')[-1].strip()
                if len(summary_only) > 120:
                    return True

            return False

    return StoppingCriteriaList([StopOnSequences(tokenizer, stop_sequences)])

def clean_summary_for_evaluation(text: str) -> str:
    stop_words = ['Summary:', 'Documentation:', 'Description:', 'Function Summary:']
    for stop in stop_words:
        if stop in text:
            text = text.split(stop)[0]

    sentences = text.split('.')
    if sentences and len(sentences) > 1:
        return sentences[0].strip() + '.'
    return text.strip()

if __name__ == "__main__":
    print("Testing utility functions...")

    test_item = {
        'code': 'def add(a, b):\n    return a + b',
        'language': 'python',
        'summary': 'Add two numbers and return the result'
    }

    print("\nTraining prompt:")
    print(create_structured_prompt(test_item, for_inference=False))
    print("\n" + "="*50 + "\n")
    print("Inference prompt:")
    print(create_structured_prompt(test_item, for_inference=True))

    print(f"\nGPU Info:")
    print(get_gpu_memory_info())

    print(f"\nTime formatting test:")
    print(f"30s: {format_time(30)}")
    print(f"90s: {format_time(90)}")
    print(f"3700s: {format_time(3700)}")

    check_dependencies()

    print("Utility functions test completed")
