#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
MODEL_PATH = "****"
DATA_PATH = Path("****")
RESULTS_PATH = Path(__file__).parent / "results"

# Language configuration
HIGH_RESOURCE_LANGUAGES = ['python', 'go', 'java', 'javascript', 'php']
LOW_RESOURCE_LANGUAGE = 'ruby'
ALL_LANGUAGES = HIGH_RESOURCE_LANGUAGES + [LOW_RESOURCE_LANGUAGE]
SUPPORTED_LANGUAGES = ALL_LANGUAGES  # For backward compatibility

# Concept layers
DEFAULT_CONCEPT_LAYERS = [8,9,10,11,12,13,14,15]

# Training configuration
TRAINING_CONFIG = {
    'batch_size': 2,
    'gradient_accumulation_steps': 16,
    'learning_rate': 5e-5,
    'min_learning_rate': 1e-6,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'max_code_length': 512,
    'max_summary_length': 128,
    'phase1_epochs': 2,
    'phase2_epochs': 3,
    'baseline_epochs': 3,
    'early_stopping_patience': 2,
    'early_stopping_threshold': 0.001,
    'adam_beta1': 0.9,
    'adam_beta2': 0.95,
    'adam_epsilon': 1e-8,
    'lr_scheduler_type': 'cosine',
    'warmup_ratio': 0.1,
    'seed': 42,
    'bf16': True,
    'dataloader_num_workers': 16,
    'pin_memory': True,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'metrics': ['bleu', 'rouge', 'meteor'],
    'rouge_types': ['rouge1', 'rouge2', 'rougeL'],
    'generation_config': {
        'max_new_tokens': 128,
        'min_new_tokens': 3,
        'do_sample': True,
        'temperature': 0.1,
        'top_p': 0.95,
        'repetition_penalty': 1.1,
    },
    'eval_batch_size': 4,
}

# Model paths
MODEL_PATHS = {
    'base1_model': RESULTS_PATH / "models" / "base1_model",
    'final_model': RESULTS_PATH / "models" / "final_model",
    'baseline_model': RESULTS_PATH / "models" / "baseline_model",
    'lora_baseline': RESULTS_PATH / "models" / "lora_baseline",
    'lora_adapters': RESULTS_PATH / "models" / "lora_baseline" / "adapters",
}

# Data sampling configuration
DATA_SAMPLING_CONFIG = {
    'phase1_max_samples_per_language': 50000,
    'validation_samples': 7000,
}

# GPU configuration
GPU_CONFIG = {
    'device': 'auto' if torch.cuda.device_count() > 1 else 'cuda',
    'mixed_precision': True,
    'gradient_checkpointing': False,
    'dataloader_pin_memory': True,
    'empty_cache_steps': 500,
}

# LoRA configuration
LORA_CONFIG = {
    'r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'bias': "none",
    'task_type': "CAUSAL_LM",
    'target_modules': [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    'inference_mode': False,
}

# LoRA training configuration
LORA_TRAINING_CONFIG = {
    'learning_rate': 1e-4,
    'min_learning_rate': 1e-5,
    'batch_size': 4,
    'eval_batch_size': 2,
    'gradient_accumulation_steps': 8,
    'phase1_epochs': 3,
    'phase2_epochs': 4,
    'baseline_epochs': 4,
    'early_stopping_patience': 3,
    'early_stopping_threshold': 0.001,
    'weight_decay': 0.005,
    'lr_scheduler_type': 'cosine',
    'warmup_ratio': 0.1,
    'seed': 42,
    'bf16': True,
}

def create_directories():
    """Create all necessary directories"""
    directories = [
        RESULTS_PATH,
        RESULTS_PATH / "models",
        RESULTS_PATH / "logs",
        RESULTS_PATH / "evaluation",
    ]
    
    for model_key in MODEL_PATHS.values():
        if isinstance(model_key, Path):
            directories.append(model_key)
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_data_path(data_path, language, split):
    """Get data file path"""
    return Path(data_path) / language / split / f"{language}_{split}.json"

def get_model_save_path(model_type):
    """Get model save path"""
    return MODEL_PATHS.get(model_type)

if __name__ == "__main__":
    create_directories()
    print("Configuration initialized, directories created")
    print(f"Results path: {RESULTS_PATH}")
    print(f"Default concept layers: {DEFAULT_CONCEPT_LAYERS}")
    print(f"High resource languages: {HIGH_RESOURCE_LANGUAGES}")
    print(f"Low resource language: {LOW_RESOURCE_LANGUAGE}")