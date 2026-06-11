#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA configuration module - imports from main config
"""

from pathlib import Path
from .config import LORA_CONFIG, LORA_TRAINING_CONFIG, RESULTS_PATH, MODEL_PATHS

# Re-export for backward compatibility
__all__ = ['LORA_CONFIG', 'LORA_TRAINING_CONFIG', 'get_lora_model_paths', 
           'create_lora_directories', 'get_lora_model_save_path', 
           'get_lora_adapters_path', 'get_lora_log_path']

def get_lora_model_paths(model_path: str, results_path: Path = None):
    """Get LoRA model paths"""
    results_path = results_path or RESULTS_PATH
    return {
        'base_model': model_path,
        'lora_baseline': MODEL_PATHS['lora_baseline'],
        'lora_adapters': MODEL_PATHS['lora_adapters'],
    }

def create_lora_directories(results_path: Path = None):
    """Create LoRA-specific directories"""
    results_path = results_path or RESULTS_PATH
    directories = [
        MODEL_PATHS['lora_baseline'],
        MODEL_PATHS['lora_adapters'],
        results_path / "evaluation" / "lora_baseline",
        results_path / "logs" / "lora_baseline",
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_lora_model_save_path(results_path: Path = None):
    """Get LoRA model save path"""
    return MODEL_PATHS['lora_baseline']

def get_lora_adapters_path(results_path: Path = None):
    """Get LoRA adapters path"""
    return MODEL_PATHS['lora_adapters']

def get_lora_log_path(results_path: Path, experiment_name: str):
    """Get LoRA log path"""
    results_path = results_path or RESULTS_PATH
    return results_path / "logs" / "lora_baseline" / f"{experiment_name}.log"