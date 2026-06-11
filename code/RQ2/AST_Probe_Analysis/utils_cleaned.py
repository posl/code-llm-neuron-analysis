#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import gzip

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    import logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

def set_random_seed(seed: int = 42):
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_humaneval_x_data(data_dir: str, languages: List[str], max_samples: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    
    language_file_mapping = {
        'python': 'humaneval_python.jsonl.gz',
        'java': 'humaneval_java.jsonl.gz',
        'cpp': 'humaneval_cpp.jsonl.gz',
        'c++': 'humaneval_cpp.jsonl.gz',
        'go': 'humaneval_go.jsonl.gz',
        'js': 'humaneval_javascript.jsonl.gz',
        'javascript': 'humaneval_javascript.jsonl.gz',
        'rust': 'humaneval_rust.jsonl.gz'
    }
    
    all_data = {}
    
    for language in languages:
        language_lower = language.lower()
        if language_lower not in language_file_mapping:
            continue
        
        file_name = language_file_mapping[language_lower]
        file_path = os.path.join(data_dir, file_name)
        
        if not os.path.exists(file_path):
            file_path_no_gz = file_path.replace('.gz', '')
            if os.path.exists(file_path_no_gz):
                file_path = file_path_no_gz
            else:
                continue
        
        try:
            language_data = load_jsonl_file(file_path)
            
            if max_samples and len(language_data) > max_samples:
                import random
                task_ids = list(language_data.keys())
                selected_ids = random.sample(task_ids, max_samples)
                language_data = {task_id: language_data[task_id] for task_id in selected_ids}
            
            all_data[language_lower] = language_data
            
        except Exception as e:
            continue
    
    return all_data

def load_jsonl_file(file_path: str) -> Dict[str, Any]:
    data = {}
    
    if file_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    try:
        with open_func(file_path, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    task_id = sample.get('task_id')
                    if task_id:
                        data[task_id] = sample
                except json.JSONDecodeError as e:
                    continue
    except Exception as e:
        raise
    
    return data

def save_json_file(data: Any, file_path: str, indent: int = 2):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)

def load_json_file(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def get_device_info() -> Dict[str, Any]:
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None,
        'device_names': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_info['device_names'].append(device_name)
    
    return device_info

def format_memory_usage(bytes_used: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_used < 1024.0:
            return f"{bytes_used:.2f} {unit}"
        bytes_used /= 1024.0
    return f"{bytes_used:.2f} PB"

def get_memory_usage() -> Dict[str, str]:
    import psutil
    
    memory = psutil.virtual_memory()
    memory_info = {
        'total_memory': format_memory_usage(memory.total),
        'available_memory': format_memory_usage(memory.available),
        'used_memory': format_memory_usage(memory.used),
        'memory_percent': f"{memory.percent:.1f}%"
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            memory_info[f'gpu_{i}_allocated'] = format_memory_usage(allocated)
            memory_info[f'gpu_{i}_reserved'] = format_memory_usage(reserved)
    
    return memory_info

def create_output_directory(base_dir: str, experiment_name: str) -> str:
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def validate_config(config: Dict[str, Any], required_keys: List[str]) -> bool:
    missing_keys = []
    for key in required_keys:
        if key not in config:
            missing_keys.append(key)
    
    if missing_keys:
        return False
    
    return True

def print_experiment_summary(config: Dict[str, Any], results: Dict[str, Any]):
    print("\n" + "="*80)
    print("AST PROBE ANALYSIS EXPERIMENT SUMMARY")
    print("="*80)
    
    print("\nEXPERIMENT CONFIGURATION:")
    print(f"  Model: {config.get('model_path', 'Unknown')}")
    print(f"  Languages: {config.get('target_languages', [])}")
    print(f"  Target Layers: {config.get('target_layers', [])}")
    print(f"  Max Samples: {config.get('max_samples_per_language', 'All')}")
    
    if 'optimal_layers' in results:
        print("\nOPTIMAL LAYERS:")
        optimal = results['optimal_layers']
        
        if 'ast' in optimal:
            print(f"  AST Node Type Prediction:")
            print(f"    Best Layer: {optimal['ast']['best_layer_accuracy']}")
            print(f"    Best Accuracy: {optimal['ast']['best_accuracy']:.4f}")
        
        if 'language' in optimal:
            print(f"  Programming Language Prediction:")
            print(f"    Best Layer: {optimal['language']['best_layer_accuracy']}")
            print(f"    Best Accuracy: {optimal['language']['best_accuracy']:.4f}")
        
        if 'layer_difference' in optimal:
            print(f"  Layer Difference: {optimal['layer_difference']['layer_gap']}")
    
    if 'layer_specialization' in results:
        spec = results['layer_specialization']
        if 'most_specialized_layer' in spec:
            print(f"\nMOST SPECIALIZED LAYER: {spec['most_specialized_layer']}")
            print(f"  Specialization Index: {spec['max_specialization_index']:.4f}")
    
    print("\n" + "="*80)

def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj