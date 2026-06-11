#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
def setup_paths():
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    if script_dir.name == "Rename":
        root_dir = script_dir.parent.parent.parent
    elif script_dir.name == "src":
        root_dir = script_dir.parent
    else:
        root_dir = script_dir
    

    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

    humaneval_x_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "humaneval-x"))
    if str(humaneval_x_dir) not in sys.path:
        sys.path.append(str(humaneval_x_dir))
    
    return {
        'ROOT_DIR': root_dir,
        'SCRIPT_DIR': script_dir,
        'HUMANEVAL_X_DIR': humaneval_x_dir
    }
def get_file_language(language: str) -> str:
    language_lower = language.lower()
    if language_lower == "js":
        return "javascript"
    return language_lower

def get_output_language(language: str) -> str:
    if language.lower() == "js":
        return "js"
    return language.lower()

def read_jsonl_file(file_path: str, compressed: bool = False) -> List[Dict[str, Any]]:
    data = []
    
    try:
        if compressed or file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
    except Exception as e:
        logging.error(f"Error reading JSONL file {file_path}: {e}")
    
    return data

def write_jsonl_file(data: List[Dict[str, Any]], file_path: str):

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_humaneval_dataset(data_dir: str, language: str) -> Dict[str, Dict[str, Any]]:

    file_language = get_file_language(language)
    dataset_path = os.path.join(data_dir, f"humaneval_{file_language}.jsonl.gz")
    
    if not os.path.exists(dataset_path):
        dataset_path_no_gz = os.path.join(data_dir, f"humaneval_{file_language}.jsonl")
        if os.path.exists(dataset_path_no_gz):
            dataset_path = dataset_path_no_gz
        else:
            return {}
    
    problems = {}
    data = read_jsonl_file(dataset_path)
    for sample in data:
        task_id = sample.get('task_id')
        if task_id:
            problems[task_id] = sample
    
    return problems

def assemble_complete_code(
    declaration: str,
    solution: str,
    language: str,
    imports: str = ""
) -> str:

    language_lower = language.lower()

    if language_lower == 'go' and imports and imports.strip():
        if not imports.endswith('\n'):
            imports += '\n'
        if declaration and not declaration.startswith('\n'):
            imports += '\n'
    else:
        imports = ""

    if not declaration:
        return imports + solution

    if language_lower in ['python', 'rust']:
        complete_code = imports + declaration + solution
    elif language_lower == 'go':
        complete_code = imports + declaration + solution
    elif language_lower in ['java', 'cpp']:
        if declaration.strip().endswith('{'):
            complete_code = declaration + solution
        else:
            complete_code = declaration + " {\n" + solution + "\n}"
    elif language_lower == 'js':
        if '=>' in declaration or declaration.strip().endswith('{'):
            complete_code = declaration + solution
        else:
            complete_code = declaration + " {\n" + solution + "\n}"
    else:
        complete_code = declaration + "\n" + solution
    
    return complete_code

def process_task_id(task_id: str) -> str:

    return task_id.split('/')[-1] if '/' in task_id else task_id

def get_universal_task_id(task_id: str) -> str:

    if '/' in task_id:
        return task_id.split('/')[1]
    return task_id

def validate_code_syntax(code: str, language: str) -> Tuple[bool, str]:

    if not code or code.isspace():
        return False, "Code is empty"
    
    bracket_pairs = {'{': '}', '[': ']', '(': ')'}
    stack = []
    
    for char in code:
        if char in bracket_pairs:
            stack.append(char)
        elif char in bracket_pairs.values():
            if not stack:
                return False, f"Unmatched closing bracket: {char}"
            if bracket_pairs[stack.pop()] != char:
                return False, f"Mismatched brackets"
    
    if stack:
        return False, f"Unclosed brackets: {stack}"
    
    return True, ""

def calculate_success_rate(samples: List[Dict[str, Any]]) -> float:
    if not samples:
        return 0.0
    success_count = sum(1 for s in samples if s.get('success', False))
    return round(success_count / len(samples) * 100, 2)

def generate_statistics(processed_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_language = defaultdict(list)
    for sample in processed_dataset:
        lang = sample.get('language')
        if lang:
            by_language[lang].append(sample)
    
    stats = {
        "total_samples": len(processed_dataset),
        "successful": sum(1 for s in processed_dataset if s.get('success', False)),
        "failed": sum(1 for s in processed_dataset if not s.get('success', False)),
        "by_language": {}
    }
    
    for lang, samples in by_language.items():
        stats["by_language"][lang] = {
            "total": len(samples),
            "success": sum(1 for s in samples if s.get('success', False)),
            "rate": calculate_success_rate(samples)
        }
    
    stats["overall_rate"] = calculate_success_rate(processed_dataset)
    
    return stats

def create_output_path(output_dir: str, language: str, filename: str) -> str:
    """Create organized output path for language-specific files"""
    lang_dir = os.path.join(output_dir, get_output_language(language))
    os.makedirs(lang_dir, exist_ok=True)
    return os.path.join(lang_dir, filename)