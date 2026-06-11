#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create variable name replacement dataset - Optimized version
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

from utils import (
    setup_paths,
    load_humaneval_dataset,
    assemble_complete_code,
    process_task_id,
    write_jsonl_file,
    create_output_path,
    read_jsonl_file
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize paths
paths = setup_paths()
ROOT_DIR = paths['ROOT_DIR']
HUMANEVAL_X_DIR = paths['HUMANEVAL_X_DIR']

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Create variable name replacement dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--renamed_dir", type=str, 
                      default=str(ROOT_DIR / "data" / "renamed_variables"),
                      help="Variable name replacement result directory")
    parser.add_argument("--original_data_dir", type=str, 
                      default=str(HUMANEVAL_X_DIR / "data"),
                      help="Original humaneval-x data directory")
    parser.add_argument("--output_dir", type=str, 
                      default=str(ROOT_DIR / "data" / "renamed_dataset"),
                      help="Output directory for renamed code")
    parser.add_argument("--original_output_dir", type=str, 
                      default=str(ROOT_DIR / "data" / "original_dataset"),
                      help="Output directory for original code")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Log level")
    parser.add_argument("--target_languages", type=str, nargs="+",
                      default=["python", "cpp", "java", "go", "js"],
                      help="Target programming languages")
    
    return parser.parse_args()

def load_renamed_code(renamed_dir: str, language: str) -> Dict[str, str]:
    """Load renamed code from result files"""
    renamed_code = {}
    jsonl_path = os.path.join(renamed_dir, language, f"renamed_samples_{language}.jsonl")
    
    if not os.path.exists(jsonl_path):
        logger.error(f"Renamed result file not found: {jsonl_path}")
        return {}
    
    data = read_jsonl_file(jsonl_path)
    for sample in data:
        if sample.get('success', False) and sample.get('renamed_code'):
            task_id = sample.get('task_id')
            renamed_code[task_id] = sample['renamed_code']
    
    logger.info(f"Loaded {len(renamed_code)} renamed samples for {language}")
    return renamed_code

def create_renamed_dataset(
    original_data: Dict[str, Dict[str, Any]], 
    renamed_code: Dict[str, str], 
    language: str
) -> List[Dict[str, Any]]:
    """Create dataset with renamed code"""
    new_dataset = []
    import_count = 0
    
    for task_id, orig_sample in original_data.items():
        pure_task_id = process_task_id(task_id)
        
        if pure_task_id in renamed_code:
            new_sample = orig_sample.copy()
            new_sample['generation'] = renamed_code[pure_task_id]
            
            # Handle Go imports
            if language.lower() == 'go':
                imports = orig_sample.get('import', '')
                if imports and imports.strip():
                    if not imports.endswith('\n'):
                        imports += '\n'
                    if not new_sample['generation'].startswith('\n'):
                        imports += '\n'
                    new_sample['generation'] = imports + new_sample['generation']
                    import_count += 1
                    logger.debug(f"Added imports for task {task_id}")
            
            new_sample['pass_at_k'] = 1
            new_dataset.append(new_sample)
        else:
            logger.warning(f"Task {task_id} (ID: {pure_task_id}) has no renamed code")
    
    if language.lower() == 'go':
        logger.info(f"Go language: {len(new_dataset)} samples, {import_count} with imports")
    else:
        logger.info(f"Created {len(new_dataset)} renamed samples for {language}")
    
    return new_dataset

def create_original_dataset(
    original_data: Dict[str, Dict[str, Any]], 
    language: str
) -> List[Dict[str, Any]]:
    """Create dataset with original code assembled"""
    original_dataset = []
    import_count = 0
    
    for task_id, orig_sample in original_data.items():
        new_sample = orig_sample.copy()
        
        declaration = orig_sample.get('declaration', '')
        solution = orig_sample.get('canonical_solution', '')
        imports = orig_sample.get('import', '') if language.lower() == 'go' else ''
        
        # Track Go imports
        if language.lower() == 'go' and imports and imports.strip():
            import_count += 1
            logger.debug(f"Added imports for task {task_id}")
        
        # Assemble complete code
        complete_code = assemble_complete_code(declaration, solution, language, imports)
        
        new_sample['generation'] = complete_code
        new_sample['pass_at_k'] = 1
        original_dataset.append(new_sample)
    
    if language.lower() == 'go':
        logger.info(f"Go language original: {len(original_dataset)} samples, {import_count} with imports")
    else:
        logger.info(f"Created {len(original_dataset)} original samples for {language}")
    
    return original_dataset

def save_dataset(dataset: List[Dict[str, Any]], output_dir: str, language: str):
    """Save dataset to file"""
    output_path = create_output_path(output_dir, language, "samples.jsonl")
    write_jsonl_file(dataset, output_path)
    logger.info(f"Saved {len(dataset)} samples to {output_path}")

def process_language(
    original_data_dir: str, 
    renamed_dir: str, 
    output_dir: str, 
    original_output_dir: str, 
    language: str
):
    """Process data for a single language"""
    logger.info(f"Processing language: {language}")
    
    # Load original data
    original_data = load_humaneval_dataset(original_data_dir, language)
    if not original_data:
        logger.error(f"Failed to load original data for {language}")
        return
    
    # Create and save original dataset
    original_dataset = create_original_dataset(original_data, language)
    save_dataset(original_dataset, original_output_dir, language)
    
    # Load renamed code
    renamed_code = load_renamed_code(renamed_dir, language)
    if not renamed_code:
        logger.error(f"Failed to load renamed code for {language}")
        return
    
    # Create and save renamed dataset
    renamed_dataset = create_renamed_dataset(original_data, renamed_code, language)
    if renamed_dataset:
        save_dataset(renamed_dataset, output_dir, language)

def main():
    """Main function"""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.original_output_dir, exist_ok=True)
    
    # Process each language
    for language in args.target_languages:
        try:
            process_language(
                args.original_data_dir, 
                args.renamed_dir, 
                args.output_dir, 
                args.original_output_dir, 
                language
            )
        except Exception as e:
            logger.error(f"Error processing {language}: {e}")
            continue
    
    logger.info("Dataset creation complete")

if __name__ == "__main__":
    main()