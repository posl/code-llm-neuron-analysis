#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from data_processor import setup_data_processor
from model_wrapper import create_model, load_model_for_phase2
from trainer import setup_trainer

def load_data(
    data_processor,
    data_path: str,
    data_type: str,
    languages: List[str] = None,
    max_samples: Optional[int] = None,
    split: str = 'train'
) -> List[Dict[str, Any]]:
    if data_type == 'high_resource':
        data = data_processor.create_high_resource_dataset(
            data_path,
            split=split,
            max_samples_per_language=max_samples
        )
    elif data_type == 'low_resource':
        data = data_processor.create_low_resource_dataset(
            data_path,
            split=split,
            max_samples=max_samples
        )
    elif data_type == 'single_language':
        if not languages or len(languages) != 1:
            raise ValueError("single_language mode requires one language")
        data = data_processor.load_language_data(
            data_path,
            language=languages[0],
            split=split,
            max_samples=max_samples
        )
    else:
        raise ValueError(f"Unsupported data type: {data_type}")
    
    return data

def fine_tune_model(
    model_path: str,
    data_path: str,
    data_type: str,
    concept_layers: List[int],
    output_path: Path,
    num_epochs: int,
    languages: List[str] = None,
    max_samples: Optional[int] = None,
    experiment_name: str = "fine_tune",
    base_model_path: Optional[str] = None,
    config_override: Dict[str, Any] = None
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    torch.manual_seed(42)
    
    data_processor = setup_data_processor(model_path)
    
    train_data = load_data(
        data_processor=data_processor,
        data_path=data_path,
        data_type=data_type,
        languages=languages,
        max_samples=max_samples,
        split='train'
    )
    
    val_data = load_data(
        data_processor=data_processor,
        data_path=data_path,
        data_type=data_type,
        languages=languages,
        max_samples=7000,
        split='valid'
    )
    
    train_stats = data_processor.get_data_statistics(train_data)
    val_stats = data_processor.get_data_statistics(val_data)
    
    train_dataset = data_processor.create_dataset(train_data)
    val_dataset = data_processor.create_dataset(val_data)
    
    train_dataloader = data_processor.create_dataloader(train_dataset, shuffle=True)
    val_dataloader = data_processor.create_dataloader(val_dataset, shuffle=False)
    
    if base_model_path:
        model = load_model_for_phase2(
            base1_model_path=Path(base_model_path),
            concept_layers=concept_layers
        )
    else:
        model = create_model(
            model_path=model_path,
            concept_layers=concept_layers
        )
    
    model_info = model.get_model_info()
    
    training_config = {
        'batch_size': 2,
        'learning_rate': 5e-5,
        'num_epochs': num_epochs,
        'gradient_accumulation_steps': 16,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'max_code_length': 512,
        'max_summary_length': 128,
        'early_stopping_patience': 2,
        'early_stopping_threshold': 0.001,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'seed': 42,
        'bf16': True,
    }
    
    if config_override:
        training_config.update(config_override)
    
    trainer = setup_trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=training_config
    )
    
    training_history = trainer.train(
        num_epochs=num_epochs,
        save_path=output_path
    )
    
    experiment_config = {
        'experiment_name': experiment_name,
        'model_path': model_path,
        'base_model_path': base_model_path,
        'data_type': data_type,
        'languages': languages,
        'concept_layers': concept_layers,
        'num_epochs': num_epochs,
        'max_samples': max_samples,
        'training_config': training_config,
        'train_stats': train_stats,
        'val_stats': val_stats,
        'model_info': model_info,
    }
    
    config_path = output_path / 'experiment_config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False, default=str)
    
    return training_history

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning script")
    
    parser.add_argument('--model_path', type=str, required=True, help='Base model path')
    parser.add_argument('--data_path', type=str, required=True, help='Data path')
    parser.add_argument('--data_type', type=str, required=True, 
                       choices=['high_resource', 'low_resource', 'single_language'],
                       help='Data type')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')
    parser.add_argument('--num_epochs', type=int, required=True, help='Training epochs')
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    
    parser.add_argument('--concept_layers', type=int, nargs='+', default=[13,14,15,16,17,18],
                       help='Concept layer list')
    parser.add_argument('--languages', type=str, nargs='+', help='Language list')
    parser.add_argument('--max_samples', type=int, help='Maximum samples')
    parser.add_argument('--base_model_path', type=str, help='Base model path for phase 2')
    
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    
    args = parser.parse_args()
    
    config_override = {}
    if args.learning_rate:
        config_override['learning_rate'] = args.learning_rate
    if args.batch_size:
        config_override['batch_size'] = args.batch_size
    
    fine_tune_model(
        model_path=args.model_path,
        data_path=args.data_path,
        data_type=args.data_type,
        concept_layers=args.concept_layers,
        output_path=Path(args.output_path),
        num_epochs=args.num_epochs,
        languages=args.languages,
        max_samples=args.max_samples,
        experiment_name=args.experiment_name,
        base_model_path=args.base_model_path,
        config_override=config_override if config_override else None
    )

if __name__ == "__main__":
    main()
