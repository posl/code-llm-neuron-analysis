#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
from pathlib import Path

from utils import (
    setup_logging, set_random_seed, load_humaneval_x_data, 
    create_output_directory, print_experiment_summary, cleanup_gpu_memory,
    save_json_file, get_device_info, get_memory_usage
)
from dataset_builder import ASTProbeDatasetBuilder
from representation_extractor import LayerRepresentationExtractor
from probe_models import ASTNodeProbe
from trainer import ProbeTrainer
from analysis import LayerAnalyzer
from visualization import ProbeVisualizer

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(str(ROOT_DIR))
from src.RQ1.model_loader import load_model_and_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="AST+Probe Analysis: Analyzing LLM's predictive ability for AST node types and programming languages at different layers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Model path")
    parser.add_argument("--device", type=str, default="auto",
                       help="Compute device (auto, cuda, cpu)")
    parser.add_argument("--device_map", type=str, default=None,
                       help="Device mapping configuration")
    
    parser.add_argument("--data_dir", type=str,
                       default=str(ROOT_DIR / "humaneval-x" / "data"),
                       help="HumanEval-X data directory")
    parser.add_argument("--target_languages", type=str, nargs="+",
                       default=["python", "java", "cpp", "go", "js"],
                       help="Target programming languages")
    parser.add_argument("--max_samples_per_language", type=int, default=None,
                       help="Maximum samples per language")
    
    parser.add_argument("--target_layers", type=str, default="auto",
                       help="Layer indices to analyze. Can be 'auto' (detect all layers), 'all' (same as auto), or specify layer indices like '0,1,2,3' or '0-10'")
    parser.add_argument("--max_sequence_length", type=int, default=768,
                       help="Maximum sequence length")
    
    parser.add_argument("--probe_types", type=str, nargs="+",
                       default=["ast"],
                       choices=["ast"],
                       help="Probe types to train (only AST prediction supported)") 
    parser.add_argument("--probe_hidden_dim", type=int, default=1024,
                       help="Probe hidden layer dimension")
    parser.add_argument("--use_simple_probe", action="store_true",
                       help="Use simple linear probe")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--early_stopping_patience", type=int, default=15,
                       help="Early stopping patience")
    parser.add_argument("--cross_validation", action="store_true",
                       help="Use cross validation")
    parser.add_argument("--k_folds", type=int, default=5,
                       help="Number of cross validation folds")
    
    parser.add_argument("--output_dir", type=str,
                       default=str(ROOT_DIR / "results" / "ast_probe_analysis"),
                       help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="ast_probe_exp",
                       help="Experiment name")
    parser.add_argument("--save_representations", action="store_true",
                       help="Save extracted representations")
    parser.add_argument("--save_models", action="store_true",
                       help="Save trained models")
    
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--disable_visualization", action="store_true",
                       help="Disable visualization generation")
    
    return parser.parse_args()

def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = device_arg
    return device

def get_model_num_layers(model) -> int:
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return len(model.transformer.h)
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return len(model.encoder.layer)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            return len(model.transformer.layers)
        else:
            return 28
    except Exception as e:
        return 28

def parse_target_layers(target_layers_str: str, num_layers: int) -> list[int]:
    if target_layers_str.lower() in ["auto", "all"]:
        return list(range(num_layers))
    elif "," in target_layers_str:
        try:
            layers = [int(x.strip()) for x in target_layers_str.split(",")]
            return [layer for layer in layers if 0 <= layer < num_layers]
        except ValueError as e:
            return list(range(num_layers))
    elif "-" in target_layers_str:
        try:
            parts = target_layers_str.split("-")
            if len(parts) == 2:
                start, end = int(parts[0]), int(parts[1])
                return list(range(max(0, start), min(end + 1, num_layers)))
            else:
                return list(range(num_layers))
        except ValueError as e:
            return list(range(num_layers))
    else:
        try:
            layer = int(target_layers_str)
            if 0 <= layer < num_layers:
                return [layer]
            else:
                return list(range(num_layers))
        except ValueError:
            return list(range(num_layers))

def main():
    args = parse_args()

    device = resolve_device(args.device)
    args.device = device

    output_dir = create_output_directory(args.output_dir, args.experiment_name)

    log_file = os.path.join(output_dir, "experiment.log")
    setup_logging(args.log_level, log_file)

    set_random_seed(args.seed)

    device_info = get_device_info()
    memory_info = get_memory_usage()

    try:
        model, tokenizer = load_model_and_tokenizer(
            model_path=args.model_path,
            device=args.device,
            device_map=args.device_map
        )
        
        num_layers = get_model_num_layers(model)
        target_layers = parse_target_layers(args.target_layers, num_layers)
        args.target_layers = target_layers
        
        humaneval_data = load_humaneval_x_data(
            args.data_dir, 
            args.target_languages, 
            args.max_samples_per_language
        )
        
        if not humaneval_data:
            return
        
        dataset_builder = ASTProbeDatasetBuilder(
            languages=args.target_languages,
            max_samples_per_language=args.max_samples_per_language
        )
        
        dataset = dataset_builder.build_dataset(humaneval_data)
        
        dataset_path = os.path.join(output_dir, "ast_probe_dataset.json")
        dataset_builder.save_dataset(dataset, dataset_path)
        
        extractor = LayerRepresentationExtractor(
            model=model,
            tokenizer=tokenizer,
            target_layers=args.target_layers,
            device=device
        )
        
        all_samples = dataset['train'] + dataset['validation'] + dataset['test']
        
        representations = extractor.batch_extract_representations(
            samples=all_samples,
            batch_size=args.batch_size,
            max_length=args.max_sequence_length
        )
        
        if args.save_representations:
            repr_path = os.path.join(output_dir, "representations.npz")
            extractor.save_representations(representations, repr_path)
        
        trainer = ProbeTrainer(
            device=device,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            early_stopping_patience=args.early_stopping_patience
        )

        all_results = {}

        for layer_idx in args.target_layers:
            layer_sample_count = sum(1 for repr_data in representations['representations']
                                    if layer_idx in repr_data.get('representations', {}))

            if layer_sample_count == 0:
                continue

            layer_results = {}

            train_loader, val_loader, test_loader = trainer.prepare_data(
                representations, dataset, layer_idx
            )

            if len(train_loader.dataset) == 0:
                continue
            
            sample_batch = next(iter(train_loader))
            input_dim = sample_batch[0].size(1)
            
            for probe_type in args.probe_types:
                
                if args.use_simple_probe:
                    probe_model = ASTNodeProbe(input_dim, len(dataset['ast_types']),
                                             dtype=torch.float32)
                else:
                    probe_model = ASTNodeProbe(input_dim, len(dataset['ast_types']),
                                             args.probe_hidden_dim, dtype=torch.float32)
                
                if args.cross_validation:
                    cv_results = trainer.cross_validate(
                        probe_model.__class__,
                        representations,
                        dataset,
                        layer_idx,
                        probe_type,
                        args.k_folds
                    )
                    layer_results[probe_type] = {'cv_result': cv_results}
                else:
                    train_result = trainer.train_probe(probe_model, train_loader, val_loader, probe_type)
                    eval_result = trainer.evaluate_probe(probe_model, test_loader, dataset, probe_type)
                    
                    layer_results[probe_type] = {
                        'train_result': train_result,
                        'eval_result': eval_result
                    }
                    
                    if args.save_models:
                        model_path = os.path.join(output_dir, f"probe_{probe_type}_layer_{layer_idx}.pt")
                        trainer.save_model(probe_model, model_path)
                
                cleanup_gpu_memory()
            
            all_results[layer_idx] = layer_results
        
        analyzer = LayerAnalyzer(all_results)
        analysis_results = analyzer.generate_summary_report()
        
        analysis_path = os.path.join(output_dir, "analysis_results.json")
        analyzer.save_analysis_results(analysis_results, analysis_path)
        
        if not args.disable_visualization:
            visualizer = ProbeVisualizer(all_results, os.path.join(output_dir, "visualizations"))
            plot_paths = visualizer.generate_comprehensive_dashboard(analysis_results, dataset)
        
        experiment_config = {
            'model_path': args.model_path,
            'target_languages': args.target_languages,
            'target_layers': args.target_layers,
            'max_samples_per_language': args.max_samples_per_language,
            'probe_types': args.probe_types,
            'training_config': {
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'early_stopping_patience': args.early_stopping_patience
            },
            'seed': args.seed
        }
        
        config_path = os.path.join(output_dir, "experiment_config.json")
        save_json_file(experiment_config, config_path)
        
        print_experiment_summary(experiment_config, analysis_results)
        
    except Exception as e:
        raise
    finally:
        cleanup_gpu_memory()

if __name__ == "__main__":
    main()