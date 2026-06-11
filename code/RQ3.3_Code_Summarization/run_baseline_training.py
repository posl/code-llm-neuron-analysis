#!/usr/bin/env python
# -*- coding: utf-8 -*-

from base_runner import BaseRunner
from pathlib import Path

class BaselineRunner(BaseRunner):
    """Runner for baseline training experiment"""
    
    def get_experiment_name(self) -> str:
        return 'baseline_ruby_training'
    
    def get_default_output_path(self) -> str:
        return str(Path(__file__).parent / "results" / "models" / "baseline_model")
    
    def get_data_type(self) -> str:
        return 'low_resource'
    
    def get_description(self) -> str:
        return "RQ4.2 Baseline Experiment: Direct Low-Resource Language Fine-tuning"
    
    def setup_parser(self, parser):
        """Add baseline-specific arguments"""
        parser.add_argument('--max_samples', type=int, default=None,
                           help='Maximum number of samples')
    
    def display_config(self, args):
        """Display baseline-specific configuration"""
        print(f"Base model: {args.model_path}")
        print(f"Data path: {args.data_path}")
        print(f"Concept layers: {args.concept_layers}")
        print(f"Low-resource language: ruby")
        print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
        print(f"Training epochs: {args.num_epochs}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Batch size: {args.batch_size}")
        print(f"Output path: {args.output_path}")
    
    def get_next_steps(self):
        """Get next steps after completion"""
        return [
            "Check training logs and model performance",
            "Run complete evaluation to compare three models:",
            "  python run_evaluation_and_baseline.py --evaluate_only",
            "View comparative analysis results"
        ]

def main():
    runner = BaselineRunner()
    runner.run()

if __name__ == "__main__":
    main()
