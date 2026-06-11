#!/usr/bin/env python
# -*- coding: utf-8 -*-

from base_runner import BaseRunner
from pathlib import Path

class Phase1Runner(BaseRunner):
    """Runner for phase 1 training experiment"""
    
    def get_experiment_name(self) -> str:
        return 'phase1_high_resource_training'
    
    def get_default_output_path(self) -> str:
        return str(Path(__file__).parent / "results" / "models" / "base1_model")
    
    def get_data_type(self) -> str:
        return 'high_resource'
    
    def get_description(self) -> str:
        return "RQ4.2 Phase 1 Training: High-Resource Language Fine-tuning"
    
    def setup_parser(self, parser):
        """Add phase1-specific arguments"""
        parser.add_argument('--max_samples_per_language', type=int, default=50000,
                           help='Maximum samples per language')
        parser.set_defaults(num_epochs=2)  # Override default epochs for phase 1
    
    def display_config(self, args):
        """Display phase1-specific configuration"""
        print(f"Model path: {args.model_path}")
        print(f"Data path: {args.data_path}")
        print(f"Concept layers: {args.concept_layers}")
        print(f"High-resource languages: python, go, java, javascript, php")
        print(f"Max samples per language: {args.max_samples_per_language}")
        print(f"Training epochs: {args.num_epochs}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Batch size: {args.batch_size}")
        print(f"Output path: {args.output_path}")
    
    def build_command(self, args):
        """Override to handle max_samples_per_language"""
        cmd = super().build_command(args)
        # Replace max_samples with max_samples_per_language value
        if hasattr(args, 'max_samples_per_language'):
            # Find and replace --max_samples value
            try:
                idx = cmd.index('--max_samples')
                cmd[idx + 1] = str(args.max_samples_per_language)
            except ValueError:
                # If not found, add it
                cmd.extend(['--max_samples', str(args.max_samples_per_language)])
        return cmd
    
    def get_next_steps(self):
        """Get next steps after completion"""
        return [
            "Check training logs and model performance",
            "Run Phase 2 training: python run_phase2_training.py",
            "Or run complete evaluation: python run_evaluation_and_baseline.py"
        ]

def main():
    runner = Phase1Runner()
    runner.run()

if __name__ == "__main__":
    main()