#!/usr/bin/env python
# -*- coding: utf-8 -*-

from base_runner import BaseRunner
from pathlib import Path
import sys

class Phase2Runner(BaseRunner):
    """Runner for phase 2 training experiment"""
    
    def get_experiment_name(self) -> str:
        return 'phase2_low_resource_training'
    
    def get_default_output_path(self) -> str:
        return str(Path(__file__).parent / "results" / "models" / "final_model")
    
    def get_data_type(self) -> str:
        return 'low_resource'
    
    def get_description(self) -> str:
        return "RQ4.2 Phase 2 Training: Low-Resource Language Fine-tuning"
    
    def setup_parser(self, parser):
        """Add phase2-specific arguments"""
        parser.add_argument('--base1_model_path', type=str, default=None,
                           help='Path to base1 model from Phase 1')
        parser.add_argument('--max_samples', type=int, default=None,
                           help='Maximum number of samples')
    
    def display_config(self, args):
        """Display phase2-specific configuration"""
        if not args.base1_model_path:
            args.base1_model_path = str(Path(__file__).parent / "results" / "models" / "base1_model")
        
        print(f"Base1 model (from Phase 1): {args.base1_model_path}")
        print(f"Data path: {args.data_path}")
        print(f"Concept layers: {args.concept_layers}")
        print(f"Low-resource language: ruby")
        print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
        print(f"Training epochs: {args.num_epochs}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Batch size: {args.batch_size}")
        print(f"Output path: {args.output_path}")
        print("\nExperiment flow:")
        print("  Phase 1: base → base1 (high-resource languages)")
        print("  Phase 2: base1 → final (low-resource language) ← Current")
    
    def validate_base1_model(self, base1_model_path: str) -> bool:
        """Check if base1 model exists"""
        base1_path = Path(base1_model_path)
        if not base1_path.exists():
            print(f"\nError: Base1 model not found at {base1_model_path}")
            print("Please run Phase 1 training first: python run_phase1_training.py")
            return False
        return True
    
    def build_command(self, args):
        """Override to use base1_model as model_path"""
        if not args.base1_model_path:
            args.base1_model_path = str(Path(__file__).parent / "results" / "models" / "base1_model")
        
        # Use base1_model_path as the model_path for phase 2
        original_model_path = args.model_path
        args.model_path = args.base1_model_path
        cmd = super().build_command(args)
        
        # Add base_model_path for tracking
        cmd.extend(['--base_model_path', args.base1_model_path])
        
        # Restore original model_path
        args.model_path = original_model_path
        return cmd
    
    def run(self):
        """Override to add base1 model validation"""
        parser = super().__class__.__bases__[0].__dict__['run'].__globals__['argparse'].ArgumentParser(
            description=self.get_description()
        )
        
        # Common arguments (from base class)
        parser.add_argument('--model_path', type=str, required=True,
                           help='Path to original base model')
        parser.add_argument('--data_path', type=str, required=True,
                           help='Path to data directory')
        parser.add_argument('--concept_layers', type=int, nargs='+', 
                           default=self.default_concept_layers,
                           help='Concept layer list')
        parser.add_argument('--num_epochs', type=int, default=3,
                           help='Number of training epochs')
        parser.add_argument('--learning_rate', type=float, default=5e-5,
                           help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=2,
                           help='Batch size')
        parser.add_argument('--output_path', type=str, default=None,
                           help='Output path for model')
        
        # Setup specific arguments
        self.setup_parser(parser)
        
        args = parser.parse_args()
        
        # Set default paths
        if not args.output_path:
            args.output_path = self.get_default_output_path()
        if not args.base1_model_path:
            args.base1_model_path = str(Path(__file__).parent / "results" / "models" / "base1_model")
        
        # Validate base1 model exists
        if not self.validate_base1_model(args.base1_model_path):
            sys.exit(1)
        
        # Create directories
        self.create_directories(args.output_path)
        
        # Validate concept layers
        self.validate_concept_layers(args.concept_layers)
        
        # Display configuration
        print("\n" + "=" * 60)
        print(self.get_description())
        print("=" * 60)
        self.display_config(args)
        
        # Confirm execution
        if not self.confirm_execution():
            print("Training cancelled")
            sys.exit(0)
        
        # Build and execute command
        cmd = self.build_command(args)
        success = self.execute_training(cmd, args.output_path)
        
        if success:
            print("\nNext steps:")
            for i, step in enumerate(self.get_next_steps(), 1):
                print(f"{i}. {step}")
        else:
            print("\nPlease check error messages and run again")
            sys.exit(1)
    
    def get_next_steps(self):
        """Get next steps after completion"""
        return [
            "Check training logs and model performance",
            "Run baseline experiment: python run_baseline_training.py",
            "Run complete evaluation to compare all models:",
            "  python run_evaluation_and_baseline.py"
        ]

def main():
    runner = Phase2Runner()
    runner.run()

if __name__ == "__main__":
    main()