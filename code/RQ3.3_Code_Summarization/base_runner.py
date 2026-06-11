#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

class BaseRunner(ABC):
    """Base class for training runners"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.default_concept_layers = [13, 14, 15, 16, 17, 18]
        self.max_layer = 32  # Llama-3.1-8B layer count
    
    @abstractmethod
    def get_experiment_name(self) -> str:
        """Get experiment name"""
        pass
    
    @abstractmethod
    def get_default_output_path(self) -> str:
        """Get default output path"""
        pass
    
    @abstractmethod
    def get_data_type(self) -> str:
        """Get data type (high_resource/low_resource)"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get experiment description"""
        pass
    
    @abstractmethod
    def setup_parser(self, parser: argparse.ArgumentParser):
        """Setup command line parser"""
        pass
    
    @abstractmethod
    def display_config(self, args: argparse.Namespace):
        """Display configuration"""
        pass
    
    @abstractmethod
    def get_next_steps(self) -> List[str]:
        """Get next steps after completion"""
        pass
    
    def validate_concept_layers(self, layers: List[int]):
        """Validate concept layers"""
        for layer in layers:
            if layer >= self.max_layer:
                print(f"Error: Concept layer {layer} exceeds model layer count {self.max_layer}")
                sys.exit(1)
    
    def create_directories(self, output_path: str):
        """Create necessary directories"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    def confirm_execution(self, prompt: str = None) -> bool:
        """Confirm execution with user"""
        if not prompt:
            prompt = f"\nStart {self.get_experiment_name()}? (y/N): "
        response = input(prompt)
        return response.lower() == 'y'
    
    def build_command(self, args: argparse.Namespace) -> List[str]:
        """Build command for fine_tune_model.py"""
        cmd = [
            sys.executable, 'fine_tune_model.py',
            '--model_path', args.model_path,
            '--data_path', args.data_path,
            '--data_type', self.get_data_type(),
            '--output_path', args.output_path,
            '--num_epochs', str(args.num_epochs),
            '--experiment_name', self.get_experiment_name(),
            '--concept_layers'] + [str(layer) for layer in args.concept_layers] + [
            '--learning_rate', str(args.learning_rate),
            '--batch_size', str(args.batch_size)
        ]
        
        if hasattr(args, 'max_samples') and args.max_samples:
            cmd.extend(['--max_samples', str(args.max_samples)])
        
        if hasattr(args, 'base_model_path') and args.base_model_path:
            cmd.extend(['--base_model_path', args.base_model_path])
        
        return cmd
    
    def execute_training(self, cmd: List[str], output_path: str) -> bool:
        """Execute training command"""
        print("Executing command:")
        print(" ".join(cmd))
        print()
        
        try:
            result = subprocess.run(cmd, check=True, cwd=self.script_dir)
            print("\n" + "=" * 60)
            print(f"{self.get_description()} completed!")
            print(f"Model saved to: {output_path}")
            print("=" * 60)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\nTraining failed with error code: {e.returncode}")
            print("Please check log files for detailed error information")
            return False
    
    def run(self):
        """Main execution method"""
        parser = argparse.ArgumentParser(description=self.get_description())
        
        # Common arguments
        parser.add_argument('--model_path', type=str, required=True,
                           help='Path to base model')
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
        
        # Set default output path if not provided
        if not args.output_path:
            args.output_path = self.get_default_output_path()
        
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