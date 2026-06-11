#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import torch
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from lora_model_wrapper import create_lora_model, LoRAModel
from lora_trainer import setup_lora_trainer
from data_processor import setup_data_processor

def main():
    parser = argparse.ArgumentParser(description='LoRA Baseline Experiment')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True, help='Base model path')
    parser.add_argument('--data_path', type=str, required=True, help='Data directory path')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--epochs', type=int, default=4, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum samples for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume_from', default=None, help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Set default output directory if not provided
    if not args.output_dir:
        args.output_dir = str(Path(__file__).parent / "results" / "lora_model")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    print("=" * 80)
    print("LoRA Baseline Experiment Starting")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Base model: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Random seed: {args.seed}")
    
    # Create LoRA configuration
    lora_config = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'r': 16,  # LoRA rank
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'target_modules': ['q_proj', 'v_proj'],
    }
    
    print("\nLoRA Configuration:")
    print(f"  rank: {lora_config['r']}")
    print(f"  alpha: {lora_config['lora_alpha']}")
    print(f"  dropout: {lora_config['lora_dropout']}")
    print(f"  target_modules: {lora_config['target_modules']}")
    
    try:
        # 1. Create data processor
        print("\nCreating data processor...")
        data_processor = setup_data_processor(args.model_path)
        
        # 2. Load Ruby dataset
        print("Loading Ruby training data...")
        train_data = data_processor.create_low_resource_dataset(
            args.data_path, 'train', max_samples=args.max_samples
        )
        print(f"Training samples: {len(train_data)}")
        
        print("Loading Ruby validation data...")
        val_data = data_processor.create_low_resource_dataset(
            args.data_path, 'valid', max_samples=7000
        )
        print(f"Validation samples: {len(val_data)}")
        
        # 3. Create datasets and dataloaders
        print("Creating datasets...")
        train_dataset = data_processor.create_dataset(train_data)
        val_dataset = data_processor.create_dataset(val_data)
        
        train_dataloader = data_processor.create_dataloader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True
        )
        val_dataloader = data_processor.create_dataloader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False
        )
        
        # 4. Create LoRA model
        print("Creating LoRA model...")
        if args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")
            model = LoRAModel.load_model(args.resume_from)
        else:
            model = create_lora_model(args.model_path)
        
        # 5. Create trainer
        print("Creating LoRA trainer...")
        trainer = setup_lora_trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=lora_config
        )
        
        # 6. Start training
        print("Starting LoRA training...")
        training_history = trainer.train(
            num_epochs=args.epochs,
            save_path=output_dir
        )
        
        # 7. Save training results
        print("Saving training results...")
        
        # Save final model
        final_model_path = output_dir / "final_model"
        model.save_model(final_model_path)
        print(f"Final model saved: {final_model_path}")
        
        # Save experiment configuration
        experiment_config = {
            'model_path': args.model_path,
            'data_path': args.data_path,
            'output_dir': str(output_dir),
            'training_config': lora_config,
            'data_config': {
                'train_samples': len(train_data),
                'val_samples': len(val_data),
                'language': 'ruby',
            },
            'training_history': training_history,
        }
        
        config_path = output_dir / "experiment_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(experiment_config, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"Experiment configuration saved: {config_path}")
        
        # 8. Print training summary
        print("=" * 80)
        print("LoRA Training Completed")
        print("=" * 80)
        print(f"Final training loss: {training_history['train_loss'][-1]:.4f}")
        if training_history['val_loss']:
            print(f"Final validation loss: {training_history['val_loss'][-1]:.4f}")
        print(f"Total training time: {sum(training_history['epoch_times']):.2f} seconds")
        print(f"Model save path: {final_model_path}")
        
        # Clean GPU memory
        torch.cuda.empty_cache()
        
        print("LoRA baseline experiment completed successfully!")
        
    except Exception as e:
        print(f"Error during LoRA training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()