#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA model wrapper for parameter-efficient fine-tuning
"""

import torch
from typing import Dict, Any, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import json
from .base_model_wrapper import BaseModelWrapper
from .config import LORA_CONFIG


class LoRAModel(BaseModelWrapper):
    """Model wrapper with LoRA adapter support"""
    
    def __init__(self, model_path: str, device: str = 'auto', load_adapters: bool = False, adapters_path: str = None):
        self.load_adapters = load_adapters
        self.adapters_path = adapters_path
        self.lora_config = None
        super().__init__(model_path, device)
    
    def _setup_tokenizer(self, model_path: str):
        """Setup tokenizer, potentially from a different path if adapters are loaded"""
        if self.load_adapters and self.adapters_path:
            final_model_dir = Path(self.adapters_path).parent
            if (final_model_dir / "tokenizer_config.json").exists():
                tokenizer_path = str(final_model_dir)
            else:
                tokenizer_path = model_path
        else:
            tokenizer_path = model_path
        
        super()._setup_tokenizer(tokenizer_path)
    
    def _setup_model(self):
        """Setup the base model and apply LoRA adapters"""
        num_gpus = torch.cuda.device_count()
        
        # Determine device mapping
        if self.device == 'auto':
            if num_gpus > 1:
                device_map = "auto"
            else:
                device_map = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_map = self.device
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        
        # Set device attributes
        self.use_multi_gpu = (device_map == "auto" and num_gpus > 1)
        if self.use_multi_gpu:
            self.data_device = 'cuda:0'
        else:
            self.data_device = device_map if device_map in ['cpu', 'cuda'] else 'cuda:0'
        
        self.input_device = self.data_device
        
        # Setup LoRA configuration
        current_inference_mode = self.load_adapters
        
        self.lora_config = LoraConfig(
            r=LORA_CONFIG['r'],
            lora_alpha=LORA_CONFIG['lora_alpha'],
            target_modules=LORA_CONFIG['target_modules'],
            lora_dropout=LORA_CONFIG['lora_dropout'],
            bias=LORA_CONFIG['bias'],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=current_inference_mode,
        )
        
        # Apply LoRA adapters
        if self.load_adapters and self.adapters_path:
            self.model = PeftModel.from_pretrained(self.base_model, self.adapters_path)
            self.model.eval()
        else:
            # Freeze base model parameters
            for param in self.base_model.parameters():
                param.requires_grad = False
            self.model = get_peft_model(self.base_model, self.lora_config)
        
        # Enable gradient checkpointing for training
        if not self.load_adapters and torch.cuda.is_available():
            self.model.gradient_checkpointing_enable()
        
        self._log_trainable_parameters()
    
    def save_model(self, save_path: Path, save_tokenizer: bool = True):
        """Save LoRA adapters and configuration"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save adapters
        adapters_path = save_path / "adapters"
        self.model.save_pretrained(adapters_path)
        
        # Save tokenizer
        if save_tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        # Save configuration
        config_info = {
            'model_type': 'lora',
            'base_model_path': str(self.model_path),
            'lora_config': {
                'r': self.lora_config.r,
                'lora_alpha': self.lora_config.lora_alpha,
                'target_modules': list(self.lora_config.target_modules),
                'lora_dropout': self.lora_config.lora_dropout,
                'bias': self.lora_config.bias,
            },
            'adapters_path': str(adapters_path),
        }
        
        with open(save_path / 'lora_model_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
    
    @classmethod
    def load_model(cls, model_path: Path, device: str = 'auto'):
        """Load LoRA model from saved path"""
        model_path = Path(model_path)
        
        config_path = model_path / 'lora_model_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_info = json.load(f)
            
            base_model_path = config_info['base_model_path']
            adapters_path = config_info['adapters_path']
            
            return cls(
                model_path=base_model_path,
                device=device,
                load_adapters=True,
                adapters_path=adapters_path
            )
        else:
            return cls(model_path=str(model_path), device=device)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including LoRA configuration"""
        trainable_params, total_params = self._log_trainable_parameters()
        
        return {
            'model_type': 'lora',
            'model_path': self.model_path,
            'lora_config': {
                'r': self.lora_config.r,
                'lora_alpha': self.lora_config.lora_alpha,
                'target_modules': list(self.lora_config.target_modules),
            },
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'device': str(next(self.model.parameters()).device),
        }


def create_lora_model(model_path: str = None, device: str = None) -> LoRAModel:
    """Create a new LoRA model"""
    device = device or 'auto'
    return LoRAModel(model_path=model_path, device=device)
