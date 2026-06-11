#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Layer-selective model wrapper for concept-based training
"""

import torch
from typing import List, Dict, Any
from pathlib import Path
from transformers import AutoModelForCausalLM
import json
from .base_model_wrapper import BaseModelWrapper
from .config import DEFAULT_CONCEPT_LAYERS


class LayerSelectiveModel(BaseModelWrapper):
    """Model wrapper with selective layer training capability"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.concept_layers = None
        self.num_layers = None
        super().__init__(model_path, device)
    
    def _setup_model(self):
        """Setup the model with optional multi-GPU support"""
        num_gpus = torch.cuda.device_count()
        
        if num_gpus > 1 and self.device == 'auto':
            device_map = self._create_device_map(num_gpus)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
            self.use_multi_gpu = True
            self.data_device = 'cuda:0'
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
            )
            self.use_multi_gpu = False
            self.data_device = self.device
            
            if self.device != 'auto':
                self.model = self.model.to(self.device)
        
        self.input_device = self.data_device
        self.num_layers = len(self.model.model.layers)
        
        if torch.cuda.is_available():
            self.model.gradient_checkpointing_enable()
        
        self._log_trainable_parameters()
    
    def set_trainable_layers(self, concept_layers: List[int] = None):
        """Set specific layers as trainable"""
        self.concept_layers = concept_layers
        
        if concept_layers is not None:
            for layer_idx in concept_layers:
                if layer_idx >= self.num_layers:
                    raise ValueError(f"Concept layer index {layer_idx} exceeds model layers {self.num_layers}")
        
        self.freeze_all_layers()
        self.unfreeze_concept_layers()
        
        self._log_trainable_parameters()
    
    def freeze_all_layers(self):
        """Freeze all model parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_concept_layers(self):
        """Unfreeze specified concept layers and lm_head"""
        if self.concept_layers is None:
            return
        
        # Unfreeze concept layers
        for layer_idx in self.concept_layers:
            try:
                layer = self.model.model.layers[layer_idx]
                for param in layer.parameters():
                    param.requires_grad = True
            except Exception:
                pass
        
        # Always unfreeze lm_head for training
        try:
            if hasattr(self.model, 'lm_head'):
                for param in self.model.lm_head.parameters():
                    param.requires_grad = True
        except Exception:
            pass
    
    def save_model(self, save_path: Path, save_tokenizer: bool = True):
        """Save model and configuration"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        
        if save_tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        config_info = {
            'concept_layers': self.concept_layers,
            'model_path': str(self.model_path),
            'num_layers': self.num_layers,
        }
        
        with open(save_path / 'concept_layers_config.json', 'w') as f:
            json.dump(config_info, f, indent=2)
    
    @classmethod
    def load_model(cls, model_path: Path, device: str = 'cuda'):
        """Load model from saved path"""
        model_path = Path(model_path)
        
        config_path = model_path / 'concept_layers_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_info = json.load(f)
            concept_layers = config_info.get('concept_layers', DEFAULT_CONCEPT_LAYERS)
        else:
            concept_layers = DEFAULT_CONCEPT_LAYERS
        
        model = cls(model_path=str(model_path), device=device)
        model.set_trainable_layers(concept_layers)
        
        return model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        trainable_params, total_params = self._log_trainable_parameters()
        
        return {
            'model_path': self.model_path,
            'concept_layers': self.concept_layers,
            'num_layers': self.num_layers,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'device': str(next(self.model.parameters()).device),
        }


def create_model(model_path: str = None, concept_layers: List[int] = None, device: str = None) -> LayerSelectiveModel:
    """Create a new LayerSelectiveModel"""
    concept_layers = concept_layers or DEFAULT_CONCEPT_LAYERS
    device = device or 'cuda'
    
    model = LayerSelectiveModel(model_path=model_path, device=device)
    model.set_trainable_layers(concept_layers)
    
    return model


def load_model_for_phase2(base1_model_path: Path, concept_layers: List[int] = None, device: str = None) -> LayerSelectiveModel:
    """Load model for phase 2 training"""
    concept_layers = concept_layers or DEFAULT_CONCEPT_LAYERS
    device = device or 'cuda'
    
    model = LayerSelectiveModel(model_path=str(base1_model_path), device=device)
    model.set_trainable_layers(concept_layers)
    
    return model
