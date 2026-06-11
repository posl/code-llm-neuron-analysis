#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Base model wrapper class with common functionality
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
from abc import ABC, abstractmethod


class BaseModelWrapper(nn.Module, ABC):
    """Base class for model wrappers with common functionality"""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.use_multi_gpu = False
        self.data_device = device
        self.input_device = device
        
        # Setup tokenizer
        self._setup_tokenizer(model_path)
        
        # Setup model with device configuration
        self._setup_model()
    
    def _setup_tokenizer(self, tokenizer_path: str):
        """Setup tokenizer with padding token"""
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @abstractmethod
    def _setup_model(self):
        """Setup the model - to be implemented by subclasses"""
        pass
    
    def _create_device_map(self, num_gpus: int) -> Dict[str, int]:
        """Create device map for multi-GPU setup"""
        device_map = {}
        
        config = AutoConfig.from_pretrained(self.model_path)
        num_layers = config.num_hidden_layers
        
        # Split model across GPUs
        device_map["model.embed_tokens"] = 0
        device_map["lm_head"] = 0
        
        # First 14 layers on GPU 0
        for layer_idx in range(min(14, num_layers)):
            device_map[f"model.layers.{layer_idx}"] = 0
        
        # Remaining layers on GPU 1
        for layer_idx in range(14, num_layers):
            device_map[f"model.layers.{layer_idx}"] = 1
        
        device_map["model.norm"] = 1
        
        return device_map
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def generate(self, input_ids, attention_mask=None, **generation_kwargs):
        """Generate text using the model"""
        with torch.no_grad():
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs
            )
    
    @abstractmethod
    def save_model(self, save_path: Path, save_tokenizer: bool = True):
        """Save model - to be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information - to be implemented by subclasses"""
        pass
    
    def _log_trainable_parameters(self):
        """Log trainable parameters count"""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        
        return trainable_params, total_params
    
    def train(self):
        """Set model to training mode"""
        if hasattr(self, 'model'):
            self.model.train()
    
    def eval(self):
        """Set model to evaluation mode"""
        if hasattr(self, 'model'):
            self.model.eval()
    
    def parameters(self):
        """Get model parameters"""
        if hasattr(self, 'model'):
            return self.model.parameters()
        return super().parameters()