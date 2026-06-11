#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LoRA trainer for code summarization using LoRAModel
"""

from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from .base_trainer import BaseTrainer
from .lora_model_wrapper import LoRAModel


class LoRATrainer(BaseTrainer):
    """Trainer for LoRA model training"""
    
    def __init__(
        self,
        model: LoRAModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None
    ):
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            trainer_type="LoRA"
        )
    
    def _get_default_learning_rate(self) -> float:
        """Get default learning rate for LoRA training"""
        return 1e-4
    
    def _get_default_weight_decay(self) -> float:
        """Get default weight decay for LoRA training"""
        return 0.005
    
    def _get_default_epochs_key(self) -> str:
        """Get default epochs config key for LoRA training"""
        return 'baseline_epochs'
    
    def _get_history_filename(self) -> str:
        """Get training history filename for LoRA training"""
        return 'lora_training_history.json'


def setup_lora_trainer(
    model: LoRAModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Dict[str, Any] = None
) -> LoRATrainer:
    """Setup and return a LoRA trainer"""
    return LoRATrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
