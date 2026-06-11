#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Standard trainer for code summarization using LayerSelectiveModel
"""

from typing import Dict, Any, Optional
from torch.utils.data import DataLoader
from .base_trainer import BaseTrainer
from .model_wrapper import LayerSelectiveModel


class CodeSummarizationTrainer(BaseTrainer):
    """Trainer for standard LayerSelectiveModel training"""
    
    def __init__(
        self,
        model: LayerSelectiveModel,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None
    ):
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            trainer_type=""  # No prefix for standard training
        )
    
    def _get_default_learning_rate(self) -> float:
        """Get default learning rate for standard training"""
        return 5e-5
    
    def _get_default_weight_decay(self) -> float:
        """Get default weight decay for standard training"""
        return 0.01
    
    def _get_default_epochs_key(self) -> str:
        """Get default epochs config key for standard training"""
        return 'phase1_epochs'
    
    def _get_history_filename(self) -> str:
        """Get training history filename for standard training"""
        return 'training_history.json'


def setup_trainer(
    model: LayerSelectiveModel,
    train_dataloader: DataLoader,
    val_dataloader: Optional[DataLoader] = None,
    config: Dict[str, Any] = None
) -> CodeSummarizationTrainer:
    """Setup and return a standard trainer"""
    return CodeSummarizationTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
