#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import time
import json
from typing import Dict, Any, Optional, Union, TYPE_CHECKING
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .model_wrapper import LayerSelectiveModel
    from .lora_model_wrapper import LoRAModel


class BaseTrainer(ABC):
    """Base trainer class with common functionality"""
    
    def __init__(
        self,
        model: Union['LayerSelectiveModel', 'LoRAModel'],
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Dict[str, Any] = None,
        trainer_type: str = "Base"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.trainer_type = trainer_type
        
        self.device = config.get('device', 'cuda')
        
        # Handle multi-GPU setup
        self.use_multi_gpu = getattr(model, 'use_multi_gpu', False)
        if self.use_multi_gpu:
            self.input_device = getattr(model, 'data_device', 'cuda:0')
        else:
            self.input_device = self.device
        
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
    
    @abstractmethod
    def _get_default_learning_rate(self) -> float:
        """Get default learning rate for specific trainer type"""
        pass
    
    @abstractmethod
    def _get_default_weight_decay(self) -> float:
        """Get default weight decay for specific trainer type"""
        pass
    
    @abstractmethod
    def _get_default_epochs_key(self) -> str:
        """Get default epochs config key for specific trainer type"""
        pass
    
    @abstractmethod
    def _get_history_filename(self) -> str:
        """Get training history filename for specific trainer type"""
        pass
    
    def _setup_optimizer(self) -> AdamW:
        """Setup optimizer with trainable parameters"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = AdamW(
            trainable_params,
            lr=self.config.get('learning_rate', self._get_default_learning_rate()),
            weight_decay=self.config.get('weight_decay', self._get_default_weight_decay()),
            betas=(self.config.get('adam_beta1', 0.9), self.config.get('adam_beta2', 0.95)),
            eps=self.config.get('adam_epsilon', 1e-8)
        )
        return optimizer
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        epochs_key = self._get_default_epochs_key()
        epochs = self.config.get('num_epochs', self.config.get(epochs_key, 3))
        gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 1)
        num_training_steps = (len(self.train_dataloader) * epochs) // gradient_accumulation_steps
        
        warmup_ratio = self.config.get('warmup_ratio', 0.1)
        num_warmup_steps = max(1, int(num_training_steps * warmup_ratio))
        
        self._setup_dynamic_steps(num_training_steps)
        
        lr_scheduler_type = self.config.get('lr_scheduler_type', 'cosine')
        
        if lr_scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        
        return scheduler
    
    def _setup_dynamic_steps(self, num_training_steps: int):
        """Setup dynamic save/eval/logging steps based on training steps"""
        save_steps = max(100, min(2000, int(num_training_steps * 0.2)))
        eval_steps = max(50, min(1000, int(num_training_steps * 0.15)))
        logging_steps = max(10, min(200, int(num_training_steps * 0.05)))
        
        self.config['save_steps'] = save_steps
        self.config['eval_steps'] = eval_steps
        self.config['logging_steps'] = logging_steps
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        epoch_start_time = time.time()
        
        # Use trainer type in progress bar description
        desc = f"{self.trainer_type} Epoch {self.epoch + 1}"
        progress_bar = tqdm(self.train_dataloader, desc=desc, leave=False)
        
        for step, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.input_device)
            attention_mask = batch['attention_mask'].to(self.input_device)
            labels = batch['labels'].to(self.input_device)
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.config.get('bf16', False)):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                original_loss = loss.item()
                
                grad_acc_steps = self.config.get('gradient_accumulation_steps', 1)
                loss = loss / grad_acc_steps
            
            loss.backward()
            total_loss += original_loss
            
            if (step + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.get('max_grad_norm', 1.0)
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Apply minimum learning rate if specified
                min_lr = self.config.get('min_learning_rate', None)
                if min_lr is not None:
                    for pg in self.optimizer.param_groups:
                        if 'lr' in pg and pg['lr'] < float(min_lr):
                            pg['lr'] = float(min_lr)
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Clear cache periodically
                if self.global_step % 100 == 0:
                    torch.cuda.empty_cache()
            
            progress_bar.set_postfix({
                'loss': f"{original_loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'epoch_time': epoch_time,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        if not self.val_dataloader:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        desc = f"{self.trainer_type} Validation"
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc=desc, leave=False):
                input_ids = batch['input_ids'].to(self.input_device)
                attention_mask = batch['attention_mask'].to(self.input_device)
                labels = batch['labels'].to(self.input_device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss.item()
                total_loss += loss
        
        avg_loss = total_loss / num_batches
        
        return {'val_loss': avg_loss}
    
    def train(self, num_epochs: int, save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Main training loop"""
        save_path = Path(save_path) if save_path else None
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            train_stats = self.train_epoch()
            val_stats = self.validate()
            
            self.train_history['train_loss'].append(train_stats['train_loss'])
            self.train_history['learning_rate'].append(train_stats['learning_rate'])
            self.train_history['epoch_times'].append(train_stats['epoch_time'])
            
            if val_stats:
                self.train_history['val_loss'].append(val_stats['val_loss'])
                
                if self._should_early_stop(val_stats['val_loss']):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if save_path:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch + 1}"
                self.save_checkpoint(checkpoint_path)
        
        if save_path:
            self.model.save_model(save_path)
            self._save_training_history(save_path)
        
        return self.train_history
    
    def _should_early_stop(self, val_loss: float) -> bool:
        """Check if should stop early based on validation loss"""
        threshold = self.config.get('early_stopping_threshold', 0.001)
        patience = self.config.get('early_stopping_patience', 3)
        
        if val_loss < self.best_val_loss - threshold:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= patience
    
    def save_checkpoint(self, checkpoint_path: Path):
        """Save model checkpoint"""
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.model.save_model(checkpoint_path)
    
    def _save_training_history(self, save_path: Path):
        """Save training history to JSON"""
        history_path = save_path / self._get_history_filename()
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.train_history, f, indent=2, ensure_ascii=False)