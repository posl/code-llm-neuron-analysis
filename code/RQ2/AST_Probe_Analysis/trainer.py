#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from typing import Dict, List, Any, Tuple

class ProbeTrainer:
    
    def __init__(self, 
                 device: str = "cuda",
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 num_epochs: int = 50,
                 early_stopping_patience: int = 10,
                 weight_decay: float = 1e-4):
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.weight_decay = weight_decay
    
    def prepare_data(self, 
                    representations: Dict[str, Any], 
                    dataset: Dict[str, Any],
                    layer_idx: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        train_X, train_ast_y = self._extract_layer_data(
            representations, dataset['train'], layer_idx, dataset
        )
        val_X, val_ast_y = self._extract_layer_data(
            representations, dataset['validation'], layer_idx, dataset
        )
        test_X, test_ast_y = self._extract_layer_data(
            representations, dataset['test'], layer_idx, dataset
        )
        
        train_loader = self._create_dataloader(train_X, train_ast_y, shuffle=True)
        val_loader = self._create_dataloader(val_X, val_ast_y, shuffle=False)
        test_loader = self._create_dataloader(test_X, test_ast_y, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def _extract_layer_data(self,
                           representations: Dict[str, Any],
                           samples: List[Dict],
                           layer_idx: int,
                           dataset: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:

        X_list = []
        ast_y_list = []

        repr_map = {}
        repr_samples_info = []

        for repr_data in representations['representations']:
            sample_info = repr_data['sample_info']
            repr_samples_info.append(sample_info)

            if layer_idx in repr_data['representations']:
                keys = [
                    (sample_info['task_id'], sample_info['token'], sample_info['language']),
                    (sample_info['task_id'], sample_info['language']),
                    (sample_info['token'], sample_info['language']),
                    sample_info['task_id']
                ]

                for key in keys:
                    if key not in repr_map:
                        repr_map[key] = []
                    repr_map[key].append((repr_data['representations'][layer_idx], sample_info))

        matched_count = 0
        unmatched_samples = []

        for sample in samples:
            matched = False

            match_keys = [
                (sample['task_id'], sample['token'], sample['language']),
                (sample['task_id'], sample['language']),
                (sample['token'], sample['language']),
                sample['task_id']
            ]

            for key in match_keys:
                if key in repr_map and repr_map[key]:
                    repr_tensor, matched_sample_info = repr_map[key][0]

                    if repr_tensor.dim() > 1:
                        repr_tensor = repr_tensor.mean(dim=0)

                    X_list.append(repr_tensor)

                    if sample['ast_type'] in dataset['ast_type_to_id']:
                        ast_label = dataset['ast_type_to_id'][sample['ast_type']]
                    else:
                        continue

                    ast_y_list.append(ast_label)
                    matched = True
                    matched_count += 1
                    break

            if not matched:
                unmatched_samples.append(sample)

        if not X_list:
            if representations['representations']:
                first_repr = representations['representations'][0]
                if layer_idx in first_repr['representations']:
                    sample_repr = first_repr['representations'][layer_idx]
                    if sample_repr.dim() > 1:
                        hidden_dim = sample_repr.size(-1)
                    else:
                        hidden_dim = sample_repr.size(0)
                else:
                    hidden_dim = 4096
            else:
                hidden_dim = 4096

            return (torch.empty(0, hidden_dim),
                   torch.empty(0, dtype=torch.long))

        X = torch.stack(X_list).float()
        ast_y = torch.tensor(ast_y_list, dtype=torch.long)

        if torch.isnan(X).any() or torch.isinf(X).any():
            pass

        return X, ast_y
    
    def _create_dataloader(self, X: torch.Tensor, ast_y: torch.Tensor, shuffle: bool = False) -> DataLoader:
        dataset = TensorDataset(X, ast_y)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def train_probe(self, 
                   probe_model: nn.Module,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   task_type: str = "ast") -> Dict[str, Any]:
        
        probe_model = probe_model.to(self.device)
        optimizer = optim.Adam(probe_model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch(
                probe_model, train_loader, optimizer, criterion, task_type
            )
            
            val_loss, val_acc = self._validate_epoch(
                probe_model, val_loader, criterion, task_type
            )
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = probe_model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= self.early_stopping_patience:
                break
        
        if best_model_state is not None:
            probe_model.load_state_dict(best_model_state)
        
        return {
            'history': history,
            'best_val_acc': best_val_acc,
            'best_model_state': best_model_state
        }
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module, task_type: str) -> Tuple[float, float]:
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_X, batch_ast_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_ast_y = batch_ast_y.to(self.device)
            
            optimizer.zero_grad()
            
            logits = model(batch_X)
            loss = criterion(logits, batch_ast_y)
            predictions = logits.argmax(dim=1)
            targets = batch_ast_y
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct_predictions += (predictions == targets).sum().item()
            total_predictions += targets.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module, task_type: str) -> Tuple[float, float]:
        model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch_X, batch_ast_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_ast_y = batch_ast_y.to(self.device)

                logits = model(batch_X)
                loss = criterion(logits, batch_ast_y)
                predictions = logits.argmax(dim=1)
                targets = batch_ast_y
                
                total_loss += loss.item()
                correct_predictions += (predictions == targets).sum().item()
                total_predictions += targets.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def evaluate_probe(self,
                      probe_model: nn.Module,
                      test_loader: DataLoader,
                      dataset: Dict[str, Any],
                      task_type: str = "ast") -> Dict[str, Any]:

        probe_model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_X, batch_ast_y in test_loader:
                batch_X = batch_X.to(self.device)

                logits = probe_model(batch_X)
                predictions = logits.argmax(dim=1).cpu().numpy()
                targets = batch_ast_y.numpy()

                all_predictions.extend(predictions)
                all_targets.extend(targets)

        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        precision_per_class = precision_score(all_targets, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(all_targets, all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)

        unique_targets = sorted(list(set(all_targets)))
        unique_predictions = sorted(list(set(all_predictions)))
        unique_labels = sorted(list(set(all_targets + all_predictions)))

        actual_label_names = []
        for label_id in unique_labels:
            if str(label_id) in dataset['id_to_ast_type']:
                actual_label_names.append(dataset['id_to_ast_type'][str(label_id)])
            else:
                actual_label_names.append(f"unknown_ast_{label_id}")

        full_label_names = []
        for i in range(len(dataset['ast_types'])):
            if str(i) in dataset['id_to_ast_type']:
                full_label_names.append(dataset['id_to_ast_type'][str(i)])
            else:
                full_label_names.append(f"unknown_ast_{i}")

        cm = confusion_matrix(all_targets, all_predictions)

        if actual_label_names:
            report = classification_report(all_targets, all_predictions,
                                         labels=unique_labels,
                                         target_names=actual_label_names,
                                         output_dict=True,
                                         zero_division=0)
        else:
            report = classification_report(all_targets, all_predictions,
                                         output_dict=True,
                                         zero_division=0)

        macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)
        micro_f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=0)

        data_quality_issues = []
        if len(unique_targets) < len(dataset['ast_types']) / 2:
            data_quality_issues.append(f"Low test set class coverage: {len(unique_targets)}/{len(dataset['ast_types'])}")
        if len(unique_predictions) < len(unique_targets) / 2:
            data_quality_issues.append(f"Low prediction class diversity: {len(unique_predictions)}/{len(unique_targets)}")
        if accuracy > 0.95:
            data_quality_issues.append(f"Abnormally high accuracy: {accuracy:.4f} (possible data leakage)")

        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'macro_f1': macro_f1,
            'micro_f1': micro_f1,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': all_predictions,
            'targets': all_targets,
            'label_names': full_label_names,
            'actual_label_names': actual_label_names,
            'unique_labels': unique_labels,
            'unique_targets': unique_targets,
            'unique_predictions': unique_predictions,
            'num_actual_classes': len(unique_labels),
            'num_total_classes': len(dataset['ast_types']),
            'data_quality_issues': data_quality_issues,
            'sample_count': len(all_targets)
        }

        return results

    def cross_validate(self,
                      probe_model_class,
                      representations: Dict[str, Any],
                      dataset: Dict[str, Any],
                      layer_idx: int,
                      task_type: str = "ast",
                      k_folds: int = 5) -> Dict[str, Any]:

        all_samples = dataset['train'] + dataset['validation']

        X, ast_y = self._extract_layer_data(representations, all_samples, layer_idx, dataset)

        if len(X) == 0:
            return {}

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = ast_y[train_idx], ast_y[val_idx]
            num_classes = len(dataset['ast_types'])

            train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

            input_dim = X.size(1)
            probe_model = probe_model_class(input_dim, num_classes)

            train_result = self.train_probe(probe_model, train_loader, val_loader, task_type)

            eval_result = self.evaluate_probe(probe_model, val_loader, dataset, task_type)

            fold_results.append({
                'fold': fold + 1,
                'train_result': train_result,
                'eval_result': eval_result
            })

        avg_accuracy = np.mean([result['eval_result']['accuracy'] for result in fold_results])
        avg_f1 = np.mean([result['eval_result']['f1_score'] for result in fold_results])
        std_accuracy = np.std([result['eval_result']['accuracy'] for result in fold_results])
        std_f1 = np.std([result['eval_result']['f1_score'] for result in fold_results])

        cv_results = {
            'fold_results': fold_results,
            'avg_accuracy': avg_accuracy,
            'avg_f1': avg_f1,
            'std_accuracy': std_accuracy,
            'std_f1': std_f1,
            'num_folds': k_folds
        }

        return cv_results

    def save_model(self, model: nn.Module, save_path: str):
        torch.save(model.state_dict(), save_path)

    def load_model(self, model: nn.Module, load_path: str):
        model.load_state_dict(torch.load(load_path, map_location=self.device))
        return model