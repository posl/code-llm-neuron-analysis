import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.distance import cosine

class LayerBasedCloneDetector:
    
    def __init__(self, target_layers: List[int]):
        self.target_layers = target_layers
        self.layer_thresholds = {}
        
    def compute_cosine_similarity(self, repr_a: torch.Tensor, repr_b: torch.Tensor) -> float:
        if isinstance(repr_a, torch.Tensor):
            repr_a = repr_a.cpu().numpy()
        if isinstance(repr_b, torch.Tensor):
            repr_b = repr_b.cpu().numpy()
        
        return 1 - cosine(repr_a.flatten(), repr_b.flatten())
    
    def compute_similarities_for_pairs(self, representation_results: List[Dict[str, Any]]) -> Dict[int, List[float]]:
        layer_similarities = {layer_idx: [] for layer_idx in self.target_layers}
        
        for result in representation_results:
            repr_a = result['representations_a']
            repr_b = result['representations_b']
            
            for layer_idx in self.target_layers:
                if layer_idx in repr_a and layer_idx in repr_b:
                    similarity = self.compute_cosine_similarity(repr_a[layer_idx], repr_b[layer_idx])
                    layer_similarities[layer_idx].append(similarity)
        
        return layer_similarities
    
    def optimize_threshold_for_layer(self, similarities: List[float], labels: List[int],
                                   layer_idx: int, return_roc_data: bool = False) -> Tuple[float, Dict[str, float]]:
        similarities = np.array(similarities)
        labels = np.array(labels)

        # Calculate AUC
        try:
            auc_score = roc_auc_score(labels, similarities)
        except ValueError:
            auc_score = 0.0

        # Adaptive threshold range
        sim_min, sim_max = similarities.min(), similarities.max()
        sim_range = sim_max - sim_min

        if sim_range < 0.1:
            threshold_min, threshold_max = 0.1, 0.9
        else:
            margin = sim_range * 0.1
            threshold_min = max(0.0, sim_min + margin)
            threshold_max = min(1.0, sim_max - margin)

        num_thresholds = 200
        thresholds = np.linspace(threshold_min, threshold_max, num_thresholds)

        best_threshold = 0.5
        best_f1 = 0.0
        best_metrics = {}
        all_metrics = []

        for threshold in thresholds:
            predictions = (similarities >= threshold).astype(int)

            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)

            metrics = {
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc_score
            }
            all_metrics.append(metrics)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = metrics.copy()

        if return_roc_data:
            fpr, tpr, roc_thresholds = roc_curve(labels, similarities)
            best_metrics['roc_data'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': auc_score
            }
            best_metrics['all_metrics'] = all_metrics

        return best_threshold, best_metrics

    def optimize_thresholds_with_similarities(self, layer_similarities: Dict[int, List[float]],
                                            labels: List[int], cv_folds: int = 5,
                                            return_roc_data: bool = False) -> Dict[int, Dict[str, float]]:
        layer_results = {}

        for layer_idx in self.target_layers:
            if layer_idx not in layer_similarities:
                continue

            similarities = layer_similarities[layer_idx]

            if len(similarities) != len(labels):
                continue

            best_threshold, best_metrics = self.optimize_threshold_for_layer(
                similarities, labels, layer_idx, return_roc_data=return_roc_data
            )

            layer_results[layer_idx] = {
                'threshold': best_threshold,
                'f1': best_metrics['f1'],
                'precision': best_metrics['precision'],
                'recall': best_metrics['recall'],
                'accuracy': best_metrics['accuracy']
            }

            if return_roc_data:
                try:
                    from sklearn.metrics import roc_curve, auc
                    fpr, tpr, _ = roc_curve(labels, similarities)
                    roc_auc = auc(fpr, tpr)
                    layer_results[layer_idx]['roc_data'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'auc': roc_auc
                    }
                    layer_results[layer_idx]['auc'] = roc_auc
                except Exception:
                    pass

        return layer_results

    def optimize_all_thresholds(self, representation_results: List[Dict[str, Any]],
                              cv_folds: int = 5, return_roc_data: bool = False) -> Dict[int, Dict[str, float]]:
        layer_similarities = self.compute_similarities_for_pairs(representation_results)
        
        labels = [1 if result['pair_info']['type'] == 'clone' else 0 
                 for result in representation_results]
        
        layer_results = {}
        
        for layer_idx in self.target_layers:
            similarities = layer_similarities[layer_idx]
            
            if len(similarities) == 0:
                continue
            
            if len(similarities) != len(labels):
                min_len = min(len(similarities), len(labels))
                similarities = similarities[:min_len]
                layer_labels = labels[:min_len]
            else:
                layer_labels = labels
            
            # Cross validation
            cv_f1_scores = []
            cv_thresholds = []
            
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            
            for train_idx, val_idx in skf.split(similarities, layer_labels):
                train_sims = [similarities[i] for i in train_idx]
                train_labels = [layer_labels[i] for i in train_idx]
                val_sims = [similarities[i] for i in val_idx]
                val_labels = [layer_labels[i] for i in val_idx]
                
                threshold, _ = self.optimize_threshold_for_layer(train_sims, train_labels, layer_idx, return_roc_data=False)
                
                val_predictions = [1 if sim >= threshold else 0 for sim in val_sims]
                val_f1 = f1_score(val_labels, val_predictions, zero_division=0)
                
                cv_f1_scores.append(val_f1)
                cv_thresholds.append(threshold)
            
            final_threshold = np.mean(cv_thresholds)
            self.layer_thresholds[layer_idx] = final_threshold
            
            final_predictions = [1 if sim >= final_threshold else 0 for sim in similarities]

            _, detailed_metrics = self.optimize_threshold_for_layer(
                similarities, layer_labels, layer_idx, return_roc_data=return_roc_data
            )

            layer_results[layer_idx] = {
                'threshold': final_threshold,
                'cv_f1_mean': np.mean(cv_f1_scores),
                'cv_f1_std': np.std(cv_f1_scores),
                'accuracy': accuracy_score(layer_labels, final_predictions),
                'precision': precision_score(layer_labels, final_predictions, zero_division=0),
                'recall': recall_score(layer_labels, final_predictions, zero_division=0),
                'f1': f1_score(layer_labels, final_predictions, zero_division=0),
                'auc': detailed_metrics.get('auc', 0.0),
                'num_samples': len(similarities)
            }

            if return_roc_data and 'roc_data' in detailed_metrics:
                layer_results[layer_idx]['roc_data'] = detailed_metrics['roc_data']
                layer_results[layer_idx]['all_metrics'] = detailed_metrics['all_metrics']
        
        return layer_results
    
    def predict_clone(self, repr_a: torch.Tensor, repr_b: torch.Tensor, 
                     layer_idx: int) -> Tuple[bool, float]:
        if layer_idx not in self.layer_thresholds:
            raise ValueError(f"Layer {layer_idx} threshold not set")
        
        similarity = self.compute_cosine_similarity(repr_a, repr_b)
        threshold = self.layer_thresholds[layer_idx]
        is_clone = similarity >= threshold
        
        return is_clone, similarity