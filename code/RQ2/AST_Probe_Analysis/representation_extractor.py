#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm
import gc

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

class LayerRepresentationExtractor:
    
    def __init__(self, model, tokenizer, target_layers: List[int], device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.target_layers = target_layers
        self.device = device

        self.model.eval()

        self.hooks = []
        self.activations = {}

        self.abnormal_dimension = 2352

        self.num_layers = self._get_num_layers()
        
        self.target_layers = [layer for layer in target_layers if 0 <= layer < self.num_layers]
    
    def _get_num_layers(self) -> int:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            return len(self.model.encoder.layer)
        else:
            return 32
    
    def _get_layer_by_index(self, layer_idx: int):
        try:
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return self.model.model.layers[layer_idx]
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                return self.model.transformer.h[layer_idx]
            elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
                return self.model.encoder.layer[layer_idx]
            else:
                return None
        except (IndexError, AttributeError) as e:
            return None
    
    def _register_hooks(self):
        self._remove_hooks()
        self.activations = {}
        
        for layer_idx in self.target_layers:
            layer = self._get_layer_by_index(layer_idx)
            if layer is not None:
                hook = layer.register_forward_hook(self._get_activation_hook(layer_idx))
                self.hooks.append(hook)
    
    def _get_activation_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            activations = hidden_states.detach().cpu().float()
            activations = self._filter_abnormal_dimensions(activations, layer_idx)
            self.activations[layer_idx] = activations

        return hook

    def _filter_abnormal_dimensions(self, activations: torch.Tensor, layer_idx: int) -> torch.Tensor:
        if activations.size(-1) > self.abnormal_dimension:
            dim_indices = list(range(activations.size(-1)))
            dim_indices.remove(self.abnormal_dimension)
            filtered_activations = activations[..., dim_indices]
            return filtered_activations
        else:
            return activations

    def _is_special_token(self, token_id: torch.Tensor) -> bool:
        special_tokens = {
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
            self.tokenizer.unk_token_id,
        }
        special_tokens = {t for t in special_tokens if t is not None}
        return token_id.item() in special_tokens

    def _filter_special_tokens(self, token_positions: List[int], input_ids: torch.Tensor) -> List[int]:
        filtered_positions = []

        for pos in token_positions:
            if 0 <= pos < len(input_ids):
                if not self._is_special_token(input_ids[pos]):
                    filtered_positions.append(pos)

        return filtered_positions

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_token_representations(self, 
                                    code: str, 
                                    token_positions: List[int],
                                    language: str,
                                    max_length: int = 512) -> Dict[int, torch.Tensor]:
        self._register_hooks()
        
        try:
            inputs = self.tokenizer(
                code, 
                return_tensors="pt", 
                truncation=True, 
                max_length=max_length,
                padding=False
            )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                if attention_mask is not None:
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids)
            
            representations = {}
            seq_length = input_ids.size(1)

            filtered_positions = self._filter_special_tokens(token_positions, input_ids[0])

            if not filtered_positions:
                return {}

            for layer_idx in self.target_layers:
                if layer_idx in self.activations:
                    layer_hidden_states = self.activations[layer_idx]

                    token_reprs = []
                    for pos in filtered_positions:
                        if 0 <= pos < seq_length:
                            if not self._is_special_token(input_ids[0, pos]):
                                token_repr = layer_hidden_states[0, pos, :]
                                token_reprs.append(token_repr)

                    if token_reprs:
                        representations[layer_idx] = torch.stack(token_reprs)
            
            return representations
            
        except Exception as e:
            return {}
        finally:
            self._remove_hooks()
            self.activations = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def batch_extract_representations(self, 
                                    samples: List[Dict[str, Any]],
                                    batch_size: int = 8,
                                    max_length: int = 512) -> Dict[str, Any]:
        all_representations = []
        
        for i in tqdm(range(0, len(samples), batch_size), desc="Extracting representations"):
            batch_samples = samples[i:i + batch_size]
            batch_representations = []
            
            for sample in batch_samples:
                try:
                    token_positions = self._get_token_positions(
                        sample['code'],
                        sample.get('token', ''),
                        sample.get('token_index', 0),
                        sample.get('ast_start_byte'),
                        sample.get('ast_end_byte')
                    )
                    
                    if not token_positions:
                        continue
                    
                    representations = self.extract_token_representations(
                        sample['code'],
                        token_positions,
                        sample['language'],
                        max_length
                    )
                    
                    sample_repr = {
                        'sample_info': sample,
                        'representations': representations,
                        'token_positions': token_positions
                    }
                    batch_representations.append(sample_repr)
                    
                except Exception as e:
                    continue
            
            all_representations.extend(batch_representations)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return {
            'representations': all_representations,
            'target_layers': self.target_layers,
            'total_samples': len(all_representations)
        }
    
    def _get_token_positions(self, code: str, target_token: str, token_index: int,
                           ast_start_byte: int = None, ast_end_byte: int = None) -> List[int]:
        try:
            alignment_result = self._align_ast_tokens_with_model_tokens(
                code, target_token, ast_start_byte, ast_end_byte
            )

            if alignment_result['success']:
                return alignment_result['positions']
            else:
                return self._fallback_token_alignment(code, target_token)

        except Exception as e:
            return []
    
    def _align_ast_tokens_with_model_tokens(self, code: str, target_token: str,
                                           ast_start_byte: int = None, ast_end_byte: int = None) -> Dict[str, Any]:
        try:
            encoded = self.tokenizer(code, return_tensors="pt", add_special_tokens=True,
                                   return_offsets_mapping=True)
            input_ids = encoded["input_ids"][0]

            if "offset_mapping" in encoded:
                offset_mapping = encoded["offset_mapping"][0]
                return self._align_with_offset_mapping(
                    code, target_token, input_ids, offset_mapping, ast_start_byte, ast_end_byte
                )
            else:
                return self._align_with_character_mapping(
                    code, target_token, input_ids, ast_start_byte, ast_end_byte
                )

        except Exception as e:
            return {
                'success': False,
                'positions': [],
                'error': f"Alignment algorithm failed: {str(e)}",
                'confidence': 0.0
            }

    def _align_with_offset_mapping(self, code: str, target_token: str, input_ids: torch.Tensor,
                                 offset_mapping: torch.Tensor, ast_start_byte: int = None,
                                 ast_end_byte: int = None) -> Dict[str, Any]:
        try:
            positions = []
            confidence = 0.0

            if ast_start_byte is not None and ast_end_byte is not None:
                for i, (start_char, end_char) in enumerate(offset_mapping):
                    if start_char == end_char and start_char == 0:
                        continue

                    if self._is_special_token(input_ids[i]):
                        continue

                    if start_char <= ast_start_byte < end_char or start_char < ast_end_byte <= end_char:
                        token_text = self.tokenizer.decode([input_ids[i]], skip_special_tokens=True)
                        if self._tokens_match(target_token, token_text):
                            positions.append(i)
                            confidence = 0.9

                if positions:
                    return {
                        'success': True,
                        'positions': positions,
                        'method': 'offset_mapping_with_ast_bytes',
                        'confidence': confidence
                    }

            return self._align_with_text_matching(code, target_token, input_ids, offset_mapping)

        except Exception as e:
            return {
                'success': False,
                'positions': [],
                'error': f"Offset mapping alignment failed: {str(e)}",
                'confidence': 0.0
            }

    def _align_with_character_mapping(self, code: str, target_token: str, input_ids: torch.Tensor,
                                    ast_start_byte: int = None, ast_end_byte: int = None) -> Dict[str, Any]:
        try:
            positions = []
            confidence = 0.0

            char_to_token_map = self._build_char_to_token_mapping(code, input_ids)

            if ast_start_byte is not None:
                char_start = ast_start_byte
                if char_start < len(char_to_token_map):
                    token_idx = char_to_token_map[char_start]
                    if token_idx is not None:
                        token_text = self.tokenizer.decode([input_ids[token_idx]], skip_special_tokens=True)
                        if self._tokens_match(target_token, token_text):
                            positions.append(token_idx)
                            confidence = 0.7

            if not positions:
                positions = self._search_token_by_text(target_token, input_ids)
                confidence = 0.5

            return {
                'success': len(positions) > 0,
                'positions': positions,
                'method': 'character_mapping',
                'confidence': confidence
            }

        except Exception as e:
            return {
                'success': False,
                'positions': [],
                'error': f"Character mapping alignment failed: {str(e)}",
                'confidence': 0.0
            }

    def _is_special_token(self, token_id: int) -> bool:
        special_token_ids = set()

        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            special_token_ids.add(self.tokenizer.bos_token_id)
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            special_token_ids.add(self.tokenizer.eos_token_id)
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            special_token_ids.add(self.tokenizer.pad_token_id)
        if hasattr(self.tokenizer, 'unk_token_id') and self.tokenizer.unk_token_id is not None:
            special_token_ids.add(self.tokenizer.unk_token_id)
        if hasattr(self.tokenizer, 'cls_token_id') and self.tokenizer.cls_token_id is not None:
            special_token_ids.add(self.tokenizer.cls_token_id)
        if hasattr(self.tokenizer, 'sep_token_id') and self.tokenizer.sep_token_id is not None:
            special_token_ids.add(self.tokenizer.sep_token_id)
        if hasattr(self.tokenizer, 'mask_token_id') and self.tokenizer.mask_token_id is not None:
            special_token_ids.add(self.tokenizer.mask_token_id)

        return int(token_id) in special_token_ids

    def _filter_special_tokens(self, positions: List[int], input_ids: torch.Tensor) -> List[int]:
        filtered_positions = []
        for pos in positions:
            if pos < len(input_ids) and not self._is_special_token(input_ids[pos]):
                filtered_positions.append(pos)

        return filtered_positions

    def _align_with_text_matching(self, code: str, target_token: str, input_ids: torch.Tensor,
                                offset_mapping: torch.Tensor) -> Dict[str, Any]:
        try:
            positions = []
            best_confidence = 0.0

            for i, (start_char, end_char) in enumerate(offset_mapping):
                if start_char == end_char and start_char == 0:
                    continue

                if self._is_special_token(input_ids[i]):
                    continue

                token_text = self.tokenizer.decode([input_ids[i]], skip_special_tokens=True)
                confidence = self._calculate_token_similarity(target_token, token_text)

                if confidence > 0.8:
                    positions.append(i)
                    best_confidence = max(best_confidence, confidence)
                elif confidence > best_confidence and confidence > 0.5:
                    positions = [i]
                    best_confidence = confidence

            return {
                'success': len(positions) > 0,
                'positions': positions,
                'method': 'text_matching',
                'confidence': best_confidence
            }

        except Exception as e:
            return {
                'success': False,
                'positions': [],
                'error': f"Text matching failed: {str(e)}",
                'confidence': 0.0
            }

    def _build_char_to_token_mapping(self, code: str, input_ids: torch.Tensor) -> List[int]:
        char_to_token = [None] * len(code)
        current_pos = 0

        for token_idx, token_id in enumerate(input_ids):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)

            if token_text and current_pos < len(code):
                found_pos = code.find(token_text, current_pos)
                if found_pos != -1:
                    for i in range(found_pos, min(found_pos + len(token_text), len(code))):
                        char_to_token[i] = token_idx
                    current_pos = found_pos + len(token_text)

        return char_to_token

    def _tokens_match(self, target_token: str, model_token: str, threshold: float = 0.8) -> bool:
        return self._calculate_token_similarity(target_token, model_token) >= threshold

    def _calculate_token_similarity(self, token1: str, token2: str) -> float:
        if not token1 or not token2:
            return 0.0

        clean_token1 = token1.strip().lower()
        clean_token2 = token2.strip().lower()

        if clean_token1 == clean_token2:
            return 1.0

        if clean_token1 in clean_token2 or clean_token2 in clean_token1:
            return 0.9

        return self._edit_distance_similarity(clean_token1, clean_token2)

    def _edit_distance_similarity(self, s1: str, s2: str) -> float:
        if len(s1) == 0 or len(s2) == 0:
            return 0.0

        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0

        common_prefix = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                common_prefix += 1
            else:
                break

        return common_prefix / max_len

    def _search_token_by_text(self, target_token: str, input_ids: torch.Tensor) -> List[int]:
        positions = []

        for i, token_id in enumerate(input_ids):
            if self._is_special_token(token_id):
                continue

            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            if self._tokens_match(target_token, token_text, threshold=0.6):
                positions.append(i)

        return positions[:1]

    def _fallback_token_alignment(self, code: str, target_token: str) -> List[int]:
        try:
            encoded = self.tokenizer(code, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"][0]

            target_encoded = self.tokenizer(target_token, add_special_tokens=False)["input_ids"]

            if not target_encoded:
                return []

            positions = []
            for i in range(len(input_ids) - len(target_encoded) + 1):
                if input_ids[i:i + len(target_encoded)].tolist() == target_encoded:
                    positions.extend(range(i, i + len(target_encoded)))

            if positions:
                return positions[:1]

            return self._search_token_by_text(target_token, input_ids)

        except Exception as e:
            return []

    def validate_token_alignment(self, code: str, target_token: str, positions: List[int],
                               confidence: float) -> Dict[str, Any]:
        validation_result = {
            'is_valid': False,
            'confidence': confidence,
            'issues': [],
            'recommendations': []
        }

        try:
            if not positions:
                validation_result['issues'].append("No alignment positions found")
                validation_result['recommendations'].append("Check if token text is correct")
                return validation_result

            encoded = self.tokenizer(code, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"][0]

            for pos in positions:
                if pos >= len(input_ids):
                    validation_result['issues'].append(f"Position {pos} exceeds sequence length {len(input_ids)}")
                    continue

                aligned_token = self.tokenizer.decode([input_ids[pos]], skip_special_tokens=True)
                similarity = self._calculate_token_similarity(target_token, aligned_token)

                if similarity < 0.5:
                    validation_result['issues'].append(
                        f"Position {pos} token '{aligned_token}' similarity to target token '{target_token}' too low: {similarity:.3f}"
                    )
                elif similarity < 0.8:
                    validation_result['recommendations'].append(
                        f"Position {pos} match quality is medium, similarity: {similarity:.3f}"
                    )

            if len(validation_result['issues']) == 0:
                validation_result['is_valid'] = True
            elif confidence > 0.7 and len(validation_result['issues']) <= 1:
                validation_result['is_valid'] = True
                validation_result['recommendations'].append("Alignment quality acceptable, but pay attention to potential issues")

            return validation_result

        except Exception as e:
            validation_result['issues'].append(f"Validation process error: {str(e)}")
            return validation_result

    def get_alignment_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        stats = {
            'total_samples': len(samples),
            'successful_alignments': 0,
            'failed_alignments': 0,
            'average_confidence': 0.0,
            'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'common_issues': [],
            'alignment_methods': {}
        }

        confidences = []
        issues_counter = {}

        for sample in samples:
            alignment_result = self._align_ast_tokens_with_model_tokens(
                sample['code'],
                sample.get('token', ''),
                sample.get('ast_start_byte'),
                sample.get('ast_end_byte')
            )

            if alignment_result['success']:
                stats['successful_alignments'] += 1
                confidence = alignment_result['confidence']
                confidences.append(confidence)

                if confidence >= 0.8:
                    stats['confidence_distribution']['high'] += 1
                elif confidence >= 0.6:
                    stats['confidence_distribution']['medium'] += 1
                else:
                    stats['confidence_distribution']['low'] += 1

                method = alignment_result.get('method', 'unknown')
                stats['alignment_methods'][method] = stats['alignment_methods'].get(method, 0) + 1

            else:
                stats['failed_alignments'] += 1
                error = alignment_result.get('error', 'unknown error')
                issues_counter[error] = issues_counter.get(error, 0) + 1

        if confidences:
            stats['average_confidence'] = sum(confidences) / len(confidences)

        stats['common_issues'] = sorted(issues_counter.items(), key=lambda x: x[1], reverse=True)[:5]

        return stats

    def save_representations(self, representations: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        serializable_data = {
            'target_layers': representations['target_layers'],
            'total_samples': representations['total_samples'],
            'representations': []
        }
        
        for repr_data in representations['representations']:
            sample_data = {
                'sample_info': repr_data['sample_info'],
                'token_positions': repr_data['token_positions'],
                'layer_representations': {}
            }
            
            for layer_idx, layer_repr in repr_data['representations'].items():
                sample_data['layer_representations'][str(layer_idx)] = layer_repr.numpy()
            
            serializable_data['representations'].append(sample_data)
        
        np.savez_compressed(output_path, **serializable_data)
    
    def load_representations(self, input_path: str) -> Dict[str, Any]:
        data = np.load(input_path, allow_pickle=True)
        
        representations = {
            'target_layers': data['target_layers'].tolist(),
            'total_samples': int(data['total_samples']),
            'representations': []
        }
        
        for repr_data in data['representations']:
            sample_data = {
                'sample_info': repr_data['sample_info'],
                'token_positions': repr_data['token_positions'],
                'representations': {}
            }
            
            for layer_idx_str, layer_repr in repr_data['layer_representations'].items():
                layer_idx = int(layer_idx_str)
                sample_data['representations'][layer_idx] = torch.from_numpy(layer_repr)
            
            representations['representations'].append(sample_data)
        
        return representations