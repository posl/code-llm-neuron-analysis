import torch
import gc
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class CodeCloneRepresentationExtractor:
    
    def __init__(self, model_path: str, target_layers: List[int], device: str = "cuda",
                 pooling_method: str = "attention_weighted"):
        self.model_path = model_path
        self.target_layers = target_layers
        self.device = device
        self.pooling_method = pooling_method  # "mean", "attention_weighted"
        self.model = None
        self.tokenizer = None
        self.hooks = []
        self.activations = {}
        self.attention_weights = {}
        
    def load_model(self):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.model.eval()
        
    def _register_hooks(self):
        self.hooks = []
        self.activations = {}
        self.attention_weights = {}

        for layer_idx in self.target_layers:
            layer = self._get_layer_by_index(layer_idx)
            if layer is not None:
                hook = layer.register_forward_hook(self._get_activation_hook(layer_idx))
                self.hooks.append(hook)

                if self.pooling_method == "attention_weighted":
                    attn_layer = self._get_attention_layer_by_index(layer_idx)
                    if attn_layer is not None:
                        attn_hook = attn_layer.register_forward_hook(self._get_attention_hook(layer_idx))
                        self.hooks.append(attn_hook)
    
    def _get_layer_by_index(self, layer_idx: int):
        try:
            if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
                return self.model.model.layers[layer_idx]
            elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
                return self.model.transformer.h[layer_idx]
            elif hasattr(self.model, "layers"):
                return self.model.layers[layer_idx]
            else:
                return None
        except (IndexError, AttributeError):
            return None

    def _get_attention_layer_by_index(self, layer_idx: int):
        try:
            layer = self._get_layer_by_index(layer_idx)
            if layer is None:
                return None

            if hasattr(layer, "self_attn"):
                return layer.self_attn
            elif hasattr(layer, "attention"):
                return layer.attention
            elif hasattr(layer, "attn"):
                return layer.attn
            else:
                return None
        except (IndexError, AttributeError):
            return None
    
    def _get_activation_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            self.activations[layer_idx] = hidden_states.detach().cpu().float()

        return hook

    def _get_attention_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attention_weights = output[1]
                if attention_weights is not None:
                    self.attention_weights[layer_idx] = attention_weights.detach().cpu().float()

        return hook
    
    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
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
        
        return int(token_id) in special_token_ids

    def _filter_abnormal_dimensions(self, hidden_states: torch.Tensor, layer_idx: int) -> torch.Tensor:
        # Filter dimension 2352 anomaly starting from layer 21
        if layer_idx >= 21 and hidden_states.shape[-1] > 2352:
            filtered_states = hidden_states.clone()
            filtered_states[:, :, 2352] = 0.0
            return filtered_states

        return hidden_states

    def _compute_attention_weights_pooling(self, hidden_states: torch.Tensor,
                                         attention_weights: torch.Tensor, layer_idx: int) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape

        hidden_states = self._filter_abnormal_dimensions(hidden_states, layer_idx)

        avg_attention = attention_weights.mean(dim=1)

        token_importance = avg_attention.sum(dim=1)

        # Skip begin_of_text token
        if seq_len > 1:
            token_importance = token_importance[:, 1:]
            hidden_states_filtered = hidden_states[:, 1:, :]
        else:
            hidden_states_filtered = hidden_states

        token_importance = torch.softmax(token_importance, dim=1)

        pooled_repr = (token_importance.unsqueeze(-1) * hidden_states_filtered).sum(dim=1)

        return pooled_repr
    
    def extract_code_representations(self, code: str, max_length: int = None) -> Dict[int, torch.Tensor]:
        self._register_hooks()
        
        try:
            if max_length is not None:
                inputs = self.tokenizer(
                    code,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=False
                )
            else:
                inputs = self.tokenizer(
                    code,
                    return_tensors="pt",
                    truncation=False,
                    padding=False
                )
            
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                output_attentions = (self.pooling_method == "attention_weighted")

                if attention_mask is not None:
                    outputs = self.model(input_ids, attention_mask=attention_mask,
                                       output_attentions=output_attentions)
                else:
                    outputs = self.model(input_ids, output_attentions=output_attentions)
            
            representations = {}

            for layer_idx in self.target_layers:
                if layer_idx in self.activations:
                    hidden_state = self.activations[layer_idx]

                    min_val = hidden_state.min().item()
                    max_val = hidden_state.max().item()

                    if abs(min_val) > 100 or abs(max_val) > 100:
                        hidden_state = torch.clamp(hidden_state, min=-50, max=50)

                    if self.pooling_method == "attention_weighted" and layer_idx in self.attention_weights:
                        attention_weights = self.attention_weights[layer_idx]
                        pooled_repr = self._compute_attention_weights_pooling(hidden_state, attention_weights, layer_idx)
                    else:
                        hidden_state = self._filter_abnormal_dimensions(hidden_state, layer_idx)

                        if hidden_state.shape[1] > 1:
                            # Skip begin_of_text token
                            pooled_repr = hidden_state[:, 1:, :].mean(dim=1)
                        else:
                            pooled_repr = hidden_state.mean(dim=1)

                    representations[layer_idx] = pooled_repr.squeeze(0).float()
            
            return representations
            
        except Exception:
            return {}
        finally:
            self._remove_hooks()
            self.activations = {}
            self.attention_weights = {}
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def extract_code_pair_representations(self, code_a: str, code_b: str) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
        repr_a = self.extract_code_representations(code_a)
        repr_b = self.extract_code_representations(code_b)
        
        return repr_a, repr_b
    
    def batch_extract_representations(self, code_pairs: List[Dict[str, Any]], batch_size: int = 4) -> List[Dict[str, Any]]:
        results = []
        
        for i in tqdm(range(0, len(code_pairs), batch_size), desc="Extracting representations"):
            batch_pairs = code_pairs[i:i + batch_size]
            
            for pair in batch_pairs:
                try:
                    repr_a, repr_b = self.extract_code_pair_representations(
                        pair['codeA'], pair['codeB']
                    )
                    
                    result = {
                        'pair_info': pair,
                        'representations_a': repr_a,
                        'representations_b': repr_b
                    }
                    results.append(result)
                    
                except Exception:
                    continue
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results