#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import Dict, Any, Callable, List, Tuple
import random

class NeuronInterventionHook:
    def __init__(self, module: nn.Module, intervention_fn: Callable, neurons: Dict[str, Any]):
        self.handle = module.register_forward_hook(intervention_fn)
        self.module = module
        self.neurons = neurons

    def remove(self):
        self.handle.remove()

def parse_neuron_key(neuron_key):
    if "_neuron_" not in neuron_key:
        raise ValueError(f"Unsupported neuron key format for intervention (missing '_neuron_'): {neuron_key}")

    parts = neuron_key.split("_neuron_")
    layer_name_part = parts[0]

    try:
        neuron_idx = int(parts[1])
    except (ValueError, IndexError):
        raise ValueError(f"Cannot extract neuron index from key: {neuron_key}")

    name_parts = layer_name_part.split("_")
    layer_idx = None
    layer_idx_pos = -1

    for i, part in enumerate(name_parts):
        if part.isdigit():
            layer_idx = int(part)
            layer_idx_pos = i
            break
    
    if layer_idx is None:
        raise ValueError(f"Cannot extract layer index from layer part: {layer_name_part}")

    if len(name_parts) > layer_idx_pos + 1:
        component = "_".join(name_parts[layer_idx_pos + 1:])
        if not component:
            raise ValueError(f"Empty component name extracted from key: {neuron_key}")
        return layer_idx, component, neuron_idx
    else:
        raise ValueError(f"Neuron key lacks specific component information: {neuron_key}")


def get_layer_by_index(model, layer_idx):
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        elif hasattr(model, "model") and hasattr(model.model, "layers") and "qwen" in str(type(model)).lower():
            return model.model.layers[layer_idx]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer[layer_idx]
        elif hasattr(model, "layers"):
            return model.layers[layer_idx]
        else:
            return None
    except (IndexError, AttributeError):
        return None

def get_module_by_layer_and_component(model, layer_idx, component):
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            layer = model.model.layers[layer_idx]
        elif hasattr(model, "model") and hasattr(model.model, "layers") and "qwen" in str(type(model)).lower():
            layer = model.model.layers[layer_idx]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            layer = model.transformer.h[layer_idx]
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layer = model.encoder.layer[layer_idx]
        elif hasattr(model, "layers"):
            layer = model.layers[layer_idx]
        else:
            return None
        component_map = {
            "attn_q": ("self_attn", "q_proj"),
            "attn_k": ("self_attn", "k_proj"),
            "attn_v": ("self_attn", "v_proj"),
            "attn_o": ("self_attn", "o_proj"),
            "self_attn_q_proj": ("self_attn", "q_proj"),
            "self_attn_k_proj": ("self_attn", "k_proj"),
            "self_attn_v_proj": ("self_attn", "v_proj"),
            "self_attn_o_proj": ("self_attn", "o_proj"),
            "attention_q_proj": ("self_attn", "q_proj"),
            "attention_k_proj": ("self_attn", "k_proj"),
            "attention_v_proj": ("self_attn", "v_proj"),
            "attention_o_proj": ("self_attn", "o_proj"),
            "mlp_gate": ("mlp", "gate_proj"),
            "mlp_up": ("mlp", "up_proj"),
            "mlp_down": ("mlp", "down_proj"),
            "mlp_down_proj": ("mlp", "down_proj"),
            "mlp_gate_proj": ("mlp", "gate_proj"),
            "mlp_up_proj": ("mlp", "up_proj"),
            "input_norm": ("input_layernorm", None),
            "input_ln": ("input_layernorm", None),
            "post_attn_norm": ("post_attention_layernorm", None),
            "post_attn_ln": ("post_attention_layernorm", None),
        }

        try:
            if component in component_map:
                attr1_name, attr2_name = component_map[component]

                if attr2_name:
                    if hasattr(layer, attr1_name):
                        parent_module = getattr(layer, attr1_name)
                        if hasattr(parent_module, attr2_name):
                            return getattr(parent_module, attr2_name)
                        else:
                            return None
                    else:
                        return None
                else:
                     if hasattr(layer, attr1_name):
                         return getattr(layer, attr1_name)
                     else:
                         return None
            else:
                if component == "mlp":
                    if hasattr(layer, "mlp"):
                        return layer.mlp
                    else:
                        return None
                elif component == "mlp_down_proj":
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                        return layer.mlp.down_proj
                    else:
                        return None
                elif component == "mlp_gate_proj":
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate_proj"):
                        return layer.mlp.gate_proj
                    else:
                        return None
                elif component == "mlp_up_proj":
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "up_proj"):
                        return layer.mlp.up_proj
                    else:
                        return None
                elif hasattr(layer, component):
                    return getattr(layer, component)
                else:
                    return None

        except AttributeError:
            return None

    except (IndexError, AttributeError):
        return None

def format_neurons_for_intervention(language_neurons: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    formatted_neurons = {}

    for neuron_key, neuron_info in language_neurons.items():
        specificity = neuron_info.get("specificity", 0)
        selectivity_index = neuron_info.get("selectivity_index", 0)
        importance = max(selectivity_index, specificity)
        if importance < threshold:
            continue
        try:
            if "_neuron_" in neuron_key:
                parts = neuron_key.split("_neuron_")
                layer_name = parts[0]
                neuron_idx = int(parts[1])

                layer_idx = None
                for part in layer_name.split("_"):
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                if layer_idx is not None:
                    formatted_neurons[neuron_key] = {
                        "layer_idx": layer_idx,
                        "neuron_idx": neuron_idx,
                        "importance": importance,
                        "effect_size": neuron_info.get("effect_size", 0),
                        "selectivity_index": selectivity_index,
                        "normalized_entropy": neuron_info.get("normalized_entropy", 1.0)
                    }
        except Exception:
            pass
    sorted_neurons = sorted(
        formatted_neurons.items(),
        key=lambda x: x[1]["importance"],
        reverse=True
    )
    return dict(sorted_neurons)

def setup_neuron_intervention(model, language_neurons, intervention_type="zero"):
  model.zero_neuron_intervention = zero_neuron_intervention.__get__(model)

def zero_neuron_intervention(self, language_neurons):
    neurons_by_component = {}
    for neuron_key in language_neurons.items():
        try:
            if isinstance(neuron_key, str) and "_neuron_" in neuron_key:
                layer_idx, component, neuron_idx = parse_neuron_key(neuron_key)
            elif isinstance(neuron_key, str) and "." in neuron_key:
                parts = neuron_key.split(".")
                if len(parts) == 2:
                    layer_idx = int(parts[0])
                    component = "mlp_up"
                    neuron_idx = int(parts[1])
                elif len(parts) == 3:
                    layer_idx = int(parts[0])
                    component = parts[1]
                    neuron_idx = int(parts[2])
                else:
                    continue
            else:
                continue
            grouping_key = (layer_idx, component)
            if grouping_key not in neurons_by_component:
                neurons_by_component[grouping_key] = []
            neurons_by_component[grouping_key].append(neuron_idx)
        except Exception:
            pass
    hooks = []
    total_intervened_neurons = 0
    for grouping_key, neurons in neurons_by_component.items():
        layer_idx, component = grouping_key
        target_module = get_module_by_layer_and_component(self, layer_idx, component)
        if target_module is None:
            continue
        def make_intervention_fn(neurons, layer_idx, component):
            def intervention_fn(module, input_tensors, output):
                output_shape = output.shape
                if component.startswith('ln') or 'layernorm' in component:
                    if len(output_shape) == 3:
                        dim = 2
                        valid_range = output.size(dim) - 1
                    else:
                        dim = output.dim() - 1
                        valid_range = output.size(dim) - 1
                else:
                    if len(output_shape) >= 2:
                        dim = 1 if len(output_shape) == 2 else 2
                        valid_range = output.size(dim) - 1
                    else:
                        dim = 0
                        valid_range = output.size(dim) - 1

                output_clone = output.clone()
                try:
                    for neuron_idx in neurons:
                        if 0 <= neuron_idx <= valid_range:
                            if dim == 0:
                                output_clone[neuron_idx] = 0.0
                            elif dim == 1:
                                output_clone[:, neuron_idx] = 0.0
                            elif dim == 2:
                                output_clone[:, :, neuron_idx] = 0.0

                    return output_clone
                except (IndexError, RuntimeError):
                    return output
            return intervention_fn

        hook = NeuronInterventionHook(
            target_module,
            make_intervention_fn(neurons, layer_idx, component),
            {idx: 0.0 for idx in neurons}
        )
        hooks.append(hook)

        total_intervened_neurons += len(neurons)

    if not hooks:
        return None

    class HookCollection:
        def __init__(self, hooks):
            self.hooks = hooks

        def remove(self):
            for hook in self.hooks:
                hook.remove()

    return HookCollection(hooks)




def get_total_neurons_per_component(model) -> Dict[Tuple[int, str], int]:
    neurons_count = {}
    num_layers = 0

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
    elif hasattr(model, "model") and hasattr(model.model, "layers") and "qwen" in str(type(model)).lower():
        num_layers = len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        num_layers = len(model.transformer.h)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        num_layers = len(model.encoder.layer)
    elif hasattr(model, "layers"):
         num_layers = len(model.layers)
    else:
        return {}

    intervenable_components = [
        "attn_q", "attn_k", "attn_v", "attn_o",
        "mlp_gate", "mlp_up", "mlp_down"
    ]

    for layer_idx in range(num_layers):
        for component_short_name in intervenable_components:
            module = get_module_by_layer_and_component(model, layer_idx, component_short_name)
            if module is not None and hasattr(module, 'weight'):
                num_neurons = 0
                if isinstance(module, nn.Linear):
                    num_neurons = module.out_features
                    if component_short_name in ["mlp_gate", "mlp_up"]:
                         try:
                             intermediate_size = model.config.intermediate_size
                             num_neurons = intermediate_size
                         except AttributeError:
                             pass

                elif hasattr(module, 'weight'):
                     num_neurons = module.weight.shape[0]

                if num_neurons > 0:
                    neurons_count[(layer_idx, component_short_name)] = num_neurons

    return neurons_count


def setup_random_neuron_intervention(model, num_neurons_to_intervene: int, intervention_type: str = "zero", **kwargs):
    total_neurons_map = get_total_neurons_per_component(model)
    if not total_neurons_map:
        return None

    all_neurons_flat = []
    for (layer_idx, component), num_neurons in total_neurons_map.items():
        for neuron_idx in range(num_neurons):
            all_neurons_flat.append((layer_idx, component, neuron_idx))
    
    if num_neurons_to_intervene > len(all_neurons_flat):
        num_neurons_to_intervene = len(all_neurons_flat)

    randomly_selected_neurons_list = random.sample(all_neurons_flat, num_neurons_to_intervene)

    random_neurons_dict = {}
    for i, (layer_idx, component, neuron_idx) in enumerate(randomly_selected_neurons_list):
        neuron_key = f"layer_{layer_idx}_{component}_neuron_{neuron_idx}"
        random_neurons_dict[neuron_key] = {
            "layer_idx": layer_idx,
            "neuron_idx": neuron_idx,
            "component": component,
            "importance": 1.0
        }

    return zero_neuron_intervention(model, random_neurons_dict)


def remove_intervention_hooks(hooks):
    for hook in hooks:
        hook.remove()

def apply_intervention(model, tokenizer, text, language_neurons, intervention_type="zero", device="cuda"):
    hook = setup_neuron_intervention(model, language_neurons, intervention_type)

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    if hook:
        hook.remove()

    return output