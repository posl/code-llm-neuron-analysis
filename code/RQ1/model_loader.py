#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizer, Qwen2ForCausalLM, Qwen2Tokenizer

def load_model_and_tokenizer(model_path, device="cuda", use_quantized=True, load_in_8bit=False, load_in_4bit=False, device_map=None):
    
    quantization_config = None
    torch_dtype = None

    if not use_quantized:
        torch_dtype = torch.float32
    elif load_in_8bit:
        quantization_config = {"load_in_8bit": True}
        torch_dtype = None
    elif load_in_4bit:
        quantization_config = {"load_in_4bit": True}
        torch_dtype = None
    else:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        actual_device_map = None
        if device_map:
            actual_device_map = device_map
        elif device == "cuda":
            actual_device_map = "auto"
        
        if "llama" in model_path.lower():
            model_kwargs = {
                "device_map": actual_device_map,
                "low_cpu_mem_usage": True,
            }

            if quantization_config:
                if "load_in_8bit" in quantization_config:
                    model_kwargs["load_in_8bit"] = True
                elif "load_in_4bit" in quantization_config:
                    model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["torch_dtype"] = torch_dtype

            model = LlamaForCausalLM.from_pretrained(model_path, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        elif "qwen" in model_path.lower():
            model_kwargs = {
                "device_map": actual_device_map,
                "low_cpu_mem_usage": True,
            }

            if quantization_config:
                if "load_in_8bit" in quantization_config:
                    model_kwargs["load_in_8bit"] = True
                elif "load_in_4bit" in quantization_config:
                    model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["torch_dtype"] = torch_dtype

            model = Qwen2ForCausalLM.from_pretrained(model_path, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            model_kwargs = {
                "device_map": actual_device_map,
                "low_cpu_mem_usage": True,
            }

            if quantization_config:
                if "load_in_8bit" in quantization_config:
                    model_kwargs["load_in_8bit"] = True
                elif "load_in_4bit" in quantization_config:
                    model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["torch_dtype"] = torch_dtype

            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer

    except Exception as e:
        raise

def get_model_hidden_layers(model):
    model_type = "unknown"
    if "llama" in str(type(model)).lower():
        model_type = "llama"
    elif "qwen" in str(type(model)).lower():
        model_type = "qwen"
    elif "gpt" in str(type(model)).lower():
        model_type = "gpt"
    elif "bert" in str(type(model)).lower():
        model_type = "bert"

    hidden_layers = {}

    if model_type == "llama":
        try:
            layers = model.model.layers
            for i, layer in enumerate(layers):
                hidden_layers[f"layer_{i}"] = layer
            return hidden_layers
        except AttributeError:
            pass
    elif model_type == "qwen":
        try:
            layers = model.model.layers
            for i, layer in enumerate(layers):
                hidden_layers[f"layer_{i}"] = layer
            return hidden_layers
        except AttributeError:
            pass

    try:
        for name, module in model.named_modules():
            if name == "" or "." not in name:
                continue

            if any(layer_type in name.lower() for layer_type in ["layer", "block", "transformer", "encoder"]):
                if hasattr(module, "mlp") or hasattr(module, "ffn") or hasattr(module, "feed_forward"):
                    hidden_layers[name] = module
    except Exception:
        pass

    return hidden_layers

def _create_enhanced_hook(layer_key, activation_dict, collect_token_level=True):
    """Create an enhanced hook for capturing layer activations."""
    def hook(module, input, output):
        try:
            output_cpu = output.detach().cpu()

            if collect_token_level and output_cpu.dim() >= 3:
                token_level_key = f"{layer_key}_token_level"
                if token_level_key not in activation_dict:
                    activation_dict[token_level_key] = []

                max_tokens = min(100, output_cpu.size(1))
                token_activations = output_cpu[0, :max_tokens, :].clone()
                activation_dict[token_level_key].append(token_activations)

                if token_activations.size(0) > 1:
                    token_diffs = torch.norm(
                        token_activations[1:] - token_activations[:-1],
                        dim=1
                    )
                    gradient_key = f"{layer_key}_token_gradient"
                    if gradient_key not in activation_dict:
                        activation_dict[gradient_key] = []
                    activation_dict[gradient_key].append(token_diffs)

            processed_output = None
            if output_cpu.dim() >= 3:
                if output_cpu.size(-2) > 0 and output_cpu.size(-1) > 0:
                    processed_output = torch.mean(output_cpu, dim=-2)
                    if processed_output.dim() != 2:
                        processed_output = None
            elif output_cpu.dim() == 2:
                processed_output = output_cpu

            if processed_output is not None and processed_output.dim() == 2:
                if layer_key not in activation_dict:
                    activation_dict[layer_key] = []
                activation_dict[layer_key].append(processed_output)

                stats_key = f"{layer_key}_stats"
                if stats_key not in activation_dict:
                    activation_dict[stats_key] = []

                mean = torch.mean(processed_output, dim=1)
                std = torch.std(processed_output, dim=1)
                max_val, _ = torch.max(processed_output, dim=1)
                min_val, _ = torch.min(processed_output, dim=1)

                stats = torch.stack([mean, std, max_val, min_val], dim=1)
                activation_dict[stats_key].append(stats)

        except Exception:
            pass

    return hook

def _create_attention_hook(layer_idx, activation_dict, collect_attention=True):
    """Create an attention hook for capturing attention weights."""
    def hook(module, input, output):
        try:
            if not collect_attention:
                return

            attn_weights = None

            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
            elif hasattr(output, 'attentions') and output.attentions is not None:
                attn_weights = output.attentions

            if attn_weights is not None:
                attn_weights_cpu = attn_weights.detach().cpu()

                attn_key = f"layer_{layer_idx}_attention_weights"
                if attn_key not in activation_dict:
                    activation_dict[attn_key] = []

                activation_dict[attn_key].append(attn_weights_cpu[0])

                if attn_weights_cpu.dim() == 4:
                    entropy = torch.zeros(attn_weights_cpu.size(1))
                    for head_idx in range(attn_weights_cpu.size(1)):
                        head_weights = attn_weights_cpu[0, head_idx]
                        head_weights = head_weights / (head_weights.sum(dim=-1, keepdim=True) + 1e-10)
                        head_entropy = -torch.sum(
                            head_weights * torch.log(head_weights + 1e-10),
                            dim=-1
                        ).mean()
                        entropy[head_idx] = head_entropy

                    entropy_key = f"layer_{layer_idx}_attention_entropy"
                    if entropy_key not in activation_dict:
                        activation_dict[entropy_key] = []
                    activation_dict[entropy_key].append(entropy)

        except Exception:
            pass

    return hook

def _register_layer_hooks(layer, idx, activation_dict, hooks, collect_token_level=True, collect_attention=True):
    """Register hooks for a single layer (works for both Llama and Qwen)."""
    # MLP hooks
    handle1 = layer.mlp.down_proj.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_mlp_down", activation_dict, collect_token_level))
    handle2 = layer.mlp.gate_proj.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_mlp_gate", activation_dict, collect_token_level))
    handle3 = layer.mlp.up_proj.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_mlp_up", activation_dict, collect_token_level))
    hooks.extend([handle1, handle2, handle3])

    # Attention hooks
    handle4 = layer.self_attn.q_proj.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_attn_q", activation_dict, collect_token_level))
    handle5 = layer.self_attn.k_proj.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_attn_k", activation_dict, collect_token_level))
    handle6 = layer.self_attn.v_proj.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_attn_v", activation_dict, collect_token_level))
    handle7 = layer.self_attn.o_proj.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_attn_o", activation_dict, collect_token_level))
    hooks.extend([handle4, handle5, handle6, handle7])

    # Attention weight hook
    if collect_attention and hasattr(layer.self_attn, 'forward'):
        try:
            handle_attn = layer.self_attn.register_forward_hook(
                _create_attention_hook(idx, activation_dict, collect_attention))
            hooks.append(handle_attn)
        except Exception:
            pass

    # LayerNorm hooks
    handle8 = layer.input_layernorm.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_input_norm", activation_dict, collect_token_level))
    handle9 = layer.post_attention_layernorm.register_forward_hook(
        _create_enhanced_hook(f"layer_{idx}_post_attn_norm", activation_dict, collect_token_level))
    hooks.extend([handle8, handle9])

def get_activation_hooks(model, activation_dict, collect_token_level=True, collect_attention=True):
    hooks = []
    hidden_layers = get_model_hidden_layers(model)

    # Handle Llama and Qwen models with unified code
    if any(model_type in str(type(model)).lower() for model_type in ["llama", "qwen"]):
        for idx, layer in enumerate(model.model.layers):
            _register_layer_hooks(layer, idx, activation_dict, hooks, collect_token_level, collect_attention)

    else:
        # Handle other model types
        for name, layer in hidden_layers.items():
            layer_idx = -1
            for part in name.split('_'):
                if part.isdigit():
                    layer_idx = int(part)
                    break

            handle = layer.register_forward_hook(
                _create_enhanced_hook(name, activation_dict, collect_token_level))
            hooks.append(handle)

    return hooks

def _process_layer_neurons(layer, layer_idx, neuron_positions):
    """Process neuron positions for a single layer (works for both Llama and Qwen)."""
    neuron_positions[layer_idx] = {}

    if hasattr(layer, "mlp"):
        # Process MLP projections
        projections = [
            ("gate_proj", "mlp_gate"),
            ("up_proj", "mlp_up"),
            ("down_proj", "mlp_down")
        ]
        
        for proj_name, component in projections:
            if hasattr(layer.mlp, proj_name):
                proj = getattr(layer.mlp, proj_name)
                if hasattr(proj, "weight"):
                    weight = proj.weight
                    for neuron_idx in range(weight.size(0)):
                        neuron_positions[layer_idx][f"{proj_name}_{neuron_idx}"] = {
                            "layer_name": f"layer_{layer_idx}",
                            "component": component,
                            "neuron_idx": neuron_idx,
                            "weight_shape": list(weight.shape)
                        }

    if hasattr(layer, "self_attn"):
        # Process attention projections
        projections = [
            ("q_proj", "attn_q"),
            ("k_proj", "attn_k"),
            ("v_proj", "attn_v"),
            ("o_proj", "attn_o")
        ]
        
        for proj_name, component in projections:
            if hasattr(layer.self_attn, proj_name):
                proj = getattr(layer.self_attn, proj_name)
                if hasattr(proj, "weight"):
                    weight = proj.weight
                    for neuron_idx in range(weight.size(0)):
                        neuron_positions[layer_idx][f"{proj_name}_{neuron_idx}"] = {
                            "layer_name": f"layer_{layer_idx}",
                            "component": component,
                            "neuron_idx": neuron_idx,
                            "weight_shape": list(weight.shape)
                        }

def get_neuron_positions(model):
    hidden_layers = get_model_hidden_layers(model)
    neuron_positions = {}

    # Handle Llama and Qwen models with unified code
    if any(model_type in str(type(model)).lower() for model_type in ["llama", "qwen"]):
        for layer_idx, layer in enumerate(model.model.layers):
            _process_layer_neurons(layer, layer_idx, neuron_positions)
        return neuron_positions

    for layer_name, layer in hidden_layers.items():
        layer_idx = int(layer_name.split("_")[1])
        neuron_positions[layer_idx] = {}

        if hasattr(layer, "mlp") and hasattr(layer.mlp, "up_proj"):
            if hasattr(layer.mlp.up_proj, "weight"):
                weight = layer.mlp.up_proj.weight
                for neuron_idx in range(weight.size(0)):
                    neuron_positions[layer_idx][neuron_idx] = {
                        "layer_name": layer_name,
                        "neuron_idx": neuron_idx,
                        "weight_shape": list(weight.shape)
                    }
        elif hasattr(layer, "mlp") and hasattr(layer.mlp, "c_fc"):
            if hasattr(layer.mlp.c_fc, "weight"):
                weight = layer.mlp.c_fc.weight
                for neuron_idx in range(weight.size(0)):
                    neuron_positions[layer_idx][neuron_idx] = {
                        "layer_name": layer_name,
                        "neuron_idx": neuron_idx,
                        "weight_shape": list(weight.shape)
                    }

    return neuron_positions
