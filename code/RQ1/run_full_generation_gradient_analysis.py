#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import torch
import argparse
import gc
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Any
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import gzip

def detect_model_type(model_name: str, model=None) -> str:
    model_name_lower = model_name.lower()
    
    if "qwen" in model_name_lower:
        return "qwen"
    elif "codellama" in model_name_lower:
        return "codellama"
    elif "llama" in model_name_lower:
        if model and hasattr(model, 'config'):
            if hasattr(model.config, '_name_or_path') and 'codellama' in model.config._name_or_path.lower():
                return "codellama"
            if hasattr(model.config, 'model_type') and model.config.model_type == 'codellama':
                return "codellama"
        return "llama"
    else:
        return "unknown"

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"


class FullGenerationGradientFinder:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16,
        max_length: int = 2048,
        batch_size: int = 1,
        offload_to_cpu: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_length = max_length
        self.batch_size = batch_size
        self.offload_to_cpu = offload_to_cpu

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if device == "auto":
            device_map = "auto"
        elif offload_to_cpu:
            device_map = "auto"
        else:
            device_map = {"": device}

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=device_map,
                low_cpu_mem_usage=True
            )

        self.model.eval()

        self.hooks = []
        self.activations = {}
        self.gradients = {}

        self._register_hooks()

    def _register_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        model_type = detect_model_type(self.model_path, self.model)
        
        if model_type == "codellama":
            self._register_llama_hooks()
        elif isinstance(self.model, LlamaForCausalLM):
            self._register_llama_hooks()
        else:
            self._register_generic_hooks()

    def _register_llama_hooks(self):
        """为Llama模型注册hooks"""
        for layer_idx, layer in enumerate(self.model.model.layers):
            # MLP组件
            mlp_components = {
                "gate_proj": layer.mlp.gate_proj,
                "up_proj": layer.mlp.up_proj,
                "down_proj": layer.mlp.down_proj
            }
            self._register_component_hooks(mlp_components, layer_idx, prefix="")
            
            # Attention组件
            attn_components = {
                "q_proj": layer.self_attn.q_proj,
                "k_proj": layer.self_attn.k_proj,
                "v_proj": layer.self_attn.v_proj,
                "o_proj": layer.self_attn.o_proj
            }
            self._register_component_hooks(attn_components, layer_idx, prefix="attention_")

    def _register_component_hooks(self, components, layer_idx, prefix=""):
        """通用的hook注册函数"""
        for comp_name, comp in components.items():
            key_name = f"layer_{layer_idx}_{prefix}{comp_name}"
            
            # 前向hook
            def make_forward_hook(key):
                def hook(module, input, output):
                    self.activations[key] = output.detach()
                return hook
            
            # 反向hook
            def make_backward_hook(key):
                def hook(module, grad_in, grad_out):
                    self.gradients[key] = grad_out[0].detach()
                return hook
            
            forward_hook = comp.register_forward_hook(make_forward_hook(key_name))
            backward_hook = comp.register_full_backward_hook(make_backward_hook(key_name))
            self.hooks.extend([forward_hook, backward_hook])

    def _register_generic_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                def get_activation(name):
                    def hook(module, input, output):
                        self.activations[name] = output.detach()
                    return hook

                def get_gradient(name):
                    def hook(module, grad_in, grad_out):
                        self.gradients[name] = grad_out[0].detach()
                    return hook

                forward_hook = module.register_forward_hook(
                    get_activation(name)
                )
                backward_hook = module.register_full_backward_hook(
                    get_gradient(name)
                )

                self.hooks.extend([forward_hook, backward_hook])

    def compute_gradients_for_full_generation(
        self,
        prompt: str,
        target_code: str,
        language: str,
        pregenerated_code: str = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        self.activations = {}
        self.gradients = {}

        if hasattr(self.model, "hf_device_map"):
            device_values = list(set(self.model.hf_device_map.values()))
            valid_devices = [d for d in device_values if d != "cpu" and not isinstance(d, int)]

            if valid_devices:
                input_device = valid_devices[0]
            else:
                input_device = "cpu"
        else:
            if self.device == "auto":
                input_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                input_device = self.device

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(input_device)

        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        target_inputs = self.tokenizer(target_code, return_tensors="pt")
        target_inputs = target_inputs.to(input_ids.device)
        target_ids = target_inputs.input_ids

        self.model.zero_grad()

        with torch.set_grad_enabled(True):
            if pregenerated_code is not None:
                if pregenerated_code.startswith(prompt):
                    generated_code = pregenerated_code[len(prompt):]
                else:
                    generated_code = pregenerated_code

                generated_inputs = self.tokenizer(generated_code, return_tensors="pt")
                generated_inputs = generated_inputs.to(input_ids.device)
                generated_ids = generated_inputs.input_ids[0]
            else:
                generation_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                }

                if temperature > 0:
                    generation_kwargs.update({
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p
                    })
                else:
                    generation_kwargs.update({
                        "do_sample": False
                    })

                outputs = self.model.generate(**generation_kwargs)
                generated_ids = outputs.sequences[0, input_ids.shape[1]:]

            generated_code = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            combined_input_ids = torch.cat([input_ids, generated_ids.unsqueeze(0)], dim=1)

            model_outputs = self.model(
                input_ids=combined_input_ids,
                attention_mask=torch.ones_like(combined_input_ids),
                return_dict=True
            )

            logits = model_outputs.logits
            min_length = min(logits.size(1) - 1, target_ids.size(1) - 1)

            shift_logits = logits[:, :min_length, :].contiguous()
            shift_labels = target_ids[:, 1:min_length+1].contiguous()

            shift_logits_float32 = shift_logits.to(torch.float32)

            try:
                loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                loss = loss_fct(
                    shift_logits_float32.view(-1, shift_logits_float32.size(-1)),
                    shift_labels.view(-1)
                )

                loss.backward()
            except RuntimeError as e:
                # 尝试使用float16处理
                if "ScalarType" in str(e) or "scalar type" in str(e).lower():
                    shift_logits_f16 = shift_logits.to(torch.float16)
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
                    loss = loss_fct(
                        shift_logits_f16.view(-1, shift_logits_f16.size(-1)),
                        shift_labels.view(-1)
                    )
                    loss.backward()
                else:
                    raise

        gradient_info = self._collect_gradient_info(language)

        return gradient_info

    def _collect_gradient_info(self, language: str) -> Dict[str, Any]:
        result = {
            "language": language,
            "model_path": self.model_path,
        }

        layers_info = {}

        for key in self.activations.keys():
            if key in self.gradients:
                activation = self.activations[key]
                gradient = self.gradients[key]

                try:
                    activation = activation.to(torch.float32)
                    gradient = gradient.to(torch.float32)
                    
                    act_mean = activation.mean(dim=1)
                    grad_mean = gradient.mean(dim=1)

                    importance = (grad_mean * act_mean).abs()

                    importance_np = importance.detach().cpu().numpy()
                except Exception:
                    try:
                        cpu_activation = activation.detach().cpu().to(torch.float32)
                        cpu_gradient = gradient.detach().cpu().to(torch.float32)

                        act_mean = cpu_activation.mean(dim=1)
                        grad_mean = cpu_gradient.mean(dim=1)
                        importance = (grad_mean * act_mean).abs()
                        importance_np = importance.numpy()
                    except Exception:
                        continue

                neurons = {}
                for neuron_idx in range(importance_np.shape[1]):
                    neuron_importance = float(importance_np[0, neuron_idx])
                    
                    neuron_activation = float(act_mean[0, neuron_idx])
                    neuron_gradient = float(grad_mean[0, neuron_idx])

                    if neuron_importance > 0:
                        neurons[f"neuron_{neuron_idx}"] = {
                            "neuron_idx": neuron_idx,
                            "importance": neuron_importance,
                            "activation": neuron_activation,
                            "gradient": neuron_gradient,
                            "importance_by_language": {
                                language: neuron_importance
                            },
                            "activation_by_language": {
                                language: neuron_activation
                            },
                            "gradient_by_language": {
                                language: neuron_gradient
                            }
                        }

                layer_info = {
                    "component_type": key.split("_")[-1] if "_" in key else "unknown",
                    "neurons": neurons,
                    "activation": act_mean.mean().item(),
                    "gradient": grad_mean.mean().item(),
                    "importance": importance.mean().item()
                }

                layers_info[key] = layer_info

        result["layers"] = layers_info

        return result

    def analyze_multiple_samples(
        self,
        samples: List[Dict[str, Any]],
        language: str,
        use_pregenerated_code: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.9,
    ) -> Dict[str, Any]:
        combined_result = {
            "language": language,
            "model_path": self.model_path,
            "layers": {}
        }

        sample_count = 0
        for task_idx, sample in enumerate(tqdm(samples, desc=f"Processing {language} tasks")):
            try:
                prompt = sample["prompt"]
                solution = sample.get("solution", sample.get("canonical_solution", ""))

                if use_pregenerated_code and "generations" in sample and sample["generations"]:
                    generations = sample["generations"]

                    for gen_idx, pregenerated_code in enumerate(generations):
                        try:
                            gradient_info = self.compute_gradients_for_full_generation(
                                prompt=prompt,
                                target_code=solution,
                                language=language,
                                pregenerated_code=pregenerated_code,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p
                            )

                            self._merge_gradient_info(combined_result, gradient_info, language, sample_count)
                            sample_count += 1

                            torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            continue
                else:
                    if "generation" in sample:
                        pregenerated_code = sample["generation"]

                        try:
                            gradient_info = self.compute_gradients_for_full_generation(
                                prompt=prompt,
                                target_code=solution,
                                language=language,
                                pregenerated_code=pregenerated_code,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p
                            )

                            self._merge_gradient_info(combined_result, gradient_info, language, sample_count)
                            sample_count += 1
                        except Exception:
                            continue
                    else:
                        try:
                            gradient_info = self.compute_gradients_for_full_generation(
                                prompt=prompt,
                                target_code=solution,
                                language=language,
                                pregenerated_code=None,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature,
                                top_p=top_p
                            )

                            self._merge_gradient_info(combined_result, gradient_info, language, sample_count)
                            sample_count += 1
                        except Exception:
                            continue
            except Exception:
                continue

            torch.cuda.empty_cache()
            gc.collect()

        return combined_result

    def _merge_gradient_info(self, combined_result: Dict[str, Any], gradient_info: Dict[str, Any], language: str, sample_count: int):
        if sample_count == 0:
            combined_result["layers"] = gradient_info["layers"]
        else:
            for layer_key, layer_info in gradient_info["layers"].items():
                if layer_key not in combined_result["layers"]:
                    combined_result["layers"][layer_key] = layer_info
                else:
                    for neuron_key, neuron_info in layer_info["neurons"].items():
                        if neuron_key not in combined_result["layers"][layer_key]["neurons"]:
                            combined_result["layers"][layer_key]["neurons"][neuron_key] = neuron_info
                        else:
                            existing_neuron = combined_result["layers"][layer_key]["neurons"][neuron_key]

                            existing_importance = existing_neuron["importance"]
                            new_importance = neuron_info["importance"]
                            combined_importance = (existing_importance * sample_count + new_importance) / (sample_count + 1)
                            existing_neuron["importance"] = combined_importance

                            if language in existing_neuron["importance_by_language"]:
                                existing_lang_importance = existing_neuron["importance_by_language"][language]
                                new_lang_importance = neuron_info["importance_by_language"][language]
                                combined_lang_importance = (existing_lang_importance * sample_count + new_lang_importance) / (sample_count + 1)
                                existing_neuron["importance_by_language"][language] = combined_lang_importance
                            else:
                                existing_neuron["importance_by_language"][language] = neuron_info["importance_by_language"][language]
                                
                            if "activation" in neuron_info:
                                if "activation" not in existing_neuron:
                                    existing_neuron["activation"] = neuron_info["activation"]
                                else:
                                    existing_neuron["activation"] = (existing_neuron["activation"] * sample_count + neuron_info["activation"]) / (sample_count + 1)
                                
                                if "activation_by_language" in neuron_info:
                                    if "activation_by_language" not in existing_neuron:
                                        existing_neuron["activation_by_language"] = {}
                                    
                                    if language in existing_neuron["activation_by_language"]:
                                        existing_lang_activation = existing_neuron["activation_by_language"][language]
                                        new_lang_activation = neuron_info["activation_by_language"][language]
                                        combined_lang_activation = (existing_lang_activation * sample_count + new_lang_activation) / (sample_count + 1)
                                        existing_neuron["activation_by_language"][language] = combined_lang_activation
                                    else:
                                        existing_neuron["activation_by_language"][language] = neuron_info["activation_by_language"][language]
                            
                            if "gradient" in neuron_info:
                                if "gradient" not in existing_neuron:
                                    existing_neuron["gradient"] = neuron_info["gradient"]
                                else:
                                    existing_neuron["gradient"] = (existing_neuron["gradient"] * sample_count + neuron_info["gradient"]) / (sample_count + 1)
                                
                                if "gradient_by_language" in neuron_info:
                                    if "gradient_by_language" not in existing_neuron:
                                        existing_neuron["gradient_by_language"] = {}
                                    
                                    if language in existing_neuron["gradient_by_language"]:
                                        existing_lang_gradient = existing_neuron["gradient_by_language"][language]
                                        new_lang_gradient = neuron_info["gradient_by_language"][language]
                                        combined_lang_gradient = (existing_lang_gradient * sample_count + new_lang_gradient) / (sample_count + 1)
                                        existing_neuron["gradient_by_language"][language] = combined_lang_gradient
                                    else:
                                        existing_neuron["gradient_by_language"][language] = neuron_info["gradient_by_language"][language]

    def save_results(self, results: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def save_combined_results(self, all_language_results: Dict[str, Dict[str, Any]], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        combined_result = {}

        all_layers = set()
        for language, language_results in all_language_results.items():
            if "layers" in language_results:
                all_layers.update(language_results["layers"].keys())

        for layer_name in all_layers:
            layer_info = self._parse_layer_name(layer_name)

            combined_result[layer_name] = {
                "layer_info": layer_info,
                "importance_by_language": {},
                "neurons": {}
            }

            for language, language_results in all_language_results.items():
                if "layers" in language_results and layer_name in language_results["layers"]:
                    layer_data = language_results["layers"][layer_name]

                    combined_result[layer_name]["importance_by_language"][language] = {}

                    if "activation" in layer_data:
                        combined_result[layer_name]["importance_by_language"][language]["activation"] = float(layer_data["activation"])
                    if "gradient" in layer_data:
                        combined_result[layer_name]["importance_by_language"][language]["gradient"] = float(layer_data["gradient"])
                    if "importance" in layer_data:
                        combined_result[layer_name]["importance_by_language"][language]["importance"] = float(layer_data["importance"])
                    
                    if "neurons" in layer_data:
                        for neuron_key, neuron_info in layer_data["neurons"].items():
                            neuron_idx = neuron_info["neuron_idx"]

                            detailed_neuron_key = f"{layer_name}_neuron_{neuron_idx}"

                            if detailed_neuron_key not in combined_result[layer_name]["neurons"]:
                                combined_result[layer_name]["neurons"][detailed_neuron_key] = {
                                    "layer_name": layer_name,
                                    "neuron_idx": neuron_idx,
                                    "component_type": layer_info["component_type"],
                                    "layer_idx": layer_info["layer_idx"],
                                    "importance_by_language": {},
                                    "gradient_by_language": {},
                                    "activation_by_language": {},
                                    "detailed_key": detailed_neuron_key
                                }

                            if "importance_by_language" in neuron_info and language in neuron_info["importance_by_language"]:
                                combined_result[layer_name]["neurons"][detailed_neuron_key]["importance_by_language"][language] = float(neuron_info["importance_by_language"][language])
                            else:
                                combined_result[layer_name]["neurons"][detailed_neuron_key]["importance_by_language"][language] = 0.0

                            if "gradient_by_language" in neuron_info and language in neuron_info["gradient_by_language"]:
                                combined_result[layer_name]["neurons"][detailed_neuron_key]["gradient_by_language"][language] = float(neuron_info["gradient_by_language"][language])
                            else:
                                combined_result[layer_name]["neurons"][detailed_neuron_key]["gradient_by_language"][language] = 0.0

                            if "activation_by_language" in neuron_info and language in neuron_info["activation_by_language"]:
                                combined_result[layer_name]["neurons"][detailed_neuron_key]["activation_by_language"][language] = float(neuron_info["activation_by_language"][language])
                            else:
                                combined_result[layer_name]["neurons"][detailed_neuron_key]["activation_by_language"][language] = 0.0

        combined_result["_metadata"] = {
            "model_path": self.model_path,
            "num_languages": len(all_language_results),
            "languages": list(all_language_results.keys()),
            "num_layers": len(all_layers),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "combined_format": True,
            "format_version": "2.0"
        }

        total_neurons = 0
        for layer_name, layer_data in combined_result.items():
            if layer_name == "_metadata":
                continue
            if "neurons" in layer_data:
                total_neurons += len(layer_data["neurons"])

        combined_result["_metadata"]["num_neurons"] = total_neurons

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined_result, f, indent=2, ensure_ascii=False)

        return output_path

    def _parse_layer_name(self, layer_name):
        layer_info = {}

        if "layer_" in layer_name:
            try:
                layer_idx_str = layer_name.split("layer_")[1].split("_")[0]
                layer_info["layer_idx"] = int(layer_idx_str)
            except (IndexError, ValueError):
                layer_info["layer_idx"] = -1
        else:
            layer_info["layer_idx"] = -1

        if "_gate_proj" in layer_name:
            layer_info["component_type"] = "gate_proj"
        elif "_up_proj" in layer_name:
            layer_info["component_type"] = "up_proj"
        elif "_down_proj" in layer_name:
            layer_info["component_type"] = "down_proj"
        elif "_q_proj" in layer_name:
            layer_info["component_type"] = "q_proj"
        elif "_k_proj" in layer_name:
            layer_info["component_type"] = "k_proj"
        elif "_v_proj" in layer_name:
            layer_info["component_type"] = "v_proj"
        elif "_o_proj" in layer_name:
            layer_info["component_type"] = "o_proj"
        elif "_mlp" in layer_name:
            layer_info["component_type"] = "mlp"
        elif "_self_attn" in layer_name:
            layer_info["component_type"] = "self_attn"
        else:
            layer_info["component_type"] = "other"

        layer_info["original_name"] = layer_name

        return layer_info


def load_humaneval_dataset(dataset_path: str) -> Dict[str, Dict[str, Any]]:
    # 统一的文件读取逻辑
    open_func = gzip.open if dataset_path.endswith('.gz') else open
    mode = 'rt' if dataset_path.endswith('.gz') else 'r'
    
    with open_func(dataset_path, mode, encoding='utf-8') as f:
        return {task['task_id']: task for task in [json.loads(line) for line in f]}


def load_samples_for_language(language: str, args) -> List[Dict[str, Any]]:
    samples = []
    task_samples = {}

    if not args.no_pregenerated_code:
        if language == "js":
            js_file = os.path.join(args.pregenerated_code_dir, "javascript", "samples.jsonl")
            if not os.path.exists(js_file):
                js_file = os.path.join(args.pregenerated_code_dir, "js", "samples.jsonl")
            pregenerated_code_file = js_file
        else:
            pregenerated_code_file = os.path.join(args.pregenerated_code_dir, language, "samples.jsonl")

        if os.path.exists(pregenerated_code_file):
            with open(pregenerated_code_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines:
                task_data = json.loads(line)
                task_id = task_data["task_id"]

                if "generation" not in task_data:
                    continue

                pass_at_k = task_data.get("pass_at_k", 1)

                if task_id not in task_samples:
                    task_samples[task_id] = {
                        "task_id": task_id,
                        "prompt": task_data["prompt"],
                        "solution": task_data.get("canonical_solution", task_data.get("solution", "")),
                        "generations": [],
                        "pass_at_k": pass_at_k
                    }

                task_samples[task_id]["generations"].append(task_data["generation"])

            if args.num_samples is not None:
                task_ids = list(task_samples.keys())[:args.num_samples]
                task_samples = {task_id: task_samples[task_id] for task_id in task_ids}

            samples = list(task_samples.values())

    if not samples:
        if args.data_dir is None:
            args.data_dir = os.path.join(args.humaneval_root, "data")

        if language == "js":
            dataset_path = os.path.join(args.data_dir, f"humaneval_javascript.jsonl.gz")
            if not os.path.exists(dataset_path):
                dataset_path = os.path.join(args.data_dir, f"humaneval_js.jsonl.gz")
        else:
            dataset_path = os.path.join(args.data_dir, f"humaneval_{language}.jsonl.gz")

        if os.path.exists(dataset_path):
            dataset = load_humaneval_dataset(dataset_path)

            for task_id, task_data in list(dataset.items())[:args.num_samples]:
                samples.append({
                    "task_id": task_id,
                    "prompt": task_data["prompt"],
                    "solution": task_data["canonical_solution"],
                    "generations": [],
                    "pass_at_k": 1
                })

    return samples


def main():
    parser = argparse.ArgumentParser(description="Full code generation gradient analysis tool")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Model path")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto' for all available GPUs, or 'cuda:0', 'cuda:1', 'cpu' etc")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "float32", "bfloat16"],
                        help="Data type, some operations may automatically convert to float32 for compatibility")
    parser.add_argument("--offload_to_cpu", action="store_true",
                        help="Whether to offload parts of the model to CPU to save GPU memory")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size, default is 1")

    parser.add_argument("--humaneval_root", type=str, required=True,
                        help="Absolute path to humaneval-x project root directory")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Data directory path")
    parser.add_argument("--languages", type=str, nargs="+",
                        default=["all"],
                        choices=["python", "cpp", "java", "go", "js", "all"],
                        help="Programming languages, specify multiple with spaces, or use 'all' for all languages")
    parser.add_argument("--num_samples", type=int, default=164,
                        help="Number of samples")
    parser.add_argument("--no_pregenerated_code", action="store_true",
                        help="Do not use pre-generated code, regenerate code instead")
    parser.add_argument("--pregenerated_code_dir", type=str, required=True,
                        help="Pre-generated code directory containing samples.jsonl files for each language")

    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p sampling parameter")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file name, default is full_generation_neurons_combined.json")
    parser.add_argument("--save_separate_languages", action="store_true",
                        help="Whether to save results for each language separately")

    args = parser.parse_args()

    try:
        if args.dtype == "float16":
            dtype = torch.float16
        elif args.dtype == "float32":
            dtype = torch.float32
        elif args.dtype == "bfloat16":
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
                args.dtype = "float16"
        else:
            raise ValueError(f"Unsupported data type: {args.dtype}")

        test_tensor = torch.zeros(1, dtype=dtype)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()

    except Exception:
        dtype = torch.float32
        args.dtype = "float32"

    if "all" in args.languages:
        args.languages = ["python", "cpp", "java", "go", "js"]

    os.makedirs(args.output_dir, exist_ok=True)

    gradient_finder = FullGenerationGradientFinder(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        batch_size=args.batch_size,
        offload_to_cpu=args.offload_to_cpu
    )

    all_language_results = {}

    for language in args.languages:
        if args.output_file is None:
            language_output_file = f"full_generation_neurons_{language}.json"
        else:
            name, ext = os.path.splitext(args.output_file)
            language_output_file = f"{name}_{language}{ext}"

        language_output_path = os.path.join(args.output_dir, language_output_file)

        language_samples = load_samples_for_language(language, args)

        if not language_samples:
            continue

        results = gradient_finder.analyze_multiple_samples(
            samples=language_samples,
            language=language,
            use_pregenerated_code=not args.no_pregenerated_code,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

        if args.save_separate_languages:
            gradient_finder.save_results(results, language_output_path)

        all_language_results[language] = results

        torch.cuda.empty_cache()
        gc.collect()

    if all_language_results:
        if args.output_file is None:
            combined_output_file = "full_generation_neurons_combined.json"
        else:
            combined_output_file = args.output_file

        combined_output_path = os.path.join(args.output_dir, combined_output_file)

        combined_output_path = gradient_finder.save_combined_results(all_language_results, combined_output_path)


if __name__ == "__main__":
    main()