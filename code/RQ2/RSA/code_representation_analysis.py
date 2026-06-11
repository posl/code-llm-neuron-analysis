import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional
import torch
from tqdm import tqdm
from scipy.spatial.distance import cosine
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

HUMANEVAL_X_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "humaneval-x"))
sys.path.append(str(HUMANEVAL_X_DIR))

try:
    from src.RQ1.utils import setup_logging, set_seed
    from src.RQ1.model_loader import load_model_and_tokenizer
except ImportError:
    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, project_root)
    from src.RQ1.utils import setup_logging, set_seed
    from src.RQ1.model_loader import load_model_and_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM layer specialization analysis - analyze layer specialization characteristics of large language models when processing cross-language code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Similarity metric description:

  Core metrics:
    cosine      - Cosine similarity, measures vector direction similarity, not affected by length
    """
    )
    parser.add_argument("--model_path", type=str, required=True,
                      help="Model path or huggingface model name")
    parser.add_argument("--data_dir", type=str, required=True,
                      help="humaneval-x data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Results output directory")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Log level")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run on")
    parser.add_argument("--device_map", type=str, default=None,
                      help="Custom device mapping, can be 'auto' or JSON format layer-to-device mapping, e.g.: '{\"0-15\":0,\"16-31\":1}'")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    parser.add_argument("--target_languages", type=str, nargs="+",
                      default=["python", "cpp", "java", "go", "js"],
                      help="Target programming languages")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum number of samples per language")
    parser.add_argument("--similarity_metrics", type=str, nargs="+",
                      default=["cosine"],
                      choices=["cosine"],
                      help="Similarity metrics to use (default: cosine)")
    parser.add_argument("--primary_metric", type=str, default="cosine",
                      choices=["cosine"],
                      help="Primary similarity metric for analysis")
    parser.add_argument("--enable_all_visualizations", action="store_true", default=True,
                      help="Enable all visualization features, including collective trend analysis and comprehensive dashboard")
    parser.add_argument("--disable_advanced_viz", action="store_true", default=False,
                      help="Disable advanced visualization features, generate basic charts only")

    args = parser.parse_args()
    return args

def load_humaneval_x_data(data_dir: str, languages: List[str], max_samples: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    task_data = defaultdict(dict)

    try:
        from src.RQ1 import read_dataset as hx_read_dataset
    except ImportError:
        def hx_read_dataset(dataset_path):
            problems = {}
            try:
                if dataset_path.endswith('.gz'):
                    import gzip
                    with gzip.open(dataset_path, 'rt', encoding='utf-8') as f:
                        for line in f:
                            sample = json.loads(line)
                            task_id = sample.get('task_id')
                            if task_id:
                                problems[task_id] = sample
                else:
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            sample = json.loads(line)
                            task_id = sample.get('task_id')
                            if task_id:
                                problems[task_id] = sample
            except Exception as e:
                pass
            return problems

    for language in languages:
        file_language = language
        if language.lower() == "js":
            file_language = "javascript"
        elif language.lower() == "cpp":
            file_language = "cpp"

        dataset_path = os.path.join(data_dir, f"humaneval_{file_language.lower()}.jsonl.gz")

        if not os.path.exists(dataset_path):
            dataset_path_no_gz = os.path.join(data_dir, f"humaneval_{file_language.lower()}.jsonl")
            if os.path.exists(dataset_path_no_gz):
                dataset_path = dataset_path_no_gz
            else:
                continue
        try:
            problems = hx_read_dataset(dataset_path)

            if max_samples:
                keys = list(problems.keys())[:max_samples]
                problems = {k: problems[k] for k in keys}

            for task_id, sample in problems.items():
                if '/' in task_id:
                    task_num = task_id.split('/')[1]
                    universal_id = task_num
                else:
                    universal_id = task_id

                task_data[universal_id][language] = {
                    'prompt': sample.get('prompt', ''),
                    'canonical_solution': sample.get('canonical_solution', ''),
                    'declaration': sample.get('declaration', ''),
                    'entry_point': sample.get('entry_point', ''),
                    'test': sample.get('test', '')
                }
        except Exception as e:
            pass

    return task_data

def preprocess_code(code: str, language: str) -> str:
    return code.strip()

def prepare_dataset(task_data: Dict[str, Dict[str, Any]], languages: List[str]) -> List[Dict[str, Any]]:
    dataset = []
    
    for task_id, langs_data in task_data.items():
        available_langs = set(langs_data.keys())
        target_langs = set(languages)
        
        task_entry = {
            'task_id': task_id,
            'languages': {}
        }
        
        for lang in available_langs:
            if lang in target_langs:
                declaration = langs_data[lang].get('declaration', '')
                solution = langs_data[lang].get('canonical_solution', '')
                
                complete_code = assemble_complete_code(declaration, solution, lang)
                processed_code = preprocess_code(complete_code, lang)
                
                task_entry['languages'][lang] = {
                    'original_code': complete_code,
                    'processed_code': processed_code
                }
        
        if len(task_entry['languages']) >= 2:
            dataset.append(task_entry)
    
    return dataset

def assemble_complete_code(declaration: str, solution: str, language: str) -> str:
    if not declaration:
        return solution
    
    if language in ['python', 'go', 'rust']:
        return declaration + solution
    elif language in ['java', 'cpp']:
        if declaration.strip().endswith('{'):
            return declaration + solution
        else:
            return declaration + " {\n" + solution + "\n}"
    elif language == 'js':
        if '=>' in declaration or declaration.strip().endswith('{'):
            return declaration + solution
        else:
            return declaration + " {\n" + solution + "\n}"
    else:
        return declaration + "\n" + solution



def get_all_layers(model):

    num_layers = get_model_num_layers(model)
    return list(range(num_layers))

def get_model_num_layers(model):

    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return len(model.encoder.layer)
    else:
        for attr_name in ["layers", "h", "layer", "blocks"]:
            if hasattr(model, attr_name):
                layers = getattr(model, attr_name)
                if isinstance(layers, list) or hasattr(layers, "__len__"):
                    return len(layers)
        return 32

def _filter_abnormal_dimensions(hidden_state, layer_idx):

    if layer_idx >= 0 and hidden_state.shape[-1] > 2352:
        filtered_state = hidden_state.clone()
        filtered_state[:, :, 2352] = 0.0
        return filtered_state

    return hidden_state

def extract_code_representations(model, tokenizer, code, language, layer_indices, device="cuda", max_length=512):

    prefixed_code = code
    inputs = tokenizer(prefixed_code, return_tensors="pt", truncation=True, max_length=max_length, padding=False)

    multi_gpu = False
    target_device = device
    if hasattr(model, "hf_device_map"):
        multi_gpu = len(set(model.hf_device_map.values())) > 1

    if multi_gpu and hasattr(model, "hf_device_map"):
        device_values = list(model.hf_device_map.values())
        valid_devices = [d for d in device_values if d != "cpu" and not isinstance(d, int)]
        if valid_devices:
            target_device = valid_devices[0]
        else:
            target_device = device_values[0] if device_values else device

    input_ids = inputs["input_ids"].to(target_device)

    representations = {}
    hooks = []
    hidden_states = {}

    def get_activation(name):
        def hook(module, input, output):
            if hasattr(module, "weight") and hasattr(module.weight, "device"):
                current_device = module.weight.device
            else:
                current_device = target_device

            if isinstance(output, (tuple, list)):
                hidden_state = output[0]
            else:
                hidden_state = output

            hidden_states[name] = hidden_state.detach().to(current_device)
        return hook

    for layer_idx in layer_indices:
        layer = get_layer_by_index(model, layer_idx)
        if layer is not None:
            hook = layer.register_forward_hook(get_activation(f"layer_{layer_idx}"))
            hooks.append(hook)

    with torch.no_grad():
        outputs = model(input_ids)

    for hook in hooks:
        hook.remove()

    for layer_idx in layer_indices:
        layer_name = f"layer_{layer_idx}"
        if layer_name in hidden_states:
            hidden_state = hidden_states[layer_name]

            hidden_state = _filter_abnormal_dimensions(hidden_state, layer_idx)

            min_val = hidden_state.min().item()
            max_val = hidden_state.max().item()

            if abs(min_val) > 100 or abs(max_val) > 100:
                hidden_state = torch.clamp(hidden_state, min=-50, max=50)

            if hidden_state.shape[1] > 1:
                avg_hidden_state = hidden_state[:, 1:, :].mean(dim=1)
            else:
                avg_hidden_state = hidden_state.mean(dim=1)

            representations[layer_idx] = {
                "hidden_state": hidden_state.to(torch.float32).cpu().numpy(),
                "pooled": avg_hidden_state.to(torch.float32).cpu().numpy()
            }

    return representations

def get_layer_by_index(model, layer_idx):
    try:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers[layer_idx]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h[layer_idx]
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer[layer_idx]
        elif hasattr(model, "layers"):
            return model.layers[layer_idx]
        else:
            return None
    except (IndexError, AttributeError) as e:
        return None

def extract_dataset_representations(model, tokenizer, dataset, layer_groups, device="cuda", batch_size=1):
    representations_data = []
    all_layer_indices = sorted(set(layer_groups["all"]))

    for task_entry in tqdm(dataset, desc="Extracting representations"):
        task_id = task_entry["task_id"]
        languages_data = task_entry["languages"]
        task_representations = {
            "task_id": task_id,
            "representations": {}
        }
        for lang, code_data in languages_data.items():
            code = code_data["processed_code"]
            representations = extract_code_representations(
                model, tokenizer, code, lang, all_layer_indices, device
            )
            task_representations["representations"][lang] = representations
        representations_data.append(task_representations)

    return representations_data

def save_representations(representations_data, output_dir, filename="code_representations.npz"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    save_dict = {}

    for i, task_data in enumerate(representations_data):
        task_id = task_data["task_id"]
        representations = task_data["representations"]
        for lang, lang_representations in representations.items():
            for layer_idx, layer_repr in lang_representations.items():
                key_pooled = f"{task_id}_{lang}_{layer_idx}_pooled"
                save_dict[key_pooled] = layer_repr["pooled"]

    np.savez_compressed(output_path, **save_dict)

class SimilarityAnalyzer:

    def __init__(self, metrics=None):
        self.core_metrics = {
            'cosine': self.cosine_similarity,
        }
        self.supplementary_metrics = {}
        self.available_metrics = self.core_metrics.copy()

        if metrics is None:
            self.metrics = ['cosine']
        else:
            self.metrics = [m for m in metrics if m in self.available_metrics]
            if not self.metrics:
                self.metrics = ['cosine']



    def cosine_similarity(self, vec1, vec2):
        return 1 - cosine(vec1.flatten(), vec2.flatten())





    def compute_all_similarities(self, repr1, repr2):
        similarities = {}
        for metric_name in self.metrics:
            metric_func = self.available_metrics.get(metric_name)
            if metric_func is None:
                continue

            try:
                similarity = metric_func(repr1, repr2)
                similarities[metric_name] = float(similarity)
            except Exception as e:
                similarities[metric_name] = 0.0
        return similarities

def compute_pairwise_similarities(representations_data, all_layers, similarity_metrics=None):
    analyzer = SimilarityAnalyzer(metrics=similarity_metrics)
    metrics_to_compute = analyzer.metrics

    similarities = {
        "by_layer": {layer_idx: {metric: [] for metric in metrics_to_compute} for layer_idx in all_layers},
        "by_task": {
            task_data["task_id"]: {
                "by_layer": {layer_idx: {metric: [] for metric in metrics_to_compute} for layer_idx in all_layers},
                "by_language_pair": {}
            } for task_data in representations_data
        },
        "by_language_pair": {},
        "metrics_used": metrics_to_compute
    }

    all_found_languages = set()
    for task_data in representations_data:
        all_found_languages.update(task_data["representations"].keys())

    for lang1_iter, lang2_iter in combinations(sorted(list(all_found_languages)), 2):
        lang_pair_iter = f"{lang1_iter}_{lang2_iter}"
        similarities["by_language_pair"][lang_pair_iter] = {
            "by_layer": {layer_idx: {metric: [] for metric in metrics_to_compute} for layer_idx in all_layers}
        }
        for task_data in representations_data:
            task_id_iter = task_data["task_id"]
            similarities["by_task"][task_id_iter]["by_language_pair"][lang_pair_iter] = {
                "by_layer": {layer_idx: {metric: [] for metric in metrics_to_compute} for layer_idx in all_layers}
            }

    total_pairs = 0
    for task_data in representations_data:
        languages_in_task = list(task_data["representations"].keys())
        pairs_in_task = len(list(combinations(languages_in_task, 2)))
        total_pairs += pairs_in_task

    for task_data in tqdm(representations_data, desc="Computing similarity"):
        task_id = task_data["task_id"]
        task_representations = task_data["representations"]
        languages_in_task = list(task_representations.keys())

        for lang1, lang2 in combinations(languages_in_task, 2):
            sorted_lang_pair = f"{min(lang1, lang2)}_{max(lang1, lang2)}"

            for layer_idx in all_layers:
                if (layer_idx in task_representations[lang1] and
                    layer_idx in task_representations[lang2]):
                    repr1 = task_representations[lang1][layer_idx]["pooled"]
                    repr2 = task_representations[lang2][layer_idx]["pooled"]

                    similarity_scores = analyzer.compute_all_similarities(repr1, repr2)

                    for metric, score in similarity_scores.items():
                        similarities["by_layer"][layer_idx][metric].append(score)
                        similarities["by_task"][task_id]["by_layer"][layer_idx][metric].append(score)
                        similarities["by_language_pair"][sorted_lang_pair]["by_layer"][layer_idx][metric].append(score)
                        similarities["by_task"][task_id]["by_language_pair"][sorted_lang_pair]["by_layer"][layer_idx][metric].append(score)

    return similarities

def save_similarities(similarities, output_dir, filename="similarities.json"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {str(k): convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    serializable_similarities = convert_to_serializable(similarities)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_similarities, f, indent=2, ensure_ascii=False)

def _validate_similarity_data(similarities, all_layers, metrics_used):

    for metric in metrics_used:
        layer_stats = []
        for layer_idx in all_layers:
            if layer_idx in similarities["by_layer"] and metric in similarities["by_layer"][layer_idx]:
                sims = similarities["by_layer"][layer_idx][metric]
                if sims:
                    mean_sim = np.mean(sims)
                    std_sim = np.std(sims)
                    min_sim = np.min(sims)
                    max_sim = np.max(sims)
                    layer_stats.append({
                        'layer': layer_idx,
                        'mean': mean_sim,
                        'std': std_sim,
                        'min': min_sim,
                        'max': max_sim,
                        'count': len(sims)
                    })

def analyze_similarities(similarities, all_layers, primary_metric='cosine'):
    metrics_used = similarities.get("metrics_used", [primary_metric])
    if primary_metric not in metrics_used:
        primary_metric = metrics_used[0] if metrics_used else 'cosine'

    _validate_similarity_data(similarities, all_layers, metrics_used)

    analysis_results = {
        "metrics_used": metrics_used,
        "primary_metric": primary_metric,
        "average_similarities": {
            "by_layer": {},
            "by_metric": {}
        },
        "language_pairs": {},
    }

    for metric in metrics_used:
        analysis_results["average_similarities"]["by_metric"][metric] = {}

    for layer_idx in all_layers:
        layer_idx_str = str(layer_idx)
        analysis_results["average_similarities"]["by_layer"][layer_idx_str] = {}
        for metric in metrics_used:
            if layer_idx in similarities["by_layer"] and metric in similarities["by_layer"][layer_idx]:
                sims = similarities["by_layer"][layer_idx][metric]
                if sims:
                    avg_sim = np.mean(sims)
                    std_sim = np.std(sims)
                    median_sim = np.median(sims)
                    layer_stats = {
                        "mean": float(avg_sim),
                        "std": float(std_sim),
                        "median": float(median_sim),
                        "count": len(sims),
                    }
                    analysis_results["average_similarities"]["by_layer"][layer_idx_str][metric] = layer_stats
                    analysis_results["average_similarities"]["by_metric"][metric][layer_idx_str] = layer_stats

    for lang_pair, pair_data in similarities["by_language_pair"].items():
        analysis_results["language_pairs"][lang_pair] = {
            "by_layer": {},
            "by_metric": {}
        }
        for metric in metrics_used:
            analysis_results["language_pairs"][lang_pair]["by_metric"][metric] = {}
        for layer_idx_str, layer_data in pair_data["by_layer"].items():
            analysis_results["language_pairs"][lang_pair]["by_layer"][layer_idx_str] = {}
            for metric in metrics_used:
                if metric in layer_data:
                    sims = layer_data[metric]
                    if sims:
                        avg_sim = np.mean(sims)
                        std_sim = np.std(sims)
                        median_sim = np.median(sims)
                        pair_stats = {
                            "mean": float(avg_sim),
                            "std": float(std_sim),
                            "median": float(median_sim),
                            "count": len(sims),
                        }
                        analysis_results["language_pairs"][lang_pair]["by_layer"][layer_idx_str][metric] = pair_stats
                        analysis_results["language_pairs"][lang_pair]["by_metric"][metric][layer_idx_str] = pair_stats

    max_layer = None
    max_similarity = -1
    for layer_idx_str, layer_data in analysis_results["average_similarities"]["by_layer"].items():
        if primary_metric in layer_data:
            mean_sim = layer_data[primary_metric]["mean"]
            if mean_sim > max_similarity:
                max_similarity = mean_sim
                max_layer = layer_idx_str

    if max_layer is not None:
        analysis_results["max_similarity_layer"] = {
            "layer_idx": max_layer,
            "metric": primary_metric,
            "mean": max_similarity
        }

    return analysis_results

def save_analysis_results(analysis_results, output_dir, filename="analysis_results.json"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

def visualize_results(similarities, analysis_results, all_layers, output_dir):
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False

    visualize_layer_similarities(analysis_results, all_layers, vis_dir)
    visualize_language_pair_similarities(analysis_results, all_layers, vis_dir)
    visualize_layer_heatmap(analysis_results, all_layers, vis_dir)

def visualize_layer_similarities(analysis_results, all_layers, output_dir):
    metrics_used = analysis_results.get("metrics_used", ["cosine"])
    primary_metric = analysis_results.get("primary_metric", metrics_used[0])
    for metric in metrics_used:
        layers = []
        means = []
        stds = []
        counts = []
        for layer_idx_str, layer_data in analysis_results["average_similarities"]["by_layer"].items():
            if metric in layer_data:
                layers.append(int(layer_idx_str))
                means.append(layer_data[metric]["mean"])
                stds.append(layer_data[metric]["std"])
                counts.append(layer_data[metric].get("count", 1))
        if not layers:
            continue
        sorted_indices = np.argsort(layers)
        layers = [layers[i] for i in sorted_indices]
        means = [means[i] for i in sorted_indices]
        stds = [stds[i] for i in sorted_indices]
        counts = [counts[i] for i in sorted_indices]
        confidence_intervals = []
        for std, count in zip(stds, counts):
            if count > 1:
                se = std / np.sqrt(count) 
                t_value = stats.t.ppf(0.975, df=count-1) 
                margin_error = t_value * se
                confidence_intervals.append(margin_error)
            else:
                confidence_intervals.append(std) 

        plt.figure(figsize=(12, 6))
        plt.plot(layers, means, '-o', markersize=8, linewidth=2, label=f'{metric.upper()} Similarity')
        lower_bounds = [m - ci for m, ci in zip(means, confidence_intervals)]
        upper_bounds = [m + ci for m, ci in zip(means, confidence_intervals)]
        plt.fill_between(layers, lower_bounds, upper_bounds, alpha=0.3,
                        label=f'95% Confidence Interval')

        plt.xlabel('Layer Index', fontsize=14, fontweight='bold')
        plt.ylabel(f'Average {metric.upper()} Similarity', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'layer_similarities_{metric}.svg'), format='svg', dpi=300, bbox_inches='tight')
        plt.close()

    if len(metrics_used) > 1:
        plt.figure(figsize=(14, 8))
        for metric in metrics_used:
            layers = []
            means = []
            for layer_idx_str, layer_data in analysis_results["average_similarities"]["by_layer"].items():
                if metric in layer_data:
                    layers.append(int(layer_idx_str))
                    means.append(layer_data[metric]["mean"])
            if layers:
                sorted_indices = np.argsort(layers)
                layers = [layers[i] for i in sorted_indices]
                means = [means[i] for i in sorted_indices]
                plt.plot(layers, means, '-o', label=f'{metric.upper()}', markersize=6, linewidth=2)
        plt.xlabel('Layer Index', fontsize=14, fontweight='bold')
        plt.ylabel('Average Similarity', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_similarities_comparison.svg'), format='svg', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_language_pair_similarities(analysis_results, all_layers, output_dir):
    language_pairs = list(analysis_results["language_pairs"].keys())
    if not language_pairs:
        return
    primary_metric = analysis_results.get("primary_metric", "cosine")

    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'text.usetex': False,
        'svg.fonttype': 'none',
    })
    
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#ff9896']


    for i, lang_pair in enumerate(language_pairs):
        pair_data = analysis_results["language_pairs"][lang_pair]["by_layer"]
        layers = []
        means = []
        for layer_idx, layer_data in pair_data.items():
            if primary_metric in layer_data:
                layers.append(int(layer_idx))
                means.append(layer_data[primary_metric]["mean"])
        if layers:
            sorted_indices = np.argsort(layers)
            layers = [layers[i] for i in sorted_indices]
            means = [means[i] for i in sorted_indices]

            color = colors[i % len(colors)]

            ax.plot(layers, means, marker='o', color=color, label=lang_pair,
                   linewidth=3.0, markersize=8, alpha=0.9, markeredgewidth=1.2, markeredgecolor='white')

    ax.set_xlabel('Layer Index', fontsize=18, fontweight='bold', labelpad=15)
    ax.set_ylabel(f'Average {primary_metric.upper()} Similarity', fontsize=18, fontweight='bold', labelpad=15)

    ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)

    ax.tick_params(axis='both', which='major', labelsize=16, width=1.2, length=8)
    ax.tick_params(axis='both', which='minor', width=0.8, length=4)

    legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                      fontsize=14, frameon=True, fancybox=True,
                      shadow=True, framealpha=0.95, ncol=1,
                      columnspacing=0.8, handlelength=2.5, handletextpad=0.5)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.0)

    for text in legend.get_texts():
        text.set_fontweight('bold')
        text.set_fontsize(16)

    if all_layers:
        ax.set_xlim(min(all_layers) - 0.5, max(all_layers) + 0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    plt.tight_layout(pad=2.0)
    
    svg_path = os.path.join(output_dir, 'language_pair_curves.svg')
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    pdf_path = os.path.join(output_dir, 'language_pair_curves.pdf')
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    
    plt.close()

def visualize_layer_heatmap(analysis_results, all_layers, output_dir):
    language_pairs = list(analysis_results["language_pairs"].keys())
    if not language_pairs:
        return
    metrics_used = analysis_results.get("metrics_used", ["cosine"])
    for metric in metrics_used:
        _create_simple_heatmap(analysis_results, all_layers, output_dir, metric, language_pairs)

def _create_simple_heatmap(analysis_results, all_layers, output_dir, metric, language_pairs):
    sorted_layers = sorted(all_layers)
    data = np.zeros((len(language_pairs), len(sorted_layers)))
    valid_data_mask = np.zeros((len(language_pairs), len(sorted_layers)), dtype=bool)

    for i, lang_pair in enumerate(language_pairs):
        if lang_pair not in analysis_results["language_pairs"]:
            continue
        pair_data = analysis_results["language_pairs"][lang_pair]["by_layer"]
        for j, layer_idx in enumerate(sorted_layers):
            layer_key_found = None
            layer_data_for_metric = None
            if str(layer_idx) in pair_data:
                layer_key_found = str(layer_idx)
                layer_data_for_metric = pair_data[str(layer_idx)]
            elif layer_idx in pair_data:
                layer_key_found = layer_idx
                layer_data_for_metric = pair_data[layer_idx]
            if layer_key_found is None:
                continue
            if not isinstance(layer_data_for_metric, dict) or metric not in layer_data_for_metric:
                continue
            metric_data_val = layer_data_for_metric[metric]
            if isinstance(metric_data_val, dict) and "mean" in metric_data_val:
                data[i, j] = metric_data_val["mean"]
                valid_data_mask[i, j] = True

    valid_data = data[valid_data_mask]
    if len(valid_data) == 0:
        return

    data_min = np.min(valid_data)
    data_max = np.max(valid_data)
    data_range = data_max - data_min

    if data_range < 0.1:
        margin = data_range * 0.1 if data_range > 0 else 0.01
        vmin = max(0, data_min - margin)
        vmax = min(1, data_max + margin)
    else:
        vmin = data_min
        vmax = data_max
    fig, ax = plt.subplots(figsize=(max(18, len(sorted_layers) * 0.6), max(12, len(language_pairs) * 0.9)))
    im = ax.imshow(data, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(len(sorted_layers)))
    ax.set_xticklabels(sorted_layers, rotation=45 if len(sorted_layers) > 20 else 0, fontsize=15, fontweight='bold', ha='right')
    ax.set_yticks(range(len(language_pairs)))
    ax.set_yticklabels([lp.replace('_', ' ↔ ') for lp in language_pairs], fontsize=15, fontweight='bold', va='center')
    ax.grid(False) 
    ax.set_axisbelow(False)
    ax.set_xticks(np.arange(-0.5, len(sorted_layers), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(language_pairs), 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5, alpha=0.9)
    ax.set_xlabel('Layer Index', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('Language Pairs', fontsize=16, fontweight='bold', labelpad=15)
    title_line1 = f'{metric.upper()} Similarity Analysis'
    title_line2 = f'Range: [{vmin:.4f}, {vmax:.4f}] • Δ = {data_range:.4f}'
    ax.set_title(f'{title_line1}\n{title_line2}', fontsize=18, fontweight='bold', pad=25)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, aspect=35, pad=0.02)
    cbar.set_label(f'{metric.upper()} Similarity', fontsize=15, fontweight='bold', labelpad=20)
    flat_data = data.flatten()
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    tick_values = np.percentile(flat_data, percentiles)
    tick_values = np.unique(tick_values)
    tick_values = np.concatenate([[vmin], tick_values, [vmax]])
    tick_values = np.unique(tick_values)
    if data_range < 0.01:
        tick_format = '%.3f'
    else:
        tick_format = '%.2f'

    cbar.set_ticks(tick_values)
    cbar.set_ticklabels([tick_format % val for val in tick_values], fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=14, width=1.2, length=6)
    plt.tight_layout(pad=2.0)
    plt.savefig(os.path.join(output_dir, f'detailed_heatmap_{metric}.png'),
                dpi=400, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()

def main():
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    task_data = load_humaneval_x_data(args.data_dir, args.target_languages, args.max_samples)
    dataset = prepare_dataset(task_data, args.target_languages)

    device_map = None
    if args.device_map:
        if args.device_map.lower() == 'auto':
            device_map = 'auto'
        else:
            try:
                device_map = json.loads(args.device_map)
            except json.JSONDecodeError:
                pass
    model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device, device_map=device_map)
    all_layers = get_all_layers(model)

    representations_data = extract_dataset_representations(
        model, tokenizer, dataset, {"all": all_layers}, device=args.device
    )
    save_representations(representations_data, args.output_dir)

    similarities = compute_pairwise_similarities(
        representations_data,
        all_layers,
        args.similarity_metrics
    )
    save_similarities(similarities, args.output_dir)

    analysis_results = analyze_similarities(similarities, all_layers, args.primary_metric)
    save_analysis_results(analysis_results, args.output_dir)

    visualize_results(similarities, analysis_results, all_layers, args.output_dir)

if __name__ == "__main__":
    main()