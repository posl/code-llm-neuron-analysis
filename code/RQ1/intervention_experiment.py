import os
import sys
import argparse
import json
import torch
from pathlib import Path

HUMANEVAL_X_DIR = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "humaneval-x"))
sys.path.append(str(HUMANEVAL_X_DIR))
from model_loader import load_model_and_tokenizer
from neuron_intervention import setup_neuron_intervention, setup_random_neuron_intervention
from utils import setup_logging, set_seed
from generate_samples import generate_samples
from human_eval.data import read_problems as read_dataset
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description="Language-specific or universal neuron intervention experiment")
    parser.add_argument("--model_path", type=str, required=True,
                          help="Model path or Hugging Face model name")
    parser.add_argument("--input_dir", type=str, required=True,
                          help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True,
                          help="Output directory for results")
    parser.add_argument("--data_dir", type=str, required=True,
                          help="Data directory")
    parser.add_argument("--log_level", type=str, default="INFO",
                          choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                          help="Log level")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                          help="Device to run on")
    parser.add_argument("--num_samples", type=int, default=1,
                          help="Number of samples per task")
    parser.add_argument("--max_samples", type=int, default=None,
                          help="Maximum number of samples to test")
    parser.add_argument("--pass_at_k", type=int, default=3,
                          help="pass@k evaluation parameter")
    parser.add_argument("--intervention_type", type=str, default="zero",
                          choices=["zero"],
                          help="Neuron intervention type (only zero intervention is supported)")
    parser.add_argument("--target_languages", type=str, nargs="+",
                          default=["python", "go", "java", "js", "cpp"],
                          help="Target programming languages")
    parser.add_argument("--seed", type=int, default=42,
                          help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.8,
                          help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                          help="top_p value for generation")
    parser.add_argument("--top_k", type=int, default=50,
                          help="top_k value for generation")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                          help="Repetition penalty for generation")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                          help="Maximum new tokens for generation")
    parser.add_argument("--neurons_dir", type=str, required=True,
                          help="Directory containing files of classified neurons")
    parser.add_argument("--intervention_language", type=str, default="java",
                          help="Specify a single language's neurons to intervene on")
    parser.add_argument("--stop_tokens", type=str, action="append", nargs=2,
                          metavar=("LANGUAGE", "TOKEN"),
                          help="Language-specific stop tokens")
    parser.add_argument("--skip-random-control", action="store_true", dest="run_random_control",
                          help="Skip the control experiment of randomly selecting the same number of neurons")
    parser.set_defaults(run_random_control=False)
    parser.add_argument("--intervention_scope", type=str, default="specific",
                          choices=["specific", "common"],
                          help="Select the scope of neurons for intervention")
    parser.add_argument("--target_layers", type=str, default=None,
                          help="Specify the layer indices to intervene on")
    parser.add_argument("--enable_early_stopping", action="store_true", default=True,
                          help="Enable early stopping")
    parser.add_argument("--disable_early_stopping", action="store_false", dest="enable_early_stopping",
                          help="Disable the early stopping feature")
    return parser.parse_args()

def find_file(directory: str, pattern: str):
    import os
    import glob

    matches = glob.glob(os.path.join(directory, pattern))

    if matches:
        return matches[0]
    return None

def load_neurons(neurons_dir, scope="specific", language=None, target_layers=None):
    if scope == "specific":
        if language is None:
            return {}
        file_pattern = f"{language}_specific_neurons.json"
    elif scope == "common":
        file_pattern = "common_neurons.json"
    else:
        return {}

    neuron_file = find_file(neurons_dir, file_pattern)

    if not neuron_file:
        return {}

    try:
        with open(neuron_file, 'r', encoding='utf-8') as f:
            neurons_data = json.load(f)

        is_list_format = isinstance(neurons_data, list)

        if is_list_format:
            neurons_dict = {}
            for neuron_info in neurons_data:
                if isinstance(neuron_info, dict) and 'neuron_key' in neuron_info:
                    neurons_dict[neuron_info['neuron_key']] = neuron_info
            neurons_data = neurons_dict

        if target_layers is not None and isinstance(target_layers, list) and len(target_layers) > 0:
            filtered_neurons = {}
            for neuron_key, neuron_info in neurons_data.items():
                try:
                    if "_neuron_" in neuron_key:
                        parts = neuron_key.split("_neuron_")
                        layer_name = parts[0]

                        layer_idx = None
                        for part in layer_name.split("_"):
                            if part.isdigit():
                                layer_idx = int(part)
                                break

                        if layer_idx is not None and layer_idx in target_layers:
                            filtered_neurons[neuron_key] = neuron_info
                except Exception as e:
                    continue

            return filtered_neurons

        return neurons_data

    except Exception as e:
        return {}

def _run_generation_for_languages(args, datasets, output_dir, model, tokenizer, stop_tokens_dict):
    """Helper function to run generation for all target languages."""
    for language in args.target_languages:
        if language not in datasets:
            continue

        problems = datasets[language]
        if not problems:
            continue

        lang_output_dir = output_dir / language
        lang_output_dir.mkdir(parents=True, exist_ok=True)

        actual_samples = max(args.num_samples, args.pass_at_k)
        samples_file = str(lang_output_dir / "samples.jsonl")

        generate_samples(
            model_name=args.model_path,
            problems=problems,
            output_file=samples_file,
            num_samples_per_task=actual_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            model=model,
            tokenizer=tokenizer,
            device=args.device,
            stop_tokens=stop_tokens_dict,
            language=language,
            pass_at_k=args.pass_at_k,
            enable_early_stopping=args.enable_early_stopping
        )

def modify_model_for_intervention(model, neurons_to_intervene):
    """Set up zero intervention for specified neurons"""
    return setup_neuron_intervention(model, neurons_to_intervene, "zero")

def run_intervention_experiment(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    datasets = {}

    for language in args.target_languages:
        file_language = language
        if language == "js":
            file_language = "javascript"

        dataset_path = os.path.join(args.data_dir, f"humaneval_{file_language}.jsonl.gz")

        if not os.path.exists(dataset_path):
            dataset_path_no_gz = os.path.join(args.data_dir, f"humaneval_{file_language}.jsonl")
            if os.path.exists(dataset_path_no_gz):
                dataset_path = dataset_path_no_gz
            else:
                continue

        try:
            problems = read_dataset(dataset_path)

            if args.max_samples:
                keys = list(problems.keys())[:args.max_samples]
                problems = {k: problems[k] for k in keys}

            datasets[language] = problems

        except Exception as e:
            continue

    # Only zero intervention is supported
    intervention_type = args.intervention_type

    torch.cuda.empty_cache()
    import gc
    gc.collect()

    stop_tokens_dict = {}
    if args.stop_tokens:
        for lang, token in args.stop_tokens:
            if lang not in stop_tokens_dict:
                stop_tokens_dict[lang] = []
            stop_tokens_dict[lang].append(bytes(token, "utf-8").decode("unicode_escape"))
    else:
        stop_tokens_dict = None

    intervention_targets = []
    if args.intervention_scope == "common":
        intervention_targets = ["common"]
    elif args.intervention_scope == "specific":
        if args.intervention_language:
            intervention_targets = [args.intervention_language]
        else:
            intervention_targets = args.target_languages

    # Create output directory for zero intervention
    intervention_dir = output_dir / intervention_type
    intervention_dir.mkdir(exist_ok=True)

    for target_scope_or_lang in intervention_targets:

        neurons_to_intervene = {}

        target_layers = None
        if args.target_layers:
            try:
                target_layers = [int(layer.strip()) for layer in args.target_layers.split(',') if layer.strip().isdigit()]
                if not target_layers:
                    target_layers = None
            except Exception as e:
                target_layers = None

        if args.intervention_scope == "common":
            neurons_to_intervene = load_neurons(args.neurons_dir, scope="common", target_layers=target_layers)
            intervention_target_name = "common"
        else:
            target_language = target_scope_or_lang
            neurons_to_intervene = load_neurons(args.neurons_dir, scope="specific", language=target_language, target_layers=target_layers)
            intervention_target_name = target_language

        if not neurons_to_intervene:
            continue

        # Set up zero intervention
        modify_model_for_intervention(model, neurons_to_intervene)
        actual_hook = None

        # Apply zero neuron intervention
        if hasattr(model, 'zero_neuron_intervention'):
            actual_hook = model.zero_neuron_intervention(neurons_to_intervene)
        
        if actual_hook is None:
            continue

        # Generate samples with intervention
        _run_generation_for_languages(
            args=args,
            datasets=datasets,
            output_dir=intervention_dir / intervention_target_name,
            model=model,
            tokenizer=tokenizer,
            stop_tokens_dict=stop_tokens_dict
        )

        if hasattr(actual_hook, 'remove'):
            actual_hook.remove()

        torch.cuda.empty_cache()
        gc.collect()

        # Random control experiment (if enabled)
        if args.run_random_control and neurons_to_intervene:
            num_intervened_neurons = len(neurons_to_intervene)

            random_hook = setup_random_neuron_intervention(
                model,
                num_neurons_to_intervene=num_intervened_neurons,
                intervention_type="zero"  # Only zero intervention
            )

            if random_hook is not None:
                # Generate samples with random control
                _run_generation_for_languages(
                    args=args,
                    datasets=datasets,
                    output_dir=intervention_dir / "random_control" / intervention_target_name,
                    model=model,
                    tokenizer=tokenizer,
                    stop_tokens_dict=stop_tokens_dict
                )

                if hasattr(random_hook, 'remove'):
                    random_hook.remove()

                torch.cuda.empty_cache()
                gc.collect()

def main():
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    processed_languages = []
    for lang_item in args.target_languages:
        if ',' in lang_item:
            for lang in lang_item.split(','):
                if lang.strip():
                    processed_languages.append(lang.strip().lower())
        else:
            processed_languages.append(lang_item.lower())

    args.target_languages = processed_languages

    actual_samples = max(args.num_samples, args.pass_at_k)

    if args.intervention_scope == "specific":
        if args.intervention_language:
            args.intervention_language = args.intervention_language.lower()
            if args.intervention_language not in processed_languages:
                processed_languages.append(args.intervention_language)
                args.target_languages = processed_languages
    elif args.intervention_scope == "common":
        if args.intervention_language:
            args.intervention_language = None

    run_intervention_experiment(args)

if __name__ == "__main__":
    main()