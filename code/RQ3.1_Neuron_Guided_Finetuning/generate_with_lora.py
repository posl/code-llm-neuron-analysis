#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm
import argparse
import glob
import re
import json
import gzip

def read_problems_unified(data_file: str, dataset_type: str = "auto") -> Dict[str, Any]:
    if dataset_type == "auto":
        if "mceval" in data_file.lower():
            dataset_type = "mceval"
        else:
            dataset_type = "humaneval-x"

    if dataset_type == "humaneval-x":
        return read_problems(data_file)

    elif dataset_type == "mceval":
        problems = {}

        if data_file.endswith('.gz'):
            with gzip.open(data_file, 'rt', encoding='utf-8') as f:
                lines = f.readlines()
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        for line in lines:
            if line.strip():
                problem = json.loads(line.strip())
                task_id = problem["task_id"]

                adapted_problem = {
                    "task_id": task_id,
                    "prompt": problem["prompt"],
                    "canonical_solution": problem["canonical_solution"],
                    "test": problem["test"],
                    "declaration": problem.get("signature", ""),
                    "example_test": problem.get("test", ""),
                    "text": problem.get("docstring", ""),
                    "entry_point": problem.get("entry_point", ""),
                    "instruction": problem.get("instruction", ""),
                    "level": problem.get("level", ""),
                }

                problems[task_id] = adapted_problem

        return problems

    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, stop_token_strings: List[str], device: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_token_strings = stop_token_strings
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stop_token_ids = [torch.tensor(tokenizer.encode(stop_str, add_special_tokens=False), dtype=torch.long, device=device) for stop_str in stop_token_strings if tokenizer.encode(stop_str, add_special_tokens=False)]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_str in self.stop_token_strings:
            if generated_text.endswith(stop_str):
                return True
        return False

class EarlyStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, patience: int = 5, ngram_size: int = 10):
        super().__init__()
        self.tokenizer = tokenizer
        self.patience = patience
        self.ngram_size = ngram_size
        self.last_tokens = []
        self.repeat_count = 0
        self.max_repeats = 3

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        current_token = input_ids[0][-1].item()
        self.last_tokens.append(current_token)
        
        if len(self.last_tokens) > 2 * self.ngram_size:
            self.last_tokens.pop(0)
        
        if len(self.last_tokens) < 2 * self.ngram_size:
            return False
        
        if len(self.last_tokens) >= 2 * self.ngram_size:
            first_ngram = self.last_tokens[-2*self.ngram_size:-self.ngram_size]
            second_ngram = self.last_tokens[-self.ngram_size:]
            
            if first_ngram == second_ngram:
                self.repeat_count += 1
                if self.repeat_count >= self.patience:
                    return True
            else:
                self.repeat_count = 0
        
        return False

def generate_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    problems: Dict[str, Any],
    output_file: str,
    language: str,
    num_samples_per_task: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 2048,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_tasks: Optional[int] = None,
    stop_tokens: Optional[Dict[str, List[str]]] = None,
    pass_at_k: int = 1,
    enable_early_stopping: bool = True
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    task_ids = list(problems.keys())
    if max_tasks is not None:
        task_ids = task_ids[:max_tasks]

    actual_samples = max(num_samples_per_task, pass_at_k)

    if stop_tokens is None:
        stop_tokens = {
            "python": ["\ndef ", "\nclass ", "\nif __name__"],
            "java": ["\npublic ", "\nclass ", "\ninterface "],
            "cpp": ["\nvoid ", "\nint ", "\nclass "],
            "javascript": ["\nfunction ", "\nconst ", "\nclass "],
            "go": ["\nfunc ", "\ntype ", "\npackage "],
            "default": ["\nclass ", "\ndef ", "\nfunction "]
        }
    current_stop_tokens = stop_tokens.get(language, stop_tokens.get("default", []))

    for task_id in tqdm(task_ids, desc=f"Generating for {language}"):
        prompt = problems[task_id]["prompt"]
        stop_criteria = StopOnTokens(tokenizer, current_stop_tokens, device)
        
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])
        
        if enable_early_stopping:
            early_stopping = EarlyStoppingCriteria(tokenizer)
            stopping_criteria_list.append(early_stopping)

        for i in range(actual_samples):
            model_input = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                raw_generated_ids = model.generate(
                    **model_input,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=stopping_criteria_list,
                )
            
            generated_text = tokenizer.decode(raw_generated_ids[0], skip_special_tokens=True)
            
            completion = generated_text[len(prompt):]
            min_stop_idx = len(completion)
            for stop in current_stop_tokens:
                stop_idx = completion.find(stop)
                if stop_idx != -1:
                    min_stop_idx = min(min_stop_idx, stop_idx)
            
            processed_completion = completion[:min_stop_idx]
            processed_code = prompt + processed_completion

            sample = {
                "task_id": task_id,
                "generation": processed_code,
                "canonical_solution": problems[task_id]["canonical_solution"],
                "declaration": problems[task_id]["declaration"],
                "example_test": problems[task_id].get("example_test", ""),
                "prompt": prompt,
                "test": problems[task_id]["test"],
                "text": problems[task_id].get("text", None),
                "pass_at_k": pass_at_k
            }
            write_jsonl(output_file, [sample], append=True)

def main():
    parser = argparse.ArgumentParser(description="Generate HumanEval-X code using LoRA fine-tuned models")
    
    parser.add_argument("--model_path", type=str, required=True, help="Base model path or Hugging Face model name")
    parser.add_argument("--data_dir", type=str, required=True, help="Data directory containing dataset files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--dataset_type", type=str, default="auto", choices=["auto", "humaneval-x", "mceval"], help="Dataset type")
    parser.add_argument("--max_tasks", type=int, default=None, help="Max number of tasks for debugging")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of code samples per task")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty")
    parser.add_argument("--model_language", type=str, default="python", choices=["python", "cpp", "java", "go", "javascript"], help="Language of the LoRA model")
    parser.add_argument("--target_languages", type=str, nargs="+", default=["all"], help="Target languages for code generation")
    parser.add_argument("--pass_at_k", type=int, default=3, help="pass@k evaluation parameter")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens for generation")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cuda:0', or 'cpu'")
    
    parser.add_argument("--lora_adapters_dir", type=str, required=True, help="Root directory containing LoRA adapters")
    parser.add_argument("--checkpoint", type=str, default="final", help="Checkpoint: 'final', 'best', 'epoch-N', or specific checkpoint name")
    parser.add_argument("--list_checkpoints", action="store_true", help="List available checkpoints and exit")
    
    parser.add_argument("--enable_early_stopping", action="store_true", default=True, help="Enable early stopping")
    parser.add_argument("--disable_early_stopping", action="store_false", dest="enable_early_stopping", help="Disable early stopping")
    
    parser.add_argument("--multi_gpu", default=True, help="Enable multi-GPU support")
    parser.add_argument("--gpu_allocation", type=str, default=None, help="Custom GPU allocation strategy")

    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            if args.multi_gpu and torch.cuda.device_count() > 1:
                args.device = "cuda"
                print(f"Multi-GPU mode enabled, detected {torch.cuda.device_count()} GPUs")
            else:
                args.device = "cuda"
                print(f"Using single GPU mode: {args.device}")
        else:
            args.device = "cpu"
            print(f"No GPU detected, using CPU")
    
    print(f"Using device: {args.device}")
    if args.device == "cuda":
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    supported_languages = ["python", "cpp", "java", "go", "javascript"]
    
    if args.model_language.lower() not in supported_languages:
        print(f"Unsupported model language: {args.model_language}. Supported: {supported_languages}")
        return
    
    if "all" in args.target_languages:
        target_languages_to_process = supported_languages
    else:
        target_languages_to_process = [lang.lower() for lang in args.target_languages if lang.lower() in supported_languages]

    if not target_languages_to_process:
        print("No valid target languages specified. Exiting.")
        return
    
    print(f"Model language: {args.model_language}")
    print(f"Target languages: {target_languages_to_process}")

    if args.list_checkpoints:
        print(f"Listing checkpoints for model language {args.model_language}:")
        possible_adapter_names = [
            f"llama-3.1-8b-{args.model_language}-mceval-lora",
            f"llama-3.1-8b-{args.model_language}-lora",
            f"llama-3.1-8b-{args.model_language}-humaneval-lora",
        ]

        base_adapter_path = None
        for adapter_name in possible_adapter_names:
            candidate_path = os.path.join(args.lora_adapters_dir, adapter_name)
            if os.path.exists(candidate_path):
                base_adapter_path = candidate_path
                break

        if base_adapter_path:
            print(f"\n=== {args.model_language.upper()} Model Adapter ===")
            print(f"Adapter path: {base_adapter_path}")
            checkpoint_dirs = glob.glob(os.path.join(base_adapter_path, "checkpoint-*"))
            checkpoint_numbers = []
            for cp_dir in checkpoint_dirs:
                match = re.search(r'checkpoint-(\d+)', cp_dir)
                if match:
                    checkpoint_numbers.append(int(match.group(1)))
            checkpoint_numbers.sort()

            for cp_num in checkpoint_numbers:
                print(f"  - epoch-{cp_num}")
            print(f"  - final (default)")
            if os.path.exists(os.path.join(base_adapter_path, "best")):
                print(f"  - best")
        else:
            print(f"LoRA adapter not found for {args.model_language}, tried paths:")
            for adapter_name in possible_adapter_names:
                candidate_path = os.path.join(args.lora_adapters_dir, adapter_name)
                print(f"  - {candidate_path}")
        return

    print(f"Loading base model: {args.model_path}")
    
    device_map = args.device
    if args.multi_gpu and torch.cuda.device_count() > 1:
        if args.gpu_allocation:
            try:
                device_map = {}
                for allocation in args.gpu_allocation.split(","):
                    gpu_id, layers = allocation.split(":")
                    start, end = map(int, layers.split("-"))
                    for layer in range(start, end + 1):
                        device_map[layer] = int(gpu_id)
                print(f"Using custom GPU allocation: {device_map}")
            except Exception as e:
                print(f"Failed to parse custom GPU allocation: {e}, using auto allocation")
                device_map = "auto"
        else:
            device_map = "auto"
            print("Using automatic multi-GPU allocation")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map
    )
    base_model_name = os.path.basename(args.model_path.rstrip('/'))
    
    if isinstance(device_map, dict) or device_map == "auto":
        print("Model layer distribution:")
        if hasattr(base_model, "hf_device_map"):
            for layer, device in base_model.hf_device_map.items():
                print(f"  {layer}: {device}")
        else:
            print("  Unable to get detailed model layer distribution")

    print(f"Loading LoRA adapter for model language: {args.model_language}")
    
    possible_adapter_names = [
        f"{base_model_name}-{args.model_language}-mceval-lora",
        f"{base_model_name}-{args.model_language}-lora",
        f"{base_model_name}-{args.model_language}-humaneval-lora",
    ]

    base_adapter_path = None
    for adapter_name in possible_adapter_names:
        candidate_path = os.path.join(args.lora_adapters_dir, adapter_name)
        if os.path.exists(candidate_path):
            base_adapter_path = candidate_path
            print(f"Found adapter: {adapter_name}")
            break

    if base_adapter_path is None:
        print(f"LoRA adapter not found for model language '{args.model_language}'. Tried paths:")
        for adapter_name in possible_adapter_names:
            candidate_path = os.path.join(args.lora_adapters_dir, adapter_name)
            print(f"  - {candidate_path}")
        return

    adapter_path = base_adapter_path
    if args.checkpoint != "final":
        if args.checkpoint == "best":
            best_path = os.path.join(base_adapter_path, "best")
            if os.path.exists(best_path):
                adapter_path = best_path
                print(f"Using best checkpoint: {adapter_path}")
            else:
                print(f"Best checkpoint not found, using final model")
        elif args.checkpoint.startswith("epoch-"):
            try:
                epoch_num = args.checkpoint.split("-")[1]
                checkpoint_path = os.path.join(base_adapter_path, f"checkpoint-{epoch_num}")
                if os.path.exists(checkpoint_path):
                    adapter_path = checkpoint_path
                    print(f"Using epoch-{epoch_num} checkpoint: {adapter_path}")
                else:
                    print(f"Checkpoint epoch-{epoch_num} not found, using final model")
            except:
                print(f"Invalid epoch format: {args.checkpoint}, using final model")
        else:
            custom_path = os.path.join(base_adapter_path, args.checkpoint)
            if os.path.exists(custom_path):
                adapter_path = custom_path
                print(f"Using custom checkpoint: {adapter_path}")
            else:
                print(f"Custom checkpoint {args.checkpoint} not found, using final model")

    print(f"Loading LoRA adapter from: {adapter_path}")
    try:
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        peft_model.eval()

        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print(f"Failed to load LoRA model for {args.model_language}: {e}")
        return

    for target_language in target_languages_to_process:
        print(f"Generating {target_language} code using {args.model_language} model")

        data_file = None
        possible_files = []

        if args.dataset_type == "mceval" or "mceval" in args.data_dir.lower():
            lang_map = {
                "python": "Python",
                "js": "JavaScript",
                "javascript": "JavaScript",
                "java": "Java",
                "cpp": "CPP",
                "go": "Go"
            }
            mceval_lang = lang_map.get(target_language, target_language.capitalize())
            possible_files = [
                os.path.join(args.data_dir, f"{mceval_lang}.jsonl"),
                os.path.join(args.data_dir, f"{mceval_lang}.jsonl.gz"),
            ]
        else:
            if target_language == "js":
                possible_files = [
                    os.path.join(args.data_dir, f"humaneval_javascript.jsonl.gz"),
                    os.path.join(args.data_dir, f"humaneval_js.jsonl.gz"),
                    os.path.join(args.data_dir, f"humaneval_javascript.jsonl"),
                    os.path.join(args.data_dir, f"humaneval_js.jsonl"),
                ]
            else:
                possible_files = [
                    os.path.join(args.data_dir, f"humaneval_{target_language}.jsonl.gz"),
                    os.path.join(args.data_dir, f"humaneval_{target_language}.jsonl"),
                ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                data_file = file_path
                break

        if data_file is None:
            print(f"Data file not found for language {target_language}, skipping. Tried paths: {possible_files}")
            continue

        try:
            problems = read_problems_unified(data_file, args.dataset_type)
            print(f"Loaded {len(problems)} {target_language} problems")
        except Exception as e:
            print(f"Error loading data file: {str(e)}")
            continue

        lang_output_dir = Path(args.output_dir) / f"{args.model_language}_model" / target_language
        lang_output_dir.mkdir(parents=True, exist_ok=True)

        if args.checkpoint == "final":
            output_file = str(lang_output_dir / "samples.jsonl")
        else:
            checkpoint_name = args.checkpoint.replace("/", "_")
            output_file = str(lang_output_dir / f"samples_{checkpoint_name}.jsonl")
        
        if os.path.exists(output_file):
            os.remove(output_file)

        generate_samples(
            model=peft_model,
            tokenizer=tokenizer,
            problems=problems,
            output_file=output_file,
            language=target_language,
            num_samples_per_task=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            max_tasks=args.max_tasks,
            pass_at_k=args.pass_at_k,
            enable_early_stopping=args.enable_early_stopping
        )
        print(f"Finished generating {target_language} code using {args.model_language} model. Results saved to {output_file}")
    
    print("All specified languages have been processed.")

if __name__ == "__main__":
    main()