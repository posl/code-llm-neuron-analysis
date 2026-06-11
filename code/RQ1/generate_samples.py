#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from pathlib import Path
from typing import Dict, Any, Optional, List
from human_eval.data import write_jsonl, read_problems
import json
import gzip
from tqdm import tqdm

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
        raise ValueError(f"不支持的数据集类型: {dataset_type}")

class StopOnTokens(StoppingCriteria):
    def __init__(self, tokenizer: AutoTokenizer, stop_token_strings: List[str], device: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_token_strings = stop_token_strings

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.stop_token_ids = []
        for stop_str in stop_token_strings:
            token_ids = tokenizer.encode(stop_str, add_special_tokens=False)
            if token_ids:
                self.stop_token_ids.append(torch.tensor(token_ids, dtype=torch.long, device=device))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        for stop_str in self.stop_token_strings:
            if generated_text.endswith(stop_str):
                return True
        return False

def generate_samples(
    model_name: str,
    problems: Dict[str, Any],
    output_file: str,
    num_samples_per_task: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    repetition_penalty: float = 1.0,
    max_new_tokens: int = 2048,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_tasks: Optional[int] = None,
    stop_tokens: Optional[Dict[str, List[str]]] = None,
    language: str = "python",
    hook_output_dict: Optional[Dict[str, List[torch.Tensor]]] = None,
    language_task_means_dict: Optional[Dict[str, List[torch.Tensor]]] = None,
    pass_at_k: int = 1,
    enable_early_stopping: bool = True
):
    if model is None or tokenizer is None:
        if device == "auto":
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if max_tasks is not None:
        task_ids = list(problems.keys())[:max_tasks]
    else:
        task_ids = list(problems.keys())

    actual_samples = max(num_samples_per_task, pass_at_k)

    # Simplified stop tokens handling
    DEFAULT_STOP_TOKENS = {
        "python": ["\ndef ", "\nclass ", "\nif __name__", "\n# ", "\n```", "\nprint(", "\nassert "],
        "java": ["\npublic ", "\nprivate ", "\nprotected ", "\nclass ", "\ninterface ", "\nenum ", "\n// ", "\n/*", "\nimport ", "\npackage "],
        "cpp": ["\nvoid ", "\nint ", "\nfloat ", "\ndouble ", "\nchar ", "\nbool ", "\nclass ", "\nstruct ", "\n// ", "\n/*"],
        "javascript": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass ", "\n// ", "\n/*", "\nexport ", "\nimport "],
        "go": ["\nfunc ", "\ntype ", "\nvar ", "\nconst ", "\npackage ", "\nimport ", "\n// "],
        "rust": ["\nfn ", "\npub ", "\nstruct ", "\nenum ", "\nimpl ", "\nuse ", "\n// ", "\nlet "],
        "default": ["\nclass ", "\ndef ", "\nfunction ", "\nif ", "\n// ", "\n/*", "\n```"]
    }
    
    if stop_tokens is None:
        stop_tokens = DEFAULT_STOP_TOKENS
    
    current_stop_tokens = stop_tokens.get(language, stop_tokens.get("default", DEFAULT_STOP_TOKENS["default"]))

    def detect_repetitive_output(text):
        if len(text) < 120:
            return False

        lines = text.split('\n')
        if len(lines) < 6:
            return False
        consecutive_same = 0
        prev_line = ""

        for line in lines:
            line_stripped = line.strip()
            if line_stripped and line_stripped == prev_line and len(line_stripped) > 20:
                if any(keyword in line_stripped.lower() for keyword in
                      ['}', 'return', 'break', 'continue', 'pass', 'end', 'else', 'if ', 'for ', 'while ']):
                    consecutive_same += 1
                    if consecutive_same >= 6:
                        return True
                else:
                    consecutive_same += 1
                    if consecutive_same >= 4:
                        return True
            else:
                consecutive_same = 0
            prev_line = line_stripped
        tail_length = min(200, len(text) // 2)
        tail = text[-tail_length:]
        sentences = [s.strip() for s in tail.replace('\n', '.').split('.') if s.strip()]
        if sentences:
            last_sentence = sentences[-1]
            if len(last_sentence) > 20:
                if any(keyword in last_sentence.lower() for keyword in ['return', 'if ', 'for ', 'while ', 'func ', 'def ', '}']):
                    count = tail.count(last_sentence)
                    if count >= 5:
                        return True
                else:
                    count = tail.count(last_sentence)
                    if count >= 3:
                        return True
        for pattern_len in [50, 80, 120]:
            if len(text) >= pattern_len * 4:
                pattern = text[-pattern_len:]
                if len(pattern.strip()) > pattern_len * 0.7:
                    code_structure_chars = (pattern.count('{') + pattern.count('}') +
                                           pattern.count('    ') + pattern.count('\n') +
                                           pattern.count('var ') + pattern.count('func ') +
                                           pattern.count('if ') + pattern.count('for '))
                    if code_structure_chars > len(pattern) * 0.4:
                        continue

                    count = text.count(pattern)
                    if count >= 5:
                        return True

        return False
    
    def generate_one_completion(task_id, prompt, stop_criteria_list):
        try:
            model_input = tokenizer(prompt, return_tensors="pt")
            input_ids_len = model_input.input_ids.shape[1]

            actual_device = device
            if device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"

            model_input = {k: v.to(actual_device) for k, v in model_input.items()}
            model.eval()
            with torch.no_grad():
                input_ids = model_input["input_ids"]
                attention_mask = model_input.get("attention_mask", None)
                for step in range(max_new_tokens):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits[:, -1, :]
                    if temperature > 0:
                        logits = logits / temperature
                        if top_k > 0:
                            top_k_logits, top_k_indices = torch.topk(logits, top_k)
                            logits = torch.full_like(logits, float('-inf'))
                            logits.scatter_(1, top_k_indices, top_k_logits)

                        if top_p < 1.0:
                            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0
                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            logits[indices_to_remove] = float('-inf')

                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)

                    input_ids = torch.cat([input_ids, next_token], dim=-1)
                    if attention_mask is not None:
                        attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
                    current_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                    if stop_criteria_list(input_ids, None):
                        break
                    if enable_early_stopping and step > 50 and step % 20 == 0:
                        completion_so_far = current_text[len(prompt):]

                        if len(completion_so_far) > 200:
                            if detect_repetitive_output(completion_so_far):
                                break
                result = tokenizer.decode(input_ids[0], skip_special_tokens=True)

                new_tokens_count = input_ids.shape[1] - input_ids_len
                completion_length = len(result) - len(prompt)

                return result

        except Exception as e:
            time.sleep(3)
            return generate_one_completion(task_id, prompt, stop_criteria_list)
    
    def post_process_code(prompt, generated_text, stop_tokens, lang):
        if len(generated_text) <= len(prompt):
            return generated_text
        completion = generated_text[len(prompt):]

        min_stop_idx = len(completion)
        for stop_token in stop_tokens:
            if stop_token in completion:
                token_idx = completion.find(stop_token)
                if token_idx >= 0 and token_idx < min_stop_idx:
                    min_stop_idx = token_idx

        if min_stop_idx < len(completion):
            completion = completion[:min_stop_idx]

        if detect_repeated_functions(completion, lang):
            completion = extract_first_function(completion, lang)

        return prompt + completion

    def detect_repeated_functions(code, lang):
        """Detect if code contains repeated function definitions."""
        patterns = {
            "python": (["\ndef ", "\nclass "], lambda counts: any(c > 1 for c in counts)),
            "javascript": (["\nfunction ", " => {", "\nclass "], lambda counts: any(c > 1 for c in counts)),
            "typescript": (["\nfunction ", " => {", "\nclass "], lambda counts: any(c > 1 for c in counts)),
            "java": (["public ", "private ", "protected ", "void ", "int ", "boolean ", "String ", "double ", "float ", "\nclass "],
                    lambda counts: sum(counts[:-1]) > 1 or counts[-1] > 1),
            "cpp": (["public ", "private ", "protected ", "void ", "int ", "boolean ", "String ", "double ", "float ", "\nclass "],
                   lambda counts: sum(counts[:-1]) > 1 or counts[-1] > 1),
            "c#": (["public ", "private ", "protected ", "void ", "int ", "boolean ", "String ", "double ", "float ", "\nclass "],
                  lambda counts: sum(counts[:-1]) > 1 or counts[-1] > 1),
            "go": (["\nfunc "], lambda counts: counts[0] > 1),
            "rust": (["\nfn ", "\nimpl "], lambda counts: any(c > 1 for c in counts))
        }
        
        # Get patterns for language or use default
        lang_patterns, check_func = patterns.get(lang,
            (["\ndef ", "\nfunction ", "\nclass "], lambda counts: any(c > 1 for c in counts)))
        
        # Count occurrences
        counts = [code.count(pattern if pattern.startswith("\n") else f"\n{pattern}")
                 if not pattern.startswith("\n") and lang in ["java", "cpp", "c#"]
                 else code.count(pattern)
                 for pattern in lang_patterns]
        
        return check_func(counts)

    def extract_first_function(code, lang):
        """Extract the first function from code."""
        patterns_map = {
            "python": ["\ndef ", "\nclass "],
            "javascript": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass "],
            "typescript": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass "],
            "java": ["\npublic ", "\nprivate ", "\nprotected ", "\nclass "],
            "cpp": ["\npublic ", "\nprivate ", "\nprotected ", "\nclass "],
            "c#": ["\npublic ", "\nprivate ", "\nprotected ", "\nclass "],
            "go": ["\nfunc ", "\ntype "],
            "rust": ["\nfn ", "\npub fn ", "\nimpl ", "\nstruct "]
        }
        
        patterns = patterns_map.get(lang, ["\ndef ", "\nfunction ", "\nclass "])

        function_starts = []
        for pattern in patterns:
            idx = code.find(pattern)
            if idx != -1:
                function_starts.append((idx, pattern))

        if not function_starts:
            return code

        start_idx, _ = min(function_starts, key=lambda x: x[0])

        if lang == "python":
            lines = code[start_idx:].split('\n')
            first_function_lines = [lines[0]]

            def_line = lines[0]
            def_indent = len(def_line) - len(def_line.lstrip())

            for line in lines[1:]:
                if not line.strip() or len(line) - len(line.lstrip()) > def_indent:
                    first_function_lines.append(line)
                else:
                    break

            extracted_code = code[:start_idx] + '\n'.join(first_function_lines)

        elif lang in ["javascript", "java", "cpp", "c#", "go", "rust"]:
            lines = code[start_idx:].split('\n')
            first_function_lines = [lines[0]]

            brace_count = 0
            in_function = False

            for line in lines[1:]:
                for char in line:
                    if char == '{':
                        in_function = True
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1

                first_function_lines.append(line)

                if in_function and brace_count <= 0:
                    break

            extracted_code = code[:start_idx] + '\n'.join(first_function_lines)

        else:
            extracted_code = code[:start_idx] + code[start_idx:].split('\n\n')[0]

        return extracted_code

    for task_id in tqdm(task_ids, desc="生成代码"):
        if hook_output_dict is not None:
            hook_output_dict.clear()

        prompt = problems[task_id]["prompt"]

        stop_criteria = StopOnTokens(tokenizer, current_stop_tokens, device)
        stopping_criteria_list = StoppingCriteriaList([stop_criteria])

        for i in range(actual_samples):
            raw_generated_code = generate_one_completion(task_id, prompt, stopping_criteria_list)

            processed_code = post_process_code(prompt, raw_generated_code, current_stop_tokens, language)

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

            if hook_output_dict is not None and language_task_means_dict is not None:
                for layer_key, step_activations_list in hook_output_dict.items():
                    if not step_activations_list:
                        continue
                    try:
                        if not all(t.dim() == 2 for t in step_activations_list):
                            continue
                        expected_dim = step_activations_list[0].size(-1)
                        if not all(t.size(-1) == expected_dim for t in step_activations_list):
                             continue

                        cat_activations = torch.cat(step_activations_list, dim=0)
                        task_mean = torch.mean(cat_activations, dim=0)

                        if layer_key not in language_task_means_dict:
                            language_task_means_dict[layer_key] = []
                        language_task_means_dict[layer_key].append(task_mean.cpu())
                    except Exception as proc_err:
                        pass

def main():
    import argparse
    import os
    from pathlib import Path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = Path(os.path.dirname(script_dir))

    parser = argparse.ArgumentParser(description="Generate HumanEval code using a large language model")
    parser.add_argument("--model_path", type=str, required=True,
                          help="Model path or Hugging Face model name")
    parser.add_argument("--data_dir", type=str, required=True,
                          help="Data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                          help="Output directory")
    parser.add_argument("--dataset_type", type=str, default="auto",
                          choices=["auto", "humaneval-x", "mceval"],
                          help="Dataset type")
    parser.add_argument("--max_tasks", type=int, default=None,
                          help="Maximum number of tasks to process")
    parser.add_argument("--num_samples", type=int, default=1,
                          help="Number of code samples to generate per task")
    parser.add_argument("--temperature", type=float, default=0.8,
                          help="Generation temperature")
    parser.add_argument("--top_p", type=float, default=0.95,
                          help="Nucleus sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                          help="Top-k sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                          help="Repetition penalty parameter")
    parser.add_argument("--languages", type=str, nargs="+", default=["all"],
                          help="List of programming languages to process")
    parser.add_argument("--pass_at_k", type=int, default=3,
                          help="pass@k evaluation parameter")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                          help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="auto",
                          help="Device to use")

    args = parser.parse_args()

    supported_languages = ["python", "cpp", "java", "go", "js"]

    if "all" in args.languages:
        languages_to_process = supported_languages
    else:
        languages_to_process = []
        for lang in args.languages:
            if lang.lower() in supported_languages:
                languages_to_process.append(lang.lower())

        if not languages_to_process:
            languages_to_process = ["python"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, device_map=args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    for language in languages_to_process:
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
            mceval_lang = lang_map.get(language, language.capitalize())
            possible_files = [
                os.path.join(args.data_dir, f"{mceval_lang}.jsonl"),
                os.path.join(args.data_dir, f"{mceval_lang}.jsonl.gz"),
            ]
        else:
            if language == "js":
                possible_files = [
                    os.path.join(args.data_dir, f"humaneval_javascript.jsonl.gz"),
                    os.path.join(args.data_dir, f"humaneval_js.jsonl.gz"),
                    os.path.join(args.data_dir, f"humaneval_javascript.jsonl"),
                    os.path.join(args.data_dir, f"humaneval_js.jsonl"),
                ]
            else:
                possible_files = [
                    os.path.join(args.data_dir, f"humaneval_{language}.jsonl.gz"),
                    os.path.join(args.data_dir, f"humaneval_{language}.jsonl"),
                ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                data_file = file_path
                break

        if data_file is None:
            continue

        lang_output_dir = output_dir / language
        lang_output_dir.mkdir(exist_ok=True)
        output_file = str(lang_output_dir / "samples.jsonl")

        try:
            problems = read_problems_unified(data_file, args.dataset_type)

            generate_samples(
                model_name=args.model_path,
                problems=problems,
                output_file=output_file,
                num_samples_per_task=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                model=model,
                tokenizer=tokenizer,
                device="cuda" if args.device == "auto" else args.device,
                max_tasks=args.max_tasks,
                language=language,
                pass_at_k=args.pass_at_k
            )

        except Exception as e:
            continue

if __name__ == "__main__":
    main()