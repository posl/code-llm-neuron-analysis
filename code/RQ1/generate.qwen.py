import os, json, gzip, gc, logging, argparse
from pathlib import Path
from typing import Dict, Any, Union, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from human_eval.data import read_problems, write_jsonl
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

def read_problems_unified(data_file: str, dataset_type: str = "auto") -> Dict[str, Any]:
    if dataset_type == "auto":
        dataset_type = "mceval" if "mceval" in data_file.lower() else "humaneval-x"
    if dataset_type == "humaneval-x":
        return read_problems(data_file)

    opener = gzip.open if data_file.endswith(".gz") else open
    problems = {}
    with opener(data_file, "rt", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            problems[p["task_id"]] = {
                "task_id": p["task_id"],
                "prompt": p["prompt"],
                "canonical_solution": p["canonical_solution"],
                "test": p["test"],
                "declaration": p.get("signature", ""),
                "text": p.get("docstring", ""),
            }
    logger.info("McEval loaded: %d tasks", len(problems))
    return problems

def extract_code_block(prompt: str, full_text: str) -> str:
    start_brace_index = prompt.rfind('{')
    if start_brace_index == -1:
        return full_text  
    prefix = full_text[:start_brace_index + 1]
    body_and_extra = full_text[start_brace_index + 1:]

    brace_count = 1
    end_brace_index = -1

    for i, char in enumerate(body_and_extra):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        
        if brace_count == 0:
            end_brace_index = i
            break
    
    if end_brace_index != -1:
        return prefix + body_and_extra[:end_brace_index + 1]
    else:

        return full_text

def post_process_code(prompt: str, generated_text: str, stop_tokens: List[str], lang: str) -> str:
    if lang in ["java", "javascript", "cpp", "go", "rust", "csharp"]:
        processed_text = extract_code_block(prompt, generated_text)
    else:
        processed_text = generated_text

    if processed_text.startswith(prompt):
        completion = processed_text[len(prompt):]
    else:
        completion = generated_text[len(prompt):] if generated_text.startswith(prompt) else processed_text

    min_stop_idx = len(completion)
    for stop_token in stop_tokens:
        if stop_token in completion:
            token_idx = completion.find(stop_token)
            if token_idx != -1 and token_idx < min_stop_idx:
                min_stop_idx = token_idx
    completion = completion[:min_stop_idx]

    return prompt + completion

def generate_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    problems: Dict[str, Any],
    output_file: str,
    language: str,
    pass_at_k: int,
    num_samples_per_task: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
):
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        use_cache=True,
    )

    stop_tokens_map = {
        "python": ["\ndef ", "\nclass ", "\nif __name__", "\n#"],
        "java": ["\npublic ", "\nprivate ", "\nprotected ", "\nclass ", "\ninterface "],
        "cpp": ["\nvoid ", "\nint ", "\nfloat ", "\ndouble ", "\nclass ", "\nstruct "],
        "javascript": ["\nfunction ", "\nconst ", "\nlet ", "\nvar ", "\nclass "],
        "go": ["\nfunc ", "\ntype ", "\nvar ", "\nconst "],
        "default": ["\n```", "\n//", "\n/*"],
    }
    current_stop_tokens = stop_tokens_map.get(language, stop_tokens_map["default"])


    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as fout:
        for task_id, problem in tqdm(problems.items(), desc=f"Generating for {language}"):
            prompt = problem["prompt"]
            for _ in range(num_samples_per_task):
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, generation_config=gen_config)
                generated_completion = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[-1] :], skip_special_tokens=True
                )
                
                full_generated_text = prompt + generated_completion

                for sp in ("<|fim_middle|>", "<|fim_prefix|>", "<|fim_suffix|>", "<|endoftext|>"):
                    full_generated_text = full_generated_text.replace(sp, "")

                # 调用后处理函数进行清理
                processed_generation = post_process_code(
                    prompt=prompt,
                    generated_text=full_generated_text,
                    stop_tokens=current_stop_tokens,
                    lang=language,
                )

                sample = {
                    "task_id": task_id,
                    "prompt": prompt, 
                    "generation": processed_generation,
                    "pass_at_k": pass_at_k,
                    "canonical_solution": problem["canonical_solution"],
                    "test": problem["test"],
                }
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")

                del inputs, outputs, generated_completion, full_generated_text, processed_generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

            if torch.cuda.is_available():
                logger.debug(
                    "Task %s finished | GPU-0 alloc %.2f GB reserved %.2f GB",
                    task_id,
                    torch.cuda.memory_allocated(0) / 1e9,
                    torch.cuda.memory_reserved(0) / 1e9,
                )

    logger.info("Completed → %s", output_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default="0,1",)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--language", default="python,java,go,cpp,javascript")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None, help="调试专用")
    parser.add_argument("--load_in_8bit", action="store_true", help="OOM 时再开")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    n_gpu = len(args.gpu.split(","))
    max_mem = {i: "40GiB" for i in range(n_gpu)} 

    quantization_config = (
        BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
        if args.load_in_8bit
        else None
    )

    logger.info("Loading model %s ... This will be done only once.", args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_mem,
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Model and Tokenizer loaded successfully.")

    os.makedirs(args.output_dir, exist_ok=True)
    languages = [lang.strip() for lang in args.language.split(",") if lang.strip()]
    lang_map = {"js": "javascript"}

    for lang in languages:
        lang = lang_map.get(lang, lang)
        candidates = [
            f"{args.data_dir}/humaneval_{lang}.jsonl.gz",
            f"{args.data_dir}/humaneval_{lang}.jsonl",
            f"{args.data_dir}/{lang.capitalize()}.jsonl.gz",
            f"{args.data_dir}/{lang.capitalize()}.jsonl",
        ]
        data_file = next((c for c in candidates if Path(c).exists()), None)
        if not data_file:
            logger.warning("Skip %s — no data file found in %s", lang, candidates)
            continue

        problems = read_problems_unified(data_file)
        if args.max_samples:
            problems = dict(list(problems.items())[: args.max_samples])
            logger.info("Debug mode: only %d tasks", len(problems))

        output_file = os.path.join(args.output_dir, lang, "samples.jsonl")
        logger.info("Start %s → %s", lang, output_file)
        
        generate_samples(
            model=model,
            tokenizer=tokenizer,
            problems=problems,
            output_file=output_file,
            language=lang,
            pass_at_k=args.num_samples,
            num_samples_per_task=args.num_samples,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    logger.info("All languages finished!")


if __name__ == "__main__":
    main()