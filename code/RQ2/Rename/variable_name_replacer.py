#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variable name replacer using LLM - Optimized version
"""

import os
import argparse
import asyncio
import aiohttp
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

from utils import (
    setup_paths,
    load_humaneval_dataset,
    assemble_complete_code,
    get_universal_task_id,
    write_jsonl_file,
    create_output_path,
    generate_statistics,
    calculate_success_rate
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize paths
paths = setup_paths()
ROOT_DIR = paths['ROOT_DIR']
HUMANEVAL_X_DIR = paths['HUMANEVAL_X_DIR']

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Replace variable names in code using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--api_token", type=str,
                      required=True,
                      help="API token")
    parser.add_argument("--model", type=str,
                      required=True,
                      help="LLM model name")
    parser.add_argument("--data_dir", type=str,
                      required=True,
                      help="humaneval-x data directory")
    parser.add_argument("--output_dir", type=str,
                      required=True,
                      help="Output directory")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Log level")
    parser.add_argument("--temperature", type=float, default=0.1,
                      help="LLM generation temperature")
    parser.add_argument("--target_languages", type=str, nargs="+",
                      default=["python", "cpp", "java", "go", "js", "rust"],
                      help="Target programming languages")
    parser.add_argument("--max_samples", type=int, default=None,
                      help="Maximum samples per language")
    parser.add_argument("--batch_size", type=int, default=3,
                      help="Async batch processing size")
    parser.add_argument("--max_retries", type=int, default=3,
                      help="Maximum API retries")
    parser.add_argument("--retry_delay", type=int, default=5,
                      help="Retry delay (seconds)")
    parser.add_argument("--validation_retries", type=int, default=2,
                      help="Maximum validation retries")
    parser.add_argument("--debug", action="store_true",
                      help="Debug mode")
    parser.add_argument("--no_stream", action="store_true",
                      help="Disable streaming output")
    
    return parser.parse_args()

def load_all_languages_data(
    data_dir: str, 
    languages: List[str], 
    max_samples: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """Load data for all specified languages"""
    task_data = defaultdict(dict)
    
    for language in languages:
        problems = load_humaneval_dataset(data_dir, language)
        
        if max_samples:
            keys = list(problems.keys())[:max_samples]
            problems = {k: problems[k] for k in keys}
        
        for task_id, sample in problems.items():
            universal_id = get_universal_task_id(task_id)
            task_data[universal_id][language] = {
                'prompt': sample.get('prompt', ''),
                'canonical_solution': sample.get('canonical_solution', ''),
                'declaration': sample.get('declaration', ''),
                'entry_point': sample.get('entry_point', ''),
                'test': sample.get('test', '')
            }
    
    logger.info(f"Loaded data for {len(task_data)} tasks")
    return task_data

def prepare_dataset(
    task_data: Dict[str, Dict[str, Any]], 
    languages: List[str]
) -> List[Dict[str, Any]]:
    """Prepare dataset for variable name replacement"""
    dataset = []
    
    for task_id, langs_data in task_data.items():
        for lang in languages:
            if lang in langs_data:
                lang_data = langs_data[lang]
                declaration = lang_data.get('declaration', '')
                solution = lang_data.get('canonical_solution', '')
                
                complete_code = assemble_complete_code(
                    declaration, solution, lang
                )
                
                if complete_code.strip():
                    dataset.append({
                        'task_id': task_id,
                        'language': lang,
                        'original_code': complete_code,
                        'renamed_code': None
                    })
    
    logger.info(f"Prepared {len(dataset)} samples for processing")
    return dataset

class VariableRenamer:
    """Handle variable renaming with LLM"""
    
    LANGUAGE_GUIDANCE = {
        "python": """
Python variable rules:
- Replace: local variables, parameters, instance variables (self.x)
- Keep: built-in functions, module names, class names
""",
        "cpp": """
C++ variable rules:
- Replace: local variables, parameters, member variables
- Keep: standard library names, class names, macros
""",
        "java": """
Java variable rules:
- Replace: local variables, parameters, member variables
- Keep: class names, package names, method names
""",
        "go": """
Go variable rules:
- Replace: local variables, parameters, struct fields
- Keep: package names, function names, type names
""",
        "js": """
JavaScript variable rules:
- Replace: let/const/var variables, parameters, object properties
- Keep: built-in objects, DOM API, library functions
""",
        "rust": """
Rust variable rules:
- Replace: let/mut variables, parameters
- Keep: trait names, struct names, standard library types
"""
    }
    
    @staticmethod
    def create_prompt(code: str, language: str) -> str:
        """Create replacement prompt"""
        guidance = VariableRenamer.LANGUAGE_GUIDANCE.get(
            language.lower(), ""
        )
        
        return f"""Replace all user-defined variable names in this {language} code with random alphanumeric combinations.

Rules:
1. Only replace user-defined variables
2. Use random names like "a7x", "zk9q", "r5t2m" (2-7 chars)
3. Keep consistency - same variable always gets same replacement
4. Don't change code structure or logic

{guidance}

Code:
```{language}
{code}
```

Return only the replaced code without explanations."""

    @staticmethod
    def post_process_response(response: str, language: str) -> str:
        """Extract code from LLM response"""
        code = response.strip()
        
        # Extract from code blocks
        markers = [
            f"```{language}", "```", 
            f"```{language.lower()}", 
            "```python", "```java", "```cpp", 
            "```javascript", "```js", "```go", "```rust"
        ]
        
        for marker in markers:
            if marker in code:
                start = code.find(marker) + len(marker)
                end = code.rfind("```")
                if end > start:
                    code = code[start:end].strip()
                    break
        
        # Remove comment blocks
        lines = []
        in_comment = False
        
        for line in code.split('\n'):
            stripped = line.strip()
            
            # Skip explanatory text
            if any(stripped.startswith(x) for x in [
                "This is", "Above", "I have", "Note"
            ]) or "replaced code" in stripped:
                continue
            
            # Handle multi-line comments
            if language.lower() in ["java", "cpp", "js", "javascript"]:
                if "/*" in stripped and "*/" in stripped:
                    continue
                elif "/*" in stripped:
                    in_comment = True
                    continue
                elif "*/" in stripped:
                    in_comment = False
                    continue
                elif in_comment:
                    continue
            
            lines.append(line)
        
        return '\n'.join(lines).strip()

async def call_llm_api(
    code: str,
    language: str,
    api_token: str,
    model: str,
    temperature: float = 0.1,
    max_retries: int = 3,
    retry_delay: int = 5,
    stream: bool = True
) -> Tuple[bool, str]:
    """Call LLM API for variable renaming"""
    prompt = VariableRenamer.create_prompt(code, language)
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": 4096,
        "temperature": temperature
    }
    
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://llm.chutes.ai/v1/chat/completions",
                    headers=headers,
                    json=body,
                    timeout=60
                ) as response:
                    if response.status != 200:
                        error = await response.text()
                        logger.error(f"API error (attempt {attempt+1}): {error}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(retry_delay)
                            continue
                        return False, f"API error: {response.status}"
                    
                    if stream:
                        full_response = await process_stream(response)
                    else:
                        data = await response.json()
                        if "choices" not in data:
                            return False, "Invalid response format"
                        full_response = data["choices"][0]["message"]["content"]
                    
                    renamed_code = VariableRenamer.post_process_response(
                        full_response, language
                    )
                    
                    if not renamed_code:
                        return False, "Empty result"
                    
                    return True, renamed_code
                    
        except Exception as e:
            logger.error(f"API call error (attempt {attempt+1}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
            else:
                return False, str(e)
    
    return False, "Max retries exceeded"

async def process_stream(response) -> str:
    """Process streaming response"""
    full_response = ""
    logger.info("Receiving LLM stream...")
    
    async for line in response.content:
        line = line.decode("utf-8").strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                json_data = json.loads(data)
                if "choices" in json_data and json_data["choices"]:
                    content = json_data["choices"][0].get("delta", {}).get("content", "")
                    if content:
                        full_response += content
            except:
                pass
    
    logger.info("Stream complete")
    return full_response

def validate_code(
    original: str, 
    renamed: str, 
    language: str
) -> Tuple[bool, str]:
    """Validate renamed code"""
    if not renamed or renamed.isspace():
        return False, "Empty code"
    
    # Check line count
    orig_lines = original.strip().split('\n')
    renamed_lines = renamed.strip().split('\n')
    
    if abs(len(renamed_lines) - len(orig_lines)) > max(2, len(orig_lines) * 0.1):
        return False, f"Line count mismatch"
    
    # Check for error markers
    error_markers = ["ERROR", "FAILED", "UNABLE TO", "COULD NOT"]
    for marker in error_markers:
        if marker in renamed.upper():
            return False, f"Contains error: {marker}"
    
    # Check syntax markers
    syntax_markers = {
        "python": ["def ", "class ", "import ", "return "],
        "java": ["public ", "private ", "class ", "return "],
        "cpp": ["#include", "int ", "void ", "return "],
        "go": ["func ", "package ", "import ", "return "],
        "js": ["function ", "const ", "let ", "return "],
        "rust": ["fn ", "struct ", "impl ", "use "]
    }
    
    for marker in syntax_markers.get(language.lower(), []):
        if marker in original and marker not in renamed:
            return False, f"Missing syntax: {marker}"
    
    # Check brackets
    for open_b, close_b in {'{': '}', '[': ']', '(': ')'}.items():
        if (original.count(open_b) != renamed.count(open_b) or 
            original.count(close_b) != renamed.count(close_b)):
            return False, "Bracket mismatch"
    
    return True, ""

async def process_single_sample(
    sample: Dict[str, Any],
    api_token: str,
    model: str,
    temperature: float,
    max_retries: int,
    retry_delay: int,
    stream: bool = True,
    validation_retries: int = 2
) -> Dict[str, Any]:
    """Process a single code sample"""
    original_code = sample['original_code']
    language = sample['language']
    
    success, renamed_code = await call_llm_api(
        original_code, language, api_token, model,
        temperature, max_retries, retry_delay, stream
    )
    
    if success:
        is_valid, error = validate_code(original_code, renamed_code, language)
        
        retries = 0
        while not is_valid and retries < validation_retries:
            logger.warning(f"Validation failed for {sample['task_id']}: {error}")
            
            adjusted_temp = min(temperature + 0.1 * (retries + 1), 0.9)
            success, renamed_code = await call_llm_api(
                original_code, language, api_token, model,
                adjusted_temp, max_retries, retry_delay, stream
            )
            
            if success:
                is_valid, error = validate_code(original_code, renamed_code, language)
            
            retries += 1
            await asyncio.sleep(1)
        
        success = success and is_valid
    
    sample['success'] = success
    if success:
        sample['renamed_code'] = renamed_code
    else:
        sample['error'] = renamed_code if isinstance(renamed_code, str) else "Failed"
    
    return sample

async def process_batch(
    batch: List[Dict[str, Any]],
    api_token: str,
    model: str,
    temperature: float,
    max_retries: int,
    retry_delay: int,
    stream: bool = True,
    validation_retries: int = 2
) -> List[Dict[str, Any]]:
    """Process a batch of samples"""
    tasks = [
        process_single_sample(
            sample, api_token, model, temperature,
            max_retries, retry_delay, stream, validation_retries
        )
        for sample in batch
    ]
    
    return await asyncio.gather(*tasks)

async def process_dataset(
    dataset: List[Dict[str, Any]],
    api_token: str,
    model: str,
    temperature: float,
    batch_size: int,
    max_retries: int,
    retry_delay: int,
    stream: bool = True,
    validation_retries: int = 2
) -> List[Dict[str, Any]]:
    """Process entire dataset"""
    processed = []
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(dataset) + batch_size - 1)//batch_size}")
        
        results = await process_batch(
            batch, api_token, model, temperature,
            max_retries, retry_delay, stream, validation_retries
        )
        processed.extend(results)
        
        if i + batch_size < len(dataset):
            await asyncio.sleep(1)
    
    return processed

def save_results(processed_dataset: List[Dict[str, Any]], output_dir: str):
    """Save processing results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by language
    by_language = defaultdict(list)
    for sample in processed_dataset:
        by_language[sample['language']].append(sample)
    
    # Save all samples
    all_path = os.path.join(output_dir, "all_renamed_samples.jsonl")
    write_jsonl_file(processed_dataset, all_path)
    
    # Save by language
    for lang, samples in by_language.items():
        lang_dir = os.path.join(output_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        
        # Save samples
        samples_path = os.path.join(lang_dir, f"renamed_samples_{lang}.jsonl")
        write_jsonl_file(samples, samples_path)
        
        # Save statistics
        stats = {
            "language": lang,
            "total_samples": len(samples),
            "successful_renames": sum(1 for s in samples if s.get('success', False)),
            "failed_renames": sum(1 for s in samples if not s.get('success', False)),
            "success_rate": calculate_success_rate(samples)
        }
        
        stats_path = os.path.join(lang_dir, "stats.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
    
    # Save overall statistics
    overall_stats = generate_statistics(processed_dataset)
    stats_path = os.path.join(output_dir, "overall_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(overall_stats, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"Overall success rate: {overall_stats['overall_rate']}%")

async def main():
    """Main function"""
    args = parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    logger.info("="*60)
    logger.info("Variable Name Replacer Started")
    logger.info(f"Model: {args.model}")
    logger.info(f"Languages: {', '.join(args.target_languages)}")
    logger.info("="*60)
    
    # Load data
    task_data = load_all_languages_data(
        args.data_dir, args.target_languages, args.max_samples
    )
    
    # Prepare dataset
    dataset = prepare_dataset(task_data, args.target_languages)
    
    # Debug mode
    if args.debug:
        logger.info("Debug mode: processing limited samples")
        debug_dataset = []
        lang_samples = defaultdict(list)
        
        for sample in dataset:
            lang_samples[sample['language']].append(sample)
        
        for lang, samples in lang_samples.items():
            debug_dataset.extend(samples[:min(2, len(samples))])
        
        dataset = debug_dataset
        logger.info(f"Debug: processing {len(dataset)} samples")
    
    # Process dataset
    logger.info(f"Starting to process {len(dataset)} samples...")
    processed = await process_dataset(
        dataset,
        args.api_token,
        args.model,
        args.temperature,
        args.batch_size,
        args.max_retries,
        args.retry_delay,
        stream=not args.no_stream,
        validation_retries=args.validation_retries
    )
    
    # Save results
    save_results(processed, args.output_dir)
    
    logger.info("="*60)
    logger.info("Processing Complete")
    logger.info("="*60)

if __name__ == "__main__":
    asyncio.run(main())