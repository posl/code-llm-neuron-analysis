#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer
from utils import create_structured_prompt, clean_summary_for_evaluation
from bleu import bleu as moses_bleu
from lora_model_wrapper import LoRAModel

ALL_LANGUAGES = ['python', 'java', 'go', 'javascript', 'php', 'ruby']
LOW_RESOURCE_LANGUAGE = 'ruby'

class SimpleModelComparator:
    
    def __init__(self):
        self.tokenizer = None
        self.original_model = None
        self.base1_model = None
        self.lora_model = None
        self.cached_original_results = None
        
    def load_models(self, original_model_path: str, compare_model_path: str, lora_model_path: str = None, skip_original: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(original_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not skip_original:
            self.original_model = AutoModelForCausalLM.from_pretrained(
                original_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
        else:
            self.original_model = None

        if lora_model_path:
            self.lora_model = LoRAModel.load_model(lora_model_path, device='auto')
            self.base1_model = None
        else:
            self.base1_model = AutoModelForCausalLM.from_pretrained(
                str(compare_model_path),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

    def load_cached_results(self, cache_file_path: str, languages: List[str] = None):
        cache_path = Path(cache_file_path)
        if not cache_path.exists():
            return False

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)

            if 'detailed_results' not in cached_data:
                return False

            if languages:
                filtered_results = {}
                for lang in languages:
                    if lang in cached_data['detailed_results']:
                        filtered_results[lang] = cached_data['detailed_results'][lang]
                self.cached_original_results = filtered_results
            else:
                self.cached_original_results = cached_data['detailed_results']

            return True

        except Exception:
            return False
    
    def load_test_data(self, data_path: str, languages: List[str] = None, max_samples_per_lang: int = None) -> Dict[str, List[Dict[str, Any]]]:
        if languages is None:
            languages = ALL_LANGUAGES
        
        test_data = {}
        
        for language in languages:
            test_file = Path(data_path) / language / 'test' / f'{language}_test.json'
            
            if not test_file.exists():
                continue
                
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if max_samples_per_lang and len(data) > max_samples_per_lang:
                data = data[:max_samples_per_lang]
            
            test_data[language] = data
        
        return test_data

    def analyze_summary_lengths(self, test_data: Dict[str, List[Dict[str, Any]]]):
        for language, samples in test_data.items():
            lengths = [len(sample['summary'].split()) for sample in samples]

            if lengths:
                avg_length = sum(lengths) / len(lengths)
                min_length = min(lengths)
                max_length = max(lengths)

                short_count = sum(1 for l in lengths if l <= 3)
                medium_count = sum(1 for l in lengths if 4 <= l <= 8)
                long_count = sum(1 for l in lengths if l > 8)

    def generate_summary(self, model, code: str, language: str, gen_config: Dict[str, Any]) -> tuple:
        try:
            item = {'code': code, 'language': language}
            prompt = create_structured_prompt(item, for_inference=True)

            if isinstance(model, LoRAModel):
                tokenizer = model.tokenizer
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=550,
                    truncation=True,
                    padding=False
                )

                device = getattr(model, 'input_device', 'cuda')
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)

                with torch.no_grad():
                    outputs = model.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=gen_config['max_new_tokens'],
                        min_new_tokens=gen_config['min_new_tokens'],
                        do_sample=gen_config['do_sample'],
                        temperature=gen_config['temperature'],
                        top_p=gen_config['top_p'],
                        repetition_penalty=gen_config['repetition_penalty'],
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )

                generated_text = tokenizer.decode(
                    outputs[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip()

            else:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=550,
                    truncation=True,
                    padding=True
                )

                device = next(model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=gen_config['max_new_tokens'],
                        min_new_tokens=gen_config['min_new_tokens'],
                        do_sample=gen_config['do_sample'],
                        temperature=gen_config['temperature'],
                        top_p=gen_config['top_p'],
                        repetition_penalty=gen_config['repetition_penalty'],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )

                generated_text = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                ).strip()

            full_output = generated_text

            if not generated_text:
                return None, full_output

            processed_summary = clean_summary_for_evaluation(generated_text)

            return processed_summary, full_output

        except Exception:
            return None, "Generation failed"

    def compute_bleu_score(self, reference: str, hypothesis: str) -> float:
        try:
            clean_reference = clean_summary_for_evaluation(reference)
            clean_hypothesis = clean_summary_for_evaluation(hypothesis)

            moses_scores = moses_bleu([clean_reference], clean_hypothesis)
            return moses_scores[0]
        except Exception:
            return 0.0

    def compute_rouge_score(self, reference: str, hypothesis: str) -> float:
        try:
            clean_reference = clean_summary_for_evaluation(reference)
            clean_hypothesis = clean_summary_for_evaluation(hypothesis)

            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
            scores = scorer.score(clean_reference, clean_hypothesis)
            return scores['rougeL'].fmeasure
        except Exception:
            return 0.0


    def compare_on_samples(self, test_data: Dict[str, List[Dict[str, Any]]], gen_config: Dict[str, Any]) -> Dict[str, Any]:
        all_results = {}

        for language, samples in test_data.items():
            use_cached_original = (self.cached_original_results is not None and
                                 language in self.cached_original_results and
                                 self.original_model is None)

            if use_cached_original:
                cached_samples = self.cached_original_results[language]['samples']
                if len(cached_samples) < len(samples):
                    samples = samples[:len(cached_samples)]
                elif len(cached_samples) > len(samples):
                    pass

            lang_results = []
            valid_references = []
            valid_original_summaries = []
            valid_base1_summaries = []

            original_failed_count = 0
            base1_failed_count = 0
            both_failed_count = 0

            for i, sample in enumerate(tqdm(samples, desc=f"Evaluating {language}")):
                code = sample['code']
                reference = sample['summary']

                if use_cached_original and i < len(cached_samples):
                    cached_result = cached_samples[i]
                    original_summary = cached_result.get('original_processed_summary') or cached_result.get('original_summary')
                    if original_summary == "GENERATION_FAILED":
                        original_summary = None
                    original_bleu = cached_result.get('original_bleu', 0.0)
                    original_rouge = cached_result.get('original_rougeL_f', 0.0)
                    original_failed = cached_result.get('original_failed', original_summary is None)
                    original_full_output = cached_result.get('original_full_output', "N/A")
                else:
                    original_summary, original_full_output = self.generate_summary(self.original_model, code, language, gen_config)
                    original_failed = original_summary is None
                    if not original_failed:
                        original_bleu = self.compute_bleu_score(reference, original_summary)
                        original_rouge = self.compute_rouge_score(reference, original_summary)
                    else:
                        original_bleu = original_rouge = 0.0

                if self.lora_model:
                    base1_summary, base1_full_output = self.generate_summary(self.lora_model, code, language, gen_config)
                else:
                    base1_summary, base1_full_output = self.generate_summary(self.base1_model, code, language, gen_config)
                base1_failed = base1_summary is None

                if original_failed:
                    original_failed_count += 1
                if base1_failed:
                    base1_failed_count += 1
                if original_failed and base1_failed:
                    both_failed_count += 1

                if not base1_failed:
                    base1_bleu = self.compute_bleu_score(reference, base1_summary)
                    base1_rouge = self.compute_rouge_score(reference, base1_summary)
                else:
                    base1_bleu = base1_rouge = 0.0

                if not original_failed and not base1_failed:
                    valid_references.append(reference)
                    valid_original_summaries.append(original_summary)
                    valid_base1_summaries.append(base1_summary)

                result = {
                    'sample_id': i + 1,
                    'language': language,
                    'code': code[:200] + "..." if len(code) > 200 else code,
                    'reference_summary': reference,
                    'original_processed_summary': original_summary if original_summary is not None else "GENERATION_FAILED",
                    'original_full_output': original_full_output,
                    'base1_processed_summary': base1_summary if base1_summary is not None else "GENERATION_FAILED",
                    'base1_full_output': base1_full_output,
                    'original_bleu': original_bleu,
                    'base1_bleu': base1_bleu,
                    'original_rougeL_f': original_rouge,
                    'base1_rougeL_f': base1_rouge,
                    'original_failed': original_failed,
                    'base1_failed': base1_failed,
                    'excluded_from_corpus': original_failed or base1_failed
                }

                lang_results.append(result)

            all_results[language] = {
                'samples': lang_results,
                'generation_stats': {
                    'total_samples': len(samples),
                    'valid_samples': len(valid_references),
                    'original_failed': original_failed_count,
                    'base1_failed': base1_failed_count,
                    'both_failed': both_failed_count,
                    'excluded_from_corpus': len(samples) - len(valid_references)
                }
            }
        
        return all_results
    
    def print_comparison_results(self, results: Dict[str, Any]):
        print("\n" + "="*100)
        print("Model Comparison Results")
        print("="*100)

        print(f"\nGeneration Statistics:")
        print("-" * 80)
        print(f"{'Language':<12} {'Total':<8} {'Valid':<8} {'Orig Failed':<12} {'Base1 Failed':<13} {'Both Failed':<12} {'Success Rate':<12}")
        print("-" * 80)

        total_samples = 0
        total_valid = 0
        total_orig_failed = 0
        total_base1_failed = 0
        total_both_failed = 0

        for language, lang_data in results.items():
            stats = lang_data['generation_stats']
            success_rate = (stats['valid_samples'] / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0

            print(f"{language.capitalize():<12} {stats['total_samples']:<8} {stats['valid_samples']:<8} "
                  f"{stats['original_failed']:<12} {stats['base1_failed']:<13} {stats['both_failed']:<12} {success_rate:<11.1f}%")

            total_samples += stats['total_samples']
            total_valid += stats['valid_samples']
            total_orig_failed += stats['original_failed']
            total_base1_failed += stats['base1_failed']
            total_both_failed += stats['both_failed']

        print("-" * 80)
        overall_success_rate = (total_valid / total_samples * 100) if total_samples > 0 else 0
        print(f"{'Overall':<12} {total_samples:<8} {total_valid:<8} "
              f"{total_orig_failed:<12} {total_base1_failed:<13} {total_both_failed:<12} {overall_success_rate:<11.1f}%")
        print("-" * 80)

        print(f"\nBLEU-4 Comparison Results:")
        print("-" * 80)

        print(f"{'Language':<12} {'Samples':<8} {'Original BLEU':<15} {'Base1 BLEU':<15} {'Improvement':<12} {'Type':<12}")
        print("-" * 80)

        high_resource_improvements = []
        low_resource_improvements = []

        for language, lang_data in results.items():
            lang_results = lang_data['samples']
            stats = lang_data['generation_stats']

            valid_results = [r for r in lang_results if not r['excluded_from_corpus']]

            if valid_results:
                original_bleus = [r['original_bleu'] for r in valid_results]
                base1_bleus = [r['base1_bleu'] for r in valid_results]

                avg_original = sum(original_bleus) / len(original_bleus)
                avg_base1 = sum(base1_bleus) / len(base1_bleus)
                improvement = ((avg_base1 - avg_original) / avg_original * 100) if avg_original > 0 else 0
            else:
                avg_original = avg_base1 = improvement = 0.0

            lang_type = "Low-resource" if language == LOW_RESOURCE_LANGUAGE else "High-resource"

            print(f"{language.capitalize():<12} {stats['valid_samples']:<8} {avg_original:<15.4f} {avg_base1:<15.4f} {improvement:+8.2f}% {lang_type:<12}")

            if language == LOW_RESOURCE_LANGUAGE:
                low_resource_improvements.append(improvement)
            else:
                high_resource_improvements.append(improvement)

        print("-" * 80)

        if high_resource_improvements:
            avg_high = sum(high_resource_improvements) / len(high_resource_improvements)
            print(f"High-resource languages average improvement: {avg_high:+.2f}%")

        if low_resource_improvements:
            avg_low = sum(low_resource_improvements) / len(low_resource_improvements)
            print(f"Low-resource language improvement: {avg_low:+.2f}%")

        print(f"\nROUGE-L F1 Comparison Results:")
        print("-" * 80)
        print(f"{'Language':<12} {'Samples':<8} {'Original ROUGE-L':<18} {'Base1 ROUGE-L':<18} {'Improvement':<12} {'Type':<12}")
        print("-" * 80)

        for language, lang_data in results.items():
            lang_results = lang_data['samples']
            stats = lang_data['generation_stats']

            valid_results = [r for r in lang_results if not r['excluded_from_corpus']]

            if valid_results:
                original_rougeL = [r['original_rougeL_f'] for r in valid_results]
                base1_rougeL = [r['base1_rougeL_f'] for r in valid_results]

                avg_original_rougeL = sum(original_rougeL) / len(original_rougeL)
                avg_base1_rougeL = sum(base1_rougeL) / len(base1_rougeL)
                rouge_improvement = ((avg_base1_rougeL - avg_original_rougeL) / avg_original_rougeL * 100) if avg_original_rougeL > 0 else 0
            else:
                avg_original_rougeL = avg_base1_rougeL = rouge_improvement = 0.0

            lang_type = "Low-resource" if language == LOW_RESOURCE_LANGUAGE else "High-resource"

            print(f"{language.capitalize():<12} {stats['valid_samples']:<8} {avg_original_rougeL:<18.4f} {avg_base1_rougeL:<18.4f} {rouge_improvement:+8.2f}% {lang_type:<12}")

        print("-" * 80)

        print("="*100)
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary_table = []
        high_resource_improvements = []
        low_resource_improvements = []

        for language, lang_data in results.items():
            lang_results = lang_data['samples']
            stats = lang_data['generation_stats']

            valid_results = [r for r in lang_results if not r['excluded_from_corpus']]

            if valid_results:
                original_bleus = [r['original_bleu'] for r in valid_results]
                base1_bleus = [r['base1_bleu'] for r in valid_results]

                avg_original = sum(original_bleus) / len(original_bleus)
                avg_base1 = sum(base1_bleus) / len(base1_bleus)
                improvement = ((avg_base1 - avg_original) / avg_original * 100) if avg_original > 0 else 0

                original_rougeL = [r['original_rougeL_f'] for r in valid_results]
                base1_rougeL = [r['base1_rougeL_f'] for r in valid_results]
                avg_original_rougeL = sum(original_rougeL) / len(original_rougeL)
                avg_base1_rougeL = sum(base1_rougeL) / len(base1_rougeL)
                rouge_improvement = ((avg_base1_rougeL - avg_original_rougeL) / avg_original_rougeL * 100) if avg_original_rougeL > 0 else 0
            else:
                avg_original = avg_base1 = improvement = 0.0
                avg_original_rougeL = avg_base1_rougeL = rouge_improvement = 0.0

            lang_type = "Low-resource" if language == LOW_RESOURCE_LANGUAGE else "High-resource"

            summary_table.append({
                'language': language.capitalize(),
                'total_samples': stats['total_samples'],
                'valid_samples': stats['valid_samples'],
                'original_failed': stats['original_failed'],
                'base1_failed': stats['base1_failed'],
                'both_failed': stats['both_failed'],
                'success_rate': round((stats['valid_samples'] / stats['total_samples'] * 100) if stats['total_samples'] > 0 else 0, 2),
                'original_bleu': round(avg_original, 4),
                'base1_bleu': round(avg_base1, 4),
                'improvement_percent': round(improvement, 2),
                'original_rougeL': round(avg_original_rougeL, 4),
                'base1_rougeL': round(avg_base1_rougeL, 4),
                'rougeL_improvement_percent': round(rouge_improvement, 2),
                'type': lang_type
            })

            if language == LOW_RESOURCE_LANGUAGE:
                low_resource_improvements.append(improvement)
            else:
                high_resource_improvements.append(improvement)

        overall_stats = {
            'high_resource_avg_improvement': round(sum(high_resource_improvements) / len(high_resource_improvements), 2) if high_resource_improvements else 0,
            'low_resource_avg_improvement': round(sum(low_resource_improvements) / len(low_resource_improvements), 2) if low_resource_improvements else 0,
            'total_samples': sum(len(lang_data['samples']) for lang_data in results.values()),
            'total_languages': len(results)
        }

        results_with_meta = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary_table': summary_table,
            'overall_statistics': overall_stats,
            'detailed_results': results
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, ensure_ascii=False, indent=2)

        csv_path = output_path.with_suffix('.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("Language,Total_Samples,Valid_Samples,Original_Failed,Base1_Failed,Both_Failed,Success_Rate,Original_BLEU,Base1_BLEU,BLEU_Improvement_Percent,Original_ROUGE_L,Base1_ROUGE_L,ROUGE_L_Improvement_Percent,Type\n")
            for row in summary_table:
                f.write(f"{row['language']},{row['total_samples']},{row['valid_samples']},{row['original_failed']},{row['base1_failed']},{row['both_failed']},{row['success_rate']},{row['original_bleu']},{row['base1_bleu']},{row['improvement_percent']},{row['original_rougeL']},{row['base1_rougeL']},{row['rougeL_improvement_percent']},{row['type']}\n")

def main():
    parser = argparse.ArgumentParser(description="Model comparison script")
    
    parser.add_argument('--original_model_path', type=str, required=True,
                       help='Original model path')
    parser.add_argument('--compare_model_path', type=str, required=True,
                       help='Fine-tuned model path')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Data directory path')
    parser.add_argument('--lora_model_path', type=str, default=None,
                       help='LoRA model path')
    
    parser.add_argument('--use_cached_original', action='store_true', default=False,
                       help='Use cached original model results')
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable cache')
    parser.add_argument('--cache_file', type=str, default=None,
                       help='Cache file path')
    
    parser.add_argument('--languages', nargs='+', default=['ruby'],
                       help='Test language list')
    parser.add_argument('--max_samples_per_lang', type=int, default=10,
                       help='Max samples per language')
    parser.add_argument('--output_file', default='simple_comparison.json',
                       help='Output file name')
    
    parser.add_argument('--results_path', type=str, default='./results',
                       help='Results directory path')
    
    args = parser.parse_args()
    
    if args.no_cache:
        args.use_cached_original = False
    
    comparator = SimpleModelComparator()
    
    try:
        print(f"\nModel Configuration:")
        print(f"Original model: {args.original_model_path}")
        if args.lora_model_path:
            print(f"LoRA model: {args.lora_model_path}")
        else:
            print(f"Base1 model: {args.compare_model_path}")
        print(f"Use cache: {args.use_cached_original}")
        if args.use_cached_original and args.cache_file:
            print(f"Cache file: {args.cache_file}")
        
        if args.use_cached_original and args.cache_file:
            cache_loaded = comparator.load_cached_results(args.cache_file, args.languages)
            if not cache_loaded:
                print(f"Cannot load cache file: {args.cache_file}")
                print("Switching to full inference mode")
                args.use_cached_original = False
        
        comparator.load_models(args.original_model_path, args.compare_model_path,
                             lora_model_path=args.lora_model_path,
                             skip_original=args.use_cached_original)
        
        test_data = comparator.load_test_data(args.data_path, args.languages, args.max_samples_per_lang)
        
        if not test_data:
            print("No test data found")
            return
        
        comparator.analyze_summary_lengths(test_data)
        
        gen_config = {
            'max_new_tokens': 128,
            'min_new_tokens': 3,
            'do_sample': True,
            'temperature': 0.1,
            'top_p': 0.95,
            'repetition_penalty': 1.1,
        }
        
        results = comparator.compare_on_samples(test_data, gen_config)
        
        comparator.print_comparison_results(results)
        
        output_path = Path(args.results_path) / "evaluation" / args.output_file
        comparator.save_results(results, output_path)
        
        print(f"\nComparison completed!")
        print(f"Results saved to: {output_path}")
        
        if args.use_cached_original:
            print(f"Used cached original model results")
    
    except Exception as e:
        print(f"Error during comparison: {e}")
        raise

if __name__ == "__main__":
    main()
