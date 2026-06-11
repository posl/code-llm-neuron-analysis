#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import json
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm
from rouge_score import rouge_scorer
import evaluate

from model_wrapper import LayerSelectiveModel
from data_processor import CodeSummarizationDataProcessor
from utils import create_structured_prompt, clean_summary_for_evaluation, get_stopping_criteria
from bleu import bleu as moses_bleu

class CodeSummarizationEvaluator:
    
    def __init__(self, model: LayerSelectiveModel, data_processor: CodeSummarizationDataProcessor, config: Dict[str, Any]):
        self.model = model
        self.data_processor = data_processor
        self.device = config.get('device', 'cuda')
        self.config = config
        
        self.rouge_scorer = rouge_scorer.RougeScorer(
            config['rouge_types'], 
            use_stemmer=True
        )
        
        try:
            self.meteor_metric = evaluate.load('meteor')
        except Exception:
            self.meteor_metric = None
        
        self.stop_sequences = [
            "\n\n```", "\n\nWrite", "\n\nSummary:", "```",
            "Parameters:", "Explanation:", "Example:", "Args:",
            ":param", ":return:", ":rtype:", ":raises:"
        ]
        self.stopping_criteria = get_stopping_criteria(self.data_processor.tokenizer, self.stop_sequences)
    
    def generate_summary(self, code: str, max_length: int = None) -> tuple:
        max_length = max_length or self.config['generation_config']['max_new_tokens']
        
        item = {'code': code, 'language': 'python'}
        input_text = create_structured_prompt(item, for_inference=True)
        
        inputs = self.data_processor.tokenizer(
            input_text,
            max_length=self.config['max_code_length'],
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        gen_config = self.config['generation_config']
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.config.get('bf16', False)):
                outputs = self.model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    stopping_criteria=self.stopping_criteria,
                    **gen_config
                )
        
        full_output = self.data_processor.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()
        
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        
        generated_text = self.data_processor.tokenizer.decode(
            generated_tokens, 
            skip_special_tokens=True
        ).strip()
        
        for stop_seq in self.stop_sequences:
            if stop_seq in generated_text:
                generated_text = generated_text.split(stop_seq)[0]

        summary = clean_summary_for_evaluation(generated_text)
        
        return summary, full_output

    def compute_bleu_score(self, clean_reference: str, clean_hypothesis: str) -> Dict[str, float]:
        try:
            moses_scores = moses_bleu([clean_reference], clean_hypothesis)
            bleu_scores = {
                'bleu': moses_scores[0],
                'bleu_1': moses_scores[1] if len(moses_scores) > 1 else 0.0,
                'bleu_2': moses_scores[2] if len(moses_scores) > 2 else 0.0,
                'bleu_3': moses_scores[3] if len(moses_scores) > 3 else 0.0,
                'bleu_4': moses_scores[0],
            }
            return bleu_scores
        except Exception:
            return {
                'bleu': 0.0,
                'bleu_1': 0.0,
                'bleu_2': 0.0,
                'bleu_3': 0.0,
                'bleu_4': 0.0,
            }
    
    def compute_rouge_score(self, clean_reference: str, clean_hypothesis: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(clean_reference, clean_hypothesis)
        
        rouge_scores = {}
        for rouge_type in self.config['rouge_types']:
            rouge_scores[f'{rouge_type}_precision'] = scores[rouge_type].precision
            rouge_scores[f'{rouge_type}_recall'] = scores[rouge_type].recall
            rouge_scores[f'{rouge_type}_fmeasure'] = scores[rouge_type].fmeasure
        
        return rouge_scores
    
    def compute_meteor_score(self, clean_reference: str, clean_hypothesis: str) -> float:
        if self.meteor_metric is None:
            return 0.0
        
        try:
            score = self.meteor_metric.compute(
                predictions=[clean_hypothesis],
                references=[clean_reference]
            )
            return score['meteor']
        except Exception:
            return 0.0
    
    def evaluate_sample(self, code: str, reference_summary: str) -> Dict[str, Any]:
        generated_summary, full_output = self.generate_summary(code)
        
        clean_reference = clean_summary_for_evaluation(reference_summary)
        
        metrics = {}
        
        bleu_scores = self.compute_bleu_score(clean_reference, generated_summary)
        metrics.update(bleu_scores)
        
        rouge_scores = self.compute_rouge_score(clean_reference, generated_summary)
        metrics.update(rouge_scores)
        
        meteor_score = self.compute_meteor_score(clean_reference, generated_summary)
        metrics['meteor'] = meteor_score
        
        return {
            'processed_summary': generated_summary,
            'full_output': full_output,
            'clean_reference': clean_reference,
            'metrics': metrics
        }
    
    def evaluate_dataset(self, test_data: List[Dict[str, Any]], max_samples: int = None) -> Dict[str, Any]:
        if max_samples:
            test_data = test_data[:max_samples]
        
        self.model.eval()
        
        all_results = []
        all_metrics = {metric: [] for metric in ['bleu', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4']}
        
        for rouge_type in self.config['rouge_types']:
            for suffix in ['_precision', '_recall', '_fmeasure']:
                all_metrics[f'{rouge_type}{suffix}'] = []
        
        all_metrics['meteor'] = []
        
        for item in tqdm(test_data, desc="Evaluating"):
            try:
                result = self.evaluate_sample(item['code'], item['summary'])
                all_results.append(result)
                
                for metric_name, value in result['metrics'].items():
                    if metric_name in all_metrics:
                        all_metrics[metric_name].append(value)
                        
            except Exception:
                continue
        
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            if values:
                avg_metrics[f'avg_{metric_name}'] = np.mean(values)
                avg_metrics[f'std_{metric_name}'] = np.std(values)
        
        main_metrics = {
            'bleu': avg_metrics.get('avg_bleu', 0.0),
            'rouge1_f': avg_metrics.get('avg_rouge1_fmeasure', 0.0),
            'rouge2_f': avg_metrics.get('avg_rouge2_fmeasure', 0.0),
            'rougeL_f': avg_metrics.get('avg_rougeL_fmeasure', 0.0),
            'meteor': avg_metrics.get('avg_meteor', 0.0),
        }
        
        return {
            'main_metrics': main_metrics,
            'detailed_metrics': avg_metrics,
            'sample_results': all_results,
            'num_samples': len(all_results),
            'num_total': len(test_data)
        }
    
    def save_evaluation_results(self, results: Dict[str, Any], save_path: Path, model_name: str):
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        main_results = {
            'model_name': model_name,
            'main_metrics': results['main_metrics'],
            'detailed_metrics': results['detailed_metrics'],
            'num_samples': results['num_samples'],
            'num_total': results['num_total']
        }
        
        with open(save_path / f'{model_name}_evaluation.json', 'w', encoding='utf-8') as f:
            json.dump(main_results, f, indent=2, ensure_ascii=False)
        
        sample_results = {
            'model_name': model_name,
            'sample_results': [
                {
                    'code': sample.get('code', ''),
                    'reference_summary': sample.get('reference_summary', ''),
                    'extracted_summary': sample.get('processed_summary', ''),
                    'full_output': sample.get('full_output', ''),
                    'metrics': sample['metrics']
                }
                for sample in results['sample_results']
            ]
        }
        
        with open(save_path / f'{model_name}_samples.json', 'w', encoding='utf-8') as f:
            json.dump(sample_results, f, indent=2, ensure_ascii=False)

def create_evaluator(model: LayerSelectiveModel, data_processor: CodeSummarizationDataProcessor, config: Dict[str, Any]) -> CodeSummarizationEvaluator:
    return CodeSummarizationEvaluator(model, data_processor, config)
