#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(str(ROOT_DIR))

from ast_parser import ASTParser

class ASTProbeDatasetBuilder:
    
    def __init__(self, languages: List[str], max_samples_per_language: int = 100):
        self.languages = [lang.lower() for lang in languages]
        self.max_samples_per_language = max_samples_per_language
        self.parsers = {}
        self.ast_type_to_id = {}
        self.language_to_id = {}
        self.id_to_ast_type = {}
        self.id_to_language = {}
        
        self._initialize_parsers()
        
    def _initialize_parsers(self):
        for language in self.languages:
            try:
                self.parsers[language] = ASTParser(language)
            except Exception as e:
                continue
    
    def build_dataset(self, humaneval_data: Dict[str, Dict]) -> Dict[str, Any]:
        all_samples = []
        language_stats = defaultdict(int)
        ast_type_stats = defaultdict(int)
        
        for language in self.languages:
            if language not in humaneval_data:
                continue
                
            if language not in self.parsers:
                continue
            
            language_data = humaneval_data[language]
            language_samples = self._process_language_data(language, language_data)
            
            all_samples.extend(language_samples)
            language_stats[language] = len(language_samples)
            
            for sample in language_samples:
                ast_type_stats[sample['ast_type']] += 1
        
        self._build_label_mappings(all_samples)
        
        balanced_samples = self._balance_dataset(all_samples, min_samples_per_type=5)
        
        train_samples, val_samples, test_samples = self._split_dataset(balanced_samples)
        
        dataset = {
            'train': train_samples,
            'validation': val_samples,
            'test': test_samples,
            'ast_types': list(self.ast_type_to_id.keys()),
            'languages': list(self.language_to_id.keys()),
            'ast_type_to_id': self.ast_type_to_id,
            'language_to_id': self.language_to_id,
            'id_to_ast_type': self.id_to_ast_type,
            'id_to_language': self.id_to_language,
            'statistics': {
                'total_samples': len(balanced_samples),
                'train_samples': len(train_samples),
                'val_samples': len(val_samples),
                'test_samples': len(test_samples),
                'language_distribution': language_stats,
                'ast_type_distribution': dict(ast_type_stats),
                'num_ast_types': len(self.ast_type_to_id),
                'num_languages': len(self.language_to_id)
            }
        }
        
        return dataset
    
    def _process_language_data(self, language: str, language_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        parser = self.parsers[language]
        samples = []
        
        task_items = list(language_data.items())[:self.max_samples_per_language]
        
        for task_id, task_data in task_items:
            try:
                code = self._extract_code_from_task(task_data, language)
                if not code or len(code.strip()) < 10:
                    continue
                
                token_ast_pairs = parser.extract_token_ast_pairs(code)
                
                for pair in token_ast_pairs:
                    sample = {
                        'task_id': task_id,
                        'language': language,
                        'token': pair['token'],
                        'ast_type': pair['ast_type'],
                        'context': pair['context'],
                        'position': pair['position'],
                        'token_index': pair['token_index'],
                        'code': code,
                        'ast_start_byte': pair.get('ast_start_byte'),
                        'ast_end_byte': pair.get('ast_end_byte'),
                        'ast_start_point': pair.get('ast_start_point'),
                        'ast_end_point': pair.get('ast_end_point')
                    }
                    samples.append(sample)
                    
            except Exception as e:
                continue
        
        return samples
    
    def _extract_code_from_task(self, task_data: Dict[str, Any], language: str) -> str:
        code_fields = ['canonical_solution', 'solution', 'code', 'generation']
        
        for field in code_fields:
            if field in task_data and task_data[field]:
                code = task_data[field].strip()
                if code:
                    return code
        
        if 'prompt' in task_data and 'declaration' in task_data:
            declaration = task_data.get('declaration', '').strip()
            if declaration:
                return declaration
        
        return ""
    
    def _build_label_mappings(self, samples: List[Dict[str, Any]]):
        ast_types = set()
        languages = set()
        
        for sample in samples:
            ast_types.add(sample['ast_type'])
            languages.add(sample['language'])
        
        self.ast_type_to_id = {ast_type: i for i, ast_type in enumerate(sorted(ast_types))}
        self.language_to_id = {language: i for i, language in enumerate(sorted(languages))}
        
        self.id_to_ast_type = {i: ast_type for ast_type, i in self.ast_type_to_id.items()}
        self.id_to_language = {i: language for language, i in self.language_to_id.items()}
    
    def _balance_dataset(self, samples: List[Dict[str, Any]], min_samples_per_type: int = 5) -> List[Dict[str, Any]]:
        ast_type_counts = Counter(sample['ast_type'] for sample in samples)
        language_counts = Counter(sample['language'] for sample in samples)

        rare_types = [ast_type for ast_type, count in ast_type_counts.items()
                     if count < min_samples_per_type]

        filtered_samples = [sample for sample in samples
                          if sample['ast_type'] not in rare_types]

        filtered_ast_counts = Counter(sample['ast_type'] for sample in filtered_samples)

        groups = defaultdict(list)
        for sample in filtered_samples:
            key = (sample['language'], sample['ast_type'])
            groups[key].append(sample)

        group_sizes = [len(group) for group in groups.values()]
        if group_sizes:
            target_size = min(max(min_samples_per_type, int(np.median(group_sizes))), 50)
        else:
            target_size = min_samples_per_type

        balanced_samples = []
        for key, group in groups.items():
            if len(group) >= target_size:
                selected = np.random.choice(group, size=target_size, replace=False).tolist()
            else:
                selected = group

            balanced_samples.extend(selected)

        return balanced_samples
    
    def _split_dataset(self, samples: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        if len(samples) < 3:
            return samples, [], []

        ast_type_counts = Counter(sample['ast_type'] for sample in samples)
        language_counts = Counter(sample['language'] for sample in samples)

        min_ast_count = min(ast_type_counts.values()) if ast_type_counts else 0
        can_stratify_by_ast = min_ast_count >= 2

        train_samples, val_samples, test_samples = self._try_split_strategies(
            samples, can_stratify_by_ast, language_counts, ast_type_counts
        )

        self._validate_split_results(train_samples, val_samples, test_samples, samples)

        return train_samples, val_samples, test_samples

    def _try_split_strategies(self, samples: List[Dict[str, Any]], can_stratify_by_ast: bool,
                            language_counts: Counter, ast_type_counts: Counter) -> Tuple[List, List, List]:
        try:
            return self._split_by_language_with_ast_stratify(samples, can_stratify_by_ast)
        except Exception as e:
            pass

        try:
            return self._split_by_language_only(samples)
        except Exception as e:
            pass

        return self._simple_random_split(samples)

    def _split_by_language_with_ast_stratify(self, samples: List[Dict[str, Any]],
                                           can_stratify_by_ast: bool) -> Tuple[List, List, List]:
        train_samples, val_samples, test_samples = [], [], []

        language_groups = defaultdict(list)
        for sample in samples:
            language_groups[sample['language']].append(sample)

        for language, lang_samples in language_groups.items():
            if len(lang_samples) < 3:
                train_samples.extend(lang_samples)
                continue

            lang_ast_types = [s['ast_type'] for s in lang_samples]
            unique_ast_types = set(lang_ast_types)
            min_ast_count_in_lang = min(Counter(lang_ast_types).values()) if lang_ast_types else 0

            can_stratify_this_lang = can_stratify_by_ast and min_ast_count_in_lang >= 2 and len(unique_ast_types) > 1

            try:
                if can_stratify_this_lang:
                    train_lang, temp_lang = train_test_split(
                        lang_samples, test_size=0.3, random_state=42,
                        stratify=lang_ast_types
                    )
                else:
                    train_lang, temp_lang = train_test_split(
                        lang_samples, test_size=0.3, random_state=42
                    )
            except ValueError as e:
                train_lang, temp_lang = train_test_split(
                    lang_samples, test_size=0.3, random_state=42
                )

            if len(temp_lang) >= 2:
                temp_ast_types = [s['ast_type'] for s in temp_lang]
                temp_unique_ast_types = set(temp_ast_types)
                min_temp_ast_count = min(Counter(temp_ast_types).values()) if temp_ast_types else 0

                can_stratify_temp = (can_stratify_this_lang and
                                   min_temp_ast_count >= 2 and
                                   len(temp_unique_ast_types) > 1)

                try:
                    if can_stratify_temp:
                        val_lang, test_lang = train_test_split(
                            temp_lang, test_size=0.5, random_state=42,
                            stratify=temp_ast_types
                        )
                    else:
                        val_lang, test_lang = train_test_split(
                            temp_lang, test_size=0.5, random_state=42
                        )
                except ValueError as e:
                    mid_point = len(temp_lang) // 2
                    val_lang = temp_lang[:mid_point]
                    test_lang = temp_lang[mid_point:]
            else:
                val_lang = temp_lang[:len(temp_lang)//2] if temp_lang else []
                test_lang = temp_lang[len(temp_lang)//2:] if temp_lang else []

            train_samples.extend(train_lang)
            val_samples.extend(val_lang)
            test_samples.extend(test_lang)

        return train_samples, val_samples, test_samples

    def _split_by_language_only(self, samples: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        languages = [sample['language'] for sample in samples]
        unique_languages = set(languages)

        if len(unique_languages) < 2:
            return self._simple_random_split(samples)

        train_samples, temp_samples = train_test_split(
            samples, test_size=0.3, random_state=42, stratify=languages
        )

        temp_languages = [sample['language'] for sample in temp_samples]
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.5, random_state=42, stratify=temp_languages
        )

        return train_samples, val_samples, test_samples

    def _simple_random_split(self, samples: List[Dict[str, Any]]) -> Tuple[List, List, List]:
        train_samples, temp_samples = train_test_split(
            samples, test_size=0.3, random_state=42
        )

        if len(temp_samples) >= 2:
            val_samples, test_samples = train_test_split(
                temp_samples, test_size=0.5, random_state=42
            )
        else:
            mid_point = len(temp_samples) // 2
            val_samples = temp_samples[:mid_point]
            test_samples = temp_samples[mid_point:]

        return train_samples, val_samples, test_samples

    def _validate_split_results(self, train_samples: List, val_samples: List,
                              test_samples: List, original_samples: List):
        total_split = len(train_samples) + len(val_samples) + len(test_samples)

        if total_split != len(original_samples):
            pass

        for name, split_samples in [("Train set", train_samples), ("Validation set", val_samples), ("Test set", test_samples)]:
            if split_samples:
                lang_dist = Counter(s['language'] for s in split_samples)
                ast_dist = Counter(s['ast_type'] for s in split_samples)
            else:
                pass
    
    def create_token_context_pairs(self, code: str, language: str) -> List[Dict[str, Any]]:
        if language not in self.parsers:
            return []
        
        parser = self.parsers[language]
        return parser.extract_token_ast_pairs(code)
    
    def save_dataset(self, dataset: Dict[str, Any], output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        self.ast_type_to_id = dataset['ast_type_to_id']
        self.language_to_id = dataset['language_to_id']
        self.id_to_ast_type = dataset['id_to_ast_type']
        self.id_to_language = dataset['id_to_language']

        return dataset

    def analyze_dataset_quality(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        analysis = {
            'basic_stats': {},
            'ast_type_analysis': {},
            'language_analysis': {},
            'filtering_effectiveness': {},
            'data_quality_metrics': {}
        }

        total_samples = dataset['statistics']['total_samples']
        num_ast_types = dataset['statistics']['num_ast_types']
        num_languages = dataset['statistics']['num_languages']

        analysis['basic_stats'] = {
            'total_samples': total_samples,
            'num_ast_types': num_ast_types,
            'num_languages': num_languages,
            'avg_samples_per_ast_type': total_samples / num_ast_types if num_ast_types > 0 else 0,
            'avg_samples_per_language': total_samples / num_languages if num_languages > 0 else 0
        }

        ast_types = dataset['ast_types']
        ast_type_dist = dataset['statistics']['ast_type_distribution']

        meaningful_types = self._categorize_ast_types(ast_types)
        analysis['ast_type_analysis'] = {
            'total_types': len(ast_types),
            'meaningful_categories': meaningful_types,
            'type_distribution': ast_type_dist,
            'most_common_types': sorted(ast_type_dist.items(), key=lambda x: x[1], reverse=True)[:10],
            'least_common_types': sorted(ast_type_dist.items(), key=lambda x: x[1])[:10]
        }

        lang_dist = dataset['statistics']['language_distribution']
        analysis['language_analysis'] = {
            'language_distribution': lang_dist,
            'language_balance': self._calculate_balance_score(list(lang_dist.values()))
        }

        analysis['filtering_effectiveness'] = self._analyze_filtering_effectiveness(ast_types, ast_type_dist)

        analysis['data_quality_metrics'] = self._calculate_quality_metrics(dataset)

        return analysis

    def _categorize_ast_types(self, ast_types: List[str]) -> Dict[str, List[str]]:
        categories = {
            'declarations': [],
            'statements': [],
            'expressions': [],
            'literals': [],
            'identifiers': [],
            'control_flow': [],
            'operators': [],
            'punctuation': [],
            'others': []
        }

        for ast_type in ast_types:
            ast_lower = ast_type.lower()

            if any(keyword in ast_lower for keyword in ['definition', 'declaration', 'class', 'function', 'method']):
                categories['declarations'].append(ast_type)
            elif any(keyword in ast_lower for keyword in ['statement', 'return', 'break', 'continue']):
                categories['statements'].append(ast_type)
            elif any(keyword in ast_lower for keyword in ['expression', 'call', 'binary', 'unary']):
                categories['expressions'].append(ast_type)
            elif any(keyword in ast_lower for keyword in ['literal', 'number', 'string', 'boolean', 'true', 'false', 'null', 'none']):
                categories['literals'].append(ast_type)
            elif 'identifier' in ast_lower:
                categories['identifiers'].append(ast_type)
            elif any(keyword in ast_lower for keyword in ['if', 'for', 'while', 'switch', 'case', 'loop']):
                categories['control_flow'].append(ast_type)
            elif any(keyword in ast_lower for keyword in ['operator', '+', '-', '*', '/', '=', '<', '>']):
                categories['operators'].append(ast_type)
            elif any(char in ast_type for char in '.,;:()[]{}'):
                categories['punctuation'].append(ast_type)
            else:
                categories['others'].append(ast_type)

        return categories

    def _calculate_balance_score(self, values: List[int]) -> float:
        if not values or len(values) <= 1:
            return 1.0

        total = sum(values)
        if total == 0:
            return 1.0

        proportions = [v / total for v in values]
        ideal_proportion = 1.0 / len(values)
        deviations = [abs(p - ideal_proportion) for p in proportions]
        avg_deviation = sum(deviations) / len(deviations)
        balance_score = 1.0 - (avg_deviation / ideal_proportion)
        return max(0.0, balance_score)

    def _analyze_filtering_effectiveness(self, ast_types: List[str], ast_type_dist: Dict[str, int]) -> Dict[str, Any]:
        punctuation_indicators = [';', ',', '.', ':', '(', ')', '[', ']', '{', '}', 'punctuation']
        remaining_punctuation = [t for t in ast_types if any(p in t.lower() for p in punctuation_indicators)]

        low_value_indicators = ['comment', 'whitespace', 'error', 'missing']
        remaining_low_value = [t for t in ast_types if any(lv in t.lower() for lv in low_value_indicators)]

        meaningful_keywords = [
            'definition', 'declaration', 'statement', 'expression', 'identifier',
            'literal', 'call', 'binary', 'unary', 'if', 'for', 'while', 'class', 'function'
        ]
        meaningful_types = [t for t in ast_types if any(kw in t.lower() for kw in meaningful_keywords)]

        return {
            'remaining_punctuation_types': remaining_punctuation,
            'remaining_low_value_types': remaining_low_value,
            'meaningful_types_count': len(meaningful_types),
            'meaningful_types_ratio': len(meaningful_types) / len(ast_types) if ast_types else 0,
            'filtering_success': len(remaining_punctuation) == 0 and len(remaining_low_value) == 0,
            'total_filtered_samples': sum(ast_type_dist[t] for t in remaining_punctuation + remaining_low_value)
        }

    def _calculate_quality_metrics(self, dataset: Dict[str, Any]) -> Dict[str, float]:
        stats = dataset['statistics']

        ast_dist = list(stats['ast_type_distribution'].values())
        lang_dist = list(stats['language_distribution'].values())

        ast_balance = self._calculate_balance_score(ast_dist)
        lang_balance = self._calculate_balance_score(lang_dist)

        min_samples_per_type = 10
        sufficient_ast_types = sum(1 for count in ast_dist if count >= min_samples_per_type)
        ast_sufficiency = sufficient_ast_types / len(ast_dist) if ast_dist else 0

        overall_quality = (ast_balance + lang_balance + ast_sufficiency) / 3

        return {
            'ast_type_balance': ast_balance,
            'language_balance': lang_balance,
            'ast_type_sufficiency': ast_sufficiency,
            'overall_quality_score': overall_quality
        }