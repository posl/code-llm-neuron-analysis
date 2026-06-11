#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class CodeSummarizationDataset(Dataset):
    
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_code_length: int = 512, max_summary_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_summary_length = max_summary_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # Build structured prompt
        full_text = self.create_structured_prompt(item, for_inference=False)

        # Tokenize full text without padding
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_code_length + self.max_summary_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # Create labels
        labels = input_ids.clone()
        prompt_without_summary = self.create_structured_prompt(item, for_inference=True)
        prompt_encoding = self.tokenizer(
            prompt_without_summary,
            add_special_tokens=True,
            return_tensors='pt'
        )
        response_start_index = prompt_encoding['input_ids'].shape[1]
        response_start_index = min(response_start_index, labels.shape[0])
        labels[:response_start_index] = -100
        
        pad_token_id = self.tokenizer.pad_token_id
        labels[labels == pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'language': item['language'],
            'func_name': item.get('func_name', ''),
        }
    
    def create_structured_prompt(self, item: Dict[str, Any], for_inference: bool = False) -> str:
        # Simple prompt structure
        language = item['language']
        code = item['code']
        
        prompt = f"Generate a concise summary for the following {language} code:\n```{language}\n{code}\n```\n\nSummary:"
        
        if not for_inference:
            summary = item.get('summary', '')
            prompt = f"{prompt} {summary}"
        
        return prompt

class CodeSummarizationDataProcessor:
    
    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_language_data(self, data_path: str, language: str, split: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        file_path = Path(data_path) / language / split / f"{language}_{split}.json"
        
        if not file_path.exists():
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Random sampling
        if max_samples and len(data) > max_samples:
            random.shuffle(data)
            data = data[:max_samples]

        return data
    
    def create_high_resource_dataset(self, data_path: str, split: str = 'train', max_samples_per_language: Optional[int] = None) -> List[Dict[str, Any]]:
        all_data = []
        high_resource_languages = ['python', 'go', 'java', 'javascript', 'php']
        
        for language in high_resource_languages:
            language_data = self.load_language_data(data_path, language, split, max_samples_per_language)
            all_data.extend(language_data)
        
        random.shuffle(all_data)
        return all_data
    
    def create_low_resource_dataset(self, data_path: str, split: str = 'train', max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        data = self.load_language_data(data_path, 'ruby', split, max_samples)
        return data
    
    def create_dataset(self, data: List[Dict[str, Any]], max_code_length: int = 512, max_summary_length: int = 128) -> CodeSummarizationDataset:
        # Use create_structured_prompt from CodeSummarizationDataset
        dataset = CodeSummarizationDataset(
            data=data,
            tokenizer=self.tokenizer,
            max_code_length=max_code_length,
            max_summary_length=max_summary_length
        )
        return dataset
    
    def create_dataloader(self, dataset: CodeSummarizationDataset, batch_size: int = 2, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        # Filter invalid samples
        valid_batch = [item for item in batch if (item['labels'] != -100).any()]
        if not valid_batch:
            raise ValueError("All samples in batch are invalid")
        
        batch = valid_batch

        # Find max length in batch
        max_length = max(item['input_ids'].shape[0] for item in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            input_ids = item['input_ids'].squeeze()
            attention_mask = item['attention_mask'].squeeze()
            labels = item['labels'].squeeze()

            current_length = input_ids.shape[0]
            pad_length = max_length - current_length

            if pad_length > 0:
                pad_token_id = self.tokenizer.pad_token_id
                input_ids = torch.cat([
                    input_ids,
                    torch.full((pad_length,), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length, dtype=attention_mask.dtype, device=attention_mask.device)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), -100, dtype=labels.dtype, device=labels.device)
                ])

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            'input_ids': torch.stack(input_ids_list),
            'attention_mask': torch.stack(attention_mask_list),
            'labels': torch.stack(labels_list),
            'languages': [item['language'] for item in batch],
            'func_names': [item.get('func_name', '') for item in batch],
            'actual_max_length': max_length,
        }

    def get_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not data:
            return {}
        
        code_lengths = [len(item['code']) for item in data]
        summary_lengths = [len(item['summary']) for item in data]
        
        language_distribution = {}
        for item in data:
            lang = item['language']
            language_distribution[lang] = language_distribution.get(lang, 0) + 1
        
        stats = {
            'total_samples': len(data),
            'language_distribution': language_distribution,
            'code_length': {
                'mean': sum(code_lengths) / len(code_lengths),
                'min': min(code_lengths),
                'max': max(code_lengths),
            },
            'summary_length': {
                'mean': sum(summary_lengths) / len(summary_lengths),
                'min': min(summary_lengths),
                'max': max(summary_lengths),
            }
        }
        
        return stats

def setup_data_processor(model_path: str) -> CodeSummarizationDataProcessor:
    random.seed(42)
    processor = CodeSummarizationDataProcessor(model_path)
    return processor