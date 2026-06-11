#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer
)
from typing import List, Tuple, Dict
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.RQ1.neuron_intervention import parse_neuron_key, get_module_by_layer_and_component

torch.backends.cuda.matmul.allow_tf32 = True

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        output_device = model.get_output_embeddings().weight.device
        
        if "labels" in inputs:
            inputs["labels"] = inputs["labels"].to(output_device)
            
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        if "loss" in logs:
            print(f"Step {self.state.global_step}: loss = {logs['loss']:.4f}, epoch = {logs.get('epoch', 'N/A')}")

        super().log(logs, start_time)

def load_language_neurons(neuron_file_path: str) -> List[dict]:
    if not os.path.exists(neuron_file_path):
        raise FileNotFoundError(f"Neuron data file not found: {neuron_file_path}")
    
    with open(neuron_file_path, 'r', encoding='utf-8') as f:
        neurons = json.load(f)
    
    print(f"Loaded {len(neurons)} language-specific neurons")
    return neurons

def parse_neuron_locations(neurons: List[dict]) -> List[Tuple[int, str, int]]:
    locations = []
    for neuron in neurons:
        neuron_key = neuron['neuron_key']
        try:
            layer_idx, component, neuron_idx = parse_neuron_key(neuron_key)
            locations.append((layer_idx, component, neuron_idx))
        except ValueError as e:
            print(f"Warning: Cannot parse neuron key {neuron_key}: {e}")
            continue
    
    print(f"Successfully parsed {len(locations)} neuron locations")
    return locations

def freeze_all_parameters(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    print("Froze all model parameters")

def unfreeze_neuron_parameters(model, neuron_locations: List[Tuple[int, str, int]]) -> Dict[str, int]:
    unfrozen_modules = set()
    processed_neurons = 0
    skipped_neurons = 0

    for layer_idx, component, neuron_idx in neuron_locations:
        try:
            module = get_module_by_layer_and_component(model, layer_idx, component)
            if module is None:
                print(f"Warning: Cannot find module layer_{layer_idx}_{component}")
                skipped_neurons += 1
                continue

            if hasattr(module, 'weight') and module.weight is not None:
                if neuron_idx < module.weight.shape[0]:
                    module.weight.requires_grad = True
                    unfrozen_modules.add(f"layer_{layer_idx}_{component}_weight")

                    if hasattr(module, 'bias') and module.bias is not None:
                        module.bias.requires_grad = True
                        unfrozen_modules.add(f"layer_{layer_idx}_{component}_bias")

                    processed_neurons += 1
                else:
                    print(f"Warning: Neuron index {neuron_idx} out of range for module {component} (shape {module.weight.shape[0]})")
                    skipped_neurons += 1

        except Exception as e:
            print(f"Warning: Error processing neuron layer_{layer_idx}_{component}_{neuron_idx}: {e}")
            skipped_neurons += 1
            continue

    stats = {
        "processed_neurons": processed_neurons,
        "skipped_neurons": skipped_neurons,
        "unfrozen_modules": len(unfrozen_modules)
    }

    print(f"Processed {processed_neurons} neurons, skipped {skipped_neurons}, unfroze {len(unfrozen_modules)} modules")
    return stats

def create_prompt_mceval_instruct(sample):
    instruction = sample['instruction']
    output = sample['output']

    prompt_text = f"{instruction}\n\ncode:\n"
    full_text = prompt_text + output

    return prompt_text, full_text

def tokenize_and_prepare_labels_mceval_instruct(examples, tokenizer, max_seq_length):
    input_ids = []
    labels = []

    sample_count = len(examples['instruction'])

    for i in range(sample_count):
        sample = {
            'instruction': examples['instruction'][i],
            'output': examples['output'][i],
            'language': examples['language'][i]
        }

        prompt_text, full_text = create_prompt_mceval_instruct(sample)

        tokenized_full = tokenizer(full_text, truncation=False, padding=False, add_special_tokens=True)
        tokenized_full_ids = tokenized_full['input_ids']

        if len(tokenized_full_ids) > max_seq_length:
            continue

        tokenized_prompt = tokenizer(prompt_text, add_special_tokens=True, truncation=False, padding=False)
        prompt_len = len(tokenized_prompt['input_ids'])

        label = list(tokenized_full_ids)
        label[:prompt_len] = [-100] * prompt_len

        input_ids.append(tokenized_full_ids)
        labels.append(label)

    return {"input_ids": input_ids, "labels": labels}

def get_trainable_parameter_count(model) -> Tuple[int, int]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune based on language-specific neurons")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--language", type=str, default="python", choices=["python", "java", "cpp", "go", "javascript"], help="Target language")
    parser.add_argument("--neuron_file", type=str, required=True, help="Path to language-specific neurons JSON file")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output_dir", type=str, default="./neuron_specific_models", help="Fine-tuned model save path")
    parser.add_argument("--max_seq_length", type=int, default=1536, help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    output_dir = os.path.join(args.output_dir, args.language)
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Language-specific neuron fine-tuning - {args.language} ===")

    print("\n1. Loading language-specific neuron data...")
    neurons = load_language_neurons(args.neuron_file)
    neuron_locations = parse_neuron_locations(neurons)

    if not neuron_locations:
        raise ValueError(f"No valid language-specific neurons found for {args.language}")

    print("\n2. Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n3. Setting parameter freezing...")
    freeze_all_parameters(model)
    unfreeze_stats = unfreeze_neuron_parameters(model, neuron_locations)

    trainable_params, total_params = get_trainable_parameter_count(model)
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)")
    print(f"Neuron processing stats: {unfreeze_stats}")

    print("\n4. Loading and processing data...")
    language_map = {
        "python": "Python-instruct.jsonl",
        "java": "Java-instruct.jsonl",
        "cpp": "CPP-instruct.jsonl",
        "go": "Go-instruct.jsonl",
        "javascript": "JavaScript-instruct.jsonl"
    }

    dataset_path = os.path.join(args.dataset_dir, language_map[args.language])
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(dataset)} training samples")

    if 'instruction' not in dataset.column_names or 'output' not in dataset.column_names:
        raise ValueError(f"Data format error, expected 'instruction' and 'output' fields, actual fields: {dataset.column_names}")

    print("Tokenizing data...")
    tokenize_func = lambda examples: tokenize_and_prepare_labels_mceval_instruct(examples, tokenizer, args.max_seq_length)
    tokenized_dataset = dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    print(f"Tokenization complete, valid samples: {len(tokenized_dataset)}")

    print("\n5. Setting training parameters...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,
        report_to=None,
        seed=args.seed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8
    )

    print("\n6. Creating trainer...")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("\n7. Starting training...")
    print(f"Training configuration:")
    print(f"  - Language: {args.language}")
    print(f"  - Neuron count: {len(neuron_locations)}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Training samples: {len(tokenized_dataset)}")
    print(f"  - Training epochs: {args.num_train_epochs}")
    print(f"  - Learning rate: {args.learning_rate}")

    trainer.train()

    print("\n8. Saving fine-tuned model...")
    final_model_path = os.path.join(output_dir, "final_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    print(f"Model saved to: {final_model_path}")

    training_info = {
        "language": args.language,
        "neuron_count": len(neuron_locations),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "parameter_efficiency": trainable_params / total_params,
        "training_samples": len(tokenized_dataset),
        "epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "model_path": args.model_path,
        "dataset_path": dataset_path
    }

    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)

    print(f"Training info saved to: {info_path}")
    print("\n=== Neuron-specific fine-tuning complete ===")

if __name__ == "__main__":
    main()
