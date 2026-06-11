import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model
from transformers import Trainer

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

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with LoRA on McEval dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Base model path")
    parser.add_argument("--dataset_dir", type=str, required=True, help="McEval dataset directory")
    parser.add_argument("--language", type=str, default="java", choices=["python", "java", "cpp", "go", "javascript"], help="Target language")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LoRA adapters")
    parser.add_argument("--max_seq_length", type=int, default=1300, help="Maximum sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()

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

def main():
    args = parse_args()

    print(f"Loading and processing McEval dataset for {args.language}")

    language_file_map = {
        "python": "Python-instruct.jsonl",
        "cpp": "CPP-instruct.jsonl",
        "java": "Java-instruct.jsonl",
        "javascript": "JavaScript-instruct.jsonl",
        "go": "Go-instruct.jsonl"
    }

    if args.language not in language_file_map:
        raise ValueError(f"Unsupported language for McEval: {args.language}")

    dataset_path = os.path.join(args.dataset_dir, language_file_map[args.language])
    print(f"Using McEval/Instruct format: {dataset_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print(f"Loaded {len(dataset)} samples from {dataset_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if 'instruction' in dataset.column_names and 'output' in dataset.column_names:
        print("Detected McEval/Instruct format, using McEval tokenization")
        tokenize_func = lambda examples: tokenize_and_prepare_labels_mceval_instruct(examples, tokenizer, args.max_seq_length)
    else:
        raise ValueError(f"McEval format expected but not found. Available columns: {dataset.column_names}")

    processed_dataset = dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count()
    )
    print(f"Original dataset size: {len(dataset)}. Processed dataset size: {len(processed_dataset)}")

    print("Configuring standard LoRA and loading model")
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "lm_head"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Configuring training")
    base_model_name = os.path.basename(args.model_path)
    training_output_dir = os.path.join(args.output_dir, f"{base_model_name}-{args.language}-mceval-lora")

    training_args = TrainingArguments(
        output_dir=training_output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=True,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=False,
        seed=args.seed,
        fp16=False,
        max_grad_norm=1.0,
        dataloader_drop_last=False,
        report_to=None,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print(f"Starting training for {args.language}")
    trainer.train()

    print(f"Saving LoRA adapter to {training_output_dir}")
    trainer.save_model(training_output_dir)
    tokenizer.save_pretrained(training_output_dir)

if __name__ == "__main__":
    main()