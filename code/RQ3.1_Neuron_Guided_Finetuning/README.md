# RQ3.1: Neuron-Guided Fine-tuning

This directory contains scripts and tools for neuron-guided fine-tuning experiments, focusing on improving model performance for low-resource programming languages.

## Quick Start

### 1. Standard LoRA Fine-tuning
```bash
python code/RQ3.1_Neuron_Guided_Finetuning/finetune_lora.py \
    --model_name meta-llama/Llama-3.1-8B \
    --dataset_path datasets/McEval/Instruct/Go-instruct.jsonl \
    --output_dir results/RQ3.1/lora_finetuned \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### 2. Neuron-Specific Fine-tuning
```bash
python code/RQ3.1_Neuron_Guided_Finetuning/finetune_neuron_specific.py \
    --model_name meta-llama/Llama-3.1-8B \
    --dataset_path datasets/McEval/Instruct/Go-instruct.jsonl \
    --neuron_file results/RQ1/Llama-3.1-8B/language_specific_neurons/go_specific_neurons.json \
    --output_dir results/RQ3.1/neuron_finetuned \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
```

### 3. Generate with Fine-tuned Model
```bash
python code/RQ3.1_Neuron_Guided_Finetuning/generate_with_lora.py \
    --base_model meta-llama/Llama-3.1-8B \
    --lora_adapter results/RQ3.1/lora_finetuned \
    --input_file datasets/McEval/generation/Python.jsonl \
    --output_file results/RQ3.1/generated_results.jsonl \
    --max_length 512 \
    --temperature 0.7
```

## Output Files

- `lora_finetuned/`: Directory containing LoRA adapter weights
  - `adapter_config.json`: LoRA configuration
  - `adapter_model.bin`: Fine-tuned adapter weights
- `neuron_finetuned/`: Directory containing neuron-specific fine-tuned model
  - `config.json`: Model configuration
  - `pytorch_model.bin`: Fine-tuned model weights
- `generated_results.jsonl`: Generated code outputs
- `training_logs.txt`: Training progress and metrics
