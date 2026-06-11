# RQ3.3: Code Summarization

This directory contains scripts and tools for code summarization experiments using phase-based training approaches.

## Quick Start

### 1. Phase-based Training Pipeline
```bash
# Phase 1: Train on high-resource languages
python code/RQ3.3_Code_Summarization/run_phase1_training.py \
    --model_name meta-llama/Llama-3.1-8B \
    --phase high_resource \
    --train_data datasets/CodeSearchNet/python/train/python_train.json \
    --valid_data datasets/CodeSearchNet/python/valid/python_valid.json \
    --output_dir results/RQ3.3/phase1_model \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 5e-5

# Phase 2: Fine-tune on low-resource languages  
python code/RQ3.3_Code_Summarization/run_phase2_training.py \
    --model_path results/RQ3.3/phase1_model \
    --phase low_resource \
    --train_data datasets/CodeSearchNet/go/train/go_train.json \
    --valid_data datasets/CodeSearchNet/go/valid/go_valid.json \
    --output_dir results/RQ3.3/phase2_model \
    --num_epochs 2 \
    --batch_size 8 \
    --learning_rate 2e-5
```

### 2. Baseline Training
```bash
python code/RQ3.3_Code_Summarization/run_baseline_training.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_data datasets/CodeSearchNet/python/train/python_train.json \
    --valid_data datasets/CodeSearchNet/python/valid/python_valid.json \
    --output_dir results/RQ3.3/baseline_model \
    --num_epochs 3 \
    --batch_size 8
```

### 3. LoRA-based Training
```bash
python code/RQ3.3_Code_Summarization/run_lora_baseline.py \
    --model_name meta-llama/Llama-3.1-8B \
    --train_data datasets/CodeSearchNet/python/train/python_train.json \
    --valid_data datasets/CodeSearchNet/python/valid/python_valid.json \
    --output_dir results/RQ3.3/lora_model \
    --lora_rank 16 \
    --lora_alpha 32 \
    --num_epochs 3
```

### 4. Evaluation
```bash
python code/RQ3.3_Code_Summarization/simple_compare.py \
    --model_path results/RQ3.3/phase2_model \
    --test_data datasets/CodeSearchNet/go/test/go_test.json \
    --output_file results/RQ3.3/evaluation_results.csv \
    --metrics bleu rouge-l
```

## Output Files

- `phase1_model/`: Phase 1 trained model (high-resource languages)
- `phase2_model/`: Phase 2 fine-tuned model (low-resource languages)  
- `baseline_model/`: Standard fine-tuned model for comparison
- `lora_model/`: LoRA adapter weights
- `evaluation_results.csv`: Evaluation metrics (BLEU, ROUGE-L scores)
- `training_logs/`: Training progress and loss curves
- `generated_summaries.json`: Generated code summaries for test set

## Notes

1. **Dataset Format**: CodeSearchNet datasets should be in JSON format with 'code' and 'docstring' fields
2. **Phase Training**: Phase 1 uses high-resource languages (Python, Java, Go, PHP), Phase 2 uses low-resource languages (Ruby)
