# RQ1: Language-Specific Neuron Analysis

This directory contains scripts and tools for identifying and analyzing language-specific neurons in LLMs.

## Quick Start

### Run Complete Neuron Analysis Pipeline
```bash
# Full analysis for multiple programming languages
python code/RQ1/run_full_generation_gradient_analysis.py \
    --model_path meta-llama/Llama-3.1-8B \
    --output_dir results/RQ1/Llama-3.1-8B \
    --sample_num 500 \
    --device cuda \
    --device_map auto
```

### Generate Code Samples
```bash
# Generate code samples for evaluation  
python code/RQ1/generate_samples.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_dir datasets/McEval/generation \
    --output_dir results/RQ1/generated_samples \
    --temperature 0.7 \
    --max_tokens 512
```

### Classify Language-Specific Neurons
```bash
# Identify language-specific neurons from gradient analysis
python code/RQ1/classify_neurons.py \
    --gradient_dir results/RQ1/Llama-3.1-8B/gradients \
    --output_dir results/RQ1/Llama-3.1-8B/language_specific_neurons \
```

### Run Neuron Intervention Experiments
```bash
# Test impact of neuron interventions (zero intervention only)
python code/RQ1/intervention_experiment.py \
    --model_path meta-llama/Llama-3.1-8B \
    --neurons_dir results/RQ1/Llama-3.1-8B/language_specific_neurons \
    --input_dir datasets/humaneval-x \
    --data_dir datasets/humaneval-x/data \
    --output_dir results/RQ1/intervention_results \
    --intervention_language python \
    --intervention_scope specific \
    --target_languages python go js java cpp \
    --num_samples 1 \
    --pass_at_k 3 \
    --temperature 0.8 \
    --device cuda
```

## Output Files
- `language_specific_neurons/`: JSON files with identified language-specific neurons
  - `python_specific_neurons.json`
  - `js_specific_neurons.json`
- `generated_samples/`: Generated code samples for evaluation
- `intervention_results/`: Results from neuron intervention experiments
  - `zero/`: Zero intervention results by language
  - `zero/random_control/`: Random control experiment results (if enabled)

## Notes

1. **GPU Memory Requirements**: Full gradient analysis requires significant GPU memory. Use `--device_map auto` for multi-GPU setups.
2. **Supported Languages**: Python, JavaScript, Go, Java, C++ (ensure datasets are available).
3. **Intervention Scope**: Can be 'specific' (language-specific neurons) or 'common' (common neurons across languages).
4. **Pass@k Evaluation**: The `--pass_at_k` parameter determines how many samples to generate for evaluation (default: 3).
