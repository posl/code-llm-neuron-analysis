# RQ2: Code Representation Analysis

This directory contains scripts and tools for analyzing code representation capabilities of large language models through three main experiments.

## Experiments

### 1. RSA (Representation Similarity Analysis)
Analyzes layer specialization characteristics of models when processing cross-language code.

### 2. AST Probe Analysis
Uses probe models to analyze LLM's predictive ability for AST node types.

### 3. Variable Renaming Analysis
Analyzes similarity changes in code representations before and after variable name replacement.

## Quick Start

### RSA Analysis
```bash
# Run code representation analysis
python code/RQ2/RSA/code_representation_analysis.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_dir datasets/humaneval-x/data \
    --output_dir results/RQ2/RSA \
    --target_languages python cpp java go js \
    --device cuda \
    --device_map auto
```

### AST Probe Analysis
```bash
# Run AST probe analysis
python code/RQ2/AST_Probe_Analysis/run_ast_probe_analysis.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_dir datasets/humaneval-x/data \
    --output_dir results/RQ2/AST_Probe \
    --target_languages python java cpp go js \
    --device cuda \
    --num_epochs 50 \
    --batch_size 32
```

### Variable Renaming Analysis
```bash
# Step 1: Create renamed dataset
python code/RQ2/Rename/create_renamed_dataset.py \
    --renamed_dir data/renamed_variables \
    --original_data_dir datasets/humaneval-x/data \
    --output_dir data/renamed_dataset \
    --original_output_dir data/original_dataset \
    --target_languages python cpp java go js

# Step 2: Run similarity analysis
python code/RQ2/Rename/variable_renaming_similarity_analysis.py \
    --model_path meta-llama/Llama-3.1-8B \
    --original_dataset_dir data/original_dataset \
    --renamed_dataset_dir data/renamed_dataset \
    --output_dir results/RQ2/Rename \
    --device cuda \
    --device_map auto
```

## Output Files

### RSA Analysis Output
- `code_representations.npz`: Extracted code representations
- `similarities.json`: Pairwise similarity results
- `analysis_results.json`: Analysis summary
- `visualizations/`: Visualization charts directory

### AST Probe Analysis Output
- `ast_probe_dataset.json`: AST probe dataset
- `representations.npz`: Extracted representations (optional)
- `analysis_results.json`: Probe performance analysis
- `experiment_config.json`: Experiment configuration
- `visualizations/`: Visualization results
- `probe_*.pt`: Trained probe models (optional)

### Variable Renaming Analysis Output
- `renaming_similarities.json`: Renaming similarity results
- `visualizations_renaming/`: Visualization charts
  - `layer_similarities_*.png`: Layer similarities
  - `all_languages_renaming_similarity_comparison_*.png`: Language comparisons
  - `*_renaming_similarity_by_layer_*.png`: Per-language detailed analysis
  - `overall_similarity_distribution_*.png`: Overall distribution

## Notes

1. **GPU Memory Requirements**: Recommend using GPU with at least 16GB VRAM, or use `device_map="auto"` for multi-GPU distributed loading
2. **Data Preparation**: Ensure humaneval-x dataset is downloaded to the specified directory
3. **Dependencies**: Need to install tree-sitter packages for AST parsing
4. **Computation Time**: RSA and AST probe analysis may take considerable time, use `--max_samples_per_language` parameter for small-scale testing
5. **Variable Renaming**: Need to run variable renaming tool first to generate renamed_variables directory
