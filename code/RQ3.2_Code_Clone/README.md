# RQ3.2: Code Clone Detection

This directory contains scripts and tools for code clone detection experiments using layer-wise representation analysis and LLM-based methods.

## Quick Start

### 1. Extract Code Representations
```bash
python code/RQ3.2_Code_Clone/representation_extractor.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_path datasets/CodeNet/java_cn.json \
    --output_dir results/RQ3.2/representations \
    --device cuda \
    --batch_size 16
```

### 2. Run Clone Detection
```bash
python code/RQ3.2_Code_Clone/run_clone_detection.py \
    --representations_dir results/RQ3.2/representations \
    --output_dir results/RQ3.2/detection_results \
    --detection_method similarity \
    --similarity_threshold 0.85
```

### 3. LLM-based Clone Detection
```bash
python code/RQ3.2_Code_Clone/llm_clone_detector.py \
    --model_path meta-llama/Llama-3.1-8B \
    --data_path datasets/CodeNet/java_cn.json \
    --output_dir results/RQ3.2/llm_detection \
    --prompt_strategy zero-shot \
    --device cuda
```

### 4. Visualize Results
```bash
python code/RQ3.2_Code_Clone/visualization.py \
    --results_dir results/RQ3.2/detection_results \
    --output_dir results/RQ3.2/visualizations
```

## Output Files

- `representations/`: Directory containing extracted code representations
  - `layer_*/`: Representations for each layer
  - `metadata.json`: Dataset and extraction metadata
- `detection_results/`: Clone detection results
  - `clone_pairs.json`: Detected clone pairs with similarity scores
  - `performance_metrics.json`: Precision, recall, F1 scores
- `llm_detection/`: LLM-based detection results
  - `predictions.json`: LLM clone predictions
  - `evaluation.json`: Performance metrics
- `visualizations/`: Generated visualization plots
  - `similarity_distribution.png`: Similarity score distribution
  - `layer_performance.png`: Performance across layers
  - `confusion_matrix.png`: Detection confusion matrix

## Notes

1. **Dataset Format**: CodeNet dataset should be in JSON format with labeled clone pairs
2. **Detection Methods**: Supports similarity-based and LLM-based detection approaches
3. **Layer Analysis**: Analyzes clone detection performance across different model layers
