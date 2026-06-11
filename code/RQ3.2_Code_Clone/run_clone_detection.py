import argparse
import os
import json
from datetime import datetime

from data_processor import CodeNetDataProcessor
from representation_extractor import CodeCloneRepresentationExtractor
from clone_detector import LayerBasedCloneDetector
from visualization import CloneDetectionVisualizer

def main():
    parser = argparse.ArgumentParser(description='Code Clone Detection Experiment')
    parser.add_argument('--model_path', type=str, 
                       required=True,
                       help='Model path')
    parser.add_argument('--data_path', type=str,
                       required=True,
                       help='Data path')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(os.path.dirname(__file__), 'results'),
                       help='Output directory')
    parser.add_argument('--target_layers', type=int, nargs='+',
                       default=list(range(28)),
                       help='Target layers')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum samples for testing')
    parser.add_argument('--cv_folds', type=int, default=5,
                       help='Cross validation folds')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    parser.add_argument('--load_similarities', type=str, default="/mnt/sda/yz/projects/XLLM4SE/src/RQ4.1_Code_Clone/results/clone_detection_20250910_130205/similarities.npz",
                       help='Load existing similarity data file path')
    parser.add_argument('--pooling_method', type=str, default='mean',
                       choices=['mean', 'attention_weighted'],
                       help='Pooling method: mean or attention_weighted')
    
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(args.output_dir, f"clone_detection_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    try:
        # Step 1: Load data
        data_processor = CodeNetDataProcessor(args.data_path)
        code_pairs = data_processor.load_data()
        
        if args.max_samples and args.max_samples < len(code_pairs):
            code_pairs = code_pairs[:args.max_samples]
        
        stats = data_processor.get_statistics(code_pairs)

        # Check if loading existing similarity data
        if args.load_similarities and os.path.exists(args.load_similarities):
            if args.load_similarities.endswith('.npz'):
                # Load numpy format
                import numpy as np
                data = np.load(args.load_similarities)
                layer_similarities = {}
                for key in data.files:
                    if key.startswith('layer_'):
                        layer_idx = int(key.split('_')[1])
                        layer_similarities[layer_idx] = data[key].tolist()
                labels = data['labels'].tolist()
            else:
                # Load JSON format
                with open(args.load_similarities, 'r', encoding='utf-8') as f:
                    similarities_data = json.load(f)
                layer_similarities = similarities_data['layer_similarities']
                labels = similarities_data['labels']

            # Convert keys to int
            layer_similarities = {int(k): v for k, v in layer_similarities.items()}

            # Step 3: Clone detection with existing data
            detector = LayerBasedCloneDetector(args.target_layers)

            layer_results = detector.optimize_thresholds_with_similarities(
                layer_similarities, labels, cv_folds=args.cv_folds, return_roc_data=True
            )

            representation_results = []
            for i, label in enumerate(labels):
                representation_results.append({
                    'pair_info': {'type': 'clone' if label == 1 else 'nonclone'}
                })

        else:
            # Step 2: Extract representations
            extractor = CodeCloneRepresentationExtractor(
                model_path=args.model_path,
                target_layers=args.target_layers,
                device=args.device,
                pooling_method=args.pooling_method
            )

            extractor.load_model()

            representation_results = extractor.batch_extract_representations(
                code_pairs, batch_size=args.batch_size
            )

            # Step 3: Clone detection
            detector = LayerBasedCloneDetector(args.target_layers)

            layer_results = detector.optimize_all_thresholds(
                representation_results, cv_folds=args.cv_folds, return_roc_data=True
            )

            layer_similarities = detector.compute_similarities_for_pairs(representation_results)
            labels = [1 if result['pair_info']['type'] == 'clone' else 0
                     for result in representation_results]
        
        # Step 4: Visualization
        visualizer = CloneDetectionVisualizer(experiment_dir)

        performance_summary = visualizer.plot_layer_performance(layer_results)

        visualizer.plot_similarity_distributions(layer_similarities, labels)

        trend_stats = visualizer.plot_similarity_trends_by_label(layer_similarities, labels)
        
        # Step 5: Save results
        def convert_to_python_types(obj):
            import numpy as np
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_python_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            else:
                return obj

        results_data = {
            'experiment_config': {
                'model_path': args.model_path,
                'data_path': args.data_path,
                'target_layers': args.target_layers,
                'batch_size': args.batch_size,
                'max_samples': args.max_samples,
                'cv_folds': args.cv_folds,
                'pooling_method': args.pooling_method,
                'timestamp': timestamp
            },
            'data_statistics': convert_to_python_types(stats),
            'layer_results': convert_to_python_types(layer_results),
            'performance_summary': convert_to_python_types(performance_summary)
        }

        results_path = os.path.join(experiment_dir, 'results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        # Save similarity data
        similarities_data = {
            'layer_similarities': convert_to_python_types(layer_similarities),
            'labels': convert_to_python_types(labels),
            'pair_info': [result['pair_info'] for result in representation_results] if 'representation_results' in locals() else [],
            'target_layers': convert_to_python_types(args.target_layers),
            'num_pairs': len(labels)
        }

        similarities_path = os.path.join(experiment_dir, 'similarities.json')
        with open(similarities_path, 'w', encoding='utf-8') as f:
            json.dump(similarities_data, f, indent=2, ensure_ascii=False)

        # Save numpy format
        import numpy as np
        similarities_array_path = os.path.join(experiment_dir, 'similarities.npz')

        save_dict = {}
        for layer, sims in layer_similarities.items():
            save_dict[f'layer_{layer}'] = np.array(sims, dtype=np.float32)

        save_dict['labels'] = np.array(labels, dtype=np.int32)
        save_dict['layers'] = np.array(args.target_layers, dtype=np.int32)

        np.savez_compressed(similarities_array_path, **save_dict)
        
        return results_data
        
    except Exception as e:
        raise

if __name__ == "__main__":
    main()