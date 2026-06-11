import os
import sys
import json
import torch
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from data_processor import CodeNetDataProcessor

class LLMCloneDetector:
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        
        self.generation_config = {
            'temperature': 0.1,
            'max_length': 1024,
        }
        
        # Prompt template for cross-language clone detection
        self.prompt_template = """
Task: Compare two code snippets from different programming languages 
and determine if they implement the SAME functionality.
CRITICAL: Be very strict. 
Only answer "yes" if both snippets perform exactly the same task with the same logic, 
regardless of programming language.
Code A:
{code_a}

Code B:
{code_b}

Answer:"""

    def load_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model.eval()
            
        except Exception as e:
            raise

    def create_prompt(self, code_a: str, code_b: str) -> str:
        max_code_length = 5000

        if len(code_a) > max_code_length:
            code_a = code_a[:max_code_length] + "\n# ... (truncated)"

        if len(code_b) > max_code_length:
            code_b = code_b[:max_code_length] + "\n# ... (truncated)"

        return self.prompt_template.format(code_a=code_a, code_b=code_b)

    def predict_clone(self, code_a: str, code_b: str) -> Tuple[bool, str, float]:
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded, please call load_model() first")

        prompt = self.create_prompt(code_a, code_b)

        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3500,
                padding=False
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            input_length = input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=20,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            generated_length = outputs[0].shape[0] - input_length

            if not generated_text:
                full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if not generated_text:
                    generated_text = "no"

            is_clone, confidence = self._parse_response(generated_text)

            return is_clone, generated_text, confidence

        except Exception as e:
            return False, f"Error: {str(e)}", 0.0

    def _parse_response(self, response: str) -> Tuple[bool, float]:
        response_clean = response.lower().strip()

        if response_clean.startswith('yes'):
            return True, 0.95
        elif response_clean.startswith('no'):
            return False, 0.95

        elif 'yes' in response_clean and 'no' not in response_clean:
            return True, 0.8
        elif 'no' in response_clean and 'yes' not in response_clean:
            return False, 0.8

        positive_indicators = ['similar', 'same', 'equivalent', 'clone', 'duplicate']
        negative_indicators = ['different', 'dissimilar', 'not', 'unlike', 'distinct']

        positive_count = sum(1 for indicator in positive_indicators if indicator in response_clean)
        negative_count = sum(1 for indicator in negative_indicators if indicator in response_clean)

        if positive_count > negative_count and positive_count > 0:
            return True, 0.6
        elif negative_count > positive_count and negative_count > 0:
            return False, 0.6

        return False, 0.1

    def batch_predict(self, code_pairs: List[Dict[str, Any]], 
                     batch_size: int = 1) -> List[Dict[str, Any]]:
        results = []
        total_time = 0
        
        for i, pair in enumerate(code_pairs):
            start_time = time.time()
            
            try:
                is_clone, raw_response, confidence = self.predict_clone(
                    pair['codeA'], 
                    pair['codeB']
                )
                
                result = {
                    'pair_info': pair,
                    'prediction': is_clone,
                    'raw_response': raw_response,
                    'confidence': confidence,
                    'ground_truth': pair.get('type') == 'clone'
                }
                results.append(result)
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
            except Exception as e:
                result = {
                    'pair_info': pair,
                    'prediction': False,
                    'raw_response': f"Error: {str(e)}",
                    'confidence': 0.0,
                    'ground_truth': pair.get('type') == 'clone'
                }
                results.append(result)
        
        return results

    def evaluate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        predictions = [r['prediction'] for r in results]
        ground_truths = [r['ground_truth'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        tp = sum(1 for p, g in zip(predictions, ground_truths) if p and g)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p and not g)
        tn = sum(1 for p, g in zip(predictions, ground_truths) if not p and not g)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if not p and g)
        
        accuracy = (tp + tn) / len(predictions) if len(predictions) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_confidence': avg_confidence,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
            'total_samples': len(predictions)
        }
        
        return metrics

    def plot_confusion_matrix(self, results: List[Dict[str, Any]],
                            save_path: str = None,
                            show_plot: bool = True) -> None:
        predictions = [r['prediction'] for r in results]
        ground_truths = [r['ground_truth'] for r in results]

        cm = confusion_matrix(ground_truths, predictions)

        plt.rcParams['font.family'] = 'DejaVu Sans'

        plt.figure(figsize=(10, 8))

        sns.heatmap(cm,
                   annot=True,
                   fmt='d',
                   cmap='Blues',
                   xticklabels=['Non-Clone', 'Clone'],
                   yticklabels=['Non-Clone', 'Clone'],
                   cbar_kws={'label': 'Number of Samples'},
                   annot_kws={'size': 14})

        plt.title('Code Clone Detection Confusion Matrix', fontsize=18, pad=20)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)

        total = np.sum(cm)
        accuracy = (cm[0,0] + cm[1,1]) / total
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        stats_text = f'Total Samples: {total}\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}'
        plt.figtext(0.02, 0.02, stats_text, fontsize=11, ha='left',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show_plot:
            plt.show()
        else:
            plt.close()

    def save_confusion_matrix_data(self, results: List[Dict[str, Any]],
                                 save_path: str) -> None:
        predictions = [r['prediction'] for r in results]
        ground_truths = [r['ground_truth'] for r in results]

        cm = confusion_matrix(ground_truths, predictions)

        tn, fp, fn, tp = cm.ravel()

        confusion_data = {
            'confusion_matrix': {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            },
            'matrix_2d': cm.tolist(),
            'labels': ['Non-Clone', 'Clone'],
            'metrics': {
                'accuracy': (tp + tn) / (tp + tn + fp + fn),
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            },
            'total_samples': int(np.sum(cm)),
            'unparseable_responses': len([r for r in results if r['confidence'] <= 0.1])
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(confusion_data, f, indent=2, ensure_ascii=False)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM Code Clone Detection')
    parser.add_argument('--model_path', type=str, 
                       required=True,
                       help='Model path')
    parser.add_argument('--data_path', type=str,
                       required=True,
                       help='Data file path')
    parser.add_argument('--max_samples', type=int, default=6000,
                       help='Maximum number of samples')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(os.path.dirname(__file__), 'results'),
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device')
    
    args = parser.parse_args()
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(args.output_dir, f'llm_clone_detection_{timestamp}')
        os.makedirs(experiment_dir, exist_ok=True)
        
        processor = CodeNetDataProcessor(args.data_path)
        all_code_pairs = processor.load_data()

        if args.max_samples > 0:
            code_pairs = all_code_pairs[:args.max_samples]
        else:
            code_pairs = all_code_pairs
        
        detector = LLMCloneDetector(args.model_path, args.device)  
        detector.load_model()
        
        results = detector.batch_predict(code_pairs)
        
        metrics = detector.evaluate_results(results)
        
        results_data = {
            'experiment_config': {
                'model_path': args.model_path,
                'data_path': args.data_path,
                'max_samples': args.max_samples,
                'generation_config': detector.generation_config,
                'timestamp': timestamp
            },
            'metrics': metrics,
            'detailed_results': results
        }
        
        results_path = os.path.join(experiment_dir, 'llm_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        confusion_matrix_path = os.path.join(experiment_dir, 'confusion_matrix.png')
        detector.plot_confusion_matrix(results,
                                     save_path=confusion_matrix_path,
                                     show_plot=False)

        confusion_data_path = os.path.join(experiment_dir, 'confusion_matrix.json')
        detector.save_confusion_matrix_data(results, confusion_data_path)

    except Exception as e:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()