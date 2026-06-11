#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from typing import Dict, List, Any
from scipy import stats
import json

class LayerAnalyzer:
    
    def __init__(self, results: Dict[int, Dict[str, Any]]):
        self.results = results
        self.layer_indices = sorted(results.keys())
    
    def find_optimal_layers(self) -> Dict[str, Any]:
        """Find optimal layers for each task"""
        performance_data = {}
        
        # Collect performance data for all tasks
        for layer_idx in self.layer_indices:
            layer_results = self.results[layer_idx]
            
            for task_name in ['ast', 'language']:
                if task_name in layer_results:
                    if task_name not in performance_data:
                        performance_data[task_name] = {}
                    performance_data[task_name][layer_idx] = {
                        'accuracy': layer_results[task_name]['eval_result']['accuracy'],
                        'f1_score': layer_results[task_name]['eval_result']['f1_score']
                    }
        
        # Calculate optimal layers for each task
        optimal_results = {}
        for task_name, task_performance in performance_data.items():
            if task_performance:
                optimal_results[task_name] = self._find_best_layers(task_performance)
        
        # Calculate layer difference if both AST and Language data exist
        if 'ast' in optimal_results and 'language' in optimal_results:
            optimal_results['layer_difference'] = {
                'ast_best_layer': optimal_results['ast']['best_layer_accuracy'],
                'language_best_layer': optimal_results['language']['best_layer_accuracy'],
                'layer_gap': abs(optimal_results['ast']['best_layer_accuracy'] - 
                               optimal_results['language']['best_layer_accuracy'])
            }
        
        return optimal_results
    
    def _find_best_layers(self, task_performance: Dict[int, Dict[str, float]]) -> Dict[str, Any]:
        """Helper method to find best layers for a specific task"""
        best_layer_acc = max(task_performance.keys(), 
                           key=lambda x: task_performance[x]['accuracy'])
        best_layer_f1 = max(task_performance.keys(), 
                          key=lambda x: task_performance[x]['f1_score'])
        
        return {
            'best_layer_accuracy': best_layer_acc,
            'best_accuracy': task_performance[best_layer_acc]['accuracy'],
            'best_layer_f1': best_layer_f1,
            'best_f1': task_performance[best_layer_f1]['f1_score'],
            'performance_curve': [(layer, perf['accuracy']) for layer, perf in task_performance.items()]
        }
    
    def analyze_layer_specialization(self) -> Dict[str, Any]:
        """Analyze how specialized each layer is for specific tasks"""
        specialization_results = {}
        
        for layer_idx in self.layer_indices:
            layer_results = self.results[layer_idx]
            
            if 'ast' in layer_results and 'language' in layer_results:
                ast_acc = layer_results['ast']['eval_result']['accuracy']
                lang_acc = layer_results['language']['eval_result']['accuracy']
                
                specialization_index = abs(ast_acc - lang_acc)
                dominant_task = 'ast' if ast_acc > lang_acc else 'language'
                
                specialization_results[layer_idx] = {
                    'ast_accuracy': ast_acc,
                    'language_accuracy': lang_acc,
                    'specialization_index': specialization_index,
                    'dominant_task': dominant_task,
                    'performance_ratio': max(ast_acc, lang_acc) / min(ast_acc, lang_acc) if min(ast_acc, lang_acc) > 0 else float('inf')
                }
        
        if specialization_results:
            most_specialized_layer = max(specialization_results.keys(), 
                                       key=lambda x: specialization_results[x]['specialization_index'])
            
            avg_specialization = np.mean([result['specialization_index'] 
                                        for result in specialization_results.values()])
            
            analysis_summary = {
                'layer_specialization': specialization_results,
                'most_specialized_layer': most_specialized_layer,
                'max_specialization_index': specialization_results[most_specialized_layer]['specialization_index'],
                'average_specialization': avg_specialization,
                'specialization_trend': self._analyze_specialization_trend(specialization_results)
            }
        else:
            analysis_summary = {'error': 'Insufficient data for specialization analysis'}
        
        return analysis_summary
    
    def _analyze_specialization_trend(self, specialization_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Analyze the trend of specialization across layers"""
        layers = sorted(specialization_results.keys())
        specialization_values = [specialization_results[layer]['specialization_index'] for layer in layers]
        
        if len(layers) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(layers, specialization_values)
            
            trend_analysis = {
                'slope': slope,
                'r_squared': r_value ** 2,
                'p_value': p_value,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                'trend_strength': abs(r_value)
            }
        else:
            trend_analysis = {'error': 'Insufficient data points to analyze trend'}
        
        return trend_analysis
    
    def compute_layer_correlations(self) -> np.ndarray:
        """Compute correlation matrix between layer performances"""
        performance_matrix = []
        
        for layer_idx in self.layer_indices:
            layer_results = self.results[layer_idx]
            layer_performance = []
            
            # Extract performance metrics in a consistent order
            for task in ['ast', 'language']:
                if task in layer_results:
                    layer_performance.extend([
                        layer_results[task]['eval_result']['accuracy'],
                        layer_results[task]['eval_result']['f1_score']
                    ])
                else:
                    layer_performance.extend([0.0, 0.0])
            
            performance_matrix.append(layer_performance)
        
        performance_matrix = np.array(performance_matrix)
        correlation_matrix = np.corrcoef(performance_matrix)
        
        return correlation_matrix
    
    def analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance patterns across layers"""
        patterns = {}
        
        # Extract performance curves for each task
        performance_curves = self._extract_performance_curves()
        
        # Analyze individual curves
        for task_name, curve in performance_curves.items():
            if curve:
                patterns[task_name] = self._analyze_curve_pattern(curve, task_name.upper())
        
        # Compare curves if both exist
        if 'ast' in performance_curves and 'language' in performance_curves:
            if performance_curves['ast'] and performance_curves['language']:
                patterns['comparison'] = self._compare_performance_curves(
                    performance_curves['ast'], 
                    performance_curves['language']
                )
        
        return patterns
    
    def _extract_performance_curves(self) -> Dict[str, List[float]]:
        """Extract performance curves for all tasks"""
        curves = {'ast': [], 'language': []}
        
        for layer_idx in self.layer_indices:
            layer_results = self.results[layer_idx]
            
            for task in curves.keys():
                if task in layer_results:
                    curves[task].append(layer_results[task]['eval_result']['accuracy'])
        
        return curves
    
    def _analyze_curve_pattern(self, curve: List[float], task_name: str) -> Dict[str, Any]:
        """Analyze pattern of a single performance curve"""
        curve = np.array(curve)
        
        pattern_analysis = {
            'max_value': float(np.max(curve)),
            'min_value': float(np.min(curve)),
            'mean_value': float(np.mean(curve)),
            'std_value': float(np.std(curve)),
            'peak_layer': int(np.argmax(curve)),
            'valley_layer': int(np.argmin(curve))
        }
        
        if len(curve) > 2:
            diff = np.diff(curve)
            pattern_analysis.update({
                'overall_trend': 'increasing' if np.sum(diff) > 0 else 'decreasing',
                'volatility': float(np.std(diff)),
                'monotonic': bool(np.all(diff >= 0)) or bool(np.all(diff <= 0))
            })
            
            # Find local maxima and minima
            local_maxima = []
            local_minima = []
            
            for i in range(1, len(curve) - 1):
                if curve[i] > curve[i-1] and curve[i] > curve[i+1]:
                    local_maxima.append(i)
                elif curve[i] < curve[i-1] and curve[i] < curve[i+1]:
                    local_minima.append(i)
            
            pattern_analysis.update({
                'local_maxima': local_maxima,
                'local_minima': local_minima,
                'num_peaks': len(local_maxima),
                'num_valleys': len(local_minima)
            })
        
        return pattern_analysis
    
    def _compare_performance_curves(self, curve1: List[float], curve2: List[float]) -> Dict[str, Any]:
        """Compare two performance curves"""
        curve1 = np.array(curve1)
        curve2 = np.array(curve2)
        
        # Align curves to same length
        min_len = min(len(curve1), len(curve2))
        curve1 = curve1[:min_len]
        curve2 = curve2[:min_len]
        
        correlation = np.corrcoef(curve1, curve2)[0, 1]
        difference = curve1 - curve2
        
        comparison = {
            'correlation': float(correlation),
            'mean_difference': float(np.mean(difference)),
            'max_difference': float(np.max(np.abs(difference))),
            'crossover_points': [],
            'ast_dominance_layers': [],
            'language_dominance_layers': []
        }
        
        # Identify dominance and crossover points
        for i in range(len(difference)):
            if difference[i] > 0:
                comparison['ast_dominance_layers'].append(i)
            elif difference[i] < 0:
                comparison['language_dominance_layers'].append(i)
            
            if i > 0 and np.sign(difference[i]) != np.sign(difference[i-1]):
                comparison['crossover_points'].append(i)
        
        return comparison
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        report = {
            'optimal_layers': self.find_optimal_layers(),
            'layer_specialization': self.analyze_layer_specialization(),
            'performance_patterns': self.analyze_performance_patterns(),
            'layer_correlations': self.compute_layer_correlations().tolist(),
            'summary_statistics': self._compute_summary_statistics()
        }
        
        return report
    
    def _compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics for all tasks"""
        stats = {
            'total_layers': len(self.layer_indices),
            'layer_range': (min(self.layer_indices), max(self.layer_indices)) if self.layer_indices else (0, 0),
            'tasks_analyzed': [],
            'overall_performance': {}
        }
        
        # Identify all tasks
        all_tasks = set()
        for layer_results in self.results.values():
            all_tasks.update(layer_results.keys())
        stats['tasks_analyzed'] = list(all_tasks)
        
        # Compute statistics for each task
        for task in all_tasks:
            task_performances = []
            for layer_idx in self.layer_indices:
                if task in self.results[layer_idx]:
                    task_performances.append(self.results[layer_idx][task]['eval_result']['accuracy'])
            
            if task_performances:
                stats['overall_performance'][task] = {
                    'mean_accuracy': float(np.mean(task_performances)),
                    'std_accuracy': float(np.std(task_performances)),
                    'min_accuracy': float(np.min(task_performances)),
                    'max_accuracy': float(np.max(task_performances))
                }
        
        return stats
    
    def save_analysis_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        serializable_results = self._make_json_serializable(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj