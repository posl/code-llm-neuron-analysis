import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class ProbeVisualizer:
   
    def __init__(self, results: Dict[int, Dict[str, Any]], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.figsize = (12, 8)
        self.dpi = 300
    
    def _adjust_y_axis_range(self, ax, values: List[float]):
        """Helper method to adjust Y-axis range for better visualization"""
        if not values:
            return
            
        min_val = min(values)
        max_val = max(values)
        range_size = max_val - min_val
        
        if range_size < 0.1:
            center = (min_val + max_val) / 2
            min_val = max(0, center - 0.05)
            max_val = min(1, center + 0.05)
        else:
            margin = range_size * 0.1
            min_val = max(0, min_val - margin)
            max_val = min(1, max_val + margin)
        
        ax.set_ylim(min_val, max_val)
    
    def _extract_task_metrics(self, task_name: str = 'ast') -> Dict[str, List[float]]:
        """Extract metrics for a specific task across all layers"""
        layers = sorted(self.results.keys())
        accuracies = []
        f1_scores = []
        
        for layer in layers:
            layer_results = self.results[layer]
            if task_name in layer_results:
                accuracies.append(layer_results[task_name]['eval_result']['accuracy'])
                f1_scores.append(layer_results[task_name]['eval_result']['f1_score'])
            else:
                accuracies.append(0)
                f1_scores.append(0)
        
        return {
            'layers': layers,
            'accuracies': accuracies,
            'f1_scores': f1_scores
        }
    
    def plot_layer_performance_curves(self, save_path: Optional[str] = None) -> str:
        """Plot accuracy and F1 score curves across layers"""
        metrics = self._extract_task_metrics('ast')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot accuracy
        ax1.plot(metrics['layers'], metrics['accuracies'], 'o-', 
                label='AST Node Type', linewidth=2, markersize=6, color='#2E86AB')
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        self._adjust_y_axis_range(ax1, metrics['accuracies'])
        
        # Plot F1 score
        ax2.plot(metrics['layers'], metrics['f1_scores'], 'o-', 
                label='AST Node Type', linewidth=2, markersize=6, color='#A23B72')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        self._adjust_y_axis_range(ax2, metrics['f1_scores'])
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "layer_performance_curves.svg"
        
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_confusion_matrices(self, layer_idx: int, save_path: Optional[str] = None) -> str:
        """Plot confusion matrices for specified layer"""
        if layer_idx not in self.results:
            return ""
        
        layer_results = self.results[layer_idx]
        
        tasks_to_plot = []
        if 'ast' in layer_results:
            tasks_to_plot.append(('ast', 'AST Node Type'))
        
        if not tasks_to_plot:
            return ""
        
        fig, axes = plt.subplots(1, len(tasks_to_plot), figsize=(8 * len(tasks_to_plot), 6))
        if len(tasks_to_plot) == 1:
            axes = [axes]
        
        for i, (task_type, task_name) in enumerate(tasks_to_plot):
            task_results = layer_results[task_type]['eval_result']
            cm = task_results['confusion_matrix']
            labels = task_results.get('label_names', None)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=labels, yticklabels=labels)
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"confusion_matrix_layer_{layer_idx}.svg"
        
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_ast_type_distribution(self, dataset: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Plot distribution of AST node types in dataset"""
        if 'statistics' in dataset and 'ast_type_distribution' in dataset['statistics']:
            ast_dist = dataset['statistics']['ast_type_distribution']
        else:
            return ""
        
        ast_types = list(ast_dist.keys())
        counts = list(ast_dist.values())
        
        plt.figure(figsize=(12, 8))
        
        bars = plt.bar(range(len(ast_types)), counts, color=sns.color_palette("husl", len(ast_types)))
        
        plt.xlabel('AST Node Types')
        plt.ylabel('Count')
        plt.xticks(range(len(ast_types)), ast_types, rotation=45, ha='right')
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "ast_type_distribution.svg"
        
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_layer_heatmap(self, analysis_results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Generate heatmap of performance across layers and tasks"""
        layers = sorted(self.results.keys())
        
        # Collect all unique tasks
        tasks = set()
        for layer_results in self.results.values():
            tasks.update(layer_results.keys())
        tasks = sorted(list(tasks))
        
        # Build performance matrix
        performance_matrix = []
        for task in tasks:
            task_performance = []
            for layer in layers:
                if task in self.results[layer]:
                    accuracy = self.results[layer][task]['eval_result']['accuracy']
                    task_performance.append(accuracy)
                else:
                    task_performance.append(0.0)
            performance_matrix.append(task_performance)
        
        plt.figure(figsize=(max(8, len(layers) * 0.5), max(6, len(tasks) * 0.8)))
        
        sns.heatmap(performance_matrix, 
                   xticklabels=[f'Layer {i}' for i in layers],
                   yticklabels=[task.upper() for task in tasks],
                   annot=True, 
                   fmt='.3f', 
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Accuracy'})
        
        plt.xlabel('Layer Index')
        plt.ylabel('Tasks')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "layer_performance_heatmap.svg"
        
        plt.savefig(save_path, format='svg', bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def plot_ast_performance_analysis(self, analysis_results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Create detailed AST performance analysis plot with publication-ready styling"""
        metrics = self._extract_task_metrics('ast')
        
        # Apply publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')
        self._apply_publication_style()
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Plot both metrics
        ax.plot(metrics['layers'], metrics['accuracies'], marker='o', label='Accuracy', linewidth=3.5,
               markersize=10, color='#1f77b4', alpha=0.9, markeredgewidth=1.5, markeredgecolor='white')
        ax.plot(metrics['layers'], metrics['f1_scores'], marker='s', label='F1 Score', linewidth=3.5,
               markersize=10, color='#ff7f0e', alpha=0.9, markeredgewidth=1.5, markeredgecolor='white')
        
        # Set labels and styling
        ax.set_xlabel('Layer Index', fontsize=24, fontweight='bold', labelpad=15)
        ax.set_ylabel('Performance Score', fontsize=24, fontweight='bold', labelpad=15)
        
        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)
        
        ax.tick_params(axis='both', which='major', labelsize=22, width=1.2, length=8)
        ax.tick_params(axis='both', which='minor', width=0.8, length=4)
        
        # Configure legend
        legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02),
                        ncol=2, fontsize=22, frameon=True, fancybox=True, 
                        framealpha=0.95, borderpad=0.8, labelspacing=0.8)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1.0)
        
        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_fontsize(22)
        
        # Adjust Y-axis range
        all_scores = metrics['accuracies'] + metrics['f1_scores']
        self._adjust_y_axis_range(ax, all_scores)
        
        # Adjust X-axis range
        if metrics['layers']:
            ax.set_xlim(min(metrics['layers']) - 0.5, max(metrics['layers']) + 0.5)
        
        # Style the plot frame
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        
        plt.tight_layout(pad=2.0)
        
        # Save in multiple formats
        if save_path is None:
            save_path = self.output_dir / "ast_performance_analysis"
        else:
            save_path = Path(save_path).with_suffix('')
        
        svg_path = str(save_path) + ".svg"
        plt.savefig(svg_path, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none', dpi=300)
        
        pdf_path = str(save_path) + ".pdf"
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none', dpi=300)
        
        plt.close()
        
        return svg_path
    
    def _apply_publication_style(self):
        """Apply publication-ready matplotlib styling"""
        plt.rcParams.update({
            'font.size': 20,
            'axes.titlesize': 24,
            'axes.labelsize': 22,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'text.usetex': False,
            'svg.fonttype': 'none',
        })
    
    def generate_comprehensive_dashboard(self, analysis_results: Dict[str, Any], dataset: Dict[str, Any]) -> List[str]:
        """Generate all visualization plots"""
        generated_plots = []
        
        try:
            # Generate each plot type
            plot_generators = [
                lambda: self.plot_layer_performance_curves(),
                lambda: self.generate_layer_heatmap(analysis_results),
                lambda: self.plot_ast_performance_analysis(analysis_results),
                lambda: self.plot_ast_type_distribution(dataset)
            ]
            
            for generator in plot_generators:
                try:
                    plot_path = generator()
                    if plot_path:
                        generated_plots.append(plot_path)
                except Exception as e:
                    # Continue with other plots if one fails
                    continue
                    
        except Exception as e:
            pass
        
        return generated_plots