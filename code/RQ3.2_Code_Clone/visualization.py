import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import os
from sklearn.metrics import roc_curve, auc

class CloneDetectionVisualizer:
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_layer_performance(self, layer_results: Dict[int, Dict[str, float]],
                             save_name: str = "layer_performance.png",
                             llm_baseline_f1: float = 0.684):
        layers = sorted(layer_results.keys())
        f1_scores = [layer_results[layer]['f1'] for layer in layers]

        plt.style.use('seaborn-v0_8-whitegrid')

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

        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(layers, f1_scores, marker='o', color='#1f77b4', linewidth=3.5,
               markersize=10, alpha=0.9, label='F1 Score',
               markeredgewidth=1.5, markeredgecolor='white')
        
        # Add LLM baseline
        if llm_baseline_f1 is not None:
            ax.axhline(y=llm_baseline_f1, color='#ff7f0e', linestyle='--',
                      linewidth=2.5, alpha=0.8)
            
            ax.text((min(layers) + max(layers)) / 2.4, llm_baseline_f1 + 0.008, 'LLM Prompting Method',
                   fontsize=20, fontweight='bold', color='black',
                   ha='center', va='bottom')

        ax.set_xlabel('Layer Index', fontsize=24, fontweight='bold', labelpad=15)
        ax.set_ylabel('F1 Score', fontsize=24, fontweight='bold', labelpad=15)

        ax.grid(True, linestyle='-', alpha=0.3, linewidth=0.8)
        ax.set_axisbelow(True)

        ax.tick_params(axis='both', which='major', labelsize=22, width=1.2, length=8)
        ax.tick_params(axis='both', which='minor', width=0.8, length=4)

        legend = ax.legend(loc='upper right',
                          fontsize=22,
                          frameon=True,
                          fancybox=True,
                          framealpha=0.95,
                          borderpad=0.8,
                          labelspacing=0.8)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(1.0)

        for text in legend.get_texts():
            text.set_fontweight('bold')
            text.set_fontsize(22)

        # Set Y axis range
        y_min = min(f1_scores)
        y_max = max(f1_scores)
        y_range = y_max - y_min
        y_margin = max(0.02, y_range * 0.1)
        ax.set_ylim(max(0, y_min - y_margin), min(1, y_max + y_margin))

        # Set X axis range
        if layers:
            ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)

        # Style spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')

        plt.tight_layout(pad=2.0)

        # Save in multiple formats
        base_name = save_name.rsplit('.', 1)[0]
        
        svg_path = os.path.join(self.output_dir, f"{base_name}.svg")
        plt.savefig(svg_path, format='svg', bbox_inches='tight',
                   facecolor='white', edgecolor='none', dpi=300)
        
        pdf_path = os.path.join(self.output_dir, f"{base_name}.pdf")
        plt.savefig(pdf_path, format='pdf', bbox_inches='tight',
                   facecolor='white', edgecolor='none', dpi=300)
        
        png_path = os.path.join(self.output_dir, save_name)
        plt.savefig(png_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        
        plt.close()

        best_layer = layers[np.argmax(f1_scores)]
        best_f1 = max(f1_scores)
        
        return {
            'best_layer': best_layer,
            'best_f1': best_f1,
            'layer_ranking': sorted(zip(layers, f1_scores), key=lambda x: x[1], reverse=True)
        }

    def plot_similarity_distributions(self, layer_similarities: Dict[int, List[float]],
                                    labels: List[int], save_name: str = "similarity_distributions.png"):
        # Select key layers for visualization
        key_layers = [0, 8, 11, 13, 16, 24, 26, 31]
        available_layers = [layer for layer in key_layers if layer in layer_similarities]

        if not available_layers:
            return

        plt.style.use('seaborn-v0_8-whitegrid')

        n_layers = len(available_layers)
        n_cols = 4
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, layer_idx in enumerate(available_layers):
            if i >= len(axes):
                break

            similarities = layer_similarities[layer_idx]

            clone_sims = [sim for sim, label in zip(similarities, labels) if label == 1]
            nonclone_sims = [sim for sim, label in zip(similarities, labels) if label == 0]

            ax = axes[i]

            if clone_sims:
                ax.hist(clone_sims, bins=30, alpha=0.7, label='Clone',
                       color='#d62728', density=True, edgecolor='white', linewidth=0.5)
            if nonclone_sims:
                ax.hist(nonclone_sims, bins=30, alpha=0.7, label='Non-clone',
                       color='#1f77b4', density=True, edgecolor='white', linewidth=0.5)

            ax.set_xlabel('Cosine Similarity', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Layer {layer_idx}', fontsize=11, fontweight='bold')

            ax.grid(True, linestyle='-', alpha=0.2, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.legend(fontsize=9, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

            ax.tick_params(axis='both', which='major', labelsize=9)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)

        # Hide unused subplots
        for i in range(len(available_layers), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

    def plot_similarity_trends_by_label(self, layer_similarities: Dict[int, List[float]],
                                      labels: List[int], save_name: str = "similarity_trends_by_label.png"):
        layers = sorted(layer_similarities.keys())
        clone_means = []
        nonclone_means = []
        clone_stds = []
        nonclone_stds = []

        for layer_idx in layers:
            similarities = layer_similarities[layer_idx]

            clone_sims = [sim for sim, label in zip(similarities, labels) if label == 1]
            nonclone_sims = [sim for sim, label in zip(similarities, labels) if label == 0]

            if clone_sims:
                clone_means.append(np.mean(clone_sims))
                clone_stds.append(np.std(clone_sims))
            else:
                clone_means.append(0)
                clone_stds.append(0)

            if nonclone_sims:
                nonclone_means.append(np.mean(nonclone_sims))
                nonclone_stds.append(np.std(nonclone_sims))
            else:
                nonclone_means.append(0)
                nonclone_stds.append(0)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(layers, clone_means, marker='o', linewidth=2.5, markersize=6,
               color='#d62728', label='Clone pairs', alpha=0.9)
        ax.plot(layers, nonclone_means, marker='o', linewidth=2.5, markersize=6,
               color='#1f77b4', label='Non-clone pairs', alpha=0.9)

        # Add error bands
        ax.fill_between(layers,
                        [m - s for m, s in zip(clone_means, clone_stds)],
                        [m + s for m, s in zip(clone_means, clone_stds)],
                        alpha=0.2, color='#d62728')
        ax.fill_between(layers,
                        [m - s for m, s in zip(nonclone_means, nonclone_stds)],
                        [m + s for m, s in zip(nonclone_means, nonclone_stds)],
                        alpha=0.2, color='#1f77b4')

        ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Cosine Similarity', fontsize=14, fontweight='bold')

        ax.grid(True, linestyle='-', alpha=0.2, linewidth=0.5)
        ax.set_axisbelow(True)

        ax.tick_params(axis='both', which='major', labelsize=12)

        legend = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0),
                          fontsize=10, frameon=True, fancybox=True,
                          shadow=True, framealpha=0.95, ncol=1,
                          columnspacing=0.8, handlelength=2.5, handletextpad=0.5)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(0.5)

        ax.set_ylim(0, 1)

        if layers:
            ax.set_xlim(min(layers) - 0.5, max(layers) + 0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

        plt.tight_layout()

        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        similarity_diff = [c - n for c, n in zip(clone_means, nonclone_means)]
        max_diff_idx = np.argmax(np.abs(similarity_diff))
        max_diff_layer = layers[max_diff_idx]
        max_diff_value = similarity_diff[max_diff_idx]

        return {
            'clone_means': clone_means,
            'nonclone_means': nonclone_means,
            'similarity_diff': similarity_diff,
            'max_gap_layer': max_diff_layer,
            'max_gap_value': max_diff_value
        }