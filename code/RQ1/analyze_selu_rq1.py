#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_selu_rq1.py
SELU × Neuron-Guided RQ1 分析・可視化スクリプト

元論文 RQ1 (PLSスコア) → SELU (TSSスコア) への読み替え:
  - 言語 (Python/Java/...) → タスク (bug_issue/sentiment/...)
  - PLSスコア              → TSSスコア
  - 言語間Jaccard重複率   → タスク間Jaccard重複率

[出力物]
  1. top_neurons_per_task.csv     : タスク別TSS上位ニューロン一覧
  2. neuron_layer_distribution.png: レイヤー別上位ニューロン数の棒グラフ
  3. component_distribution.png   : コンポーネント種別(MLP/Attn)分布
  4. jaccard_heatmap.png          : タスク間ニューロン重複率ヒートマップ
  5. tss_heatmap.png              : 上位ニューロンのTSSスコアヒートマップ
  6. rq1_summary.json             : 数値サマリー (テーブル・統計)

使い方:
  python analyze_selu_rq1.py \\
      --input selu_neurons_combined.json \\
      --output_dir ./rq1_results \\
      --top_k 100 \\
      --tss_threshold 2.0
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────
# 定数
# ──────────────────────────────────────────
COMPONENT_GROUPS = {
    "mlp":  ["gate_proj", "up_proj", "down_proj"],
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
FIGSIZE_WIDE  = (14, 5)
FIGSIZE_SQUARE = (10, 8)
DPI = 150


# ──────────────────────────────────────────
# 1. JSONロード & フラット化
# ──────────────────────────────────────────

def load_combined_json(path: str) -> Tuple[Dict, Dict, List[str]]:
    """
    selu_neurons_combined.json を読み込み、
    (layers_dict, metadata, tasks) を返す。
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    meta  = data.pop("_metadata", {})
    tasks = meta.get("tasks", [])
    log.info(f"Loaded: {len(data)} layers, {len(tasks)} tasks → {tasks}")
    return data, meta, tasks


def flatten_neurons(layers_dict: Dict, tasks: List[str]) -> pd.DataFrame:
    """
    全レイヤー・全ニューロンを1行1ニューロンのDataFrameに展開する。

    列:
      layer_name, neuron_key, layer_idx, component_type, component_group,
      importance_{task}, gradient_{task}, activation_{task},
      tss_{task}  (各タスク)
    """
    rows = []
    for layer_name, layer_data in tqdm(layers_dict.items(), desc="Flattening neurons"):
        li = layer_data.get("layer_info", {})
        layer_idx       = li.get("layer_idx", -1)
        component_type  = li.get("component_type", "other")
        component_group = _comp_group(component_type)

        for neuron_key, nd in layer_data.get("neurons", {}).items():
            row = {
                "layer_name":      layer_name,
                "neuron_key":      neuron_key,
                "layer_idx":       layer_idx,
                "component_type":  component_type,
                "component_group": component_group,
            }
            for task in tasks:
                row[f"importance_{task}"] = nd.get("importance_by_language", {}).get(task, 0.0)
                row[f"gradient_{task}"]   = nd.get("gradient_by_language",   {}).get(task, 0.0)
                row[f"activation_{task}"] = nd.get("activation_by_language", {}).get(task, 0.0)
                row[f"tss_{task}"]        = nd.get("tss_by_task",            {}).get(task, 0.0)
            rows.append(row)

    df = pd.DataFrame(rows)
    log.info(f"Flattened: {len(df)} neurons")
    return df


def _comp_group(comp_type: str) -> str:
    for grp, comps in COMPONENT_GROUPS.items():
        if comp_type in comps:
            return grp
    return "other"


# ──────────────────────────────────────────
# 2. タスク別 TSS 上位ニューロン
# ──────────────────────────────────────────

def get_top_neurons(
    df: pd.DataFrame,
    tasks: List[str],
    top_k: int,
    tss_threshold: float,
) -> Dict[str, pd.DataFrame]:
    """
    タスクごとにTSSスコア上位top_k件 (かつ tss >= tss_threshold) を返す。
    戻り値: {task: DataFrame}
    """
    top: Dict[str, pd.DataFrame] = {}
    for task in tasks:
        col = f"tss_{task}"
        if col not in df.columns:
            log.warning(f"  {task}: TSS column not found, skipping")
            continue
        filtered = df[df[col] >= tss_threshold].nlargest(top_k, col).copy()
        filtered["task"] = task
        top[task] = filtered
        log.info(f"  {task}: {len(filtered)} top neurons (tss >= {tss_threshold})")
    return top


def save_top_neurons_csv(top: Dict[str, pd.DataFrame], output_dir: str):
    rows = []
    for task, tdf in top.items():
        col = f"tss_{task}"
        for _, r in tdf.iterrows():
            rows.append({
                "task":            task,
                "layer_name":      r["layer_name"],
                "neuron_key":      r["neuron_key"],
                "layer_idx":       r["layer_idx"],
                "component_type":  r["component_type"],
                "component_group": r["component_group"],
                "tss_score":       r.get(col, 0.0),
                "importance":      r.get(f"importance_{task}", 0.0),
            })
    out = pd.DataFrame(rows)
    path = os.path.join(output_dir, "top_neurons_per_task.csv")
    out.to_csv(path, index=False)
    log.info(f"Saved → {path}")
    return out


# ──────────────────────────────────────────
# 3. レイヤー分布
# ──────────────────────────────────────────

def plot_layer_distribution(
    top: Dict[str, pd.DataFrame],
    tasks: List[str],
    output_dir: str,
):
    """
    タスクごとに「上位ニューロンが何層目に集中しているか」を積み上げ棒グラフで表示。
    横軸: layer_idx、縦軸: ニューロン数、色: タスク
    """
    # 最大レイヤー数を取得
    all_dfs = [v for v in top.values() if len(v) > 0]
    if not all_dfs:
        return
    max_layer = int(max(d["layer_idx"].max() for d in all_dfs if len(d) > 0))

    counts = {}
    for task in tasks:
        if task not in top or len(top[task]) == 0:
            continue
        c = top[task]["layer_idx"].value_counts().reindex(range(max_layer + 1), fill_value=0)
        counts[task] = c

    if not counts:
        return

    count_df = pd.DataFrame(counts).fillna(0).astype(int)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    count_df.plot(kind="bar", stacked=True, ax=ax, width=0.85)
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("# Top Neurons", fontsize=12)
    ax.set_title("Top-TSS Neuron Distribution by Layer", fontsize=14, fontweight="bold")
    ax.legend(title="Task", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, "neuron_layer_distribution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path}")


def plot_component_distribution(
    top: Dict[str, pd.DataFrame],
    tasks: List[str],
    output_dir: str,
):
    """
    コンポーネント種別(MLP gate/up/down, Attn q/k/v/o)ごとの上位ニューロン数。
    """
    records = []
    for task in tasks:
        if task not in top or len(top[task]) == 0:
            continue
        vc = top[task]["component_type"].value_counts()
        for comp, cnt in vc.items():
            records.append({"task": task, "component": comp, "count": int(cnt)})

    if not records:
        return

    cdf = pd.DataFrame(records)
    pivot = cdf.pivot_table(index="component", columns="task", values="count", fill_value=0)

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("Component Type", fontsize=12)
    ax.set_ylabel("# Top Neurons", fontsize=12)
    ax.set_title("Top-TSS Neuron Distribution by Component", fontsize=14, fontweight="bold")
    ax.legend(title="Task", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, "component_distribution.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path}")


# ──────────────────────────────────────────
# 4. Jaccard 類似度ヒートマップ
# ──────────────────────────────────────────

def compute_jaccard(top: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    タスクペアごとに「上位ニューロン集合のJaccard類似度」を計算。
    ニューロン識別キー = layer_name + "_" + neuron_key
    """
    tasks = list(top.keys())
    neuron_sets = {}
    for task in tasks:
        if len(top[task]) > 0:
            neuron_sets[task] = set(
                top[task]["layer_name"] + "::" + top[task]["neuron_key"]
            )
        else:
            neuron_sets[task] = set()

    mat = np.zeros((len(tasks), len(tasks)))
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            s1, s2 = neuron_sets[t1], neuron_sets[t2]
            if not s1 and not s2:
                mat[i, j] = 0.0
            else:
                mat[i, j] = len(s1 & s2) / len(s1 | s2)

    return pd.DataFrame(mat, index=tasks, columns=tasks)


def plot_jaccard_heatmap(jaccard_df: pd.DataFrame, output_dir: str):
    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    sns.heatmap(
        jaccard_df,
        annot=True, fmt=".2f", cmap="YlOrRd",
        vmin=0, vmax=1, linewidths=0.5,
        ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Jaccard Similarity of Top-TSS Neurons Between Tasks",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "jaccard_heatmap.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path}")


# ──────────────────────────────────────────
# 5. TSS ヒートマップ (上位ニューロン × タスク)
# ──────────────────────────────────────────

def plot_tss_heatmap(
    df: pd.DataFrame,
    top: Dict[str, pd.DataFrame],
    tasks: List[str],
    output_dir: str,
    max_neurons: int = 50,
):
    """
    全タスクのTSS上位ニューロンをまとめ、ニューロン×タスクのTSSスコアをヒートマップで表示。
    行: ニューロン (最大 max_neurons 件)
    列: タスク
    """
    # 全タスク横断で最もTSSが高いニューロンを選ぶ
    all_tops = pd.concat(list(top.values()), ignore_index=True)
    if all_tops.empty:
        return

    tss_cols = [f"tss_{t}" for t in tasks if f"tss_{t}" in all_tops.columns]
    all_tops["max_tss"] = all_tops[tss_cols].max(axis=1)
    all_tops["uid"] = all_tops["layer_name"] + "::" + all_tops["neuron_key"]
    top_uids = (
        all_tops.groupby("uid")["max_tss"]
        .max()
        .nlargest(max_neurons)
        .index.tolist()
    )

    # uid をキーにしてTSSスコアを集める
    heatmap_data = []
    for uid in top_uids:
        row_data = {"uid": uid}
        matched = all_tops[all_tops["uid"] == uid]
        if matched.empty:
            # df 全体から引く
            layer_name, neuron_key = uid.split("::", 1)
            matched = df[(df["layer_name"] == layer_name) & (df["neuron_key"] == neuron_key)]
        for task in tasks:
            col = f"tss_{task}"
            if col in matched.columns and len(matched) > 0:
                row_data[task] = float(matched[col].iloc[0])
            else:
                row_data[task] = 0.0
        heatmap_data.append(row_data)

    hm_df = pd.DataFrame(heatmap_data).set_index("uid")[tasks]

    # ラベルを短縮 (layer_idx + component)
    short_labels = []
    for uid in hm_df.index:
        parts = uid.split("::")
        lname = parts[0] if parts else uid
        # layer_N_component → L{N}_comp
        tokens = lname.split("_")
        try:
            lidx = tokens[1]
            comp = "_".join(tokens[2:])
            short_labels.append(f"L{lidx}_{comp}")
        except IndexError:
            short_labels.append(lname[:20])

    fig_h = max(6, len(top_uids) * 0.22)
    fig, ax = plt.subplots(figsize=(max(8, len(tasks) * 1.1), fig_h))
    sns.heatmap(
        hm_df,
        annot=False, cmap="Blues",
        linewidths=0.3, linecolor="#ddd",
        yticklabels=short_labels,
        ax=ax,
    )
    ax.set_title(f"TSS Scores of Top Neurons (n={len(top_uids)}) × Task",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Task", fontsize=11)
    ax.set_ylabel("Neuron (Layer_Component)", fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
    plt.tight_layout()
    path = os.path.join(output_dir, "tss_heatmap.png")
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    log.info(f"Saved → {path}")


# ──────────────────────────────────────────
# 6. サマリーJSON
# ──────────────────────────────────────────

def save_summary(
    top: Dict[str, pd.DataFrame],
    jaccard_df: pd.DataFrame,
    tasks: List[str],
    meta: Dict,
    output_dir: str,
):
    summary = {
        "model_path": meta.get("model_path", "unknown"),
        "num_tasks":  len(tasks),
        "tasks":      tasks,
        "per_task": {},
        "jaccard": {},
    }

    for task in tasks:
        if task not in top:
            continue
        tdf = top[task]
        col = f"tss_{task}"
        tss_vals = tdf[col].values if col in tdf.columns else np.array([])
        summary["per_task"][task] = {
            "num_top_neurons":        int(len(tdf)),
            "tss_mean":               float(np.mean(tss_vals)) if len(tss_vals) else 0.0,
            "tss_max":                float(np.max(tss_vals))  if len(tss_vals) else 0.0,
            "tss_min":                float(np.min(tss_vals))  if len(tss_vals) else 0.0,
            "layer_idx_most_common":  int(tdf["layer_idx"].mode()[0]) if len(tdf) else -1,
            "component_most_common":  str(tdf["component_type"].mode()[0]) if len(tdf) else "n/a",
            "mlp_ratio":              float((tdf["component_group"] == "mlp").mean()) if len(tdf) else 0.0,
            "attn_ratio":             float((tdf["component_group"] == "attn").mean()) if len(tdf) else 0.0,
        }

    # Jaccard 統計
    jmat = jaccard_df.values.copy()
    np.fill_diagonal(jmat, np.nan)
    off_diag = jmat[~np.isnan(jmat)]
    summary["jaccard"] = {
        "mean":    float(np.nanmean(off_diag)) if len(off_diag) else 0.0,
        "max":     float(np.nanmax(off_diag))  if len(off_diag) else 0.0,
        "min":     float(np.nanmin(off_diag))  if len(off_diag) else 0.0,
        "matrix":  {t: {t2: float(jaccard_df.loc[t, t2]) for t2 in tasks if t2 in jaccard_df.columns}
                    for t in tasks if t in jaccard_df.index},
    }

    path = os.path.join(output_dir, "rq1_summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"Saved → {path}")
    return summary


# ──────────────────────────────────────────
# メイン
# ──────────────────────────────────────────

def print_summary_table(summary: Dict):
    """ターミナルにサマリーをテーブル表示。"""
    print("\n" + "=" * 70)
    print(" RQ1 Summary: Top-TSS Neurons per Task")
    print("=" * 70)
    header = f"{'Task':<28} {'#Top':>5} {'TSSmean':>8} {'TSSmax':>8} {'TopLayer':>9} {'MLP%':>6}"
    print(header)
    print("-" * 70)
    for task, s in summary.get("per_task", {}).items():
        print(
            f"{task:<28} {s['num_top_neurons']:>5} "
            f"{s['tss_mean']:>8.3f} {s['tss_max']:>8.3f} "
            f"L{s['layer_idx_most_common']:<7} "
            f"{s['mlp_ratio']*100:>5.1f}%"
        )
    j = summary.get("jaccard", {})
    print("-" * 70)
    print(f"\nJaccard (off-diag) — mean: {j.get('mean',0):.3f}, "
          f"max: {j.get('max',0):.3f}, min: {j.get('min',0):.3f}")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SELU RQ1: TSS neuron analysis")
    parser.add_argument("--input",          required=True,
                        help="selu_neurons_combined.json のパス")
    parser.add_argument("--output_dir",     default="./rq1_results",
                        help="出力ディレクトリ")
    parser.add_argument("--top_k",          type=int, default=100,
                        help="タスクあたりのTSS上位ニューロン数 (default: 100)")
    parser.add_argument("--tss_threshold",  type=float, default=1.5,
                        help="TSSスコアの下限 (default: 1.5; 1.0=平均と同等)")
    parser.add_argument("--heatmap_neurons", type=int, default=50,
                        help="TSSヒートマップに表示するニューロン数 (default: 50)")
    parser.add_argument("--tasks",          nargs="+", default=None,
                        help="分析するタスク名を絞る (省略時: JSON内の全タスク)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load ---
    layers_dict, meta, tasks = load_combined_json(args.input)
    if args.tasks:
        tasks = [t for t in args.tasks if t in tasks]
        log.info(f"Filtered tasks: {tasks}")

    # --- Flatten ---
    df = flatten_neurons(layers_dict, tasks)

    # --- Top neurons per task ---
    log.info(f"Selecting top-{args.top_k} neurons (TSS >= {args.tss_threshold}) per task...")
    top = get_top_neurons(df, tasks, args.top_k, args.tss_threshold)
    save_top_neurons_csv(top, args.output_dir)

    # --- Plots ---
    log.info("Plotting layer distribution...")
    plot_layer_distribution(top, tasks, args.output_dir)

    log.info("Plotting component distribution...")
    plot_component_distribution(top, tasks, args.output_dir)

    log.info("Computing Jaccard similarity...")
    jaccard_df = compute_jaccard(top)
    plot_jaccard_heatmap(jaccard_df, args.output_dir)

    log.info("Plotting TSS heatmap...")
    plot_tss_heatmap(df, top, tasks, args.output_dir, args.heatmap_neurons)

    # --- Summary ---
    summary = save_summary(top, jaccard_df, tasks, meta, args.output_dir)
    print_summary_table(summary)

    log.info(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
