#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_selu_gradient_analysis.py
ベース: run_full_generation_gradient_analysis.py (RQ1)

[変更点まとめ]
1. language → task への読み替え
   PLSスコア(プログラミング言語特化)をTSS(タスク特化スコア)に読み替え

2. 入力/ターゲットの再設計
   元コード : prompt + generated_code → target_code(正解コード) に対するCE
   本スクリプト: input_text → label_text(正解ラベルトークン列) に対するCE
     * label_text は LABEL_TEXT_MAP でタスクごとにラベル値→文字列へ変換
     * generated_code 部分は不要のため削除

3. 損失関数
   元コード : 次トークン予測 CrossEntropy (言語モデリング損失)
   本スクリプト: input_text に対する forward + 最終トークン位置の logits で
                 label_text の先頭トークンへの CrossEntropy
                 → 「このテキストを読んだ後、正解ラベルを出力するのに重要なニューロン」

4. NER(se_entities) / MLM(requirement_completion) はスキップ
   理由: decoder-only では構造的に解析不可

[維持した設計]
- detect_model_type, _register_llama_hooks, _register_generic_hooks の構造
- フックキー命名規則: "layer_{idx}_{comp}" / "layer_{idx}_attention_{comp}"
- _parse_layer_name の実装
- save_combined_results の JSON フォーマット(後続スクリプトとの互換性)
- dtype フォールバック (bfloat16 → float16 → float32)
- input_device 解決ロジック (hf_device_map 優先)
- _merge_gradient_info の累積平均方式

[TSSの定義]
  TSS(neuron n, task t) = importance(n,t) / (mean_{t'} importance(n,t') + ε)
  値が大きいほど「そのタスクにだけ反応するニューロン」
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import argparse
import gc
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ラベル → テキスト変換マップ
# ---------------------------------------------------------------------------
# 元コードの target_code に相当する「正解ラベルのテキスト表現」を定義する。
# モデルは input_text を読んだ後にこのトークンを出力することが期待される。
# タスクごとに {ラベル値: ターゲット文字列} の辞書を用意する。

LABEL_TEXT_MAP: Dict[str, Dict] = {
    # ---- 2値分類 ----
    "bug_issue":              {0: "normal",  1: "a bug"},
    "requirement_type": {0: "optional",  1: "required"},
    "incivility":             {0: "mean", 1: "civil"},
    "tone_bearing":           {0: "technical", 1: "social"},
    # ---- 多値分類 ----
    "closed_question": {
        0: "open", 1: "invalid", 2: "spam",
        3: "rant",  4: "specific",
    },
    "commit_intent": {
        0: "other", 1: "optimize", 2: "fix",
    },
    "issue_type": {
        0: "bug", 1: "feature", 2: "question",
    },
    "question_quality": {
        0: "close", 1: "edit", 2: "high",
    },
    "sentiment": {
        0: "negative", 1: "neutral", 2: "positive",
    },
    # ---- 多ラベル分類 (代表ラベルトークンをカンマ区切りで返す) ----
    # label は [0,1,0,...] のバイナリベクトルなので、
    # アクティブなクラス名を空白区切りで並べた文字列をターゲットにする。
    "comment_type_java":   None,  # _multilabel_to_text で処理
    "comment_type_pharo":  None,
    "comment_type_python": None,
    "review_aspect":       None,
    "smell_doc":           None,
    # ---- 回帰 ----
    "story_points": None,  # _regression_to_text で処理
}

# 多ラベルタスクのクラス名定義
# ※ parquet の label_* 列名から "label_" prefix を除いた名前を列順通りに記載 1token
#
# comment_type_java  : label_summary, label_ownership, label_expand,
#                      label_usage, label_pointer, label_deprecation, label_rational
# comment_type_pharo : label_keyimplementationpoints, label_example, label_responsibilities,
#                      label_classreferences, label_intent, label_keymessages, label_collaborators
# comment_type_python: label_usage, label_parameters, label_developmentNotes,
#                      label_expand, label_Summary
# review_aspect      : label_usability, label_others, label_onlysentiment, label_bug,
#                      label_performance, label_community, label_documentation,
#                      label_compatibility, label_legal, label_portability, label_security
# smell_doc          : label_fragmented, label_tangled, label_excessive,
#                      label_bloated, label_lazy
MULTILABEL_CLASSES: Dict[str, List[str]] = {
    "comment_type_java": [
        "summary", "ownership", "expand", "usage",
        "pointer", "deprecated", "reason",
    ],
    "comment_type_pharo": [
        "implementation", "example", "role",
        "reference", "intent", "message", "partner",
    ],
    "comment_type_python": [
        "usage", "parameters", "note", "expand", "Summary",
    ],
    "review_aspect": [
        "usable", "others", "tone", "bug", "performance",
        "community", "documentation", "compatible", "legal",
        "migration", "security",
    ],
    "smell_doc": [
        "split", "mixed", "verbose", "heavy", "lazy",
    ],
}

# decoder-only では解析不可なタスク
UNSUPPORTED_TASKS = {
    "se_entities":            "NER: BIOタグはトークン単位のラベルが必要 → encoder-only 必須",
    "requirement_completion": "MLM: 双方向マスク予測 → encoder-only 必須",
}

# サポートされる全タスク (unsupported 除く)
ALL_SUPPORTED_TASKS = [t for t in LABEL_TEXT_MAP]


# ---------------------------------------------------------------------------
# ラベル変換ヘルパー
# ---------------------------------------------------------------------------

def label_to_text(task: str, label) -> Optional[str]:
    """
    ラベル値をターゲット文字列に変換する。
    変換できない場合は None を返す（サンプルをスキップ）。
    """
    if task in UNSUPPORTED_TASKS:
        return None

    # 多ラベル
    if task in MULTILABEL_CLASSES:
        classes = MULTILABEL_CLASSES[task]
        try:
            arr = np.array(label, dtype=int)
            active = [classes[i] for i in range(min(len(arr), len(classes))) if arr[i] == 1]
            return " ".join(active) if active else "none"
        except Exception:
            return None

    # 回帰 (story_points): 整数に丸めてテキスト化
    if task == "story_points":
        try:
            return str(int(round(float(label))))
        except Exception:
            return None

    # 通常の分類
    mapping = LABEL_TEXT_MAP.get(task)
    if mapping is None:
        return None
    try:
        return mapping[int(label)]
    except (KeyError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# モデル種別判定 (元コードから維持)
# ---------------------------------------------------------------------------

def detect_model_type(model_name: str, model=None) -> str:
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        return "qwen"
    elif "codellama" in model_name_lower:
        return "codellama"
    elif "llama" in model_name_lower:
        if model and hasattr(model, "config"):
            if hasattr(model.config, "_name_or_path") and \
               "codellama" in model.config._name_or_path.lower():
                return "codellama"
            if hasattr(model.config, "model_type") and \
               model.config.model_type == "codellama":
                return "codellama"
        return "llama"
    return "unknown"


# ---------------------------------------------------------------------------
# メインクラス
# ---------------------------------------------------------------------------

class SELUGradientFinder:
    """
    元コードの FullGenerationGradientFinder を SELU タスク向けに改変。

    主な変更:
    - analyze_multiple_samples → analyze_task_samples (language → task)
    - compute_gradients_for_full_generation
        → compute_gradients_for_classification
          (generated_code 廃止、label_text をターゲットに使用)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
        offload_to_cpu: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype

        log.info(f"Loading tokenizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "auto" or offload_to_cpu:
            device_map = "auto"
        else:
            device_map = {"": device}

        log.info(f"Loading model: {model_path} (dtype={dtype})")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )
        except Exception:
            log.warning("bfloat16 failed, falling back to float32")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                device_map=device_map,
                low_cpu_mem_usage=True,
            )

        self.model.eval()
        self.hooks: list = []
        self.activations: Dict[str, torch.Tensor] = {}
        self.gradients: Dict[str, torch.Tensor] = {}
        self._register_hooks()
        log.info("Model loaded and hooks registered.")

    # ------------------------------------------------------------------
    # フック登録 (元コードの構造を維持)
    # ------------------------------------------------------------------

    def _register_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.activations = {}
        self.gradients = {}

        model_type = detect_model_type(self.model_path, self.model)
        if model_type in ("codellama", "llama") or isinstance(self.model, LlamaForCausalLM):
            self._register_llama_hooks()
        else:
            self._register_generic_hooks()

    def _register_llama_hooks(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            mlp_components = {}
            for name in ("gate_proj", "up_proj", "down_proj"):
                if hasattr(layer.mlp, name):
                    mlp_components[name] = getattr(layer.mlp, name)
            self._register_component_hooks(mlp_components, layer_idx, prefix="")

            attn_components = {}
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(layer.self_attn, name):
                    attn_components[name] = getattr(layer.self_attn, name)
            self._register_component_hooks(attn_components, layer_idx, prefix="attention_")

    def _register_component_hooks(self, components, layer_idx, prefix=""):
        for comp_name, comp in components.items():
            key_name = f"layer_{layer_idx}_{prefix}{comp_name}"

            def make_forward_hook(key):
                def hook(module, input, output):
                    self.activations[key] = output.detach()
                return hook

            def make_backward_hook(key):
                def hook(module, grad_in, grad_out):
                    if grad_out[0] is not None:
                        self.gradients[key] = grad_out[0].detach()
                return hook

            fh = comp.register_forward_hook(make_forward_hook(key_name))
            bh = comp.register_full_backward_hook(make_backward_hook(key_name))
            self.hooks.extend([fh, bh])

    def _register_generic_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                def make_fwd(n):
                    def hook(module, input, output):
                        self.activations[n] = output.detach()
                    return hook

                def make_bwd(n):
                    def hook(module, grad_in, grad_out):
                        if grad_out[0] is not None:
                            self.gradients[n] = grad_out[0].detach()
                    return hook

                self.hooks.append(module.register_forward_hook(make_fwd(name)))
                self.hooks.append(module.register_full_backward_hook(make_bwd(name)))

    # ------------------------------------------------------------------
    # input_device 解決 (元コードから維持)
    # ------------------------------------------------------------------

    def _resolve_input_device(self) -> str:
        if hasattr(self.model, "hf_device_map"):
            device_values = list(set(self.model.hf_device_map.values()))
            valid = [d for d in device_values if d != "cpu" and not isinstance(d, int)]
            if valid:
                return valid[0]
        if self.device == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        return self.device

    # ------------------------------------------------------------------
    # グラジェント計算 (元コードの compute_gradients_for_full_generation を改変)
    #
    # 元コード:
    #   prompt + generated_code → forward → target_code に対する CE
    #
    # 本コード:
    #   input_text → forward → label_text の先頭トークンに対する CE
    #   (「このテキストを読んだ後、正解ラベルトークンを出力するのに重要なニューロン」)
    # ------------------------------------------------------------------

    def compute_gradients_for_sample(
        self,
        input_text: str,
        label_text: str,
        task: str,
    ) -> Dict[str, Any]:
        self.activations = {}
        self.gradients = {}

        input_device = self._resolve_input_device()

        # --- 入力テキストのトークナイズ ---
        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        ).to(input_device)

        # --- ターゲットラベルのトークナイズ (先頭トークンのみ使用) ---
        label_enc = self.tokenizer(
            label_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(input_device)
        # 先頭トークンIDを正解ラベルとして使う
        target_token_id = label_enc["input_ids"][0, 0].unsqueeze(0)  # (1,)

        self.model.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = self.model(**enc, return_dict=True)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # 最後のトークン位置のlogits → 次に何を出力するか
            last_logits = logits[0, -1, :].float().unsqueeze(0)  # (1, vocab_size)

            try:
                loss = torch.nn.functional.cross_entropy(last_logits, target_token_id)
                loss.backward()
            except RuntimeError as e:
                if "ScalarType" in str(e) or "scalar type" in str(e).lower():
                    last_logits_f16 = logits[0, -1, :].half().unsqueeze(0)
                    loss = torch.nn.functional.cross_entropy(last_logits_f16, target_token_id)
                    loss.backward()
                else:
                    raise

        return self._collect_gradient_info(task)

    # ------------------------------------------------------------------
    # 勾配情報の収集 (元コードの _collect_gradient_info から維持)
    # ------------------------------------------------------------------

    def _collect_gradient_info(self, task: str) -> Dict[str, Any]:
        result = {
            "task": task,
            "model_path": self.model_path,
            "layers": {},
        }

        for key in self.activations:
            if key not in self.gradients:
                continue
            try:
                activation = self.activations[key].to(torch.float32)
                gradient = self.gradients[key].to(torch.float32)

                act_mean = activation.mean(dim=1)   # (1, hidden)
                grad_mean = gradient.mean(dim=1)    # (1, hidden)
                importance = (grad_mean * act_mean).abs()
                importance_np = importance.detach().cpu().numpy()
            except Exception:
                try:
                    act_mean = self.activations[key].detach().cpu().float().mean(dim=1)
                    grad_mean = self.gradients[key].detach().cpu().float().mean(dim=1)
                    importance = (grad_mean * act_mean).abs()
                    importance_np = importance.numpy()
                except Exception:
                    continue

            neurons = {}
            for neuron_idx in range(importance_np.shape[1]):
                neuron_importance = float(importance_np[0, neuron_idx])
                neuron_activation = float(act_mean[0, neuron_idx])
                neuron_gradient  = float(grad_mean[0, neuron_idx])

                if neuron_importance > 0:
                    neurons[f"neuron_{neuron_idx}"] = {
                        "neuron_idx": neuron_idx,
                        "importance": neuron_importance,
                        "activation": neuron_activation,
                        "gradient":   neuron_gradient,
                        "importance_by_language": {task: neuron_importance},
                        "activation_by_language": {task: neuron_activation},
                        "gradient_by_language":   {task: neuron_gradient},
                    }

            result["layers"][key] = {
                "component_type": key.split("_")[-1] if "_" in key else "unknown",
                "neurons":    neurons,
                "activation": float(act_mean.mean().item()),
                "gradient":   float(grad_mean.mean().item()),
                "importance": float(importance.mean().item()),
            }

        return result

    # ------------------------------------------------------------------
    # タスク単位の集計 (元コードの analyze_multiple_samples を改変)
    # ------------------------------------------------------------------

    def analyze_task_samples(
        self,
        samples: List[Dict[str, Any]],
        task: str,
    ) -> Dict[str, Any]:
        combined = {
            "task": task,
            "model_path": self.model_path,
            "layers": {},
        }
        sample_count = 0

        for sample in tqdm(samples, desc=f"  {task}", leave=False):
            try:
                input_text = sample["input_text"]
                label_text = sample["label_text"]

                gradient_info = self.compute_gradients_for_sample(
                    input_text=input_text,
                    label_text=label_text,
                    task=task,
                )
                self._merge_gradient_info(combined, gradient_info, task, sample_count)
                sample_count += 1
            except Exception as e:
                log.debug(f"  sample error ({task}): {e}")
                continue
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        log.info(f"  {task}: {sample_count} samples processed, {len(combined['layers'])} layers")
        return combined

    # ------------------------------------------------------------------
    # 累積平均マージ (元コードの _merge_gradient_info を維持)
    # ------------------------------------------------------------------

    def _merge_gradient_info(
        self,
        combined: Dict,
        new_info: Dict,
        task: str,
        sample_count: int,
    ):
        if sample_count == 0:
            combined["layers"] = new_info["layers"]
            return

        for layer_key, layer_info in new_info["layers"].items():
            if layer_key not in combined["layers"]:
                combined["layers"][layer_key] = layer_info
                continue

            for neuron_key, neuron_info in layer_info["neurons"].items():
                if neuron_key not in combined["layers"][layer_key]["neurons"]:
                    combined["layers"][layer_key]["neurons"][neuron_key] = neuron_info
                    continue

                ex = combined["layers"][layer_key]["neurons"][neuron_key]
                n  = sample_count

                # importance
                ex["importance"] = (ex["importance"] * n + neuron_info["importance"]) / (n + 1)
                if task in ex["importance_by_language"]:
                    ex["importance_by_language"][task] = (
                        ex["importance_by_language"][task] * n
                        + neuron_info["importance_by_language"][task]
                    ) / (n + 1)
                else:
                    ex["importance_by_language"][task] = neuron_info["importance_by_language"][task]

                # activation
                ex["activation"] = (ex["activation"] * n + neuron_info["activation"]) / (n + 1)
                if task in ex["activation_by_language"]:
                    ex["activation_by_language"][task] = (
                        ex["activation_by_language"][task] * n
                        + neuron_info["activation_by_language"][task]
                    ) / (n + 1)
                else:
                    ex["activation_by_language"][task] = neuron_info["activation_by_language"][task]

                # gradient
                ex["gradient"] = (ex["gradient"] * n + neuron_info["gradient"]) / (n + 1)
                if task in ex["gradient_by_language"]:
                    ex["gradient_by_language"][task] = (
                        ex["gradient_by_language"][task] * n
                        + neuron_info["gradient_by_language"][task]
                    ) / (n + 1)
                else:
                    ex["gradient_by_language"][task] = neuron_info["gradient_by_language"][task]

    # ------------------------------------------------------------------
    # TSS 計算
    # ------------------------------------------------------------------

    @staticmethod
    def compute_tss(
        all_task_results: Dict[str, Dict],
        eps: float = 1e-8,
    ) -> Dict[str, Dict]:
        """
        TSS(neuron n, task t) = importance(n,t) / (mean_{t'} importance(n,t') + ε)

        all_task_results: {task: combined_result} (analyze_task_samples の返り値)
        返り値: {task: {layer_key: {neuron_key: tss_score}}}
        """
        tasks = list(all_task_results.keys())

        # 全レイヤーキーを収集
        all_layer_keys: set = set()
        for r in all_task_results.values():
            all_layer_keys.update(r.get("layers", {}).keys())

        # ニューロンごとに全タスク平均 importance を計算（分母）
        # denom[layer_key][neuron_key] = float
        denom: Dict[str, Dict[str, float]] = {}
        for lk in all_layer_keys:
            denom[lk] = {}
            # そのレイヤーに存在する全ニューロンキーを収集
            all_neuron_keys: set = set()
            for r in all_task_results.values():
                all_neuron_keys.update(r.get("layers", {}).get(lk, {}).get("neurons", {}).keys())

            for nk in all_neuron_keys:
                vals = []
                for t in tasks:
                    imp = (all_task_results[t]
                           .get("layers", {})
                           .get(lk, {})
                           .get("neurons", {})
                           .get(nk, {})
                           .get("importance", 0.0))
                    vals.append(imp)
                denom[lk][nk] = float(np.mean(vals)) + eps

        # TSS を計算
        tss_results: Dict[str, Dict] = {}
        for task in tasks:
            tss_results[task] = {}
            for lk, layer_data in all_task_results[task].get("layers", {}).items():
                tss_results[task][lk] = {}
                for nk, neuron_data in layer_data.get("neurons", {}).items():
                    imp = neuron_data.get("importance", 0.0)
                    tss_results[task][lk][nk] = imp / denom[lk][nk]

        return tss_results

    # ------------------------------------------------------------------
    # 保存 (元コードの save_combined_results のフォーマットを維持しつつ TSS を追加)
    # ------------------------------------------------------------------

    def save_combined_results(
        self,
        all_task_results: Dict[str, Dict],
        tss_results: Dict[str, Dict],
        output_path: str,
    ) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        tasks = list(all_task_results.keys())
        all_layers: set = set()
        for r in all_task_results.values():
            all_layers.update(r.get("layers", {}).keys())

        combined: Dict = {}

        for layer_name in all_layers:
            layer_info = self._parse_layer_name(layer_name)
            combined[layer_name] = {
                "layer_info": layer_info,
                "importance_by_language": {},  # 元コードとの互換性のため key 名を維持
                "neurons": {},
            }

            for task in tasks:
                r = all_task_results[task]
                if "layers" not in r or layer_name not in r["layers"]:
                    continue
                ld = r["layers"][layer_name]

                combined[layer_name]["importance_by_language"][task] = {
                    "activation": float(ld.get("activation", 0)),
                    "gradient":   float(ld.get("gradient", 0)),
                    "importance": float(ld.get("importance", 0)),
                }

                for neuron_key, ni in ld.get("neurons", {}).items():
                    nidx = ni["neuron_idx"]
                    detailed_key = f"{layer_name}_neuron_{nidx}"

                    if detailed_key not in combined[layer_name]["neurons"]:
                        combined[layer_name]["neurons"][detailed_key] = {
                            "layer_name": layer_name,
                            "neuron_idx": nidx,
                            "component_type": layer_info["component_type"],
                            "layer_idx":      layer_info["layer_idx"],
                            "importance_by_language": {},
                            "gradient_by_language":   {},
                            "activation_by_language": {},
                            "tss_by_task":            {},
                            "detailed_key": detailed_key,
                        }

                    nd = combined[layer_name]["neurons"][detailed_key]
                    nd["importance_by_language"][task] = float(ni["importance_by_language"].get(task, 0))
                    nd["gradient_by_language"][task]   = float(ni["gradient_by_language"].get(task, 0))
                    nd["activation_by_language"][task] = float(ni["activation_by_language"].get(task, 0))

                    # TSS スコアを追記
                    tss_val = (tss_results
                               .get(task, {})
                               .get(layer_name, {})
                               .get(neuron_key, 0.0))
                    nd["tss_by_task"][task] = float(tss_val)

        # メタデータ
        total_neurons = sum(
            len(v["neurons"])
            for k, v in combined.items()
            if k != "_metadata"
        )
        combined["_metadata"] = {
            "model_path":       self.model_path,
            "num_tasks":        len(tasks),
            "tasks":            tasks,
            "num_layers":       len(all_layers),
            "num_neurons":      total_neurons,
            "created_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "combined_format":  True,
            "format_version":   "3.0-selu",
            "tss_formula":      "TSS(n,t) = importance(n,t) / (mean_t'[importance(n,t')] + eps)",
            "skipped_tasks":    list(UNSUPPORTED_TASKS.keys()),
            "skip_reason":      UNSUPPORTED_TASKS,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

        log.info(f"Saved combined results → {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # _parse_layer_name (元コードから維持)
    # ------------------------------------------------------------------

    def _parse_layer_name(self, layer_name: str) -> Dict:
        info: Dict = {}
        if "layer_" in layer_name:
            try:
                info["layer_idx"] = int(layer_name.split("layer_")[1].split("_")[0])
            except (IndexError, ValueError):
                info["layer_idx"] = -1
        else:
            info["layer_idx"] = -1

        for comp in ("gate_proj","up_proj","down_proj","q_proj","k_proj","v_proj","o_proj","mlp","self_attn"):
            if f"_{comp}" in layer_name:
                info["component_type"] = comp
                break
        else:
            info["component_type"] = "other"

        info["original_name"] = layer_name
        return info

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


# ---------------------------------------------------------------------------
# データロード
# ---------------------------------------------------------------------------

def resolve_text_column(df: pd.DataFrame, task: str) -> Optional[str]:
    candidates = ["text", "body", "title", "comment", "content", "sentence", "input"]
    for c in candidates:
        if c in df.columns:
            return c
    log.warning(f"  {task}: no text column found in {list(df.columns)}")
    return None


def clean_text(text: str) -> str:
    """
    テキストの前処理。
    comment_type_* は "コメント本文 | ファイル名" の形式なので
    | 以降のファイル名部分を除去する。
    """
    text = str(text).strip()
    if "|" in text:
        text = text.split("|")[0].strip()
    return text


def extract_label(df: pd.DataFrame, task: str, row: pd.Series):
    """
    タスクに応じてラベル値を取り出す。

    通常タスク    : label 列の値をそのまま返す
    多ラベルタスク: label_* 列をバイナリリストにまとめて返す
                   列の順序は MULTILABEL_CLASSES[task] と一致させる
    """
    if task in MULTILABEL_CLASSES:
        class_names = MULTILABEL_CLASSES[task]
        # 実際に存在する label_* 列だけを使う（列順を class_names に揃える）
        label_vec = []
        for cls in class_names:
            col = f"label_{cls}"
            label_vec.append(int(row[col]) if col in df.columns else 0)
        return label_vec

    # 通常: label 列
    if "label" in df.columns:
        return row["label"]

    return None


def load_selu_samples(
    data_dir: str,
    task: str,
    num_samples: int,
) -> List[Dict[str, str]]:
    """
    SELUのparquetを読み込み、{input_text, label_text} のリストを返す。
    label_text は label_to_text で変換済み。

    対応するデータ形式:
      通常タスク    : id / text / label の3列
      多ラベルタスク: id / text / label_summary / label_ownership / ... の複数列
                     (label 列は存在しない)
    """
    if task in UNSUPPORTED_TASKS:
        log.warning(f"[SKIP] {task}: {UNSUPPORTED_TASKS[task]}")
        return []

    # parquet を探す
    for fname in ("train.parquet", "test.parquet", f"{task}.parquet"):
        p = Path(data_dir) / task / fname
        if p.exists():
            parquet_path = p
            break
    else:
        log.warning(f"[SKIP] {task}: parquet not found under {Path(data_dir) / task}")
        return []

    df = pd.read_parquet(parquet_path)
    log.info(f"  {task}: {len(df)} rows loaded from {parquet_path.name}")

    text_col = resolve_text_column(df, task)
    if text_col is None:
        return []

    # ラベル列の存在確認
    is_multilabel = task in MULTILABEL_CLASSES
    if not is_multilabel and "label" not in df.columns:
        log.warning(f"[SKIP] {task}: 'label' column not found in {list(df.columns)}")
        return []
    if is_multilabel:
        existing_label_cols = [c for c in df.columns if c.startswith("label_")]
        if not existing_label_cols:
            log.warning(f"[SKIP] {task}: no 'label_*' columns found in {list(df.columns)}")
            return []
        log.info(f"  {task}: multilabel columns = {existing_label_cols}")

    if num_samples and num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42)

    samples = []
    for _, row in df.iterrows():
        text = clean_text(row[text_col])
        if not text:
            continue
        label_val = extract_label(df, task, row)
        if label_val is None:
            continue
        lt = label_to_text(task, label_val)
        if lt is None:
            continue
        samples.append({"input_text": text, "label_text": lt})

    log.info(f"  {task}: {len(samples)} valid samples")
    return samples


# ---------------------------------------------------------------------------
# エントリポイント
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SELU Task Specificity Score (TSS) analysis")
    parser.add_argument("--model_path", type=str, required=True,
                        help="HuggingFace モデルパス (ローカル or Hub)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="SELUの datasets/ ディレクトリ")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="結果の出力先ディレクトリ")
    parser.add_argument("--output_file", type=str, default=None,
                        help="出力ファイル名 (省略時: selu_neurons_combined.json)")
    parser.add_argument("--tasks", type=str, nargs="+",
                        default=ALL_SUPPORTED_TASKS,
                        help="解析するタスク名 (省略時: 全サポートタスク)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="タスクあたりのサンプル数 (0 = 全件)")
    parser.add_argument("--device", type=str, default="auto",
                        help="cuda / cpu / auto")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--offload_to_cpu", action="store_true")
    args = parser.parse_args()

    # dtype 解決 (元コードのフォールバック方式を維持)
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    if args.dtype == "bfloat16":
        if not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
            log.warning("bfloat16 not supported, falling back to float16")
            dtype = torch.float16

    # unsupported タスクを事前に除外して報告
    requested = set(args.tasks)
    for t in sorted(requested & set(UNSUPPORTED_TASKS.keys())):
        log.warning(f"[SKIP] {t}: {UNSUPPORTED_TASKS[t]}")
    runnable = sorted(requested - set(UNSUPPORTED_TASKS.keys()))

    if not runnable:
        log.error("No runnable tasks. Exiting.")
        return

    log.info(f"Tasks to analyze ({len(runnable)}): {runnable}")
    os.makedirs(args.output_dir, exist_ok=True)

    finder = SELUGradientFinder(
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
        offload_to_cpu=args.offload_to_cpu,
    )

    all_task_results: Dict[str, Dict] = {}

    for task in runnable:
        log.info(f"=== {task} ===")
        samples = load_selu_samples(args.data_dir, task, args.num_samples)
        if not samples:
            continue

        result = finder.analyze_task_samples(samples, task)
        if result.get("layers"):
            all_task_results[task] = result

        torch.cuda.empty_cache()
        gc.collect()

    if not all_task_results:
        log.error("No results collected. Check --data_dir and task names.")
        finder.cleanup()
        return

    log.info("Computing TSS scores...")
    tss_results = SELUGradientFinder.compute_tss(all_task_results)

    output_file = args.output_file or "selu_neurons_combined.json"
    output_path = os.path.join(args.output_dir, output_file)
    finder.save_combined_results(all_task_results, tss_results, output_path)

    finder.cleanup()
    log.info("Done.")


if __name__ == "__main__":
    main()