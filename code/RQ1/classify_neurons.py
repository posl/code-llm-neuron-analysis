#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import argparse
import numpy as np
from pathlib import Path
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import tqdm

def normalize_neuron_key(key):
    """Normalize neuron key format for consistency"""
    parts = key.split('_')
    try:
        idx = -1
        if "down_proj_neuron" in key:
            idx = parts.index("down")
        elif "up_proj_neuron" in key:
            idx = parts.index("up")
        elif "gate_proj_neuron" in key:
            idx = parts.index("gate")

        if idx > 0 and parts[idx-1] != "mlp":
            parts.insert(idx, "mlp")
            return "_".join(parts)
        elif idx == 0:
            parts.insert(idx, "mlp")
            return "_".join(parts)

    except ValueError:
        pass
    return key

def normalize_gradient_data_keys(gradient_data):
    """Normalize all keys in gradient data"""
    normalized_data = {}
    keys_modified_count = 0
    for layer_name, layer_content in gradient_data.items():
        new_layer_name = normalize_neuron_key(layer_name)
        if new_layer_name != layer_name:
            keys_modified_count +=1
        
        if "neurons" in layer_content:
            normalized_neurons = {}
            for neuron_key, neuron_value in layer_content["neurons"].items():
                new_neuron_key = normalize_neuron_key(neuron_key)
                if new_neuron_key != neuron_key:
                    keys_modified_count += 1
                normalized_neurons[new_neuron_key] = neuron_value
            normalized_data[new_layer_name] = {**layer_content, "neurons": normalized_neurons}
        else:
            normalized_data[new_layer_name] = layer_content
    return normalized_data

def load_gradient_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_neuron_features(gradient_data, min_importance=0.0001):
    """Extract and compute features for each neuron"""
    languages = set()
    neuron_features = {}
    
    # Initialize feature statistics
    feature_stats = {
        "importance": {"values": []},
        "gradient": {"values": []},
        "activation": {"values": []},
        "contrast_score": {"values": []},
        "importance_ratio": {"values": []},
        "importance_average": {"values": []},
        "importance_min": {"values": []},
        "importance_max_min_ratio": {"values": []},
        "importance_std": {"values": []},
        "min_avg_importance_ratio": {"values": []},
        "significant_language_ratio": {"values": []}
    }

    # Extract features for each neuron
    for layer_name, layer_data in gradient_data.items():
        if "neurons" in layer_data:
            for neuron_key, neuron_data in layer_data["neurons"].items():
                if "importance_by_language" in neuron_data:
                    languages.update(neuron_data["importance_by_language"].keys())

                    features = {}

                    # Process importance by language
                    for language, importance_data in neuron_data["importance_by_language"].items():
                        if isinstance(importance_data, list):
                            if isinstance(importance_data[0], list):
                                flat_values = [val for sublist in importance_data for val in sublist]
                                features[language] = sum(flat_values) / len(flat_values) if flat_values else 0.0
                            else:
                                features[language] = sum(importance_data) / len(importance_data) if importance_data else 0.0
                        else:
                            features[language] = importance_data
                    
                    # Copy gradient and activation data
                    features["gradient_by_language"] = neuron_data.get("gradient_by_language", {})
                    features["activation_by_language"] = neuron_data.get("activation_by_language", {})

                    # Copy metadata
                    for key in ["layer_name", "neuron_idx", "component_type", "layer_idx", 
                               "max_language", "max_importance", "importance_ratio", "avg_other_importance"]:
                        if key in neuron_data:
                            features[key] = neuron_data[key]

                    language_importances = [features[lang] for lang in neuron_data["importance_by_language"].keys() if lang in features]
                    max_importance = max(language_importances) if language_importances else 0.0

                    if max_importance >= min_importance:
                        neuron_features[neuron_key] = features
                        feature_stats["importance"]["values"].append(max_importance)
                        
                        if "gradient_by_language" in features:
                            max_gradient = max([abs(val) for val in features["gradient_by_language"].values()]) if features["gradient_by_language"] else 0
                            feature_stats["gradient"]["values"].append(max_gradient)
                        
                        if "activation_by_language" in features:
                            max_activation = max([abs(val) for val in features["activation_by_language"].values()]) if features["activation_by_language"] else 0
                            feature_stats["activation"]["values"].append(max_activation)

    languages = sorted(list(languages))

    # Compute additional features for each neuron
    for neuron_id, features in neuron_features.items():
        for language in languages:
            if language not in features:
                features[language] = 0.0

        if "max_importance" not in features:
            importance_values = [features[lang] for lang in languages]
            max_importance = max(importance_values)
            features["max_importance"] = max_importance
            max_lang_idx = np.argmax(importance_values)
            max_lang = languages[max_lang_idx]
            features["max_language"] = max_lang
            other_importance = [features[l] for l in languages if l != max_lang]
            avg_other_importance = sum(other_importance) / len(other_importance) if other_importance else 0
            importance_ratio = max_importance / (avg_other_importance + 1e-10)
            features["importance_ratio"] = importance_ratio
            features["avg_other_importance"] = avg_other_importance
            feature_stats["importance_ratio"]["values"].append(importance_ratio)

        importance_values = [features[lang] for lang in languages]
        importance_variance = np.var(importance_values)
        features["importance_variance"] = importance_variance
        
        # Compute contrast score
        max_lang = features["max_language"]
        target_score = features.get(max_lang, 0.0)
        other_scores = [features.get(lang, 0.0) for lang in languages if lang != max_lang]
        avg_other = sum(other_scores) / len(other_scores) if other_scores else 0
        
        if target_score + avg_other > 0:
            contrast_score = target_score / (target_score + avg_other)
        else:
            contrast_score = 0.5
            
        features["contrast_score"] = contrast_score
        feature_stats["contrast_score"]["values"].append(contrast_score)
        
        # Compute statistical metrics
        avg_importance = sum(importance_values) / len(importance_values) if importance_values else 0
        features["importance_average"] = avg_importance
        feature_stats["importance_average"]["values"].append(avg_importance)
        
        min_importance_val = min(importance_values) if importance_values else 0
        features["importance_min"] = min_importance_val
        feature_stats["importance_min"]["values"].append(min_importance_val)
        
        if min_importance_val > 0:
            max_min_ratio = max_importance / min_importance_val
        else:
            max_min_ratio = 1000.0
        features["importance_max_min_ratio"] = max_min_ratio
        feature_stats["importance_max_min_ratio"]["values"].append(max_min_ratio)
        
        importance_std = np.std(importance_values)
        features["importance_std"] = importance_std
        feature_stats["importance_std"]["values"].append(importance_std)
        
        if avg_importance > 0:
            features["importance_cv"] = importance_std / avg_importance
        else:
            features["importance_cv"] = 0.0

        if avg_importance > 0:
            min_avg_ratio = min_importance_val / avg_importance
        else:
            min_avg_ratio = 0.0
        features["min_avg_importance_ratio"] = min_avg_ratio
        feature_stats["min_avg_importance_ratio"]["values"].append(min_avg_ratio)
        
        # Compute significant language ratio
        significance_threshold = max_importance * 0.4
        significant_langs = sum(1 for val in importance_values if val >= significance_threshold)
        significant_lang_ratio = significant_langs / len(languages) if languages else 0
        features["significant_language_ratio"] = significant_lang_ratio
        feature_stats["significant_language_ratio"]["values"].append(significant_lang_ratio)

    # Compute statistics for each feature
    for feature_name, stats in feature_stats.items():
        if stats["values"]:
            values = np.array(stats["values"])
            stats["min"] = float(np.min(values))
            stats["max"] = float(np.max(values))
            stats["mean"] = float(np.mean(values))
            stats["median"] = float(np.median(values))
            stats["std"] = float(np.std(values))
            stats["percentiles"] = {
                "25": float(np.percentile(values, 25)),
                "50": float(np.percentile(values, 50)),
                "60": float(np.percentile(values, 60)),
                "75": float(np.percentile(values, 75)),
                "85": float(np.percentile(values, 85)),
                "90": float(np.percentile(values, 90)),
                "95": float(np.percentile(values, 95)),
                "99": float(np.percentile(values, 99))
            }
    
    return neuron_features, languages, feature_stats

def contrastive_classification(neuron_features, languages, feature_stats=None, max_percentage=0.015):
    """Classify neurons using contrastive method"""
    results = {
        "language_specific": {lang: [] for lang in languages},
        "general": [],
        "irrelevant": []
    }
    
    # Set thresholds based on feature statistics
    if feature_stats and 'contrast_score' in feature_stats and 'importance' in feature_stats:
        contrast_threshold = feature_stats['contrast_score']['percentiles']['90']
        importance_threshold_specific = (feature_stats['importance']['percentiles']['50'] + feature_stats['importance']['percentiles']['75']) / 2
        importance_threshold_general = feature_stats['importance']['percentiles']['25']
    else:
        contrast_threshold = 0.65
        importance_threshold_specific = 0.00003
        importance_threshold_general = 0.00001

    language_specific_candidates = {lang: [] for lang in languages}
    general_candidates = []
    irrelevant_candidates = []

    for neuron_id, features in neuron_features.items():
        if "contrast_score" in features:
            max_lang = features["max_language"]
            max_score = features["contrast_score"]
        else:
            # Compute contrast scores
            contrast_scores = {}
            for target_lang in languages:
                target_score = features.get(target_lang, 0.0)
                other_scores = [features.get(lang, 0.0) for lang in languages if lang != target_lang]
                avg_other = sum(other_scores) / len(other_scores) if other_scores else 0
                if target_score + avg_other > 0:
                    contrast_score = target_score / (target_score + avg_other)
                else:
                    contrast_score = 0.5
                contrast_scores[target_lang] = contrast_score
            max_lang, max_score = max(contrast_scores.items(), key=lambda x: x[1])

        importance_value = features.get(max_lang, 0.0)
        
        # Classify based on thresholds
        if max_score > contrast_threshold and importance_value > importance_threshold_specific:
            max_min_ratio = features.get("importance_max_min_ratio", 1.0)
            if max_min_ratio > 5.0:
                language_specific_candidates[max_lang].append({
                    "neuron_id": neuron_id,
                    "contrast_score": max_score,
                    "importance": importance_value,
                    "importance_ratio": features.get("importance_ratio", 0.0),
                    "max_min_ratio": max_min_ratio,
                    "classification_method": "contrastive",
                    "gradient": features.get("gradient_by_language", {}).get(max_lang, 0.0),
                    "activation": features.get("activation_by_language", {}).get(max_lang, 0.0)
                })
            elif max_score > 0.55:
                general_candidates.append({
                    "neuron_id": neuron_id,
                    "contrast_score": max_score,
                    "max_language": max_lang,
                    "importance": importance_value,
                    "max_min_ratio": max_min_ratio,
                    "classification_method": "contrastive",
                    "gradient": features.get("gradient_by_language", {}).get(max_lang, 0.0),
                    "activation": features.get("activation_by_language", {}).get(max_lang, 0.0)
                })
            else:
                irrelevant_candidates.append({
                    "neuron_id": neuron_id,
                    "contrast_score": max_score,
                    "max_min_ratio": max_min_ratio,
                    "max_importance": features.get("max_importance", 0.0)
                })
        elif (importance_value > importance_threshold_general and 
              features.get("importance_max_min_ratio", 1000.0) < 2.0 and 
              (features.get("significant_language_ratio", 0) > 0.6 or 
               features.get("min_avg_importance_ratio", 0) > 0.5)):
            general_candidates.append({
                "neuron_id": neuron_id,
                "max_language": max_lang,
                "importance": importance_value,
                "max_min_ratio": features.get("importance_max_min_ratio", 1.0),
                "min_avg_ratio": features.get("min_avg_importance_ratio", 0.0),
                "significant_language_ratio": features.get("significant_language_ratio", 0), 
                "importance_cv": features.get("importance_cv", 0.0),
                "avg_importance": features.get("importance_average", 0.0),
                "classification_method": "contrastive",
                "gradient": features.get("gradient_by_language", {}).get(max_lang, 0.0),
                "activation": features.get("activation_by_language", {}).get(max_lang, 0.0)
            })
        else:
            irrelevant_candidates.append({
                "neuron_id": neuron_id,
                "contrast_score": max_score,
                "max_importance": features.get("max_importance", 0.0)
            })
    
    # Apply limits to results
    total_neurons = len(neuron_features)
    max_neurons_per_language = max(50, int(total_neurons * max_percentage))
    
    for lang in languages:
        candidates = sorted(language_specific_candidates[lang], key=lambda x: x["contrast_score"], reverse=True)
        results["language_specific"][lang] = candidates[:max_neurons_per_language]
    
    max_general_neurons = max(50, int(total_neurons * 0.025))
    general_candidates = sorted(general_candidates, key=lambda x: x["importance"], reverse=True)
    results["general"] = general_candidates[:max_general_neurons]
    
    results["irrelevant"] = irrelevant_candidates

    return results

def statistical_classification(neuron_features, languages, feature_stats=None, max_percentage=0.015):
    """Classify neurons using statistical method"""
    results = {
        "language_specific": {lang: [] for lang in languages},
        "general": [],
        "irrelevant": []
    }

    # Set thresholds based on feature statistics
    if feature_stats and 'importance' in feature_stats and 'importance_ratio' in feature_stats:
        importance_threshold = feature_stats['importance']['percentiles']['60']
        general_threshold = feature_stats['importance']['percentiles']['25']
        specificity_threshold = feature_stats['importance_ratio']['percentiles']['85']
    else:
        importance_threshold = 0.000015
        specificity_threshold = 2.0
        general_threshold = 0.00001

    language_specific_candidates = {lang: [] for lang in languages}
    general_candidates = []
    irrelevant_candidates = []

    for neuron_id, features in neuron_features.items():
        language_scores = {lang: features.get(lang, 0.0) for lang in languages}
        max_lang, max_score = max(language_scores.items(), key=lambda x: x[1])
        other_scores = [score for lang, score in language_scores.items() if lang != max_lang]
        avg_other_score = sum(other_scores) / len(other_scores) if other_scores else 0
        specificity_ratio = max_score / (avg_other_score + 1e-10)
        scores_array = np.array(list(language_scores.values()))
        cv = np.std(scores_array) / (np.mean(scores_array) + 1e-10)

        # Classify based on thresholds
        if max_score >= importance_threshold and specificity_ratio >= specificity_threshold:
            language_specific_candidates[max_lang].append({
                "neuron_id": neuron_id,
                "importance": float(max_score),
                "contrast_score": float(specificity_ratio),
                "specificity_cv": float(cv),
                "gradient": features.get("gradient_by_language", {}).get(max_lang, 0.0),
                "activation": features.get("activation_by_language", {}).get(max_lang, 0.0)
            })
        elif max_score >= general_threshold:
            cv = np.std(scores_array) / (np.mean(scores_array) + 1e-10)
            significance_threshold = max_score * 0.4
            significant_langs = sum(1 for score in scores_array if score >= significance_threshold)
            significant_ratio = significant_langs / len(languages) if languages else 0
            
            if cv < 0.6 and significant_ratio > 0.6:
                general_candidates.append({
                    "neuron_id": neuron_id,
                    "importance": float(max_score),
                    "max_language": max_lang,
                    "cv": float(cv),
                    "significant_language_ratio": float(significant_ratio),
                    "gradient": features.get("gradient_by_language", {}).get(max_lang, 0.0),
                    "activation": features.get("activation_by_language", {}).get(max_lang, 0.0)
                })
            else:
                irrelevant_candidates.append({
                    "neuron_id": neuron_id,
                    "max_importance": float(max_score),
                    "cv": float(cv),
                    "significant_language_ratio": float(significant_ratio)
                })
        else:
            irrelevant_candidates.append({
                "neuron_id": neuron_id,
                "max_importance": float(max_score)
            })

    # Apply limits to results
    total_neurons = len(neuron_features)
    max_neurons_per_language = max(50, int(total_neurons * max_percentage))
    
    for lang in languages:
        candidates = sorted(language_specific_candidates[lang], key=lambda x: x["contrast_score"], reverse=True)
        results["language_specific"][lang] = candidates[:max_neurons_per_language]
    
    max_general_neurons = max(50, int(total_neurons * 0.025))
    general_candidates = sorted(general_candidates, key=lambda x: x["importance"], reverse=True)
    results["general"] = general_candidates[:max_general_neurons]
    
    results["irrelevant"] = irrelevant_candidates

    return results

def ensemble_classification(contrastive_results, statistical_results, num_total_neurons_from_features, neuron_features=None, feature_stats=None, **parallel_params):
    """Combine contrastive and statistical classification results"""
    languages = list(contrastive_results["language_specific"].keys())

    method_weights = {
        "contrastive": 1.5,
        "statistical": 1.5
    }

    # Set normalization scales
    if feature_stats:
        importance_scale = 1.0 / (feature_stats['importance']['percentiles']['95'] + 1e-10)
        gradient_scale = 1.0 / (feature_stats['gradient']['percentiles']['95'] + 1e-10) if 'gradient' in feature_stats else 1e5
        activation_scale = 1.0 / (feature_stats['activation']['percentiles']['95'] + 1e-10) if 'activation' in feature_stats else 10
    else:
        importance_scale = 1e4
        gradient_scale = 1e5
        activation_scale = 10

    neuron_votes = {}
    
    # Collect all neuron keys using set comprehensions
    all_neuron_keys = set()
    
    # Add neurons from contrastive results
    all_neuron_keys.update(
        neuron["neuron_id"] 
        for lang_neurons in contrastive_results["language_specific"].values() 
        for neuron in lang_neurons
    )
    all_neuron_keys.update(neuron["neuron_id"] for neuron in contrastive_results["general"])
    all_neuron_keys.update(neuron["neuron_id"] for neuron in contrastive_results["irrelevant"])
    
    # Add neurons from statistical results
    all_neuron_keys.update(
        neuron["neuron_id"] 
        for lang_neurons in statistical_results["language_specific"].values() 
        for neuron in lang_neurons
    )
    all_neuron_keys.update(neuron["neuron_id"] for neuron in statistical_results["general"])
    all_neuron_keys.update(neuron["neuron_id"] for neuron in statistical_results["irrelevant"])
    
    # Initialize votes for all neurons
    for neuron_id in all_neuron_keys:
        neuron_votes[neuron_id] = {
            "language_specific": {
                lang: {
                    "votes": 0, "importance": 0, "contrast_score": 0,
                    "gradient_sum": 0, "activation_sum": 0, "count": 0
                } for lang in languages
            },
            "general": {"votes": 0, "importance": 0, "gradient_sum": 0, "activation_sum": 0, "count": 0},
            "irrelevant": {"votes": 0}
        }

    # Process contrastive results
    contrastive_weight = method_weights["contrastive"]
    for lang, neurons in contrastive_results["language_specific"].items():
        for neuron in neurons:
            neuron_id = neuron["neuron_id"]
            vote_info = neuron_votes[neuron_id]["language_specific"][lang]
            vote_info["votes"] += contrastive_weight
            vote_info["importance"] = max(vote_info["importance"], neuron.get("importance", 0) * contrastive_weight)
            vote_info["contrast_score"] = max(vote_info["contrast_score"], neuron.get("contrast_score", 0) * contrastive_weight)
            vote_info["gradient_sum"] += neuron.get("gradient", 0) * contrastive_weight
            vote_info["activation_sum"] += neuron.get("activation", 0) * contrastive_weight
            vote_info["count"] += contrastive_weight

    for neuron in contrastive_results["general"]:
        neuron_id = neuron["neuron_id"]
        vote_info = neuron_votes[neuron_id]["general"]
        vote_info["votes"] += contrastive_weight
        vote_info["importance"] = max(vote_info["importance"], neuron.get("importance", 0) * contrastive_weight)
        vote_info["gradient_sum"] += neuron.get("gradient", 0) * contrastive_weight
        vote_info["activation_sum"] += neuron.get("activation", 0) * contrastive_weight
        vote_info["count"] += contrastive_weight

    for neuron in contrastive_results["irrelevant"]:
        neuron_id = neuron["neuron_id"]
        neuron_votes[neuron_id]["irrelevant"]["votes"] += contrastive_weight

    # Process statistical results
    statistical_weight = method_weights["statistical"]
    for lang, neurons in statistical_results["language_specific"].items():
        for neuron in neurons:
            neuron_id = neuron["neuron_id"]
            vote_info = neuron_votes[neuron_id]["language_specific"][lang]
            vote_info["votes"] += statistical_weight
            vote_info["importance"] = max(vote_info["importance"], neuron.get("importance", 0) * statistical_weight)
            vote_info["contrast_score"] = max(vote_info["contrast_score"], neuron.get("contrast_score", 0) * statistical_weight)
            vote_info["gradient_sum"] += neuron.get("gradient", 0) * statistical_weight
            vote_info["activation_sum"] += neuron.get("activation", 0) * statistical_weight
            vote_info["count"] = vote_info.get("count", 0) + statistical_weight

    for neuron in statistical_results["general"]:
        neuron_id = neuron["neuron_id"]
        vote_info = neuron_votes[neuron_id]["general"]
        vote_info["votes"] += statistical_weight
        vote_info["importance"] = max(vote_info["importance"], neuron.get("importance", 0) * statistical_weight)
        vote_info["gradient_sum"] += neuron.get("gradient", 0) * statistical_weight
        vote_info["activation_sum"] += neuron.get("activation", 0) * statistical_weight
        vote_info["count"] = vote_info.get("count", 0) + statistical_weight

    for neuron in statistical_results["irrelevant"]:
        neuron_id = neuron["neuron_id"]
        neuron_votes[neuron_id]["irrelevant"]["votes"] += statistical_weight

    results = {
        "language_specific": {lang: [] for lang in languages},
        "general": [],
        "irrelevant": []
    }
    
    language_specific_candidates = []
    general_candidates = []
    irrelevant_candidates = []

    batch_data = list(neuron_votes.items())
    
    # Process neurons in parallel if needed
    use_parallel = parallel_params.get("use_parallel", True)
    num_cores_param = parallel_params.get("num_cores", 0)
    batch_size_factor = parallel_params.get("batch_size_factor", 10)
    use_threads = parallel_params.get("use_threads", False)
    
    if use_parallel and len(batch_data) > 1000:
        num_cores = num_cores_param if num_cores_param > 0 else max(1, mp.cpu_count() - 1)
        batch_size = max(1, len(batch_data) // (num_cores * batch_size_factor))
        
        batches = []
        for i in range(0, len(batch_data), batch_size):
            batch_votes = batch_data[i:i+batch_size]
            if neuron_features:
                batch_neuron_ids = [neuron_id for neuron_id, _ in batch_votes]
                batch_features = {neuron_id: neuron_features[neuron_id] for neuron_id in batch_neuron_ids if neuron_id in neuron_features}
            else:
                batch_features = None
            
            batches.append((batch_votes, batch_features, languages))
        
        pool_cls = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
        
        with pool_cls(max_workers=num_cores) as executor:
            all_results = []
            for batch_results in tqdm.tqdm(
                executor.map(process_neuron_batch, batches),
                total=len(batches),
                desc="Processing neuron batches"
            ):
                all_results.extend(batch_results)
        
        for neuron_id, category, result in all_results:
            if category == "language_specific":
                language_specific_candidates.append(result)
            elif category == "general":
                general_candidates.append(result)
            elif category == "irrelevant":
                irrelevant_candidates.append(result)
    else:
        single_batch = (batch_data, neuron_features, languages)
        all_results = process_neuron_batch(single_batch)
        
        for neuron_id, category, result in all_results:
            if category == "language_specific":
                language_specific_candidates.append(result)
            elif category == "general":
                general_candidates.append(result)
            elif category == "irrelevant":
                irrelevant_candidates.append(result)

    # Set limits for each category
    min_neurons_per_category = 50 
    max_neurons_per_language_percentage = 0.015
    max_general_neurons_percentage = 0.025
    
    if feature_stats and 'importance' in feature_stats:
        if len(feature_stats['importance']['values']) > 0:
            high_importance_values = [v for v in feature_stats['importance']['values'] 
                                    if v > feature_stats['importance']['percentiles']['95']]
            high_importance_ratio = len(high_importance_values) / len(feature_stats['importance']['values'])
            
            if high_importance_ratio < 0.01:
                max_neurons_per_language_percentage = 0.02
                max_general_neurons_percentage = 0.02

    max_neurons_per_language = max(min_neurons_per_category, int(num_total_neurons_from_features * max_neurons_per_language_percentage))
    max_general_neurons = max(min_neurons_per_category, int(num_total_neurons_from_features * max_general_neurons_percentage))

    # Compute confidence scores
    confidence_weights = {
        "vote": 0.3,
        "importance": 0.2,
        "contrast": 0.1,
        "gradient": 0.2,
        "activation": 0.2
    }
    
    # Process language-specific candidates
    for candidate in language_specific_candidates:
        vote_confidence = candidate.get("confidence", 0) 
        importance = candidate.get("importance", 0)
        contrast = candidate.get("contrast_score", 0)
        gradient = candidate.get("gradient", 0)
        activation = candidate.get("activation", 0)
        
        normalized_importance = importance * importance_scale
        normalized_gradient = abs(gradient) * gradient_scale
        normalized_activation = abs(activation) * activation_scale
        
        candidate["normalized_importance"] = normalized_importance
        candidate["normalized_gradient"] = normalized_gradient
        candidate["normalized_activation"] = normalized_activation
        
        combined_confidence = (
            vote_confidence * confidence_weights["vote"] +
            normalized_importance * confidence_weights["importance"] +
            contrast * confidence_weights["contrast"] +
            normalized_gradient * confidence_weights["gradient"] +
            normalized_activation * confidence_weights["activation"]
        )
        candidate["confidence"] = combined_confidence
    
    # Process general candidates
    for candidate in general_candidates:
        vote_confidence = candidate.get("confidence", 0)
        importance = candidate.get("importance", 0)
        gradient = candidate.get("gradient", 0)
        activation = candidate.get("activation", 0)
        
        normalized_importance = importance * importance_scale
        normalized_gradient = abs(gradient) * gradient_scale
        normalized_activation = abs(activation) * activation_scale
        
        candidate["normalized_importance"] = normalized_importance
        candidate["normalized_gradient"] = normalized_gradient
        candidate["normalized_activation"] = normalized_activation
        
        combined_confidence = (
            vote_confidence * confidence_weights["vote"] +
            normalized_importance * confidence_weights["importance"] +
            normalized_gradient * confidence_weights["gradient"] +
            normalized_activation * confidence_weights["activation"]
        )
        candidate["confidence"] = combined_confidence

    # Sort and apply limits
    language_specific_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    candidates_by_language = {}
    for candidate in language_specific_candidates:
        lang = candidate["language"]
        if lang not in candidates_by_language:
            candidates_by_language[lang] = []
        candidates_by_language[lang].append(candidate)

    language_quotas = {}
    all_langs = list(candidates_by_language.keys())
    if all_langs:
        for lang in all_langs:
            candidates_count = len(candidates_by_language.get(lang, []))
            language_quotas[lang] = min(max_neurons_per_language, candidates_count)

    selected_candidates = []
    for lang, quota in language_quotas.items():
        if lang in candidates_by_language:
            selected_candidates.extend(candidates_by_language[lang][:quota])

    language_specific_candidates = selected_candidates

    # Build final results
    for candidate in language_specific_candidates:
        results["language_specific"][candidate["language"]].append({
            "neuron_id": candidate["neuron_id"],
            "votes": candidate["votes"],
            "confidence": candidate["confidence"],
            "importance": candidate["importance"],
            "contrast_score": candidate.get("contrast_score", 0),
            "normalized_importance": candidate.get("normalized_importance", 0),
            "normalized_gradient": candidate.get("normalized_gradient", 0),
            "normalized_activation": candidate.get("normalized_activation", 0)
        })

    general_candidates.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    general_candidates = general_candidates[:max_general_neurons]

    for candidate in general_candidates:
        results["general"].append({
            "neuron_id": candidate["neuron_id"],
            "votes": candidate["votes"],
            "confidence": candidate["confidence"],
            "importance": candidate["importance"],
            "normalized_importance": candidate.get("normalized_importance", 0),
            "normalized_gradient": candidate.get("normalized_gradient", 0),
            "normalized_activation": candidate.get("normalized_activation", 0),
            "max_min_ratio": float(candidate.get("max_min_ratio", 1.0)),
            "min_avg_ratio": float(candidate.get("min_avg_ratio", 0.0)),
            "significant_language_ratio": float(candidate.get("significant_language_ratio", 0.0)),
            "importance_cv": float(candidate.get("cv", 1.0))
        })

    for candidate in irrelevant_candidates:
        results["irrelevant"].append(candidate)

    return results

def save_results(results, output_path, gradient_data=None, feature_stats=None):
    """Save classification results to file"""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results_with_meta = {
        "classification_results": results,
        "meta": {
            "timestamp": str(datetime.datetime.now()),
            "total_neurons": {
                "language_specific": sum(len(neurons) for neurons in results["language_specific"].values()),
                "general": len(results["general"]),
                "irrelevant": len(results["irrelevant"]),
            }
        }
    }
    
    # Add feature statistics summary
    if feature_stats:
        feature_stats_summary = {}
        for feature_name, stats in feature_stats.items():
            if feature_name in ['importance', 'contrast_score', 'gradient', 'activation', 'importance_ratio']:
                feature_stats_summary[feature_name] = {
                    k: v for k, v in stats.items() if k != 'values'
                }
        results_with_meta["meta"]["feature_stats"] = feature_stats_summary
    
    # Add irrelevant analysis
    if results["irrelevant"]:
        irrelevant_analysis = {
            "count": len(results["irrelevant"]),
            "vote_distribution": {},
        }
        
        votes_data = [n.get("irrelevant_votes", 0) for n in results["irrelevant"] if "irrelevant_votes" in n]
        if votes_data:
            irrelevant_analysis["vote_distribution"] = {
                "min": min(votes_data),
                "max": max(votes_data),
                "mean": sum(votes_data) / len(votes_data),
                "median": sorted(votes_data)[len(votes_data) // 2]
            }
        
        results_with_meta["meta"]["irrelevant_analysis"] = irrelevant_analysis

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, indent=2, ensure_ascii=False)

    # Save classified neurons by language
    classified_dir = Path(output_dir) / "classification"
    classified_dir.mkdir(parents=True, exist_ok=True)

    for lang, neurons in results["language_specific"].items():
        simplified_neurons = []

        for neuron in neurons:
            neuron_id = neuron["neuron_id"]
            importance = neuron.get("importance", 0.0)
            contrast_score = neuron.get("contrast_score", 0.0)
            confidence = neuron.get("confidence", 0.0)
            
            simplified_neuron = {
                "neuron_key": neuron_id,
                "overall_importance": float(importance),
                "preferred_language": lang,
                "weight_for_preferred_lang": float(importance),
                "specificity_cv": float(contrast_score),
                "selectivity_score": float(contrast_score) if contrast_score > 0 else 0.5,
                "classification": "Specific",
                "confidence": float(confidence),
                "normalized_importance": float(neuron.get("normalized_importance", 0.0)),
                "normalized_gradient": float(neuron.get("normalized_gradient", 0.0)),
                "normalized_activation": float(neuron.get("normalized_activation", 0.0))
            }

            # Add gradient data if available
            if gradient_data:
                for layer_name, layer_data in gradient_data.items():
                    if "neurons" in layer_data and neuron_id in layer_data["neurons"]:
                        neuron_data = layer_data["neurons"][neuron_id]
                        if "importance_by_language" in neuron_data:
                            simplified_neuron["importance_by_language"] = neuron_data["importance_by_language"]
                        if "gradient_by_language" in neuron_data:
                            simplified_neuron["gradient_by_language"] = neuron_data["gradient_by_language"]
                        if "activation_by_language" in neuron_data:
                            simplified_neuron["activation_by_language"] = neuron_data["activation_by_language"]
                        break

            simplified_neurons.append(simplified_neuron)

        lang_file = classified_dir / f"{lang}_specific_neurons.json"
        with open(lang_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_neurons, f, indent=2, ensure_ascii=False)

    # Save general neurons
    if results["general"]:
        simplified_general = []

        for neuron in results["general"]:
            neuron_id = neuron["neuron_id"]
            importance = neuron.get("importance", 0.0)
            confidence = neuron.get("confidence", 0.0)
            
            simplified_neuron = {
                "neuron_key": neuron_id,
                "overall_importance": float(importance),
                "classification": "General",
                "confidence": float(confidence),
                "normalized_importance": float(neuron.get("normalized_importance", 0.0)),
                "normalized_gradient": float(neuron.get("normalized_gradient", 0.0)),
                "normalized_activation": float(neuron.get("normalized_activation", 0.0)),
                "max_min_ratio": float(neuron.get("max_min_ratio", 1.0)),
                "min_avg_ratio": float(neuron.get("min_avg_ratio", 0.0)),
                "significant_language_ratio": float(neuron.get("significant_language_ratio", 0.0)),
                "importance_cv": float(neuron.get("cv", 1.0))
            }

            # Add gradient data if available
            if gradient_data:
                for layer_name, layer_data in gradient_data.items():
                    if "neurons" in layer_data and neuron_id in layer_data["neurons"]:
                        neuron_data = layer_data["neurons"][neuron_id]
                        if "importance_by_language" in neuron_data:
                            simplified_neuron["importance_by_language"] = neuron_data["importance_by_language"]
                        if "gradient_by_language" in neuron_data:
                            simplified_neuron["gradient_by_language"] = neuron_data["gradient_by_language"]
                        if "activation_by_language" in neuron_data:
                            simplified_neuron["activation_by_language"] = neuron_data["activation_by_language"]
                        break

            simplified_general.append(simplified_neuron)

        common_file = classified_dir / "common_neurons.json"
        with open(common_file, 'w', encoding='utf-8') as f:
            json.dump(simplified_general, f, indent=2, ensure_ascii=False)

def process_neuron_batch(batch_data):
    """Process a batch of neurons for classification"""
    results = []
    batch_votes, batch_features, languages = batch_data
    
    for neuron_id, votes in batch_votes:
        lang_specific_votes = {lang: info["votes"] for lang, info in votes["language_specific"].items()}
        lang_specific_total = sum(lang_specific_votes.values())

        general_votes = votes["general"]["votes"]
        irrelevant_votes = votes["irrelevant"]["votes"]

        lang_specific_importance = {lang: info["importance"] for lang, info in votes["language_specific"].items()}
        general_importance = votes["general"]["importance"]

        lang_specific_contrast = {lang: info["contrast_score"] for lang, info in votes["language_specific"].items()}

        # Calculate average gradients and activations
        avg_lang_specific_gradient = {}
        avg_lang_specific_activation = {}
        for lang, info in votes["language_specific"].items():
            count = info.get("count", 0)
            avg_lang_specific_gradient[lang] = info["gradient_sum"] / count if count > 0 else 0
            avg_lang_specific_activation[lang] = info["activation_sum"] / count if count > 0 else 0
        
        general_count = votes["general"].get("count", 0)
        avg_general_gradient = votes["general"]["gradient_sum"] / general_count if general_count > 0 else 0
        avg_general_activation = votes["general"]["activation_sum"] / general_count if general_count > 0 else 0

        specific_vote_threshold = 0.8
        
        # Classify neuron based on votes
        if lang_specific_total > general_votes * specific_vote_threshold and lang_specific_total > irrelevant_votes:
            max_lang = None
            max_votes = 0
            for lang, votes_count in lang_specific_votes.items():
                if votes_count > max_votes:
                    max_votes = votes_count
                    max_lang = lang
                    
            max_importance = lang_specific_importance.get(max_lang, 0.0)
            max_contrast = lang_specific_contrast.get(max_lang, 0.0)
            max_gradient = avg_lang_specific_gradient.get(max_lang, 0.0)
            max_activation = avg_lang_specific_activation.get(max_lang, 0.0)

            total_votes = lang_specific_total + general_votes + irrelevant_votes
            vote_confidence = max_votes / total_votes if total_votes > 0 else 0

            results.append((neuron_id, "language_specific", {
                "neuron_id": neuron_id,
                "language": max_lang,
                "votes": max_votes,
                "importance": max_importance,
                "contrast_score": max_contrast,
                "gradient": max_gradient,
                "activation": max_activation,
                "confidence": vote_confidence
            }))
            
        elif general_votes > irrelevant_votes * 1.2 and general_votes > lang_specific_total * 1.1:
            cv = 1.0
            significant_ratio = 0.0
            min_avg_ratio = 0.0
            max_min_ratio = 1000.0
            
            if batch_features and neuron_id in batch_features:
                features = batch_features[neuron_id]
                cv = features.get("importance_cv", 1.0)
                significant_ratio = features.get("significant_language_ratio", 0.0)
                min_avg_ratio = features.get("min_avg_importance_ratio", 0.0)
                max_min_ratio = features.get("importance_max_min_ratio", 1000.0)
            
            if (cv < 0.7 or 
                significant_ratio > 0.5 or 
                min_avg_ratio > 0.4 or 
                max_min_ratio < 2.5):
                
                total_votes = lang_specific_total + general_votes + irrelevant_votes
                vote_confidence = general_votes / total_votes if total_votes > 0 else 0

                results.append((neuron_id, "general", {
                    "neuron_id": neuron_id,
                    "votes": general_votes,
                    "importance": general_importance,
                    "cv": cv,
                    "significant_language_ratio": significant_ratio,
                    "min_avg_ratio": min_avg_ratio,
                    "max_min_ratio": max_min_ratio,
                    "gradient": avg_general_gradient,
                    "activation": avg_general_activation,
                    "confidence": vote_confidence
                }))
            else:
                total_votes = lang_specific_total + general_votes + irrelevant_votes
                confidence = irrelevant_votes / total_votes if total_votes > 0 else 0

                results.append((neuron_id, "irrelevant", {
                    "neuron_id": neuron_id,
                    "votes": irrelevant_votes,
                    "confidence": confidence,
                    "cv": cv,
                    "significant_language_ratio": significant_ratio
                }))
        else:
            total_votes = lang_specific_total + general_votes + irrelevant_votes
            confidence = irrelevant_votes / total_votes if total_votes > 0 else 0
            
            max_lang_votes = 0
            max_lang = None
            for lang, info in votes["language_specific"].items():
                if info["votes"] > max_lang_votes:
                    max_lang_votes = info["votes"]
                    max_lang = lang

            results.append((neuron_id, "irrelevant", {
                "neuron_id": neuron_id,
                "votes": irrelevant_votes,
                "confidence": confidence,
                "irrelevant_votes": irrelevant_votes,
                "general_votes": general_votes,
                "language_specific_votes": lang_specific_total,
                "max_language": max_lang,
                "max_language_votes": max_lang_votes if max_lang else 0
            }))
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Neuron Classification Tool")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Input gradient data file path")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output classification results file path")
    parser.add_argument("--method", type=str, default="ensemble",
                        choices=["contrastive", "statistical", "ensemble"],
                        help="Classification method")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--min_importance", type=float, default=0.0000035,
                        help="Minimum importance threshold for neurons")
    parser.add_argument("--use_data_driven_thresholds", action="store_true", default=True,
                        help="Use data-driven thresholds")
    parser.add_argument("--save_feature_stats", action="store_true", default=False,
                        help="Save feature statistics to file")
    parser.add_argument("--num_cores", type=int, default=0,
                        help="Number of CPU cores for parallel processing (0=auto)")
    parser.add_argument("--disable_parallel", action="store_true", default=False,
                        help="Disable parallel processing")
    parser.add_argument("--batch_size_factor", type=int, default=10,
                        help="Batch size factor for parallel processing")
    parser.add_argument("--use_threads", action="store_true", default=False,
                        help="Use threads instead of processes")
    parser.add_argument("--max_language_percentage", type=float, default=0.01,
                        help="Maximum percentage of neurons per language")
    
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and normalize gradient data
    gradient_data = load_gradient_data(args.input_file)
    gradient_data = normalize_gradient_data_keys(gradient_data)

    # Extract features
    neuron_features, languages, feature_stats = extract_neuron_features(gradient_data, min_importance=args.min_importance)
    
    # Save feature statistics if requested
    if args.save_feature_stats and feature_stats:
        stats_file = output_dir / "feature_statistics.json"
        stats_to_save = {}
        for feature_name, stats in feature_stats.items():
            stats_to_save[feature_name] = {k: v for k, v in stats.items() if k != 'values'}
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False)
        print(f"Feature statistics saved to {stats_file}")

    # Classify neurons
    feature_stats_arg = feature_stats if args.use_data_driven_thresholds else None
    if args.method == "contrastive":
        results = contrastive_classification(neuron_features, languages, feature_stats_arg, max_percentage=args.max_language_percentage)
    elif args.method == "statistical":
        results = statistical_classification(neuron_features, languages, feature_stats_arg, max_percentage=args.max_language_percentage)
    elif args.method == "ensemble":
        contrastive_results = contrastive_classification(neuron_features, languages, feature_stats_arg, max_percentage=args.max_language_percentage)
        statistical_results = statistical_classification(neuron_features, languages, feature_stats_arg, max_percentage=args.max_language_percentage)
        
        parallel_params = {
            "use_parallel": not args.disable_parallel,
            "num_cores": args.num_cores,
            "batch_size_factor": args.batch_size_factor,
            "use_threads": args.use_threads
        }
        
        results = ensemble_classification(
            contrastive_results, 
            statistical_results, 
            len(neuron_features), 
            neuron_features=neuron_features,
            feature_stats=feature_stats,
            **parallel_params
        )

    # Save results
    save_results(results, args.output_file, gradient_data, feature_stats_arg)

    # Print summary
    total_specific = sum(len(neurons) for neurons in results["language_specific"].values())
    print(f"Classification summary:")
    print(f"  - Total neurons analyzed: {len(neuron_features)}")
    print(f"  - Language-specific neurons: {total_specific} ({total_specific/len(neuron_features)*100:.2f}%)")
    for lang, neurons in results["language_specific"].items():
        print(f"    - {lang}: {len(neurons)}")
    print(f"  - General neurons: {len(results['general'])} ({len(results['general'])/len(neuron_features)*100:.2f}%)")
    print(f"  - Irrelevant neurons: {len(results['irrelevant'])} ({len(results['irrelevant'])/len(neuron_features)*100:.2f}%)")

    print(f"Classification complete. Results saved to {args.output_file}")
    print(f"Language-specific neurons saved to {args.output_dir}/classification/ directory")

if __name__ == "__main__":
    main()