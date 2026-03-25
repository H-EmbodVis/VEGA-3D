
import os
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import lru_cache, partial
import numpy as np
import pandas as pd

import datasets

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}

with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

task_config = yaml.safe_load("".join(safe_data))
dataset_path = task_config["dataset_path"]
cache_name = task_config["dataset_kwargs"]["cache_dir"]


def _expand_name_variants(name):
    if not name:
        return []
    normalized = name.strip().strip("/\\")
    if not normalized:
        return []

    variants = []
    seen = set()

    def _add(value):
        if value and value not in seen:
            variants.append(value)
            seen.add(value)

    _add(normalized)
    _add(normalized.replace("-", "_"))
    _add(normalized.replace("_", "-"))
    _add(normalized.lower())
    _add(normalized.upper())
    _add(normalized.lower().replace("-", "_"))
    _add(normalized.lower().replace("_", "-"))
    _add(normalized.replace("-", ""))
    _add(normalized.replace("_", ""))
    _add(normalized.lower().replace("-", ""))
    _add(normalized.lower().replace("_", ""))
    return variants


@lru_cache(maxsize=1)
def _resolve_cache_dir():
    specific_path = os.getenv("LMMS_EVAL_VSIBENCH_DATASET_PATH")
    if specific_path:
        return os.path.expanduser(specific_path)

    global_path = os.getenv("LMMS_EVAL_DATASET_PATH")
    if global_path:
        return os.path.expanduser(global_path)

    candidate_names = _expand_name_variants(os.path.basename(dataset_path.rstrip("/\\")))
    for name in _expand_name_variants("vsibench"):
        if name not in candidate_names:
            candidate_names.append(name)

    search_roots = []
    data_root = os.getenv("LMMS_EVAL_DATA_ROOT")
    if data_root:
        search_roots.append(os.path.expanduser(data_root))
    search_roots.extend(["data/evaluation", "data/media"])

    for root in search_roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for candidate in candidate_names:
            candidate_path = os.path.join(root, candidate)
            if os.path.isdir(candidate_path):
                return candidate_path

    local_dataset_path = os.path.expanduser(dataset_path)
    if os.path.isdir(local_dataset_path):
        return local_dataset_path

    hf_home = os.path.expanduser(os.getenv("HF_HOME", "~/.cache/huggingface/"))
    return os.path.join(hf_home, cache_name)

def vsibench_doc_to_visual(doc):
    cache_dir = _resolve_cache_dir()
    video_path = doc["dataset"] + "/" + str(doc["scene_name"]) + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist. Please check local dataset path settings.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    return dataset

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):
    
    doc['prediction'] = results[0]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    output['object_rel_direction_accuracy'] = sum([
        output.pop('object_rel_direction_easy_accuracy'),
        output.pop('object_rel_direction_medium_accuracy'),
        output.pop('object_rel_direction_hard_accuracy'),
    ]) / 3.
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output['overall'] * 100.
