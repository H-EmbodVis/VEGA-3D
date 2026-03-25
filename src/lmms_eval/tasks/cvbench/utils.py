import re
import os
import pandas as pd
from pathlib import Path
import yaml
import string
from PIL import Image
from loguru import logger as eval_logger

with open(Path(__file__).parent / "cvbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

def cvbench_doc_to_visual(doc):
    # img_path = os.path.join(cache_dir, doc["filename"])
    # return [Image.open(img_path).convert("RGB")]
    image = doc["image"]
    return [image.convert("RGB")]

def cvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
    chars = string.ascii_uppercase
    options = "Options:\n" + "\n".join([f"{chars[i]}. {c}" for i, c in enumerate(doc["choices"])])
    return "\n".join([pre_prompt, question, options, post_prompt])


def extract_characters_regex(s):
    # the choices include ABCDEF
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is" "The correct option is",
        "Best answer:" "Best option:",
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, "")

    if len(s.split()) > 10 and not re.search(r"[ABCDEF]", s):
        return ""

    matches = re.search(r"[ABCDEF]", s)
    if matches is None:
        return ""
    return matches[0]

def cvbench_process_results(doc, results):
    doc["pred_answer"] = extract_characters_regex(results[0])
    doc["result"] = 1 if doc["pred_answer"] == doc["answer"][1] else 0
    return {"cvbench_score": doc}


def cvbench_aggregate_results(results):
    df = pd.DataFrame(results)
    
        # Define a function to calculate accuracy for a given source
    def calculate_accuracy(df, source):
        source_df = df[df['source'] == source]
        accuracy = source_df['result'].mean()  # Assuming 'result' is 1 for correct and 0 for incorrect
        return accuracy

    # Calculate accuracy for each source
    accuracy_2d_ade = calculate_accuracy(df, 'ADE20K')
    accuracy_2d_coco = calculate_accuracy(df, 'COCO')
    accuracy_3d_omni = calculate_accuracy(df, 'Omni3D')

    # Calculate the accuracy for each type
    accuracy_2d = (accuracy_2d_ade + accuracy_2d_coco) / 2
    accuracy_3d = accuracy_3d_omni

    # Compute the combined accuracy as specified
    combined_accuracy = (accuracy_2d + accuracy_3d) / 2

    output = {
        "accuracy_2d": accuracy_2d,
        "accuracy_3d": accuracy_3d,
        "combined_accuracy": combined_accuracy,
    }

    for task in ["Count", "Relation", "Distance", "Depth"]:
        output[task] = df[df["task"] == task]["result"].mean()

    eval_logger.info(f"Evaluation results: {output}")
    return output["combined_accuracy"] * 100
