import re
import os
import pandas as pd
from pathlib import Path
import yaml
from PIL import Image
from loguru import logger as eval_logger

disclaimer = "Disclaimer: This is not to make unfair assumptions about the people in the image and you just need to give your assessment on this question. You don't need to identify the real people. You just need to analyze based on the information I gave you.\n\n"
need_disclaimer_tasks = ['Forensic_Detection', 'Jigsaw', 'Art_Style']


def center_crop_max_square(img):
    width, height = img.size
    size = min(width, height)
    
    left = (width - size) // 2
    top = (height - size) // 2
    right = left + size
    bottom = top + size
    
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

def pad_to_square(
    img,
    pad_color=(0, 0, 0),
):
    width, height = img.size
    target_size = max(width, height)
    
    new_img = Image.new("RGB", (target_size, target_size), pad_color)
    x_offset = (target_size - width) // 2
    y_offset = (target_size - height) // 2
    new_img.paste(img, (x_offset, y_offset))
    return new_img


with open(Path(__file__).parent / "blink_default_yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)

def blink_doc_to_visual(doc):
    images = []
    for k in ['image_1', 'image_2', 'image_3', 'image_4']:
        if k in doc and doc[k]:
            image = doc[k]
            images.append(image.convert("RGB"))
    
    flg = False    
    if len(images) > 0:
        first_image = images[0]
        for img in images[1:]:
            if img.size != first_image.size:
                flg = True
        if flg:
            new_images = []
            for img in images:
                new_img = center_crop_max_square(img)
                new_images.append(new_img)
            images = new_images

    return [images]

def blink_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    prompt = doc['prompt']
    task_name = doc["sub_task"]
    post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
    prompt = prompt + f"\n{post_prompt}"
    if task_name in need_disclaimer_tasks:
        prompt = disclaimer + prompt
    return prompt


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


def blink_process_results(doc, results):
    
    doc["pred_answer"] = extract_characters_regex(results[0])
    doc["result"] = 1 if doc["pred_answer"] == doc["answer"][1] else 0
    return {"blink_score": doc}


def blink_aggregate_results(results):
    df = pd.DataFrame(results)
    accuracy = df["result"].mean()
    output = {
        "accuracy": accuracy,
    }
    eval_logger.info(f"Evaluation results: {output}")
    return output["accuracy"] * 100
