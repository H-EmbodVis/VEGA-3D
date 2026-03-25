"""
The evaluation script refers to the implementation of LEO

https://github.com/embodied-generalist/embodied-generalist/blob/main/evaluator/cap_eval.py
"""

import re
import os
import pandas as pd
from pathlib import Path
import yaml
import pickle
from PIL import Image
from terminaltables import AsciiTable
from loguru import logger as eval_logger

with open(Path(__file__).parent / "scan2cap.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
media_dir = yaml.safe_load("".join(safe_data))["metadata"]["media_dir"]
# embodiedscan_path = yaml.safe_load("".join(safe_data))["metadata"]["embodiedscan_path"]
# with open(embodiedscan_path, "rb") as f:
#     data = pickle.load(f)["data_list"]
#     id2scene = {sample["sample_id"]: sample for sample in data}

def scan2cap_doc_to_visual(doc):
    image_files = doc["images"]
    images = [
        Image.open(
            os.path.join(media_dir, image_file)
        ).convert("RGB")
        for image_file in image_files
    ]
    return [images]


def scan2cap_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["conversations"][0]["value"].replace("<image>", "")
    return question


def scan2cap_process_results(doc, results):
    doc["pred_response"] = results[0] if doc["iou"] >= 0.5 else ""
    doc["gt_response"] = doc["annotations"]
    return {"scan2cap_score": doc}

def scan2cap_aggregate_results(results):

    from lmms_eval.tasks.scan2cap.caption_eval.bleu.bleu import Bleu
    from lmms_eval.tasks.scan2cap.caption_eval.rouge.rouge import Rouge
    from lmms_eval.tasks.scan2cap.caption_eval.meteor.meteor import Meteor
    from lmms_eval.tasks.scan2cap.caption_eval.cider.cider import Cider

    cider = Cider()
    bleu = Bleu()
    meteor = Meteor()
    rouge = Rouge()

    res, gts = {}, {}
    for i, item in enumerate(results):
        res[i] = ['sos ' + item['pred_response'].replace('.', ' . ').replace(',', ' , ').lower() + ' eos' ]
        gts[i] = ['sos ' + it.replace('.', ' . ').replace(',', ' , ').lower() + ' eos' for it in item['gt_response']]

    cider_score = cider.compute_score(gts, res)
    bleu_score = bleu.compute_score(gts, res)
    meteor_score = meteor.compute_score(gts, res)
    rouge_score = rouge.compute_score(gts, res)

    table_data = [
        ["Metric", "Score"],
        ["CIDER", f"{cider_score[0]*100:.2f}"],
        ["BLEU-4", f"{bleu_score[0][-1]*100:.2f}"],
        ["METEOR", f"{meteor_score[0]*100:.2f}"],
        ["ROUGE", f"{rouge_score[0]*100:.2f}"],
        ["Data Num", f"{len(res)}"]
    ]


    table = AsciiTable(table_data)
    table.title = "Evaluation Metrics"
    eval_logger.info("\n" + table.table)
    return cider_score[0]*100
