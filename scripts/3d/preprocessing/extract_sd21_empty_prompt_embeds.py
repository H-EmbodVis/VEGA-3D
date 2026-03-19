#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import os.path as osp

import torch
from transformers import CLIPTextModel, CLIPTokenizer


def _parse_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower().strip()
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def main():
    parser = argparse.ArgumentParser(description="Extract SD2.1 empty prompt embeddings and save as .pt")
    parser.add_argument("--checkpoint_dir", type=str, default="data/models/stable-diffusion-2-1-base")
    parser.add_argument("--output_path", type=str, default=None, help="Defaults to <checkpoint_dir>/empty_prompt_embeds.pt")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="bfloat16|float32|float16")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        output_path = osp.join(args.checkpoint_dir, "empty_prompt_embeds.pt")
    os.makedirs(osp.dirname(output_path), exist_ok=True)

    dtype = _parse_dtype(args.dtype)
    device = torch.device(args.device)

    tokenizer = CLIPTokenizer.from_pretrained(
        args.checkpoint_dir,
        subfolder="tokenizer",
        local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.checkpoint_dir,
        subfolder="text_encoder",
        local_files_only=args.local_files_only,
    ).eval()
    text_encoder.to(device=device, dtype=torch.float32)

    tokenized = tokenizer(
        [args.prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = tokenized.input_ids.to(device)
    attention_mask = tokenized.attention_mask.to(device) if "attention_mask" in tokenized else None

    with torch.inference_mode():
        out = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        prompt_embeds = out.last_hidden_state

    prompt_embeds = prompt_embeds.to(dtype=dtype).cpu().contiguous()
    payload = {
        "prompt_embeds": prompt_embeds,
        "prompt": args.prompt,
        "checkpoint_dir": args.checkpoint_dir,
        "model_max_length": int(tokenizer.model_max_length),
        "dtype": str(prompt_embeds.dtype),
        "shape": list(prompt_embeds.shape),
    }
    torch.save(payload, output_path)
    print(f"Saved empty prompt embeds to: {output_path}")
    print(f"shape={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}")


if __name__ == "__main__":
    main()
