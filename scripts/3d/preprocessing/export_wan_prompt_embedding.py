#!/usr/bin/env python3
import argparse
import os

import torch

from llava.model.multimodal_generative_encoder.wan.configs import WAN_CONFIGS
from llava.model.multimodal_generative_encoder.wan.text2video import T5EncoderModel


def main():
    parser = argparse.ArgumentParser(description="Export WAN prompt embedding to a .pt file.")
    parser.add_argument("--task", type=str, default="vace-1.3B", help="WAN task name, e.g. vace-1.3B")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="WAN checkpoint directory")
    parser.add_argument("--prompt", type=str, default="", help="Prompt text")
    parser.add_argument("--output", type=str, default="llava/model/multimodal_generative_encoder/wan_prompt_embedding.pt")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    if args.task not in WAN_CONFIGS:
        raise ValueError(f"Unsupported task: {args.task}")
    cfg = WAN_CONFIGS[args.task]

    device = torch.device(args.device)
    text_encoder = T5EncoderModel(
        text_len=cfg.text_len,
        dtype=cfg.t5_dtype,
        device=device,
        checkpoint_path=os.path.join(args.checkpoint_dir, cfg.t5_checkpoint),
        tokenizer_path=os.path.join(args.checkpoint_dir, cfg.t5_tokenizer),
        shard_fn=None,
    )

    with torch.inference_mode():
        context = text_encoder([args.prompt], device)[0].detach().cpu().contiguous()

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(context, args.output)
    print(f"Saved prompt embedding: {args.output}, shape={tuple(context.shape)}, dtype={context.dtype}")


if __name__ == "__main__":
    main()
