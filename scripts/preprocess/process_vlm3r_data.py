import os
import glob
import json
import random
import numpy as np
import argparse
from tqdm import tqdm

BASE_MEDIA_DIR = "data/media/"


def sample_images_for_vlm3r(sample, frame_num):
    data_source = sample["data_source"]
    scene_name = sample["scene_name"]

    if data_source == "scannet":
        image_files = glob.glob(os.path.join(BASE_MEDIA_DIR, "scannet/posed_images/", scene_name, "*.jpg"))
    elif data_source == "scannetpp":
        image_files = glob.glob(os.path.join(BASE_MEDIA_DIR, "scannetpp/data/", scene_name, "dslr/resized_undistorted_images", "*.JPG"))
    elif data_source == "arkitscenes":
        image_files = glob.glob(os.path.join(BASE_MEDIA_DIR, "arkitscenes/3dod/Training/", scene_name, f"{scene_name}_frames", "lowres_wide", "*.png"))
    else:
        raise NotImplementedError(f"Data source {data_source} is not implemented.")

    image_files = [os.path.join(*image_file.split("/")[2:]) for image_file in image_files]
    image_files = sorted(image_files)

    frame_num = min(frame_num, len(image_files))
    idx = random.randint(0, len(image_files) - 1)
    image_files = image_files[idx:] + image_files[:idx]
    sampling_indices = np.linspace(0, len(image_files) - 1, num=frame_num, dtype=int)
    image_files = [image_files[i] for i in sampling_indices]
    image_files = sorted(image_files)


    return image_files


def process_sample(sample, frame_num=32):


    del sample["video"]
    image_files = sample_images_for_vlm3r(sample, frame_num)
    sample["images"] = image_files
    sample["conversations"][0]["value"] = sample["conversations"][0]["value"].replace("<image>", "")
    sample["conversations"][0]["value"] = "<image>"*len(image_files) + sample["conversations"][0]["value"]
    return sample

def main(args):

    data_dir = os.path.join(args.vlm_3r_data_path, "vsibench_train")
    files = []
    for filename in os.listdir(data_dir):
        files.append(os.path.join(data_dir, filename))
    
    data = []
    for filename in files:
        print(f"Loading {filename}...")
        with open(filename, 'r') as f:
            data.extend(json.load(f))

    print(f"Loaded {len(data)} samples from {len(files)} files.")

    for data_source in ["scannet", "scannetpp", "arkitscenes"]:
        L = [sample for sample in data if sample["data_source"] == data_source]
        scenes = list(set([sample["scene_name"] for sample in L]))
        print(f"Data source: {data_source}, Number of samples: {len(L)}, Number of scenes: {len(scenes)}")
    
    data_source = args.data_source.split(",")
    data = [sample for sample in data if sample["data_source"] in data_source]
    new_data = []
    for sample in tqdm(data):
        new_sample = process_sample(sample, frame_num=args.frame_num)
        new_data.append(new_sample)

    output_file = os.path.join(args.output_dir, f"vlm_3r_{args.data_source}_{args.frame_num}f_{round(len(new_data)/1000)}k.json")
    print(f"Saving processed data to {output_file}")

    with open(output_file, 'w') as f:
        json.dump(new_data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm_3r_data_path", type=str, default="/mnt/data0/zhengduo/data/VLM-3R-DATA/")
    parser.add_argument("--data_source", type=str, default="scannet-scannetpp-arkitscenes")
    parser.add_argument("--frame_num", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="data/train")
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)

    main(args)

