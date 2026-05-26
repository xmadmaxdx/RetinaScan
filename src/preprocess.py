import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import yaml
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from skimage import exposure


def ben_graham_preprocess(image, radius=30, alpha=2.0, beta=0.5):
    img = image.astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    filtered = img - blurred
    scaled = filtered * alpha + blurred * beta
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def crop_to_circle(image):
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    radius = min(h, w) // 2 - 10
    cv2.circle(mask, center, radius, 255, -1)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


def preprocess_image(img_path, output_path, target_size=(512, 512)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
    img = crop_to_circle(img)
    img = ben_graham_preprocess(img)
    img = apply_clahe(img)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main(config):
    raw_dir = config["data"]["raw_path"]
    processed_dir = config["data"]["processed_path"]
    size = config["data"]["image_size"]

    image_paths = glob(os.path.join(raw_dir, "*.jpeg")) + glob(os.path.join(raw_dir, "*.png")) + glob(os.path.join(raw_dir, "*.jpg"))
    image_paths = [p for p in image_paths if "train" in p.lower() or "test" in p.lower()] or image_paths

    for img_path in tqdm(image_paths, desc="Preprocessing"):
        rel = os.path.relpath(img_path, raw_dir)
        out_path = os.path.join(processed_dir, rel)
        preprocess_image(img_path, out_path, target_size=(size, size))

    print(f"Processed {len(image_paths)} images -> {processed_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
