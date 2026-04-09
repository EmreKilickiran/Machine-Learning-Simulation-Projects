# =============================================================================
# 01_preprocess.py — Data Preprocessing & Mask Extraction
# =============================================================================
#
# Extracts cell and nuclear annotation masks from the TissueNet v1.1 NPZ
# archives and saves them as individual PNG images for the tf.data pipeline.
#
# TissueNet: A large-scale dataset of tissue microscopy images with
# pixel-level annotations for cell and nuclear segmentation.
# Reference: Greenwald et al., Nature Biotechnology, 2022.
#
# Input:  tissuenet_v1.1_{train,val,test}.npz
# Output: {train,val,test}_{cell,nuclear}/ directories with PNG masks
#
# Author : Yunus Emre Kılıçkıran
# Course : AI in Health Sciences (EE4069), Spring 2024
# =============================================================================

import os
import numpy as np
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR   = "data"
OUTPUT_DIR = "data/processed"

SPLITS = {
    "train": os.path.join(DATA_DIR, "tissuenet_v1.1_train.npz"),
    "val":   os.path.join(DATA_DIR, "tissuenet_v1.1_val.npz"),
    "test":  os.path.join(DATA_DIR, "tissuenet_v1.1_test.npz"),
}

ANNOTATION_CHANNELS = {
    "cell":    0,   # Channel 0: cell-level annotations
    "nuclear": 1,   # Channel 1: nuclear-level annotations
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def convert_to_rgb(image):
    """Ensure image has 3 channels (RGB) for visualization."""
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 2:
        extra = np.zeros_like(image[..., 0:1])
        return np.concatenate([image, extra], axis=-1)
    return image


def create_overlay(image, annotations, color):
    """Overlay colored annotation mask on the original image."""
    image = convert_to_rgb(image)
    overlay = image.astype(float) / (image.max() + 1e-8)
    overlay[annotations > 0, :] = color
    return (overlay * 255).astype(np.uint8)


# =============================================================================
# MASK EXTRACTION
# =============================================================================

def extract_masks(npz_path, split_name):
    """Extract and save annotation masks from NPZ archive."""
    print(f"\n  Processing {split_name}...")
    data = np.load(npz_path)
    X, y = data["X"], data["y"]
    print(f"    Images: {X.shape[0]} | Size: {X.shape[1]}×{X.shape[2]}")

    for ann_name, channel in ANNOTATION_CHANNELS.items():
        out_dir = os.path.join(OUTPUT_DIR, f"{split_name}_{ann_name}")
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(y.shape[0]):
            mask = y[idx, :, :, channel]
            # Normalize to 0-255 for PNG
            if mask.max() > 0:
                mask_norm = (mask / mask.max() * 255).astype(np.uint8)
            else:
                mask_norm = mask.astype(np.uint8)

            Image.fromarray(mask_norm).save(
                os.path.join(out_dir, f"{ann_name}_{idx}.png")
            )

        print(f"    Saved {y.shape[0]} {ann_name} masks → {out_dir}")


# =============================================================================
# VISUALIZATION (sample overlay)
# =============================================================================

def visualize_sample(npz_path, save_path="results/sample_overlay.png"):
    """Generate a sample overlay visualization."""
    import matplotlib.pyplot as plt

    data = np.load(npz_path)
    X, y = data["X"], data["y"]

    idx = np.random.randint(0, X.shape[0])
    image = X[idx]
    cell_mask = y[idx, :, :, 0]
    nuclear_mask = y[idx, :, :, 1]

    cell_overlay = create_overlay(image, cell_mask, color=[1, 0, 0])
    nuclear_overlay = create_overlay(image, nuclear_mask, color=[0, 1, 0])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(convert_to_rgb(image))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(cell_overlay)
    axes[1].set_title("Cell Annotations (Red)")
    axes[1].axis("off")

    axes[2].imshow(nuclear_overlay)
    axes[2].set_title("Nuclear Annotations (Green)")
    axes[2].axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  Sample overlay saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(" TissueNet Data Preprocessing")
    print("=" * 60)

    for split_name, npz_path in SPLITS.items():
        if os.path.exists(npz_path):
            extract_masks(npz_path, split_name)
        else:
            print(f"\n  [SKIP] {npz_path} not found")

    # Generate sample visualization
    test_path = SPLITS["test"]
    if os.path.exists(test_path):
        visualize_sample(test_path)

    print("\n  Preprocessing complete.")
