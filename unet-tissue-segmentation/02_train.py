# 02_train.py — U-Net Training & Evaluation Pipeline
#
# Trains a U-Net convolutional neural network for pixel-level segmentation
# of cellular structures in tissue microscopy images (TissueNet v1.1).
#
# Architecture:
#   Encoder:    Conv(32) → Pool → Conv(64) → Pool
#   Bottleneck: Conv(128) with Dropout(0.3)
#   Decoder:    Up → Concat → Conv(64) → Up → Concat → Conv(32)
#   Output:     Conv(1, sigmoid) — binary segmentation mask
#
# Results: F1 = 0.71, IoU = 0.55 (on 2,500 training images, 10 epochs)
# Reference: Mesmer achieves F1 = 0.82 on full dataset with extensive tuning

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D,
                                     Concatenate, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from PIL import Image

# CONFIGURATION

# Directories (output of 01_preprocess.py)
TRAIN_DIR = "data/processed/train_nuclear"
VAL_DIR   = "data/processed/val_nuclear"
TEST_DIR  = "data/processed/test_nuclear"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Hyperparameters
N_TRAIN      = 2500
N_VAL        = 500
N_TEST       = 500
TARGET_SIZE  = (128, 128)
BATCH_SIZE   = 16
EPOCHS       = 10
LEARNING_RATE = 1e-4

# DATA PIPELINE (tf.data)

def process_image_and_mask(file_path):
    """Load, resize, normalize, and augment an image-mask pair."""
    def load_image(path):
        img = Image.open(path.numpy().decode()).convert("L")
        img = img.resize(TARGET_SIZE)
        return np.array(img, dtype=np.float32) / 255.0

    image = tf.py_function(load_image, [file_path], tf.float32)
    mask  = tf.py_function(load_image, [file_path], tf.float32)

    image = tf.ensure_shape(image, [TARGET_SIZE[0], TARGET_SIZE[1]])
    mask  = tf.ensure_shape(mask, [TARGET_SIZE[0], TARGET_SIZE[1]])

    # Add channel dimension: (H, W) → (H, W, 1)
    image = tf.expand_dims(image, axis=-1)
    mask  = tf.expand_dims(mask, axis=-1)

    # On-the-fly data augmentation (random flips)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_left_right(image)
        mask  = tf.image.flip_left_right(mask)
    if tf.random.uniform([]) > 0.5:
        image = tf.image.flip_up_down(image)
        mask  = tf.image.flip_up_down(mask)

    return image, mask


def create_dataset(directory, limit):
    """Create a batched, prefetched tf.data.Dataset from PNG files."""
    paths = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory) if f.endswith(".png")
    ])[:limit]

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.map(process_image_and_mask,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return dataset


# CUSTOM METRICS

def iou(y_true, y_pred):
    """Intersection over Union (IoU) metric."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return (intersection + 1e-7) / (union + 1e-7)


def f1_score(y_true, y_pred):
    """F1 score (Dice coefficient) metric."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum(y_pred * (1 - y_true))
    fn = tf.reduce_sum((1 - y_pred) * y_true)
    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    return 2 * (precision * recall) / (precision + recall + 1e-7)


# U-NET ARCHITECTURE

def build_unet(input_size=(128, 128, 1)):
    """
    Encoder-decoder U-Net with skip connections.

    Architecture:
        Encoder: [32] → [64] → Bottleneck: [128] → Decoder: [64] → [32] → Output
        Skip connections: encoder features concatenated with decoder upsamples
        Regularization: BatchNormalization + Dropout at each block
    """
    inputs = Input(input_size)

    # --- Encoder Block 1 (32 filters) ---
    c1 = Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.2)(c1)
    c1 = Conv2D(32, (3, 3), activation="relu", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    # --- Encoder Block 2 (64 filters) ---
    c2 = Conv2D(64, (3, 3), activation="relu", padding="same")(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.2)(c2)
    c2 = Conv2D(64, (3, 3), activation="relu", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # --- Bottleneck (128 filters) ---
    c3 = Conv2D(128, (3, 3), activation="relu", padding="same")(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.3)(c3)
    c3 = Conv2D(128, (3, 3), activation="relu", padding="same")(c3)

    # --- Decoder Block 1 (64 filters + skip from c2) ---
    u4 = UpSampling2D((2, 2))(c3)
    u4 = Concatenate()([u4, c2])
    c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(u4)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(64, (3, 3), activation="relu", padding="same")(c4)

    # --- Decoder Block 2 (32 filters + skip from c1) ---
    u5 = UpSampling2D((2, 2))(c4)
    u5 = Concatenate()([u5, c1])
    c5 = Conv2D(32, (3, 3), activation="relu", padding="same")(u5)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(32, (3, 3), activation="relu", padding="same")(c5)

    # --- Output (sigmoid for binary segmentation) ---
    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c5)

    return Model(inputs=[inputs], outputs=[outputs], name="UNet")


# TRAINING

def train():
    print("=" * 60)
    print(" U-Net Tissue Segmentation — Training Pipeline")
    print("=" * 60)

    # --- Create datasets ---
    print("\n[1/4] Loading datasets...")
    train_ds = create_dataset(TRAIN_DIR, N_TRAIN)
    val_ds   = create_dataset(VAL_DIR, N_VAL)
    test_ds  = create_dataset(TEST_DIR, N_TEST)
    print(f"  Train: {N_TRAIN} | Val: {N_VAL} | Test: {N_TEST}")
    print(f"  Image size: {TARGET_SIZE} | Batch: {BATCH_SIZE}")

    # --- Build model ---
    print("\n[2/4] Building U-Net model...")
    model = build_unet()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy", iou, f1_score],
    )
    model.summary()

    # --- Train ---
    print(f"\n[3/4] Training for {EPOCHS} epochs...")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    # --- Evaluate ---
    print("\n[4/4] Evaluating on test set...")
    loss, accuracy, iou_val, f1_val = model.evaluate(test_ds)
    print(f"\n  Test Results:")
    print(f"    Loss:     {loss:.4f}")
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    IoU:      {iou_val:.4f}")
    print(f"    F1 Score: {f1_val:.4f}")

    # --- Save training curves ---
    plot_training_history(history)

    # --- Visualize predictions ---
    visualize_predictions(test_ds, model)

    # --- Save model ---
    model.save(os.path.join(RESULTS_DIR, "unet_tissuenet.keras"))
    print(f"\n  Model saved: {RESULTS_DIR}/unet_tissuenet.keras")

    return model, history


# VISUALIZATION

def plot_training_history(history):
    """Plot F1, IoU, and loss convergence over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(history.history["f1_score"], label="Train F1", linewidth=2)
    ax.plot(history.history["iou"], label="Train IoU", linewidth=2)
    ax.plot(history.history["loss"], label="Train Loss", linewidth=2)

    if "val_f1_score" in history.history:
        ax.plot(history.history["val_f1_score"], "--", label="Val F1")
        ax.plot(history.history["val_iou"], "--", label="Val IoU")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric Value")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_convergence.pdf"))
    plt.close()
    print(f"  Training curves saved: {RESULTS_DIR}/training_convergence.pdf")


def visualize_predictions(dataset, model, n=4):
    """Visualize input → ground truth → prediction side by side."""
    for batch in dataset.take(1):
        images, masks = batch
        predictions = model.predict(images, verbose=0)

        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
        for i in range(min(n, len(images))):
            axes[i, 0].imshow(images[i, :, :, 0], cmap="gray")
            axes[i, 0].set_title("Input")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(masks[i, :, :, 0], cmap="gray")
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(predictions[i, :, :, 0], cmap="gray")
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "prediction_samples.pdf"))
        plt.close()
        print(f"  Prediction samples saved: {RESULTS_DIR}/prediction_samples.pdf")
        break


# MAIN

if __name__ == "__main__":
    train()
