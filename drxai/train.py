"""Training of one model on one rotation iteration (Section 3.8).

* tf.data pipeline reading the cached, preprocessed 224x224 PNGs.
* Online augmentation is applied to *training* batches only, after
  partitioning; validation and test images are never augmented.
* Inverse-frequency class weights computed from the training folds only:
  w_c = N_train / (K * N_train,c).
* Adam (lr 1e-4), ReduceLROnPlateau (x0.5, patience 5), EarlyStopping on
  validation loss (patience 10, best weights restored), max 50 epochs.
* Outputs: best weights, per-epoch history (CSV), validation and test
  probabilities (NPZ) keyed by image_id.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .models import build_model
from .utils import ensure_dir, set_seed

AUTOTUNE = tf.data.AUTOTUNE


def _decode(path):
    img = tf.io.decode_png(tf.io.read_file(path), channels=3)
    return tf.cast(img, tf.float32)                          # 0..255


def _augment(img, a: dict):
    img = tf.image.random_flip_left_right(img) if a["flip_p"] > 0 else img
    img = tf.image.random_flip_up_down(img) if a["flip_p"] > 0 else img
    ang = tf.random.uniform([], -a["rotation_deg"], a["rotation_deg"]) * math.pi / 180.0
    img = _rotate(img, ang)
    img = img / 255.0
    img = tf.image.random_brightness(img, a["brightness"])
    img = tf.image.random_contrast(img, 1 - a["contrast"], 1 + a["contrast"])
    img = img + tf.random.normal(tf.shape(img), stddev=a["gaussian_sigma"])
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0


def _rotate(img, angle):
    """Continuous rotation about the image centre (bilinear, zero fill)."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    c, s = tf.cos(angle), tf.sin(angle)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    # inverse map: output -> input
    tx = cx - c * cx + s * cy
    ty = cy - s * cx - c * cy
    transform = tf.stack([c, -s, tx, s, c, ty, 0.0, 0.0])[None]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None], transforms=transform, output_shape=tf.shape(img)[:2],
        interpolation="BILINEAR", fill_mode="CONSTANT", fill_value=0.0)
    return out[0]


def make_dataset(paths, labels, cfg, training: bool, batch: int):
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        ds = ds.shuffle(len(paths), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: (_decode(p), tf.one_hot(y, cfg["num_classes"])),
                num_parallel_calls=AUTOTUNE)
    if training:
        a = cfg["augment"]
        ds = ds.map(lambda x, y: (_augment(x, a), y), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch).prefetch(AUTOTUNE)


def class_weights(y_train: np.ndarray, k: int) -> dict[int, float]:
    n = len(y_train)
    counts = np.bincount(y_train, minlength=k)
    return {c: float(n / (k * max(counts[c], 1))) for c in range(k)}


def train_one(model_name: str, iteration: int, table: pd.DataFrame, cfg: dict,
              out_root: str) -> dict:
    """``table`` has columns image_id, cached, grade, role (train/val/test)."""
    set_seed(cfg["seed"] + iteration)
    t = cfg["train"]
    out = ensure_dir(Path(out_root) / model_name / f"iter{iteration}")
    tr, va, te = (table[table.role == r] for r in ("train", "val", "test"))
    cw = class_weights(tr.grade.to_numpy(), cfg["num_classes"])

    model = build_model(model_name, cfg)
    model.compile(optimizer=tf.keras.optimizers.Adam(t["lr"], t["beta_1"], t["beta_2"]),
                  loss="categorical_crossentropy",
                  metrics=["accuracy", tf.keras.metrics.AUC(multi_label=True, name="auc")])
    cbs = [
        tf.keras.callbacks.EarlyStopping("val_loss", patience=t["early_stopping_patience"],
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau("val_loss", factor=t["reduce_lr_factor"],
                                             patience=t["reduce_lr_patience"]),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    hist = model.fit(make_dataset(tr.cached, tr.grade, cfg, True, t["batch_size"]),
                     validation_data=make_dataset(va.cached, va.grade, cfg, False, 64),
                     epochs=t["max_epochs"], class_weight=cw, callbacks=cbs, verbose=2)
    model.save_weights(str(out / "best.weights.h5"))

    for role, sub in (("val", va), ("test", te)):
        p = model.predict(make_dataset(sub.cached, sub.grade, cfg, False, 64), verbose=0)
        np.savez_compressed(out / f"probs_{role}.npz", image_id=sub.image_id.to_numpy(),
                            y=sub.grade.to_numpy(), p=p)
    return {"epochs_run": len(hist.history["loss"]), "class_weights": cw}
