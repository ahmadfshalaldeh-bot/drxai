"""Post-hoc explainability and its quantitative audit (Section 3.11).

All maps are computed on *real* held-out images that pass through the trained
networks; nothing here draws or synthesises retinal content. Every figure
panel produced from these functions is saved together with the image_id,
rotation iteration and predicted/reference grade, so that each panel can be
traced to its source photograph.

Grad-CAM : terminal attention-refined feature map ``*_attn_out`` (7x7 VGG and
           ResNet, 5x5 InceptionV3), bilinear up-sampling to 224x224.
LIME     : SLIC superpixels (K=64, compactness 10, sigma 1), 1000 perturbations,
           ridge surrogate; superpixel weights painted back to pixels.
SHAP     : DeepSHAP per branch with a shared background of training-fold
           images, combined by Proposition 3: phi_ens = sum_i w_i phi_i.
           If DeepExplainer cannot trace a custom layer, GradientExplainer
           (expected gradients, also linear in the model) is used and logged.
"""
from __future__ import annotations

import time

import numpy as np
import tensorflow as tf
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def _attn_layer(model: tf.keras.Model) -> str:
    names = [l.name for l in model.layers if l.name.endswith("attn_out")]
    if not names:
        raise ValueError(f"{model.name}: no *_attn_out layer")
    return names[-1]


def gradcam(model: tf.keras.Model, x: np.ndarray, class_idx: int, size: int = 224) -> np.ndarray:
    layer = model.get_layer(_attn_layer(model))
    g = tf.keras.Model(model.input, [layer.output, model.output])
    with tf.GradientTape() as tape:
        fmap, prob = g(x[None].astype(np.float32), training=False)
        score = prob[:, class_idx]
    grads = tape.gradient(score, fmap)[0]
    alpha = tf.reduce_mean(grads, axis=(0, 1))                     # GAP of gradients
    cam = tf.nn.relu(tf.reduce_sum(fmap[0] * alpha, axis=-1))
    cam = tf.image.resize(cam[..., None], (size, size), method="bilinear")[..., 0].numpy()
    return cam / (cam.max() + 1e-12), tuple(fmap.shape[1:3])


def ensemble_gradcam(members, weights, x, class_idx):
    maps = [gradcam(m, x, class_idx)[0] for m in members]
    return np.tensordot(np.asarray(weights), np.stack(maps), axes=1)


# ---------------------------------------------------------------------------
# LIME with SLIC
# ---------------------------------------------------------------------------
def lime_map(predict_fn, x: np.ndarray, class_idx: int, cfg: dict, seed: int = 42):
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    seg = SegmentationAlgorithm("slic", n_segments=cfg["lime_segments"],
                                compactness=cfg["lime_compactness"], sigma=cfg["lime_sigma"],
                                start_label=0, random_seed=seed)
    expl = lime_image.LimeImageExplainer(random_state=seed)
    e = expl.explain_instance(x.astype(np.double), predict_fn, labels=(class_idx,),
                              hide_color=0, num_samples=cfg["lime_samples"],
                              segmentation_fn=seg, batch_size=32)
    weights = dict(e.local_exp[class_idx])
    m = np.zeros(e.segments.shape, dtype=float)
    for sp, w in weights.items():
        m[e.segments == sp] = w
    return m, e.segments


# ---------------------------------------------------------------------------
# SHAP for the soft-voting ensemble (Proposition 3)
# ---------------------------------------------------------------------------
def shap_branch(model, x, background, class_idx, n_samples: int = 200):
    import shap

    try:
        ex = shap.DeepExplainer(model, background)
        sv = ex.shap_values(x[None], check_additivity=False)
        method = "deep"
    except Exception:  # custom layers not traceable by DeepExplainer
        ex = shap.GradientExplainer(model, background)
        sv = ex.shap_values(x[None], nsamples=n_samples)
        method = "expected_gradients"
    sv = sv[class_idx] if isinstance(sv, list) else sv[..., class_idx]
    return np.asarray(sv)[0].sum(-1), method                    # sum over RGB -> H x W


def shap_ensemble(members, weights, x, background, class_idx, n_samples=200):
    parts, methods = [], []
    for m in members:
        phi, meth = shap_branch(m, x, background, class_idx, n_samples)
        parts.append(phi)
        methods.append(meth)
    return np.tensordot(np.asarray(weights), np.stack(parts), axes=1), methods


# ---------------------------------------------------------------------------
# Quantitative audit
# ---------------------------------------------------------------------------
def binarize_top(att: np.ndarray, percentile: float = 85, fov: np.ndarray | None = None):
    """A_q = {pixels whose positive attribution exceeds the q-th percentile of the
    positive attributions inside the FOV} (Eq. (24))."""
    a = att.copy()
    if fov is not None:
        a[~fov] = 0
    pos = a[a > 0]
    if pos.size == 0:
        return np.zeros_like(a, bool)
    return a > np.percentile(pos, percentile)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else np.nan


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else np.nan


def timed(fn, repeats: int = 3):
    """Return (result of last call, mean seconds, sd seconds) over ``repeats``."""
    ts, out = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.mean(ts)), float(np.std(ts, ddof=1) if repeats > 1 else 0.0)


def randomize_top(model: tf.keras.Model, n_layers: int, seed: int = 0) -> tf.keras.Model:
    """Cascading parameter randomisation (Adebayo et al., 2018): re-initialise the
    top ``n_layers`` weight-bearing layers (head first, then attention, then the
    trainable backbone stage)."""
    clone = tf.keras.models.clone_model(model)
    clone.set_weights(model.get_weights())
    rng = np.random.default_rng(seed)
    weighted = [l for l in _flatten(clone) if l.weights]
    for layer in weighted[::-1][:n_layers]:
        layer.set_weights([rng.normal(0, 0.05, w.shape).astype(w.dtype)
                           for w in layer.get_weights()])
    return clone


def _flatten(model):
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            yield from _flatten(l)
        else:
            yield l


def rank_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of absolute attributions (sanity checks)."""
    return float(spearmanr(np.abs(a).ravel(), np.abs(b).ravel()).correlation)
