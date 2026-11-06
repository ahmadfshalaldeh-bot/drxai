"""Backbones, partial fine-tuning, attention placement and classification heads.

Every single-network model has the same head (Section 3.5):
    F -> [SE | CBAM | identity] -> GAP -> Dropout(0.5) -> Dense(256, ReLU, L2)
      -> Dense(5, softmax)
The terminal attention-refined feature map is exposed as the layer named
``attn_out`` (7x7 for VGG/ResNet, 5x5 for InceptionV3 at 224x224) and is the
Grad-CAM target layer.

Fine-tuning policy (identical for baselines and attention variants):
    VGG-16/19   : blocks 1-4 frozen, block 5 trainable
    ResNet-50   : stages conv1-conv4 frozen, stage conv5 trainable
    InceptionV3 : modules up to mixed8 frozen, mixed9-mixed10 trainable
    scratch     : all layers trainable, random initialisation
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import applications as A
from tensorflow.keras import layers, regularizers

from .attention import CBAM, SEBlock

_BACKBONES = {
    "vgg16": (A.VGG16, A.vgg16.preprocess_input, "block5"),
    "vgg19": (A.VGG19, A.vgg19.preprocess_input, "block5"),
    "resnet50": (A.ResNet50, A.resnet50.preprocess_input, "conv5"),
    "resnet152": (A.ResNet152, A.resnet.preprocess_input, "conv5"),
    "inceptionv3": (A.InceptionV3, A.inception_v3.preprocess_input, "mixed9"),
}


def _set_trainable(base: tf.keras.Model, first_trainable_prefix: str, pretrained: bool) -> None:
    if not pretrained:
        base.trainable = True
        return
    trainable = False
    for layer in base.layers:
        if layer.name.startswith(first_trainable_prefix):
            trainable = True
        layer.trainable = trainable


def _features(inp, backbone: str, attention: str, pretrained: bool, cfg: dict, tag: str):
    ctor, prep, prefix = _BACKBONES[backbone]
    base = ctor(include_top=False, weights="imagenet" if pretrained else None,
                input_shape=(cfg["preprocess"]["size"],) * 2 + (3,))
    base._name = f"{tag}_{backbone}"
    _set_trainable(base, prefix, pretrained)
    x = layers.Lambda(prep, name=f"{tag}_prep")(inp)        # backbone-native scaling
    f = base(x, training=False)                             # frozen BN statistics
    a = cfg["attention"]
    if attention == "se":
        f = SEBlock(a["se_ratio"], name=f"{tag}_se")(f)
    elif attention == "cbam":
        f = CBAM(a["cbam_ratio"], a["cbam_kernel"], name=f"{tag}_cbam")(f)
    f = layers.Activation("linear", name=f"{tag}_attn_out")(f)
    return f


def build_single(name: str, spec: dict, cfg: dict) -> tf.keras.Model:
    t = cfg["train"]
    size = cfg["preprocess"]["size"]
    inp = layers.Input((size, size, 3), name="image_0_255")   # CLAHE output, float 0..255
    f = _features(inp, spec["backbone"], spec["attention"], spec["pretrained"], cfg, "b")
    g = layers.GlobalAveragePooling2D(name="gap")(f)
    g = layers.Dropout(t["dropout"], name="dropout")(g)
    g = layers.Dense(t["dense_units"], activation="relu",
                     kernel_regularizer=regularizers.l2(t["l2"]), name="dense")(g)
    out = layers.Dense(cfg["num_classes"], activation="softmax", name="probs")(g)
    return tf.keras.Model(inp, out, name=name)


def build_dual_branch(cfg: dict) -> tf.keras.Model:
    """Feature-level fusion comparator: ResNet-50+CBAM || InceptionV3+SE.

    Both branches receive the *same* input image (there is no second input
    modality); their pooled descriptors are concatenated before a shared
    512-unit head. This is an ablation comparator, not the proposed model.
    """
    t = cfg["train"]
    size = cfg["preprocess"]["size"]
    inp = layers.Input((size, size, 3), name="image_0_255")
    f1 = _features(inp, "resnet50", "cbam", True, cfg, "r")
    f2 = _features(inp, "inceptionv3", "se", True, cfg, "i")
    g = layers.Concatenate(name="concat")([layers.GlobalAveragePooling2D()(f1),
                                           layers.GlobalAveragePooling2D()(f2)])
    g = layers.Dropout(t["dropout"])(g)
    g = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(t["l2"]))(g)
    out = layers.Dense(cfg["num_classes"], activation="softmax", name="probs")(g)
    return tf.keras.Model(inp, out, name="dual_branch")


def build_model(name: str, cfg: dict) -> tf.keras.Model:
    spec = cfg["models"][name]
    if spec["backbone"] == "dual":
        return build_dual_branch(cfg)
    return build_single(name, spec, cfg)


def build_ensemble_graph(members: list[tf.keras.Model], weights: list[float]) -> tf.keras.Model:
    """Differentiable soft-voting ensemble P(x) = sum_i w_i p_i(x) (Eq. (8)).

    Used for SHAP attribution of the ensemble output and for latency timing.
    """
    inp = members[0].input
    probs = [m(inp) for m in members]
    w = tf.constant(weights, dtype=tf.float32)
    out = layers.Lambda(lambda ps: tf.add_n([w[i] * p for i, p in enumerate(ps)]),
                        name="soft_vote")(probs)
    return tf.keras.Model(inp, out, name="ensemble")
