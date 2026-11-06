"""Parameter counts, FLOPs and GPU latency (Table 6, last column)."""
from __future__ import annotations

import time

import numpy as np
import tensorflow as tf


def count_params(model: tf.keras.Model) -> dict:
    tr = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    nt = int(sum(np.prod(w.shape) for w in model.non_trainable_weights))
    return {"total": tr + nt, "trainable": tr, "non_trainable": nt}


def flops(model: tf.keras.Model, size: int = 224) -> int:
    """Forward-pass FLOPs for one image via the TF profiler (2 x MACs)."""
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2_as_graph)

    spec = tf.TensorSpec([1, size, size, 3], tf.float32)
    fn = tf.function(lambda x: model(x, training=False)).get_concrete_function(spec)
    _, graph_def = convert_variables_to_constants_v2_as_graph(fn)
    with tf.Graph().as_default() as g:
        tf.graph_util.import_graph_def(graph_def, name="")
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        prof = tf.compat.v1.profiler.profile(graph=g, cmd="op", options=opts)
    return int(prof.total_float_ops)


def latency_ms(model: tf.keras.Model, size: int = 224, warmup: int = 20, runs: int = 200,
               batch: int = 1) -> dict:
    x = tf.random.uniform([batch, size, size, 3], 0, 255)
    f = tf.function(lambda t: model(t, training=False))
    for _ in range(warmup):
        f(x).numpy()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        f(x).numpy()
        ts.append((time.perf_counter() - t0) * 1000 / batch)
    return {"mean_ms": float(np.mean(ts)), "sd_ms": float(np.std(ts, ddof=1))}
