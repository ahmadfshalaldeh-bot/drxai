"""drxai: explainable attention-augmented ensemble for five-class DR grading.

Companion code for the manuscript "An Explainable Ensemble Framework Combining
Transfer Learning, Attention Mechanisms, and Post-Hoc Interpretability for
Diabetic Retinopathy Grading" (submitted to Mathematics, MDPI).

Modules that need TensorFlow (models, train, xai, complexity) are imported
lazily so that the statistics can be used and unit-tested without a GPU stack.
"""

__version__ = "1.0.0"
