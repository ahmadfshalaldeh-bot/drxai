from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'results/raw/oof_predictions.csv'
FOLDS = ROOT / 'results/raw/fold_assignments.csv'
OUT = ROOT / 'results/metrics/confidence_intervals_extended.csv'
REPS = 1000
SEED = 2026
CLASSES = np.arange(5)
PROB = [f'prob_class_{i}' for i in range(5)]

oof = pd.read_csv(RAW)
fa = pd.read_csv(FOLDS)
meta = fa[['image_id', 'subject_id', 'fold']]
df = oof.merge(meta, on='image_id', how='left', validate='one_to_one')
assert len(df) == 35126
assert df.subject_id.nunique() == 17563

sids = df.subject_id.astype(str).to_numpy()
ordered_ids = sorted(set(sids), key=lambda x: int(x) if x.isdigit() else x)
groups = [np.flatnonzero(sids == sid) for sid in ordered_ids]

def metrics(sub):
    y = sub.label.to_numpy(int)
    pred = sub.oof_pred.to_numpy(int)
    p = sub[PROB].to_numpy(float)
    p = p / p.sum(axis=1, keepdims=True)
    d = y - pred
    return {
        'accuracy': accuracy_score(y, pred),
        'balanced_accuracy': balanced_accuracy_score(y, pred),
        'macro_f1': f1_score(y, pred, average='macro', labels=CLASSES, zero_division=0),
        'weighted_f1': f1_score(y, pred, average='weighted', labels=CLASSES, zero_division=0),
        'macro_precision': precision_score(y, pred, average='macro', labels=CLASSES, zero_division=0),
        'macro_recall': recall_score(y, pred, average='macro', labels=CLASSES, zero_division=0),
        'log_loss': log_loss(y, p, labels=CLASSES),
        'qwk': cohen_kappa_score(y, pred, weights='quadratic'),
        'macro_auc_ovr': roc_auc_score(y, p, multi_class='ovr', average='macro', labels=CLASSES),
        'mae_grade': np.abs(d).mean(),
        'rmse_grade': np.sqrt(np.mean(d ** 2)),
        'within_one_grade': (np.abs(d) <= 1).mean(),
        'large_ordinal_error_rate': (np.abs(d) >= 2).mean(),
        'undergrading_rate': (pred < y).mean(),
        'overgrading_rate': (pred > y).mean(),
    }

values = {k: [] for k in metrics(df)}
rng = np.random.default_rng(SEED)
for b in range(REPS):
    chosen = rng.integers(0, len(groups), size=len(groups))
    idx = np.concatenate([groups[i] for i in chosen])
    row = metrics(df.iloc[idx])
    for key, value in row.items():
        values[key].append(float(value))

rows = []
for key, vals in values.items():
    arr = np.asarray(vals)
    rows.append({
        'metric': key,
        'mean': float(arr.mean()),
        'lower_95': float(np.quantile(arr, 0.025)),
        'upper_95': float(np.quantile(arr, 0.975)),
        'seed': SEED,
        'replicates': REPS,
        'grouping': 'filename-derived grouping key (legacy manifest column subject_id); paired eye-side records retained together',
    })

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(out.to_string(index=False))
print(f'Wrote {OUT}')
