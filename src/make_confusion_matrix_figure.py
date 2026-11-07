import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
p = pd.read_csv(sys.argv[1]); out = sys.argv[2]
cm = pd.crosstab(p["label"], p["oof_pred"]).reindex(index=range(5), columns=range(5), fill_value=0).values
rn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.7), constrained_layout=True)
for a, M, title, fmt, vmax in [(ax[0], cm, "Counts", "{:d}", cm.max()), (ax[1], rn, "Row-normalized recall", "{:.2f}", 1.0)]:
    im = a.imshow(M, cmap="Blues", vmin=0, vmax=vmax)
    for i in range(5):
        for j in range(5):
            v = M[i, j]
            a.text(j, i, fmt.format(int(v)) if fmt == "{:d}" else fmt.format(v), ha="center", va="center",
                   fontsize=8, color="white" if v > 0.55 * vmax else "black")
    a.set_xticks(range(5)); a.set_yticks(range(5))
    a.set_xlabel("Predicted grade"); a.set_ylabel("True grade"); a.set_title(title, fontsize=10)
    fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
fig.savefig(out, dpi=600)
print(cm)
