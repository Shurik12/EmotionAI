import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------
# 1. Загрузка данных
# ---------------------------------------------------------------
df = pd.read_csv("auc_roc_audio.csv")

# ---------------------------------------------------------------
# 2. Бинарная истина из 'expected'
# ---------------------------------------------------------------
df["y_true"] = df["expected"].map({"low": 0, "high": 1})

# ---------------------------------------------------------------
# 3. Числовой score из 'model'
# ---------------------------------------------------------------
score_map = {"low": 0.0, "moderate": 0.5, "high": 1.0}
df["y_score"] = df["model"].map(score_map)

df = df.dropna(subset=["y_true", "y_score"])
df["y_true"] = df["y_true"].astype(int)

# ---------------------------------------------------------------
# 4. ROC и AUC
# ---------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(df["y_true"], df["y_score"])
roc_auc = auc(fpr, tpr)

# ---------------------------------------------------------------
# 5. График
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(fpr, tpr, color="darkorange", lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--",
        label="Chance (AUC = 0.5)")

# Индивидуальные смещения подписей, чтобы не наезжали
label_offsets = {
    0.0: (8, -18, "left"),
    0.5: (8, 10, "left"),
    1.0: (-8, -25, "right"),
}

for t in sorted(np.unique(df["y_score"])):
    idx = np.argmin(np.abs(thresholds - t))
    x, y = fpr[idx], tpr[idx]
    ax.scatter(x, y, marker="o", color="red", s=90, zorder=5)

    dx, dy, ha = label_offsets.get(t, (8, 8, "left"))
    name = ["low", "moderate", "high"][int(round(t * 2))]
    ax.annotate(
        f"score={t:.1f} ({name})",   # <-- одна строка, без дублирования
        (x, y),
        textcoords="offset points",
        xytext=(dx, dy),
        ha=ha,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                  ec="gray", alpha=0.95),
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
    )

ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.set_title("ROC/AUC Curve")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("auc_roc_curve.png", dpi=150)
plt.show()