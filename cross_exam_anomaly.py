"""
MHT-CET 2026 - CROSS-EXAM ANOMALY DETECTION
===========================================
Compares:
    1. CET  vs JEE
    2. CET  vs HSC
    3. JEE  vs HSC
    4. JEE  vs HSC + SSC
    5. CET  vs HSC + SSC

RUN:
    pip install pandas numpy scikit-learn matplotlib
    python cross_exam_anomaly.py first_10k_records.csv

IDEA IN SIMPLE WORDS:
    Students who do well in one exam usually do reasonably well in the others.
    For each comparison we learn this "normal pattern" from all students,
    then calculate for every student:  EXPECTED score  vs  ACTUAL score.
    If the gap is far bigger than for other students -> anomaly.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor

path = sys.argv[1] if len(sys.argv) > 1 else "first_10k_records.csv"
OUT = "cross_exam_output"
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(path)
df = df.rename(columns={
    "Merit No": "merit_no",
    "Application ID": "app_id",
    "Candidate's Full Name": "name",
    "Percentile /Mark": "jee",
    "MHT-CET PCM Total Percentile": "cet",
    "HSC PCM %": "hsc",
    "SSC Total %": "ssc",
})
df = df[["merit_no", "app_id", "name", "jee", "cet", "hsc", "ssc"]].copy()

# ---------------------------------------------------------------------
# STEP 1: convert marks into "rank score" (0-100) so that all exams
# are on the same scale. 99 means "better than 99% of these students".
# (Needed because CET percentiles are crowded near 99 while HSC % are spread.)
# ---------------------------------------------------------------------
for c in ["jee", "cet", "hsc", "ssc"]:
    df[c + "_rank"] = df[c].rank(pct=True) * 100   # NaN stays NaN

# ---------------------------------------------------------------------
# STEP 2: the 5 comparisons  (target = what we predict, based = what we use)
# ---------------------------------------------------------------------
comparisons = {
    "CET_vs_JEE":      ("cet_rank", ["jee_rank"]),
    "CET_vs_HSC":      ("cet_rank", ["hsc_rank"]),
    "JEE_vs_HSC":      ("jee_rank", ["hsc_rank"]),
    "JEE_vs_HSC_SSC":  ("jee_rank", ["hsc_rank", "ssc_rank"]),
    "CET_vs_HSC_SSC":  ("cet_rank", ["hsc_rank", "ssc_rank"]),
}

TOP_PERCENT = 2.5   # flag the 2.5% students with the BIGGEST gap (change if you want)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.ravel()
summary = []

for i, (name, (target, based)) in enumerate(comparisons.items()):
    d = df.dropna(subset=[target] + based).copy()
    X = d[based].values
    y = d[target].values

    # Huber regression = a line that is NOT pulled by the weird students
    model = HuberRegressor().fit(X, y)
    d["expected"] = model.predict(X)
    d["gap"] = d[target] - d["expected"]          # actual - expected

    # flag the students whose gap is in the most extreme TOP_PERCENT
    cutoff = np.percentile(d["gap"].abs(), 100 - TOP_PERCENT)
    d["z"] = d["gap"]
    d["flag"] = d["gap"].abs() >= cutoff

    # label: is the target much LOWER or much HIGHER than expected?
    d["direction"] = np.where(d["gap"] < 0, "LOWER", "HIGHER")
    d["direction"] = (target[:-5].upper() + " is " + d["direction"]
                      + " than expected by " + d["gap"].abs().round(0).astype(int).astype(str)
                      + " rank pts")

    df[name + "_gap"] = d["gap"]
    df[name + "_flag"] = d["flag"]
    df[name + "_direction"] = d["direction"].where(d["flag"])
    df[name + "_flag"] = df[name + "_flag"].fillna(False).astype(bool)

    n_low = int(((d["flag"]) & (d["z"] < 0)).sum())
    n_high = int(((d["flag"]) & (d["z"] > 0)).sum())
    corr = d[based[0]].corr(d[target])
    summary.append((name, len(d), corr, n_low, n_high))

    # chart (single predictor -> scatter. two predictors -> actual vs expected)
    ax = axes[i]
    xplot = d[based[0]] if len(based) == 1 else d["expected"]
    ok = ~d["flag"]
    ax.scatter(xplot[ok], d.loc[ok, target], s=3, alpha=.25, color="#4c72b0")
    ax.scatter(xplot[~ok], d.loc[~ok, target], s=12, color="red", label="anomaly")
    ax.set_title(name.replace("_", " "))
    ax.set_xlabel(("rank of " + based[0][:-5].upper()) if len(based) == 1
                  else "EXPECTED rank (from HSC+SSC)")
    ax.set_ylabel("rank of " + target[:-5].upper())
    ax.legend(loc="lower right")

axes[5].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/charts_all_comparisons.png", dpi=130)
plt.close()

# ---------------------------------------------------------------------
# STEP 3: final result per student
# ---------------------------------------------------------------------
flag_cols = [c + "_flag" for c in comparisons]
df["n_comparisons_flagged"] = df[flag_cols].sum(axis=1)
df["which_comparisons"] = df.apply(
    lambda r: "; ".join(
        f"{c}: {r[c + '_direction']}" for c in comparisons if r[c + "_flag"]),
    axis=1)

df["level"] = np.select(
    [df["n_comparisons_flagged"] >= 3, df["n_comparisons_flagged"] == 2,
     df["n_comparisons_flagged"] == 1],
    ["HIGH", "MEDIUM", "LOW"], default="NORMAL")

keep = ["merit_no", "app_id", "name", "jee", "cet", "hsc", "ssc",
        "n_comparisons_flagged", "level", "which_comparisons"]
df[keep].to_csv(f"{OUT}/all_students_result.csv", index=False)

review = df[df["n_comparisons_flagged"] >= 2].sort_values(
    "n_comparisons_flagged", ascending=False)
review[keep].to_csv(f"{OUT}/REVIEW_LIST_check_these.csv", index=False)

# ---------------------------------------------------------------------
# STEP 4: summary text
# ---------------------------------------------------------------------
lines = ["CROSS-EXAM ANOMALY SUMMARY", "=" * 60,
         f"{'Comparison':18s}{'Students':>9s}{'Link':>7s}{'Too LOW':>9s}{'Too HIGH':>10s}"]
for n, cnt, corr, lo, hi in summary:
    lines.append(f"{n:18s}{cnt:9d}{corr:7.2f}{lo:9d}{hi:10d}")
lines += ["", "Link  = how strongly the two exams move together (0 weak, 1 strong)",
          "Too LOW  = student's score is far BELOW what other marks predict",
          "Too HIGH = student's score is far ABOVE what other marks predict",
          "", "Students by level:", str(df["level"].value_counts())]
text = "\n".join(lines)
open(f"{OUT}/summary.txt", "w").write(text)
print(text)
print(f"\nDone. Open the folder '{OUT}'.")
