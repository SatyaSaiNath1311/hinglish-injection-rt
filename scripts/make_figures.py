# Figures for the handover package (PNG + PDF) from results/tables.
#
#   python scripts/make_figures.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / (name + ".png"), dpi=200)
    fig.savefig(FIG / (name + ".pdf"))
    plt.close(fig)
    print("saved " + name)


def recall_by_lang(res, split="all"):
    d = res[(res.split == split) & (res.group_by == "lang")]
    if d.empty:
        return
    dets = sorted(d.detector.unique())
    x = np.arange(len(dets))
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, lang in enumerate(["en", "hinglish"]):
        sub = d[d.group == lang].set_index("detector").reindex(dets)
        err = np.vstack([sub.recall - sub.recall_ci_lo, sub.recall_ci_hi - sub.recall]).clip(min=0)
        ax.bar(x + (k - 0.5) * 0.38, sub.recall, 0.38, yerr=err, capsize=3, label=lang)
    ax.set_xticks(x)
    ax.set_xticklabels(dets, rotation=15)
    ax.set_ylabel("Injection recall")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Recall: English vs Hinglish (split=" + split + ")")
    _save(fig, "fig1_recall_en_vs_hinglish")


def recall_by_cmi(res, split="all"):
    d = res[(res.split == split) & (res.group_by == "cmi_bin")]
    if d.empty:
        return
    order = ["0", "(0,20]", "(20,40]", "(40,100]"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for det, g in d.groupby("detector"):
        g = g.set_index("group").reindex(order)
        ax.plot(order, g.recall, marker="o", label=det)
    ax.set_xlabel("Code-Mixing Index bin")
    ax.set_ylabel("Injection recall")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Recall vs code-mixing degree")
    _save(fig, "fig2_recall_vs_cmi")


def attack_curves():
    files = sorted(TAB.glob("attack_efficiency_*.csv"))
    if not files:
        return
    d = pd.concat([pd.read_csv(f) for f in files])
    if "tag" not in d.columns:
        d["tag"] = ""
    d["tag"] = d["tag"].fillna("")
    for (det, tag), g in d.groupby(["detector", "tag"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for m, gm in g.groupby("method"):
            ax.plot(gm.budget, gm.asr, marker="o", label=m)
        ax.set_xlabel("Query budget")
        ax.set_ylabel("Attack success rate")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Attack efficiency vs " + det + (" " + tag if tag else ""))
        _save(fig, "fig3_attack_efficiency_" + det + tag)


def main():
    f = TAB / "baseline_by_group.csv"
    if f.exists():
        res = pd.read_csv(f)
        recall_by_lang(res)
        recall_by_cmi(res)
    attack_curves()


if __name__ == "__main__":
    main()
