# Build the master dataset (English + Hinglish variants + deterministic splits).
#
#   python scripts/build_dataset.py                      # real sources (needs internet + datasets)
#   python scripts/build_dataset.py --demo               # offline synthetic test of the pipeline
#   python scripts/build_dataset.py --sources deepset xtram1 gandalf harelix dolly --cap 2500
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hinglish_rt.data.build import build_master  # noqa: E402
from hinglish_rt.data.sources import SOURCES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["deepset", "xtram1", "gandalf"],
                    choices=sorted(SOURCES))
    ap.add_argument("--cap", type=int, default=2000, help="max rows per (source, label)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ood", default="roleplay", help="held-out injection category ('' = none)")
    ap.add_argument("--mt", action="store_true", help="add machine-translated Devanagari variants")
    ap.add_argument("--local_csv", default=None, help="CSV with text,label[,source] instead of HF")
    ap.add_argument("--demo", action="store_true", help="synthetic corpus (pipeline test only)")
    ap.add_argument("--hard_neg_per_template", type=int, default=40,
                    help="synthetic benign prompts with attack-style words, per template (20 templates)")
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    a = ap.parse_args()
    build_master(a.out, a.sources, a.cap, a.seed, a.ood or None, a.mt, a.local_csv, a.demo,
                 a.hard_neg_per_template)


if __name__ == "__main__":
    main()
