# Lost in Code-Mixing

Reinforcement Learning-driven red-teaming of prompt-injection detectors in Hinglish.

Question: do prompt-injection detectors lose recall when the same malicious intent is written in
Hinglish (Hindi + English code-mixing) instead of standard English, and can an RL attacker find
those weaknesses systematically?

## Layout

```
configs/default.yaml        detectors, attacker, reward settings
data/gold/seed.csv          tiny starter set (replace with your gold set)
data/raw, data/processed    full datasets (git-ignored)
src/hinglish_rt/
  lexicon.py                EN->Hinglish / Devanagari word tables, fillers
  cmi.py                    Code-Mixing Index + script detection
  detectors/                base, cache wrapper, regex, HF classifier, hybrid, registry
  attacker/                 edit actions, reward, Gymnasium env
  eval/metrics.py           recall / FPR / F1 / AUROC / bootstrap CI
scripts/
  smoke_test.py             end-to-end check, CPU only, no downloads
  run_baseline.py           English-vs-Hinglish detector evaluation
  train_attacker.py         PPO attacker against one detector
kaggle/                     GPU runner (kernel-metadata.json + run.py)
docs/                       threat model, methods, findings, handover checklist
results/                    tables/ figures/ raw/ (raw is git-ignored)
```

## Quick start

```
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/smoke_test.py
pytest -q
```

## Kaggle (GPU jobs)

```
pip install kaggle
# put kaggle.json in ~/.kaggle (never in the repo)
# 1. edit kaggle/kernel-metadata.json (your username) and kaggle/run.py (REPO_URL, CMD)
kaggle kernels push -p kaggle
kaggle kernels status <username>/hinglish-rt-runner
kaggle kernels output <username>/hinglish-rt-runner -p results/raw
```

Add HF_TOKEN in Kaggle Secrets for gated models. Turn Internet ON in the kernel settings.

## Safety note

This repo contains adversarial prompt data for defensive evaluation of detectors only.
