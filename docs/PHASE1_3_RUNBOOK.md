# Phase 1-3 runbook (run in order, venv active, from the repo root)

```
pip install -r requirements.txt && pip install -e .
python -m pytest -q                                   # 12 tests should pass

# 1. Build the frozen master dataset (needs internet for Hugging Face)
python scripts/build_dataset.py --sources deepset xtram1 gandalf harelix dolly --cap 2000
#    Read data/processed/dataset_card.md. Sources that fail to load are skipped with a message.
#    Quick offline pipeline check: python scripts/build_dataset.py --demo --out /tmp/demo

# 2. Baseline: English vs Hinglish on test, test_ood and gold splits
python scripts/run_baseline.py --only regex_baseline protectai_deberta llama_prompt_guard_2

# 3. Attackers at equal query budgets (random vs hill-climb; add PPO after step 4)
python scripts/run_attackers.py --detector protectai_deberta --judge embedding --n_seeds 100 --budget 50

# 4. PPO attacker (train seeds -> evaluated on held-out test seeds)
python scripts/train_attacker.py --detector protectai_deberta --judge embedding --timesteps 50000
python scripts/run_attackers.py --detector protectai_deberta --judge embedding --n_seeds 100 --budget 50 \
    --ppo_model results/raw/protectai_deberta_seed42/ppo_attacker

# 5. Figures for the handover package
python scripts/make_figures.py
```

Split rules: train = fine-tuning + RL seeds; validation = tuning; test/test_ood = final numbers;
gold = fresh author-written pairs (extend src/hinglish_rt/data/manual_pairs.py, never train on it).

Hinglish quality: send data/processed/annotation_sample.csv to 2-3 bilingual annotators and compute
Cohen's kappa. To add better Hinglish (LLM-assisted or human-written), put rows in a CSV with
columns group_id,text[,mix_level,gen_method] and run scripts/import_hinglish.py --csv file.csv.
