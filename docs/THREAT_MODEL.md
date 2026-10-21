# Threat model

- Attacker knowledge: black-box. Sees only the detector's score or label for a submitted text.
- Attacker goal: a rewrite that (a) the detector classifies as benign and (b) still carries the
  original malicious intent (intent score >= tau and, later, an independent malicious-intent check).
- Attacker budget: fixed number of detector queries per seed (report attack-efficiency curves).
- Attacker capabilities: word-level edits, transliteration, script switching, filler insertion
  (extend in attacker/actions.py).
- Defender: any of the target detectors listed in configs/default.yaml.
- Out of scope: attacks on the downstream LLM itself, white-box gradient attacks.
