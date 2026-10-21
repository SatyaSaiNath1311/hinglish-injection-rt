# Kaggle runner: clones the GitHub repo, runs JOBS in order, and always saves outputs.
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/SatyaSaiNath1311/hinglish-injection-rt.git"
BRANCH = "main"
JOBS = [  # edit this list for each run
    "python -m pytest -q",
    "python scripts/build_dataset.py --sources deepset xtram1 gandalf dolly --cap 2000",
    "python scripts/run_baseline.py --only regex_baseline protectai_deberta",
    "python scripts/make_figures.py",
]

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as e:
    print("No HF_TOKEN secret (fine for public models):", e)


def sh(cmd, cwd=None):
    print("$ " + cmd, flush=True)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


work = "/kaggle/working"
repo = os.path.join(work, "repo")
if os.path.exists(repo):
    shutil.rmtree(repo)
sh("git clone --depth 1 -b " + BRANCH + " " + REPO_URL + " " + repo)
sh("pip install -q datasets sentence-transformers gymnasium stable-baselines3 pyyaml scipy", cwd=repo)
sh("pip install -q -e . --no-deps", cwd=repo)

failed = None
try:
    for job in JOBS:
        sh(job, cwd=repo)
except subprocess.CalledProcessError as e:
    failed = e
finally:
    for src, dst in (("results", "results"), ("data/processed", "data_processed")):
        p = os.path.join(repo, src)
        if os.path.exists(p):
            shutil.copytree(p, os.path.join(work, dst), dirs_exist_ok=True)
    shutil.rmtree(repo, ignore_errors=True)

if failed:
    print("A job failed:", failed)
    sys.exit(1)
print("Done. Outputs in /kaggle/working/results and /kaggle/working/data_processed")
