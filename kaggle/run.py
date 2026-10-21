# Kaggle runner: clones the repo, installs it, runs one command, copies results to /kaggle/working.
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/YOUR_GITHUB_USERNAME/hinglish-injection-rt.git"  # EDIT
BRANCH = "main"
CMD = "python scripts/run_baseline.py --only protectai_deberta"  # EDIT per experiment

try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    for key in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            os.environ[key] = secrets.get_secret(key)
        except Exception:
            pass
except ImportError:
    pass


def sh(cmd, cwd=None):
    print("$ " + cmd, flush=True)
    subprocess.check_call(cmd, shell=True, cwd=cwd)


work = "/kaggle/working"
repo = os.path.join(work, "repo")
if os.path.exists(repo):
    shutil.rmtree(repo)
sh("git clone --depth 1 -b " + BRANCH + " " + REPO_URL + " " + repo)
sh(sys.executable + " -m pip install -q -r requirements.txt", cwd=repo)
sh(sys.executable + " -m pip install -q -e .", cwd=repo)
sh(CMD, cwd=repo)

out = os.path.join(work, "results")
shutil.copytree(os.path.join(repo, "results"), out, dirs_exist_ok=True)
shutil.rmtree(repo)  # keep only /kaggle/working/results as the kernel output
print("Done. Results in " + out)
