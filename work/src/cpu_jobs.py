"""CPU baseline queue (scheduled task new1_cpu): naive + tuned LightGBM for every split / split seed; resumable."""
import os, signal, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from common import *
for s in ("SIGINT", "SIGBREAK"):
    if hasattr(signal, s): signal.signal(getattr(signal, s), signal.SIG_IGN)
os.environ.setdefault("OMP_NUM_THREADS", "4")
jobs = []
for split in SPLITS:
    jobs.append(["--split", split, "--split_seed", "0", "--seeds", "0", "1", "2", "3", "4"])
for ss in SPLIT_SEEDS[1:]:
    for split in SPLITS:
        jobs.append(["--split", split, "--split_seed", str(ss), "--seeds", str(ss)])
for j in jobs:
    subprocess.run([sys.executable, "-u", "-W", "ignore", os.path.join(HERE, "baselines.py"), "--threads", "4"] + j,
                   creationflags=0x00000200)
print("CPU_ALL_DONE", flush=True)
