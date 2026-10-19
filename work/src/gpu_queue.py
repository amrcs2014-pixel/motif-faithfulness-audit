"""Resumable GPU job queue for the motif-faithfulness audit (run by the self-restarting scheduled task new1_gpu).

Jobs run strictly one at a time (MAXPAR = 1: the laptop has crashed under sustained dual-GPU load). Two kinds:
  train   -> train.py <args>      done when results/runs/<name>/metrics.json exists
  explain -> explain.py --run <n> done when results/explain/<name>/summary.json exists; waits for the run and
                                  for the marker logs/explain_ready (explain.py tested)
The queue re-picks the highest-priority runnable job after every job; a job that fails twice is skipped.
Logs: logs/gpu_queue.log, per-job logs in logs/jobs/. Writes ALL DONE when every job has a result.
"""
import os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import *  # noqa

MAXPAR = 1


def jobs():
    J = []

    def train(model, vocab, split, seed, ss=0):
        J.append(["train", "--model", model, "--vocab", vocab, "--split", split, "--seed", str(seed), "--split_seed", str(ss)])

    def explain(model, vocab, split, seed, ss=0):
        J.append(["explain", "--run", run_name(model, vocab, split, seed, ss)])

    # 1. main set: CCMG under both motif vocabularies, fixed split (split seed 0), five training seeds
    for vocab in VOCABS:
        for split in SPLITS:
            for seed in SEEDS:
                train("ccmg", vocab, split, seed); explain("ccmg", vocab, split, seed)
    # 2. late-fusion baseline
    for split in SPLITS:
        for seed in SEEDS:
            train("latefusion", "brics", split, seed)
    # 3. data-resampling sensitivity set: split seed = training seed (BRICS only)
    for ss in SPLIT_SEEDS[1:]:
        for split in SPLITS:
            train("ccmg", "brics", split, ss, ss); explain("ccmg", "brics", split, ss, ss)
    # (late-fusion runs on re-drawn splits were dropped to keep the queue within one night of GPU time)

    # ---------------- revision (17 Oct 2026): attention controls, backbone tuning, best-config re-run ----------------
    def trainx(extra, seed, tag):
        J.append(["train", "--model", "ccmg", "--vocab", "brics", "--split", "random", "--seed", str(seed), "--split_seed", "0"] + extra)
        J.append(["explain", "--run", run_name("ccmg", "brics", "random", seed, 0, tag)])

    # 4. attention controls on the random regime: four heads; single head with an entropy bonus
    for seed in SEEDS:
        trainx(["--heads", "4"], seed, "h4")
    for seed in SEEDS:
        trainx(["--attn_entropy", "0.05"], seed, "ent0.05")
    # 5. backbone tuning grid (seed 0, random regime; the default hidden 128 / dropout 0.1 / lr 1e-3 / wd 1e-5 already exists)
    for extra in TUNING_GRID:
        J.append(["train", "--model", "ccmg", "--vocab", "brics", "--split", "random", "--seed", "0", "--split_seed", "0"] + extra)
    # 6. best tuned configuration (by validation RMSE of the grid) re-trained with five seeds and explained,
    #    added only once the whole grid has finished and only if the best configuration is not the default
    best = best_tuning_config()
    if best:
        extra, tag = best
        for seed in SEEDS:
            trainx(extra, seed, tag)
    return J


TUNING_GRID = [["--hidden", "64"], ["--hidden", "256"], ["--dropout", "0"], ["--dropout", "0.3"],
               ["--hidden", "64", "--dropout", "0"], ["--hidden", "64", "--dropout", "0.3"],
               ["--hidden", "256", "--dropout", "0"], ["--hidden", "256", "--dropout", "0.3"],
               ["--lr", "0.0003"], ["--lr", "0.003"], ["--wd", "0.0001"]]


def grid_results():
    """(extra args, tag, best validation RMSE) for every finished grid run, plus the default; None if any is missing."""
    from train import parse, tag_of
    out = []
    for extra in [[]] + TUNING_GRID:
        a = parse(["--model", "ccmg", "--vocab", "brics", "--split", "random", "--seed", "0", "--split_seed", "0"] + extra)
        tag = tag_of(a)
        f = os.path.join(RUNS, run_name("ccmg", "brics", "random", 0, 0, tag), "metrics.json")
        if not os.path.exists(f):
            return None
        out.append((extra, tag, jload(f)["best_val_rmse"]))
    return out


def best_tuning_config():
    res = grid_results()
    if res is None:
        return None
    extra, tag, _ = min(res, key=lambda r: r[2])
    return (extra, tag) if tag else None


def name_of(argv):
    if argv[0] == "train":
        from train import parse, tag_of
        a = parse(argv[1:])
        return run_name(a.model, a.vocab, a.split, a.seed, a.split_seed, tag_of(a))
    return argv[2]


def result_path(argv):
    n = name_of(argv)
    return os.path.join(RUNS, n, "metrics.json") if argv[0] == "train" else os.path.join(EXPL, n, "summary.json")


def runnable(argv):
    if argv[0] == "explain":
        return os.path.exists(os.path.join(RUNS, argv[2], "model.pt")) and os.path.exists(os.path.join(LOGS, "explain_ready"))
    return True


def job_log(argv):
    return os.path.join(LOGS, "jobs", f"{argv[0]}_{name_of(argv)}.log")


def live_workers():
    """pids of python processes running train.py / explain.py (ours or orphans)."""
    import psutil
    out = {}
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if "python" not in (p.info["name"] or "").lower():
                continue
            cl = p.info["cmdline"] or []
            for c in cl:
                c2 = c.replace("\\", "/")
                if c2.endswith("/train.py") or c2.endswith("/explain.py") or c in ("train.py", "explain.py"):
                    out[p.info["pid"]] = tuple(cl)
        except Exception:
            pass
    return out


def main():
    for s in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, s):
            signal.signal(getattr(signal, s), signal.SIG_IGN)
    import psutil
    os.makedirs(os.path.join(LOGS, "jobs"), exist_ok=True)
    log = open(os.path.join(LOGS, "gpu_queue.log"), "a", buffering=1)
    log.write(time.strftime("%m-%d %H:%M:%S ") + "queue start\n")
    failed, mine, last_fail, idle_since = {}, {}, time.time(), None
    orphans = set(live_workers())
    while True:
        orphans = {p for p in orphans if psutil.pid_exists(p)}
        for key in list(mine):
            pr, t0, fo = mine[key]
            if pr.poll() is not None:
                fo.close()
                try:
                    tail = "\n".join(open(job_log(list(key)), encoding="utf-8", errors="replace").read().strip().splitlines()[-2:])
                except Exception:
                    tail = ""
                log.write(f"   [{' '.join(key)}] rc={pr.returncode} {time.time() - t0:.0f}s\n   " + tail.replace("\n", "\n   ") + "\n")
                if pr.returncode != 0 or not os.path.exists(result_path(list(key))):
                    failed[key] = failed.get(key, 0) + 1
                    last_fail = time.time()
                del mine[key]
        if failed and time.time() - last_fail > 3600:
            failed.clear()  # a transient cause (missing package, memory pressure) gets another chance every hour
        if len(mine) + len(orphans) >= MAXPAR:
            time.sleep(15); continue
        nxt, waiting = None, 0
        for argv in jobs():
            if os.path.exists(result_path(argv)) or failed.get(tuple(argv), 0) >= 2 or tuple(argv) in mine:
                continue
            if not runnable(argv):
                waiting += 1; continue
            nxt = argv; break
        if nxt is None:
            if waiting == 0 and not mine:
                log.write(time.strftime("%m-%d %H:%M:%S ") + "ALL DONE\n"); return
            time.sleep(60); continue
        log.write(time.strftime("%m-%d %H:%M:%S ") + "RUN " + " ".join(nxt) + "\n")
        fo = open(job_log(nxt), "w")
        script = os.path.join(HERE, "train.py" if nxt[0] == "train" else "explain.py")
        pr = subprocess.Popen([sys.executable, "-u", "-W", "ignore", script] + nxt[1:], stdout=fo, stderr=subprocess.STDOUT,
                              creationflags=0x00008000 | 0x00000200)
        mine[tuple(nxt)] = (pr, time.time(), fo)
        time.sleep(10)


if __name__ == "__main__":
    if "--count" in sys.argv:
        J = jobs(); done = sum(os.path.exists(result_path(a)) for a in J)
        print(len(J), "jobs,", done, "done")
        for a in J:
            if not os.path.exists(result_path(a)):
                print("  pending:", " ".join(a))
    else:
        main()
