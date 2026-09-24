# -*- coding: utf-8 -*-
"""并行版多种子重复 + 消融 (与 replicates.py 同定义, 进程池按 run 分发)"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replicates import metrics_from_sim, route_choice, ABLATION_SCEN, N_SEEDS  # noqa: E402
import generate_scenarios as G  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios", "analysis")
CFGS = {c["name"]: c for c in G.build_scenarios()}


def one_replicate(args):
    name, seed = args
    import generate_scenarios as GG  # 子进程独立导入
    c = dict(CFGS[name]); c["seed"] = seed
    sim = GG.simulate(c, verbose=False)
    m = metrics_from_sim(sim)
    return dict(scenario=name, seed=seed, **m)


def one_ablation(args):
    name, seed, phero_on = args
    import generate_scenarios as GG
    GG.PHEROMONE_DROP_RATE = 0.25 if phero_on else 0.0
    c = dict(CFGS[name]); c["seed"] = seed
    sim = GG.simulate(c, verbose=False)
    m = metrics_from_sim(sim)
    rc = route_choice(name, sim)
    row = dict(scenario=name, seed=seed, pheromone=phero_on, **m)
    if rc:
        A_, B_ = int(rc["optA"].sum()), int(rc["optB"].sum())
        row.update({rc["labels"][0]: A_, rc["labels"][1]: B_, "other": rc["total"] - A_ - B_})
    return row


if __name__ == "__main__":
    t0 = time.time()
    rep_jobs = [(n, CFGS[n]["seed"] + 100 * k) for n in CFGS for k in range(N_SEEDS)]
    abl_jobs = [(n, CFGS[n]["seed"] + 100 * k, p)
                for n in ABLATION_SCEN for k in range(5) for p in (True, False)]
    with ProcessPoolExecutor(max_workers=40) as ex:
        rep_rows = list(ex.map(one_replicate, rep_jobs, chunksize=1))
        abl_rows = list(ex.map(one_ablation, abl_jobs, chunksize=1))
    df = pd.DataFrame(rep_rows)
    df.to_csv(os.path.join(OUT, "replicate_metrics_raw.csv"), index=False)
    df.groupby("scenario").agg(["mean", "std"]).round(3).to_csv(
        os.path.join(OUT, "replicate_metrics.csv"))
    pd.DataFrame(abl_rows).to_csv(os.path.join(OUT, "pheromone_ablation.csv"), index=False)
    print(f"replicates {len(rep_rows)} + ablation {len(abl_rows)} done in {time.time()-t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)
