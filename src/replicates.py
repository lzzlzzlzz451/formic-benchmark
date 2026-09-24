# -*- coding: utf-8 -*-
"""
多种子重复实验 + 信息素消融 (内存中运行, 不写 CSV/GIF):
  1) 每场景 10 个种子 -> analysis/replicate_metrics.csv (mean +- std)
  2) 关键场景 关闭信息素沉积(PHEROMONE_DROP_RATE=0) -> analysis/pheromone_ablation.csv
     对照: 路径选择比例与稳态携食比例是否改变 (路径锁定是否为信息素介导)
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_scenarios as G  # noqa: E402

OUT = os.path.join(HERE, "scenarios", "analysis")
os.makedirs(OUT, exist_ok=True)

N_SEEDS = 10
ABLATION_SCEN = ["wall_with_door", "central_vertical_wall", "ring_with_gaps", "side_food_detour"]
N_SEEDS_ABL = 5


def run_durations(st):
    ret, out = [], []
    for col in range(st.shape[1]):
        s = st[:, col]
        d = np.diff(s)
        starts = np.where(d != 0)[0]
        bounds = np.concatenate([[-1], starts, [len(s) - 1]])
        runs, prev = [], 0
        for b in bounds[1:]:
            runs.append((prev, b - bounds[list(bounds).index(b) - 1]))
            prev = 1 - prev
        for i, (val, ln) in enumerate(runs):
            if val == 1 and ln > 1:
                ret.append(ln)
                if i + 1 < len(runs) and runs[i + 1][1] > 1:
                    out.append(runs[i + 1][1])
    return ret, out


def metrics_from_sim(sim):
    st = sim["states"]
    T = st.shape[0]
    d = np.diff(st, axis=0)
    first_pickup = int(np.argwhere(st == 1)[:, 0].min())
    first_delivery = int(np.argwhere(d == -1)[:, 0].min()) + 1
    deliveries = int((d == -1).sum())
    carry = float(st[T - 300:].mean())
    ret, out = run_durations(st)
    return dict(first_pickup=first_pickup, first_delivery=first_delivery,
                deliveries=deliveries, carry=carry,
                outbound=float(np.mean(out)), ret=float(np.mean(ret)))


def crossing_ys(xs, ys, x_line=50.0):
    a = xs - x_line
    ti, ai = np.where(a[:-1] * a[1:] < 0)
    return 0.5 * (ys[ti, ai] + ys[ti + 1, ai])


def route_choice(name, sim):
    xs, ys = sim["xs"], sim["ys"]
    if name == "wall_with_door":
        y = crossing_ys(xs, ys)
        tot = max(len(y), 1)
        return dict(optA=(y > 46) & (y < 60), optB=y <= 8,
                    total=tot, labels=("door", "bottom_gap"))
    if name == "central_vertical_wall":
        y = crossing_ys(xs, ys)
        return dict(optA=y > 84.5, optB=y < 39.5, total=max(len(y), 1),
                    labels=("top_endaround", "bottom_endaround"))
    if name == "side_food_detour":
        y = crossing_ys(xs, ys)
        return dict(optA=y > 36, optB=y < 8, total=max(len(y), 1),
                    labels=("above", "below"))
    if name == "ring_with_gaps":
        r = np.hypot(xs - 50, ys - 50)
        ti, ai = np.where((r[:-1] - 22) * (r[1:] - 22) < 0)
        ang = np.degrees(np.arctan2(0.5 * (ys[ti, ai] + ys[ti + 1, ai]) - 50,
                                    0.5 * (xs[ti, ai] + xs[ti + 1, ai]) - 50)) % 360
        d_top = np.minimum(np.abs(ang - 72), 360 - np.abs(ang - 72))
        d_bot = np.minimum(np.abs(ang - 252), 360 - np.abs(ang - 252))
        designed = (d_top < 18) | (d_bot < 18)
        return dict(optA=~designed, optB=designed, total=len(ang),
                    labels=("pillar_squeezes", "designed_gaps"))
    return None


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定场景 (其余沿用既有 raw CSV)")
    args = ap.parse_args()

    cfgs = {c["name"]: c for c in G.build_scenarios()}
    if args.only:
        cfgs = {k: v for k, v in cfgs.items() if k in args.only}
    rows, abl_rows = [], []

    for name, cfg in cfgs.items():
        for k in range(N_SEEDS):
            seed = cfg["seed"] + 100 * k
            c = dict(cfg); c["seed"] = seed
            t0 = time.time()
            sim = G.simulate(c, verbose=False)
            m = metrics_from_sim(sim)
            rows.append(dict(scenario=name, seed=seed, **m))
            print(f"[{name} s{seed}] carry={m['carry']:.3f} deliv={m['deliveries']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    raw_path = os.path.join(OUT, "replicate_metrics_raw.csv")
    if args.only and os.path.exists(raw_path):  # 与既有结果合并去重
        old = pd.read_csv(raw_path)
        df = pd.concat([old[~old["scenario"].isin(df["scenario"])], df], ignore_index=True)
    df.to_csv(raw_path, index=False)
    agg = df.groupby("scenario").agg(["mean", "std"]).round(3)
    agg.to_csv(os.path.join(OUT, "replicate_metrics.csv"))
    print("replicate metrics saved", flush=True)

    # ---- 信息素消融 (--only 局部重跑时跳过, 结果已在 pheromone_ablation.csv) ----
    for name in ([] if args.only else ABLATION_SCEN):
        for k in range(N_SEEDS_ABL):
            seed = cfgs[name]["seed"] + 100 * k
            for phero_on in (True, False):
                G.PHEROMONE_DROP_RATE = 0.25 if phero_on else 0.0
                c = dict(cfgs[name]); c["seed"] = seed
                sim = G.simulate(c, verbose=False)
                m = metrics_from_sim(sim)
                rc = route_choice(name, sim)
                row = dict(scenario=name, seed=seed, pheromone=phero_on, **m)
                if rc:
                    A = int(rc["optA"].sum()); B = int(rc["optB"].sum())
                    row.update({rc["labels"][0]: A, rc["labels"][1]: B,
                                "other": rc["total"] - A - B})
                abl_rows.append(row)
                print(f"[abl {name} s{seed} phero={phero_on}] carry={m['carry']:.3f}", flush=True)
    G.PHEROMONE_DROP_RATE = 0.25
    pd.DataFrame(abl_rows).to_csv(os.path.join(OUT, "pheromone_ablation.csv"), index=False)
    print("ablation saved\nALL DONE", flush=True)


if __name__ == "__main__":
    main()
