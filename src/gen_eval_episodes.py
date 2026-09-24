# -*- coding: utf-8 -*-
"""为 4 个代表场景各生成 2 条额外种子轨迹 (供基线实验多轨迹测试), 只存节点 CSV。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import generate_scenarios as G  # noqa: E402

OUT = os.path.join(HERE, "scenarios", "eval_episodes")
os.makedirs(OUT, exist_ok=True)

REP = ["single_circle", "two_circles_slalom", "wall_with_door", "s_shape_maze"]
cfgs = {c["name"]: c for c in G.build_scenarios()}

for name in REP:
    for k in (1, 2):
        seed = cfgs[name]["seed"] + 100 * k
        c = dict(cfgs[name]); c["seed"] = seed
        sim = G.simulate(c, verbose=False)
        df = pd.DataFrame({
            "frame": sim["frames"], "agent_id": sim["ids"],
            "x": sim["xs"].ravel(), "y": sim["ys"].ravel(), "state": sim["states"].ravel(),
        })
        p = os.path.join(OUT, f"{name}_s{seed}.csv")
        df.to_csv(p, index=False)
        print(f"{name} seed {seed} -> {os.path.basename(p)}", flush=True)
print("ALL DONE")
