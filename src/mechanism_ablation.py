# -*- coding: utf-8 -*-
"""机制消融: 关闭拥挤探索/软避让/扩散偏置后拥堵是否回归 (证明 v2 机制必要性)"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CFGS_SCEN = ["wall_with_door", "narrow_corridor", "central_vertical_wall", "s_shape_maze"]
CONFIGS = {
    "full_v2": {},
    "no_exploration": {"EXPLORE_BASE": 0.0, "EXPLORE_GAIN": 0.0, "EXPLORE_DIFFUSE": 0.0},
    "no_soft_separation": {"SOFT_SEP_GAIN": 0.0},
    "neither_v1": {"EXPLORE_BASE": 0.0, "EXPLORE_GAIN": 0.0, "EXPLORE_DIFFUSE": 0.0, "SOFT_SEP_GAIN": 0.0},
}


def one(args):
    import generate_scenarios as GG
    from replicates import metrics_from_sim
    scen, seed, cfg_name, overrides = args
    for k, v in overrides.items():
        setattr(GG, k, v)
    cfgs = {c["name"]: c for c in GG.build_scenarios()}
    c = dict(cfgs[scen]); c["seed"] = seed
    sim = GG.simulate(c, verbose=False)
    m = metrics_from_sim(sim)
    return dict(scenario=scen, seed=seed, config=cfg_name, carry=m["carry"], deliveries=m["deliveries"])


if __name__ == "__main__":
    jobs = [(s, 42 + 100 * k, c, o) for s in CFGS_SCEN for k in range(3) for c, o in CONFIGS.items()]
    with ProcessPoolExecutor(max_workers=36) as ex:
        rows = list(ex.map(one, jobs, chunksize=1))
    out = os.path.join("scenarios", "analysis", "mechanism_ablation.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    df = pd.DataFrame(rows)
    print(df.pivot_table(index=["scenario", "config"], values=["carry", "deliveries"],
                         aggfunc="mean").round(3).to_string())
    print("ALL DONE")
