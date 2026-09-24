# -*- coding: utf-8 -*-
"""由 nodes.csv 重建 edges.csv, 使边流与节点流逐帧逐位置一致 (修复早期版本的半步错位)。
节点轨迹不受影响 (边只是输出, 不参与动力学)。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "scenarios")
R_INT = 10.0

for name in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, name)
    nodes_p = os.path.join(d, "ant_swarm_trajectory_data.csv")
    if not os.path.isfile(nodes_p):
        continue
    nd = pd.read_csv(nodes_p)
    T, N = nd["frame"].nunique(), nd["agent_id"].nunique()
    xs = nd["x"].values.reshape(T, N)
    ys = nd["y"].values.reshape(T, N)

    ef, es, et, ed = [], [], [], []
    for f in range(T):
        dx = xs[f][None, :] - xs[f][:, None]
        dy = ys[f][None, :] - ys[f][:, None]
        dist = np.hypot(dx, dy)
        iu = np.triu_indices(N, k=1)
        dvec = dist[iu]
        keep = dvec < R_INT
        if keep.any():
            s, t = iu[0][keep], iu[1][keep]
            ef.append(np.full(len(s), f)); es.append(s); et.append(t); ed.append(dvec[keep])

    out = pd.DataFrame({
        "frame": np.concatenate(ef).astype(np.int32),
        "source": np.concatenate(es).astype(np.int32),
        "target": np.concatenate(et).astype(np.int32),
        "distance": np.concatenate(ed).round(3),
    })
    out.to_csv(os.path.join(d, "ant_swarm_dynamic_edges.csv"), index=False)
    print(f"{name}: {len(out)} edges rebuilt", flush=True)

print("all done")
