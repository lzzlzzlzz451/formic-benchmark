# -*- coding: utf-8 -*-
"""从重建后的 nodes/edges CSV 重渲染 13 个 *_edges.gif, 使动画与数据完全一致。"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from generate_scenarios import Obstacle, NEST_RADIUS, FOOD_RADIUS  # noqa: E402

ROOT = os.path.join(HERE, "scenarios")
FPS, STRIDE = 20, 3

for name in sorted(os.listdir(ROOT)):
    d = os.path.join(ROOT, name)
    nodes_p = os.path.join(d, "ant_swarm_trajectory_data.csv")
    if not os.path.isfile(nodes_p):
        continue
    nd = pd.read_csv(nodes_p)
    ed = pd.read_csv(os.path.join(d, "ant_swarm_dynamic_edges.csv"))
    import json
    cfg = json.load(open(os.path.join(d, "scenario_config.json")))
    T, N = nd["frame"].nunique(), nd["agent_id"].nunique()
    xs = nd["x"].values.reshape(T, N)
    ys = nd["y"].values.reshape(T, N)
    st = nd["state"].values.reshape(T, N)
    ef = ed["frame"].values
    es, et = ed["source"].values, ed["target"].values
    obstacles = [Obstacle(s) for s in cfg["obstacles"]]
    nest, food = np.array(cfg["nest"]), np.array(cfg["food"])

    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    ax.add_patch(plt.Circle(nest, NEST_RADIUS, color="blue", alpha=0.2))
    ax.add_patch(plt.Circle(food, FOOD_RADIUS, color="green", alpha=0.2))
    for o in obstacles:
        ax.add_patch(o.patch())
    ax.plot(*nest, "b+", markersize=10); ax.plot(*food, "g+", markersize=10)
    scatter = ax.scatter([], [], s=12, zorder=5)
    lc = LineCollection([], colors="gray", linewidths=0.5, alpha=0.5, zorder=1)
    ax.add_collection(lc)

    def update(frame):
        pts = np.column_stack([xs[frame], ys[frame]])
        scatter.set_offsets(pts)
        scatter.set_color(np.where(st[frame] == 1, "red", "black"))
        lo = np.searchsorted(ef, frame, side="left")
        hi = np.searchsorted(ef, frame, side="right")
        segs = (np.stack([pts[es[lo:hi]], pts[et[lo:hi]]], axis=1)
                if hi > lo else np.empty((0, 2, 2)))
        lc.set_segments(segs)
        ax.set_title(f"{name} | Dynamic Edges - Frame {frame:04d}/{T-1:04d}")

    ani = animation.FuncAnimation(fig, update, frames=list(range(0, T, STRIDE)), interval=50)
    out = os.path.join(d, f"{name}_edges.gif")
    ani.save(out, writer="pillow", fps=FPS)
    plt.close(fig)
    print(f"{name}: edges GIF re-rendered", flush=True)

print("ALL DONE")
