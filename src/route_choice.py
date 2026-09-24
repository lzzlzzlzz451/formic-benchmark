# -*- coding: utf-8 -*-
"""从 canonical 场景数据生成 analysis/route_choice.csv (路径选择统计工件, 与论文 Sec. IV-B 对应)"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "scenarios")

def crossing_ys(xs, ys, x_line=50.0):
    a = xs - x_line
    ti, ai = np.where(a[:-1] * a[1:] < 0)
    return 0.5 * (ys[ti, ai] + ys[ti + 1, ai])

rows = []
for name in ["wall_with_door", "central_vertical_wall", "ring_with_gaps", "side_food_detour"]:
    nd = pd.read_csv(os.path.join(ROOT, name, "ant_swarm_trajectory_data.csv"))
    T, N = nd["frame"].nunique(), nd["agent_id"].nunique()
    xs, ys = nd["x"].values.reshape(T, N), nd["y"].values.reshape(T, N)
    if name == "ring_with_gaps":
        r = np.hypot(xs - 50, ys - 50)
        ti, ai = np.where((r[:-1] - 22) * (r[1:] - 22) < 0)
        ang = np.degrees(np.arctan2(0.5*(ys[ti,ai]+ys[ti+1,ai])-50, 0.5*(xs[ti,ai]+xs[ti+1,ai])-50)) % 360
        d_top = np.minimum(np.abs(ang-72), 360-np.abs(ang-72)); d_bot = np.minimum(np.abs(ang-252), 360-np.abs(ang-252))
        designed = (d_top < 18) | (d_bot < 18)
        A, B, lab = int((~designed).sum()), int(designed.sum()), ("pillar_squeezes", "designed_gaps")
    else:
        y = crossing_ys(xs, ys)
        if name == "wall_with_door":
            A, B, lab = int(((y > 46) & (y < 60)).sum()), int((y <= 8).sum()), ("door", "bottom_gap")
        elif name == "central_vertical_wall":
            A, B, lab = int((y < 39.5).sum()), int((y > 84.5).sum()), ("bottom_endaround", "top_endaround")
        else:
            A, B, lab = int((y < 8).sum()), int((y > 36).sum()), ("below", "above")
    denom = max(A + B, 1)
    rows.append(dict(scenario=name, option_major=lab[0], count_major=A, share_major=round(A/denom, 3),
                     option_minor=lab[1], count_minor=B, share_minor=round(B/denom, 3)))
os.makedirs(os.path.join(ROOT, "analysis"), exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT, "analysis", "route_choice.csv"), index=False)
print("saved scenarios/analysis/route_choice.csv")
