# -*- coding: utf-8 -*-
"""
把 generate_scenarios.py 生成的各场景 CSV 批量转换为 ST-GNN 友好的 JSON。

约定 (与早期版本不同, 采用固定尺度归一化, 保证跨场景可比):
  - x, y   除以世界边长 100.0  -> [0, 1]
  - vx, vy 除以蚂蚁速度上限 1.0 (每帧单位) -> 大致 [-1, 1]
  - 输出为 T x N 矩阵 (T 帧, N 蚂蚁), 含 state (0/1 是否携食物)
  - 边不进入 JSON: 邻接可由坐标精确重建, 对任意帧 t 取
        A_t = {(i,j): i<j, ||p_i(t) - p_j(t)|| < R_int},  R_int = 10.0
    (与 nodes.csv 完全一致, 这正是数据集的审计性质)

用法:
    python3 convert_scenarios.py [--only a b]
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN_ROOT = os.path.join(HERE, "scenarios")

WORLD_EXTENT = 100.0   # 固定尺度, 勿按文件最大值归一化
SPEED_UNIT = 1.0
R_INT = 10.0


def convert_one(scene_dir):
    csv_path = os.path.join(scene_dir, "ant_swarm_trajectory_data.csv")
    if not os.path.exists(csv_path):
        print(f"  skip (no CSV): {scene_dir}")
        return

    df = pd.read_csv(csv_path)
    df = df.sort_values(by=["agent_id", "frame"])

    df["vx"] = df.groupby("agent_id")["x"].diff().fillna(0)
    df["vy"] = df.groupby("agent_id")["y"].diff().fillna(0)

    df["x"] = df["x"] / WORLD_EXTENT
    df["y"] = df["y"] / WORLD_EXTENT
    df["vx"] = df["vx"] / SPEED_UNIT
    df["vy"] = df["vy"] / SPEED_UNIT

    piv = lambda col: df.pivot(index="frame", columns="agent_id", values=col).fillna(0).values
    data = {k: piv(k).tolist() for k in ("x", "y", "vx", "vy")}
    data["state"] = df.pivot(index="frame", columns="agent_id", values="state").fillna(0).astype(int).values.tolist()

    out_path = os.path.join(scene_dir, "ant_swarm_trajectory_with_velocity.json")
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))  # compact
    print(f"  converted: {os.path.basename(scene_dir)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    for name in sorted(os.listdir(SCEN_ROOT)):
        scene_dir = os.path.join(SCEN_ROOT, name)
        if not os.path.isdir(scene_dir):
            continue
        if args.only and name not in args.only:
            continue
        convert_one(scene_dir)


if __name__ == "__main__":
    main()
