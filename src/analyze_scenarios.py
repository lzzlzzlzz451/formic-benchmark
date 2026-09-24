# -*- coding: utf-8 -*-
"""
场景效果评估:
  1. 关键指标: 首次取食/首次投递时间、总投递次数、稳态携食比例、往返耗时
  2. 路径选择统计: 带门墙/竖墙/环形壁垒等场景中蚂蚁实际走了哪个缺口
  3. 可视化: 全程轨迹热力图 (路径成型效果) + 携食比例收敛曲线
输出到 scenarios/analysis/
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm

from generate_scenarios import build_scenarios, Obstacle, NEST_RADIUS, FOOD_RADIUS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "scenarios")
OUT = os.path.join(ROOT, "analysis")
os.makedirs(OUT, exist_ok=True)


def load_scenario(name):
    df = pd.read_csv(os.path.join(ROOT, name, "ant_swarm_trajectory_data.csv"))
    T, N = df["frame"].nunique(), df["agent_id"].nunique()
    # 行序为 frame 主序 (agent_id 0..N-1), 可直接 reshape
    assert (np.diff(df["frame"].values) >= 0).all(), "行序非 frame 主序"
    xs = df["x"].values.reshape(T, N)
    ys = df["y"].values.reshape(T, N)
    st = df["state"].values.reshape(T, N)
    cfg = json.load(open(os.path.join(ROOT, name, "scenario_config.json")))
    return xs, ys, st, cfg


def run_durations(st):
    """返回 (归巢段时长列表, 觅食段时长列表): 1-游程 = 携食归巢, 投递后到下次取食的 0-游程 = 外出觅食"""
    ret, out = [], []
    for col in range(st.shape[1]):
        s = st[:, col]
        d = np.diff(s)
        # 游程边界
        starts = np.where(d != 0)[0]
        bounds = np.concatenate([[-1], starts, [len(s) - 1]])
        # 重构游程序列: (值, 长度)
        runs = []
        prev = 0
        for b in bounds[1:]:
            seg_len = b - (bounds[np.searchsorted(bounds, b) - 1])
            runs.append((prev, seg_len))
            prev = 1 - prev
        # 逐游程配对: 1-游程计为归巢; 其后的 0-游程计为觅食
        for i, (val, ln) in enumerate(runs):
            if val == 1 and ln > 1:
                ret.append(ln)
                if i + 1 < len(runs):
                    ln2 = runs[i + 1][1]
                    if ln2 > 1:
                        out.append(ln2)
    return ret, out


def first_frame(mask_2d):
    idx = np.argwhere(mask_2d)
    return int(idx[:, 0].min()) if len(idx) else None


def crossing_ys(xs, ys, x_line):
    """统计穿越竖线 x=x_line 时的 y 值 (用于判断走了哪个缺口)"""
    a = xs - x_line
    cross = a[:-1] * a[1:] < 0
    ti, ai = np.where(cross)
    if len(ti) == 0:
        return np.zeros(0)
    y_cross = 0.5 * (ys[ti, ai] + ys[ti + 1, ai])
    return y_cross


def metrics_one(name):
    xs, ys, st, cfg = load_scenario(name)
    T = st.shape[0]
    d = np.diff(st, axis=0)
    deliveries = int((d == -1).sum())
    pickups = int((d == 1).sum())
    first_pickup = first_frame(st == 1)
    first_delivery = first_frame(d == -1)
    ret, out = run_durations(st)
    last = slice(T - 300, T)
    m = {
        "场景": name,
        "首次取食帧": first_pickup,
        "首次投递帧": first_delivery + 1 if first_delivery is not None else None,
        "总投递(趟)": deliveries,
        "稳态携食比例": round(float(st[last].mean()), 3),
        "归巢段耗时(帧)": round(float(np.mean(ret)), 1) if ret else None,
        "觅食段耗时(帧)": round(float(np.mean(out)), 1) if out else None,
    }
    return m, xs, ys, st, cfg


def main():
    rows, data = [], {}
    for cfg in build_scenarios():
        name = cfg["name"]
        m, xs, ys, st, c = metrics_one(name)
        rows.append(m)
        data[name] = (xs, ys, st, c)
        print(f"[{name}] 指标计算完成", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "metrics.csv"), index=False)
    print("\n===== 关键指标 =====")
    print(df.to_string(index=False))

    # ---- 特定场景的路径选择统计 ----
    print("\n===== 路径选择统计 =====")
    choices = {}

    # wall_with_door: x=50 处穿越高度 -> 门洞(48-58) vs 底部缺口(<8)
    ys_c = crossing_ys(*data["wall_with_door"][:2], 50.0)
    door = ((ys_c > 46) & (ys_c < 60)).sum()
    bottom = (ys_c <= 8).sum()
    others = len(ys_c) - door - bottom
    choices["wall_with_door"] = f"门洞 {door} 次 / 底部缺口 {bottom} 次 / 其他(沿墙折返) {others} 次"

    # central_vertical_wall: x=50 穿越高度 -> 上端(>84.5) vs 下端(<39.5)
    ys_c = crossing_ys(*data["central_vertical_wall"][:2], 50.0)
    top = (ys_c > 84.5).sum()
    bot = (ys_c < 39.5).sum()
    choices["central_vertical_wall"] = f"上端绕行 {top} 次 / 下端绕行 {bot} 次"

    # ring_with_gaps: 穿越 r=22 圆时的角度 -> 顶缺口72° / 底缺口252° / 柱间缝隙
    xs, ys = data["ring_with_gaps"][:2]
    r = np.hypot(xs - 50, ys - 50)
    cross = (r[:-1] - 22) * (r[1:] - 22) < 0
    ti, ai = np.where(cross)
    ang = np.degrees(np.arctan2(0.5 * (ys[ti, ai] + ys[ti + 1, ai]) - 50,
                                0.5 * (xs[ti, ai] + xs[ti + 1, ai]) - 50)) % 360
    # 缺口中心 72/252 ±18°, 其余视为柱间缝隙穿越
    d_top = np.minimum(np.abs(ang - 72), 360 - np.abs(ang - 72))
    d_bot = np.minimum(np.abs(ang - 252), 360 - np.abs(ang - 252))
    g_top, g_bot, squeeze = (d_top < 18).sum(), (d_bot < 18).sum(), ((d_top >= 18) & (d_bot >= 18)).sum()
    choices["ring_with_gaps"] = f"顶缺口 {g_top} / 底缺口 {g_bot} / 柱间缝隙挤过 {squeeze}"

    # side_food_detour: x=50 穿越高度 -> 上绕(>36) vs 下绕(<8)
    ys_c = crossing_ys(*data["side_food_detour"][:2], 50.0)
    above, below = (ys_c > 36).sum(), (ys_c < 8).sum()
    choices["side_food_detour"] = f"上方绕行 {above} / 下方绕行 {below}"

    # two_circles_slalom: x=45 与 x=55 处穿越高度 (看 S 形走位)
    slalom_txt = ""
    for xl in (45.0, 55.0):
        ys_c = crossing_ys(*data["two_circles_slalom"][:2], xl)
        if len(ys_c):
            slalom_txt += f"x={xl:.0f} 穿越高度中位数 {np.median(ys_c):.1f} (对角线 y={xl:.0f}); "
    choices["two_circles_slalom"] = slalom_txt

    for k, v in choices.items():
        print(f"  {k}: {v}")

    # ---- 轨迹热力图 ----
    cfgs = build_scenarios()
    fig, axes = plt.subplots(4, 4, figsize=(20, 20), dpi=90)
    for ax, cfg in zip(axes.ravel(), cfgs):
        name = cfg["name"]
        xs, ys, st, c = data[name]
        ax.hist2d(xs.ravel(), ys.ravel(), bins=100, range=[[0, 100], [0, 100]], norm=LogNorm(vmin=1))
        for spec in cfg["obstacles"]:
            ax.add_patch(Obstacle(spec).patch())
        nest, food = np.array(cfg["nest"]), np.array(cfg["food"])
        ax.add_patch(mpatches.Circle(nest, NEST_RADIUS, color="blue", alpha=0.5))
        ax.add_patch(mpatches.Circle(food, FOOD_RADIUS, color="lime", alpha=0.5))
        ax.set_title(name, fontsize=11)
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal")
    for ax in axes.ravel()[len(cfgs):]:
        ax.axis("off")
    fig.suptitle("Ant Visit Density (log scale) - all frames", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "path_heatmaps.png"), bbox_inches="tight")
    plt.close(fig)
    print("\n热力图已保存: analysis/path_heatmaps.png")

    # ---- 携食比例收敛曲线 ----
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)
    for cfg in cfgs:
        st = data[cfg["name"]][2]
        frac = st.mean(axis=1)
        ax.plot(np.convolve(frac, np.ones(50) / 50, mode="valid"), label=cfg["name"], lw=1.4)
    ax.set_xlabel("Frame (50-frame rolling mean)")
    ax.set_ylabel("Fraction of ants carrying food")
    ax.set_title("Foraging convergence across scenarios")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "carrying_fraction_over_time.png"))
    plt.close(fig)
    print("收敛曲线已保存: analysis/carrying_fraction_over_time.png")


if __name__ == "__main__":
    main()
