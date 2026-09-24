# -*- coding: utf-8 -*-
"""
多场景蚁群仿真数据生成器
==========================================
在 gener_ant.py 的基础上泛化：
  1. 障碍物支持任意数量 / 任意位置 / 多种形状
     - 圆形 (circle)
     - 凸多边形 (poly)：矩形墙、三角形、菱形、正 N 边形均由它派生
  2. 每个场景输出到 scenarios/<场景名>/ 下：
     - ant_swarm_trajectory_data.csv   节点状态 (frame, agent_id, x, y, state)
     - ant_swarm_dynamic_edges.csv     动态网络边 (frame, source, target, distance)
     - <场景名>_swarm.gif               基础点位动画
     - <场景名>_edges.gif               动态边网络动画
     - scenario_config.json            场景配置 (障碍物/参数元数据)
  3. 蚂蚁行为逻辑 (信息素觅食、归巢、防拥挤、边界反弹、避障) 与原版保持一致

用法:
    python3 generate_scenarios.py                 # 跑全部场景
    python3 generate_scenarios.py --only a b      # 只跑指定场景
    python3 generate_scenarios.py --steps 60 --stride 1 --out-root /tmp/smoke   # 冒烟测试
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import matplotlib.animation as animation

# ==========================================
# 全局仿真参数 (与原版 gener_ant.py 一致)
# ==========================================
BOUND_MIN = 0.0
BOUND_MAX = 100.0
FOOD_RADIUS = 6.0
NEST_RADIUS = 6.0

ANT_SPEED = 1.0
NOISE_WEIGHT = 0.3
PHEROMONE_WEIGHT = 0.8
INERTIA_WEIGHT = 0.5
SEPARATION_WEIGHT = 0.6

PHEROMONE_DECAY = 0.985
PHEROMONE_DROP_RATE = 0.25
PHEROMONE_SENSE_RAD = 18.0
PHERO_CAP = 8000            # 信息素点数上限 (防止 sensing 矩阵过大, 超限淘汰强度最低者)

INTERACTION_RADIUS = 10.0   # 动态网络连边半径
REPULSE_DIST = 2.0          # 防拥挤距离 (接触式)
EXPLORE_BASE = 0.05         # 基础探索概率 (真实蚁群总有部分个体不跟队侦察)
EXPLORE_GAIN = 0.004        # 每个感知半径内邻居增加的探索概率 (拥挤触发再探索, 缓解路线锁定)
EXPLORE_CAP = 0.5           # 探索概率上限
SOFT_SEP_RAD = 5.0          # 前瞻性软避让半径 (看到前方拥挤提前侧移, 使拥堵可消散)
SOFT_SEP_GAIN = 0.08        # 软避让增益 (单位向量求和, 温和)
EXPLORE_DIFFUSE = 0.35     # 探索者扩散偏置 (向同伴稀疏方向偏移, 促成分流与再平衡)

STEER_MARGIN = 4.0          # 障碍物软斥力生效的表面外缓冲带宽度
STEER_STRENGTH = 0.15       # 软斥力强度 (主要靠硬推挤出面 + 沿面滑动绕行)

GIF_FPS = 20
GIF_STRIDE = 3              # 每 3 帧渲染 1 帧 (数据仍全量记录)


# ==========================================
# 障碍物几何
# ==========================================
def circle_obstacle(center, radius):
    return {"type": "circle", "center": [float(center[0]), float(center[1])], "radius": float(radius)}


def rect_obstacle(cx, cy, w, h, angle=0.0):
    """矩形障碍 (可旋转, angle 单位为度), 内部转成凸多边形顶点 (逆时针 CCW)"""
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    th = np.radians(angle)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    verts = local @ R.T + np.array([cx, cy])
    return {"type": "poly", "vertices": verts.tolist()}


def polygon_obstacle(cx, cy, r, n, rot=0.0):
    """正 n 边形障碍, r 为外接圆半径, rot 为旋转角 (弧度)"""
    ang = rot + np.linspace(0, 2 * np.pi, n, endpoint=False)
    verts = np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])
    return {"type": "poly", "vertices": verts.tolist()}


class Obstacle:
    """统一障碍物接口: closest(pts) 返回 (最近表面点 cp, 带符号距离 sdist), sdist<0 表示点在障碍内部"""

    def __init__(self, spec):
        self.spec = spec
        if spec["type"] == "circle":
            self.center = np.array(spec["center"], dtype=float)
            self.radius = float(spec["radius"])
        elif spec["type"] == "poly":
            self.verts = np.array(spec["vertices"], dtype=float)
        else:
            raise ValueError(f"未知障碍物类型: {spec['type']}")

    def closest(self, pts):
        pts = np.atleast_2d(pts)
        if self.spec["type"] == "circle":
            v = pts - self.center
            d = np.linalg.norm(v, axis=1)
            d_safe = np.where(d < 1e-9, 1.0, d)
            cp = self.center + v / d_safe[:, None] * self.radius
            return cp, d - self.radius

        V = self.verts
        n_pts = pts.shape[0]
        best_d2 = np.full(n_pts, np.inf)
        cp = pts.copy()
        inside = np.ones(n_pts, dtype=bool)
        E = len(V)
        for k in range(E):
            a, b = V[k], V[(k + 1) % E]
            ab = b - a
            ab2 = float(ab @ ab)
            if ab2 <= 0:
                continue
            t = np.clip(((pts - a) @ ab) / ab2, 0.0, 1.0)
            q = a + t[:, None] * ab
            d2 = np.einsum("ij,ij->i", pts - q, pts - q)
            m = d2 < best_d2
            best_d2[m] = d2[m]
            cp[m] = q[m]
            # 逆时针多边形: 内部点对所有边的叉积 >= 0
            cross = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
            inside &= cross >= 0
        sdist = np.sqrt(best_d2)
        sdist[inside] = -sdist[inside]
        return cp, sdist

    def patch(self):
        if self.spec["type"] == "circle":
            return mpatches.Circle(self.center, self.radius, color="dimgray", alpha=0.6)
        return mpatches.Polygon(self.verts, closed=True, facecolor="dimgray", edgecolor="none", alpha=0.6)


def push_out_of_obstacles(obstacles, positions):
    """硬碰撞: 把陷入障碍物内部的点推到表面外一点点 (防穿模)"""
    for obs in obstacles:
        cp, sd = obs.closest(positions)
        m = sd < 0
        if m.any():
            p = positions[m]
            c = cp[m]
            depth = -sd[m]
            v = c - p  # 从内部点指向最近表面点
            vn = np.linalg.norm(v, axis=1)
            vn[vn == 0] = 1.0
            positions[m] = p + v / vn[:, None] * (depth + 0.05)[:, None]
    return positions


def random_circles(seed, count=8, area=(15.0, 85.0)):
    """种子化随机撒圆形障碍: 避开巢穴/食物, 彼此留出间隙"""
    rng = np.random.default_rng(seed)
    nest, food = np.array([15.0, 15.0]), np.array([85.0, 85.0])
    obs, tries = [], 0
    while len(obs) < count and tries < 20000:
        tries += 1
        r = rng.uniform(4.0, 10.0)
        c = rng.uniform(area[0], area[1], 2)
        if np.linalg.norm(c - nest) < r + 11 or np.linalg.norm(c - food) < r + 11:
            continue
        if any(np.linalg.norm(c - np.array(o["center"])) < r + o["radius"] + 5.0 for o in obs):
            continue
        obs.append(circle_obstacle(c, r))
    return obs


# ==========================================
# 场景定义
# ==========================================
def build_scenarios():
    S = []

    # 1. 基线: 单个大圆 (偏心, 复现原版布局)
    S.append(dict(name="single_circle", seed=42,
                  obstacles=[circle_obstacle((55, 60), 20)]))

    # 2. 双圆回转: 沿对角线两侧各一个, 走 S 形绕行
    S.append(dict(name="two_circles_slalom", seed=43,
                  obstacles=[circle_obstacle((35, 55), 12), circle_obstacle((65, 35), 12)]))

    # 3. 五圆散布: 多障碍遮挡多条路径
    S.append(dict(name="five_circles_scatter", seed=44,
                  obstacles=[circle_obstacle((30, 40), 8), circle_obstacle((58, 58), 9),
                             circle_obstacle((75, 30), 7), circle_obstacle((35, 72), 6),
                             circle_obstacle((68, 78), 8)]))

    # 4. 中央竖墙: 只有上下两端能绕过
    S.append(dict(name="central_vertical_wall", seed=45,
                  obstacles=[rect_obstacle(50, 62, 4, 45)]))

    # 5. 带门的墙: 蚂蚁必须从中间的门洞挤过去 (漏斗效应/拥塞)
    S.append(dict(name="wall_with_door", seed=46, num_ants=300,
                  obstacles=[rect_obstacle(50, 28, 4, 40), rect_obstacle(50, 79, 4, 42)]))

    # 6. S 形迷宫: 两道错位横墙, 强制大绕行
    S.append(dict(name="s_shape_maze", seed=47,
                  obstacles=[rect_obstacle(35, 35, 50, 4), rect_obstacle(65, 65, 50, 4)]))

    # 7. 中央大三角形 (平顶朝上)
    S.append(dict(name="triangle_center", seed=48,
                  obstacles=[polygon_obstacle(50, 52, 24, 3, -np.pi / 2)]))

    # 8. 中央菱形 (旋转 45° 的正方形)
    S.append(dict(name="diamond_center", seed=49,
                  obstacles=[polygon_obstacle(52, 50, 22, 4, 0.0)]))

    # 9. 混合形状: 圆 + 矩形 + 三角形
    S.append(dict(name="mixed_shapes", seed=50,
                  obstacles=[circle_obstacle((30, 60), 10), rect_obstacle(62, 30, 18, 6),
                             polygon_obstacle(55, 70, 12, 3, -np.pi / 2)]))

    # 10. 柱林: 3x3 小圆柱阵列, 蚂蚁在柱间穿行
    S.append(dict(name="pillar_grid", seed=51, num_ants=200,
                  obstacles=[circle_obstacle((x, y), 5) for x in (35, 50, 65) for y in (35, 50, 65)]))

    # 11. 破缺环形壁垒: 小柱围成一圈, 顶部/底部各留一个缺口
    ring_obs = []
    for i in range(10):
        if i in (2, 7):  # 72° 与 252° 处留缺口
            continue
        a = np.radians(i * 36.0)
        ring_obs.append(circle_obstacle((50 + 22 * np.cos(a), 50 + 22 * np.sin(a)), 4))
    S.append(dict(name="ring_with_gaps", seed=52, obstacles=ring_obs))

    # 12. 侧向食物: 食物放到右侧偏下, 大圆挡住直线路径
    S.append(dict(name="side_food_detour", seed=53,
                  nest=(15, 15), food=(85, 20),
                  obstacles=[circle_obstacle((50, 22), 14)]))

    # 13. 随机散布 (固定种子, 可复现)
    S.append(dict(name="random_scatter", seed=54, num_ants=200,
                  obstacles=random_circles(seed=2026, count=8)))

    # 14. 斜置墙: 与竖直方向倾斜 -15° 的墙, 两端绕行距离不对称 (用到 rect 的可旋转能力)
    S.append(dict(name="angled_wall", seed=55,
                  obstacles=[rect_obstacle(52, 48, 5, 34, -15)]))

    # 15. 正六边形: 正 n 边形族中未出现的边数 (3/4 已有 triangle/diamond)
    S.append(dict(name="hexagon_center", seed=56,
                  obstacles=[polygon_obstacle(51, 50, 21, 6, np.pi / 6)]))

    # 16. 窄走廊: 双平行墙形成 12 宽通道, 通道流/漏斗拥塞
    S.append(dict(name="narrow_corridor", seed=57,
                  obstacles=[rect_obstacle(50, 42, 52, 4), rect_obstacle(50, 58, 52, 4)]))

    # 公共默认值
    for cfg in S:
        cfg.setdefault("num_ants", 250)
        cfg.setdefault("num_steps", 1200)
        cfg.setdefault("nest", (15.0, 15.0))
        cfg.setdefault("food", (85.0, 85.0))
    return S


def validate_scenario(cfg):
    """确保巢穴/食物不被障碍物覆盖, 出生区域安全"""
    nest = np.array(cfg["nest"], dtype=float)
    food = np.array(cfg["food"], dtype=float)
    pad = NEST_RADIUS + 2.0
    for spec in cfg["obstacles"]:
        obs = Obstacle(spec)
        for p, label in [(nest, "巢穴"), (food, "食物")]:
            _, sd = obs.closest(p[None, :])
            if sd[0] < pad:
                raise ValueError(f"场景 {cfg['name']}: 障碍物 {spec} 距离{label}太近 (sdist={sd[0]:.2f})")


# ==========================================
# 仿真核心 (向量化, 逻辑对齐 gener_ant.py)
# ==========================================
def simulate(cfg, verbose=True):
    rng = np.random.default_rng(cfg["seed"])
    N = cfg["num_ants"]
    T = cfg["num_steps"]
    nest = np.array(cfg["nest"], dtype=float)
    food = np.array(cfg["food"], dtype=float)
    obstacles = [Obstacle(s) for s in cfg["obstacles"]]

    positions = np.clip(rng.normal(nest, 3.0, size=(N, 2)), BOUND_MIN + 0.5, BOUND_MAX - 0.5)
    positions = push_out_of_obstacles(obstacles, positions)
    angles = rng.uniform(0, 2 * np.pi, N)
    has_food = np.zeros(N, dtype=bool)
    phero = np.zeros((0, 3))  # 每行 [x, y, strength]

    node_chunks = []   # 每帧 (frame, ids, x, y, state)
    edge_chunks = []   # 每帧 (frame, src, tgt, dist)

    for step in range(T):
        d_food = np.linalg.norm(positions - food, axis=1)
        d_nest = np.linalg.norm(positions - nest, axis=1)
        has_food[d_food < FOOD_RADIUS] = True
        has_food[d_nest < NEST_RADIUS] = False

        # ---- 成对距离 (移动前位置; 供拥挤度统计/防拥挤/边提取基准共用) ----
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        neigh = (dists < PHEROMONE_SENSE_RAD).sum(axis=1)  # 感知半径内邻居数 = 拥挤度

        base = np.zeros((N, 2))

        # ---- 携带食物者: 直奔巢穴 (带角度噪声), 并按概率释放信息素 ----
        carry = has_food.copy()
        if carry.any():
            vec = nest - positions[carry]
            nv = np.linalg.norm(vec, axis=1)
            nv[nv == 0] = 1.0
            dirv = vec / nv[:, None]
            th = rng.uniform(-0.4, 0.4, size=carry.sum())
            c, s = np.cos(th), np.sin(th)
            rot = dirv.copy()
            rot[:, 0] = c * dirv[:, 0] - s * dirv[:, 1]
            rot[:, 1] = s * dirv[:, 0] + c * dirv[:, 1]
            base[carry] = rot
            angles[carry] = np.arctan2(rot[:, 1], rot[:, 0])
            drop = rng.random(carry.sum()) < PHEROMONE_DROP_RATE
            if drop.any():
                phero = np.vstack([phero, np.column_stack([positions[carry][drop], np.ones(drop.sum())])])

        # ---- 觅食者: 感知并跟随信息素 ----
        free = ~carry
        phero_dir = np.zeros((N, 2))
        has_pdir = np.zeros(N, dtype=bool)
        if len(phero) > 0:
            dx = phero[None, :, 0] - positions[:, None, 0]
            dy = phero[None, :, 1] - positions[:, None, 1]
            dist = np.hypot(dx, dy)
            d_ph_nest = np.linalg.norm(phero[:, :2] - nest, axis=1)
            # 只跟随比自己更远离巢穴、且不在巢穴内部的信息素 (指向食物方向的踪迹)
            mask = (dist < PHEROMONE_SENSE_RAD) & \
                   (d_ph_nest[None, :] > d_nest[:, None]) & \
                   (d_ph_nest[None, :] > NEST_RADIUS)
            w = phero[:, 2][None, :] * np.where(mask, 1.0 / np.maximum(dist, 1e-3), 0.0)
            gx = (dx * w).sum(axis=1)
            gy = (dy * w).sum(axis=1)
            gn = np.hypot(gx, gy)
            has_pdir = gn > 1e-9
            phero_dir[has_pdir, 0] = gx[has_pdir] / gn[has_pdir]
            phero_dir[has_pdir, 1] = gy[has_pdir] / gn[has_pdir]

        # 拥挤触发的再探索: 周围越挤越倾向不跟队 (侦察/分流, 缓解路线锁定与门洞死堵)
        explore_p = np.clip(EXPLORE_BASE + EXPLORE_GAIN * neigh, 0.0, EXPLORE_CAP)
        explore = rng.random(N) < explore_p
        follow = free & has_pdir & ~explore
        if follow.any():
            ra = rng.uniform(0, 2 * np.pi, N)
            rand_dir = np.column_stack([np.cos(ra), np.sin(ra)])
            cur_dir = np.column_stack([np.cos(angles), np.sin(angles)])
            move = PHEROMONE_WEIGHT * phero_dir + NOISE_WEIGHT * rand_dir + INERTIA_WEIGHT * cur_dir
            mn = np.linalg.norm(move, axis=1)
            mn[mn == 0] = 1.0
            move = move / mn[:, None]
            base[follow] = move[follow]
            angles[follow] = np.arctan2(base[follow, 1], base[follow, 0])

        wander = free & ~follow
        if wander.any():
            angles[wander] += rng.uniform(-0.5, 0.5, size=wander.sum())
            wdir = np.column_stack([np.cos(angles[wander]), np.sin(angles[wander])])
            # 扩散偏置: 指向同伴稀疏方向的单位向量 (拥挤主路上的探索者被推向另一侧)
            near_s = dists < PHEROMONE_SENSE_RAD
            caway = np.sum(diffs / np.maximum(dists, 1e-3)[..., None] * near_s[..., None], axis=1)
            cn = np.linalg.norm(caway, axis=1); cn[cn == 0] = 1.0
            caway = caway / cn[:, None]
            wmove = wdir + EXPLORE_DIFFUSE * caway[wander]
            wn = np.linalg.norm(wmove, axis=1); wn[wn == 0] = 1.0
            base[wander] = wmove / wn[:, None]
            angles[wander] = np.arctan2(base[wander, 1], base[wander, 0])

        # ---- 防拥挤: 接触式硬避让 (r<2) + 前瞻性软避让 (2<=r<5, 单位向量温和侧移) ----
        repulse = dists < REPULSE_DIST
        separation = np.sum(diffs / (dists ** 2 + 0.001)[..., None] * repulse[..., None], axis=1)
        soft = (dists >= REPULSE_DIST) & (dists < SOFT_SEP_RAD)
        separation = separation + SOFT_SEP_GAIN * np.sum(
            diffs / np.maximum(dists, 1e-3)[..., None] * soft[..., None], axis=1)

        # ---- 障碍物软斥力 (转向) ----
        steer = np.zeros((N, 2))
        for obs in obstacles:
            cp, sd = obs.closest(positions)
            m = (sd >= 0) & (sd < STEER_MARGIN)
            if m.any():
                away = positions[m] - cp[m]
                an = np.linalg.norm(away, axis=1)
                an[an == 0] = 1.0
                k = STEER_STRENGTH * (1.0 - sd[m] / STEER_MARGIN)
                steer[m] += away / an[:, None] * k[:, None]

        final = base + SEPARATION_WEIGHT * separation + steer
        fn = np.linalg.norm(final, axis=1)
        fn[fn == 0] = 1.0
        positions += final / fn[:, None] * ANT_SPEED

        # ---- 边界反弹 ----
        hit_x = (positions[:, 0] < BOUND_MIN) | (positions[:, 0] > BOUND_MAX)
        positions[:, 0] = np.clip(positions[:, 0], BOUND_MIN, BOUND_MAX)
        angles[hit_x] = np.pi - angles[hit_x]
        hit_y = (positions[:, 1] < BOUND_MIN) | (positions[:, 1] > BOUND_MAX)
        positions[:, 1] = np.clip(positions[:, 1], BOUND_MIN, BOUND_MAX)
        angles[hit_y] = -angles[hit_y]
        angles = angles % (2 * np.pi)

        # ---- 障碍物硬推挤 ----
        positions = push_out_of_obstacles(obstacles, positions)

        # ---- 动态网络提取 (与节点记录使用同一组最终位置, 保证 edges.csv 可由 nodes.csv 复算) ----
        diffs_f = positions[:, None, :] - positions[None, :, :]
        dists_f = np.linalg.norm(diffs_f, axis=2)
        inter = dists_f < INTERACTION_RADIUS
        src, tgt = np.where(inter)
        keep = src < tgt  # 无向图只记一次
        if keep.any():
            edge_chunks.append((step, src[keep], tgt[keep], dists_f[src[keep], tgt[keep]]))

        # ---- 信息素挥发 ----
        if len(phero) > 0:
            phero[:, 2] *= PHEROMONE_DECAY
            phero = phero[phero[:, 2] > 0.05]
            if len(phero) > PHERO_CAP:
                order = np.argsort(phero[:, 2])
                phero = phero[order[len(phero) - PHERO_CAP:]]

        # ---- 记录节点状态 ----
        node_chunks.append((np.full(N, step), np.arange(N),
                            positions[:, 0].round(3), positions[:, 1].round(3),
                            has_food.astype(int)))

        if verbose and (step + 1) % 200 == 0:
            print(f"    仿真进度 {step + 1}/{T}", flush=True)

    # 汇总为连续数组
    frames = np.concatenate([c[0] for c in node_chunks])
    ids = np.concatenate([c[1] for c in node_chunks])
    xs = np.concatenate([c[2] for c in node_chunks]).reshape(T, N)
    ys = np.concatenate([c[3] for c in node_chunks]).reshape(T, N)
    states = np.concatenate([c[4] for c in node_chunks]).reshape(T, N)

    if edge_chunks:
        ef = np.concatenate([np.full(len(c[1]), c[0]) for c in edge_chunks])
        es = np.concatenate([c[1] for c in edge_chunks]).astype(np.int32)
        et = np.concatenate([c[2] for c in edge_chunks]).astype(np.int32)
        ed = np.concatenate([c[3] for c in edge_chunks]).round(3)
    else:
        ef = es = et = ed = np.zeros(0)

    return dict(frames=frames, ids=ids, xs=xs, ys=ys, states=states,
                ef=ef, es=es, et=et, ed=ed)


# ==========================================
# GIF 渲染
# ==========================================
def _base_figure(cfg, obstacles):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=80)
    ax.set_xlim(BOUND_MIN, BOUND_MAX)
    ax.set_ylim(BOUND_MIN, BOUND_MAX)
    ax.set_aspect("equal")
    nest, food = np.array(cfg["nest"]), np.array(cfg["food"])
    ax.add_patch(mpatches.Circle(nest, NEST_RADIUS, color="blue", alpha=0.2))
    ax.add_patch(mpatches.Circle(food, FOOD_RADIUS, color="green", alpha=0.2))
    for obs in obstacles:
        ax.add_patch(obs.patch())
    ax.plot(*nest, "b+", markersize=10)
    ax.plot(*food, "g+", markersize=10)
    return fig, ax


def render_gif(out_path, cfg, sim, obstacles, mode, stride, fps):
    """mode: 'swarm' 基础点位动画 / 'edges' 动态网络动画"""
    xs, ys, states = sim["xs"], sim["ys"], sim["states"]
    ef, es, et = sim["ef"], sim["es"], sim["et"]
    T = xs.shape[0]
    name = cfg["name"]

    fig, ax = _base_figure(cfg, obstacles)
    scatter = ax.scatter([], [], s=12, zorder=5)
    lc = None
    if mode == "edges":
        lc = LineCollection([], colors="gray", linewidths=0.5, alpha=0.5, zorder=1)
        ax.add_collection(lc)

    def update(frame):
        pts = np.column_stack([xs[frame], ys[frame]])
        scatter.set_offsets(pts)
        scatter.set_color(np.where(states[frame] == 1, "red", "black"))
        if lc is not None:
            lo = np.searchsorted(ef, frame, side="left")
            hi = np.searchsorted(ef, frame, side="right")
            if hi > lo:
                s, t = es[lo:hi], et[lo:hi]
                segs = np.stack([pts[s], pts[t]], axis=1)
            else:
                segs = np.empty((0, 2, 2))
            lc.set_segments(segs)
        label = "Dynamic Edges" if mode == "edges" else "Flexible Swarm Dynamics"
        ax.set_title(f"{name} | {label} - Frame {frame:04d}/{T - 1:04d}")

    frames_to_render = list(range(0, T, stride))
    ani = animation.FuncAnimation(fig, update, frames=frames_to_render, interval=50)
    ani.save(out_path, writer="pillow", fps=fps)
    plt.close(fig)


# ==========================================
# 单场景完整流程
# ==========================================
def run_scenario(cfg, out_root, stride=GIF_STRIDE, fps=GIF_FPS):
    validate_scenario(cfg)
    out_dir = os.path.join(out_root, cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    obstacles = [Obstacle(s) for s in cfg["obstacles"]]

    t0 = time.time()
    print(f"[{cfg['name']}] 开始仿真: {cfg['num_ants']} 蚂蚁 x {cfg['num_steps']} 帧, "
          f"{len(cfg['obstacles'])} 个障碍物", flush=True)
    sim = simulate(cfg)

    # 1. 节点 CSV
    df_nodes = pd.DataFrame({
        "frame": sim["frames"], "agent_id": sim["ids"],
        "x": sim["xs"].ravel(), "y": sim["ys"].ravel(), "state": sim["states"].ravel(),
    })
    nodes_path = os.path.join(out_dir, "ant_swarm_trajectory_data.csv")
    df_nodes.to_csv(nodes_path, index=False)

    # 2. 动态边 CSV
    df_edges = pd.DataFrame({
        "frame": sim["ef"], "source": sim["es"], "target": sim["et"], "distance": sim["ed"],
    })
    edges_path = os.path.join(out_dir, "ant_swarm_dynamic_edges.csv")
    df_edges.to_csv(edges_path, index=False)

    t1 = time.time()
    print(f"[{cfg['name']}] 仿真+CSV 完成 ({t1 - t0:.0f}s): 节点 {len(df_nodes)} 行, "
          f"边 {len(df_edges)} 行", flush=True)

    # 3. 两个 GIF
    render_gif(os.path.join(out_dir, f"{cfg['name']}_swarm.gif"), cfg, sim, obstacles, "swarm", stride, fps)
    render_gif(os.path.join(out_dir, f"{cfg['name']}_edges.gif"), cfg, sim, obstacles, "edges", stride, fps)

    # 4. 场景配置元数据
    meta = {
        "name": cfg["name"],
        "seed": cfg["seed"],
        "num_ants": cfg["num_ants"],
        "num_steps": cfg["num_steps"],
        "nest": list(cfg["nest"]),
        "food": list(cfg["food"]),
        "obstacles": cfg["obstacles"],
        "params": {
            "bound": [BOUND_MIN, BOUND_MAX], "ant_speed": ANT_SPEED,
            "pheromone_decay": PHEROMONE_DECAY, "pheromone_drop_rate": PHEROMONE_DROP_RATE,
            "pheromone_sense_radius": PHEROMONE_SENSE_RAD, "interaction_radius": INTERACTION_RADIUS,
            "separation_weight": SEPARATION_WEIGHT,
            "explore_base": EXPLORE_BASE, "explore_gain": EXPLORE_GAIN, "explore_cap": EXPLORE_CAP,
            "soft_sep_radius": SOFT_SEP_RAD, "soft_sep_gain": SOFT_SEP_GAIN,
            "explore_diffuse": EXPLORE_DIFFUSE,
            "gif_stride": stride, "gif_fps": fps,
        },
    }
    with open(os.path.join(out_dir, "scenario_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[{cfg['name']}] 全部完成, 总耗时 {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description="多场景蚁群仿真数据生成器")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定场景 (按名称)")
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "scenarios"))
    ap.add_argument("--steps", type=int, default=None, help="覆盖仿真步数 (冒烟测试用)")
    ap.add_argument("--stride", type=int, default=GIF_STRIDE, help="GIF 渲染跳帧间隔")
    ap.add_argument("--fps", type=int, default=GIF_FPS)
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    scenarios = build_scenarios()

    index_rows = []
    for cfg in scenarios:
        if args.only and cfg["name"] not in args.only:
            continue
        if args.steps:
            cfg["num_steps"] = args.steps
        run_scenario(cfg, args.out_root, stride=args.stride, fps=args.fps)
        index_rows.append({
            "name": cfg["name"], "num_ants": cfg["num_ants"], "num_steps": cfg["num_steps"],
            "num_obstacles": len(cfg["obstacles"]),
            "obstacle_types": ",".join(sorted({o["type"] for o in cfg["obstacles"]})),
            "nest_x": cfg["nest"][0], "nest_y": cfg["nest"][1],
            "food_x": cfg["food"][0], "food_y": cfg["food"][1],
            "seed": cfg["seed"],
        })

    if index_rows:
        idx_path = os.path.join(args.out_root, "index.csv")
        df_idx = pd.DataFrame(index_rows)
        if os.path.exists(idx_path):  # --only 局部重跑时与既有索引合并
            old = pd.read_csv(idx_path)
            df_idx = pd.concat([old[~old["name"].isin(df_idx["name"])], df_idx], ignore_index=True)
        df_idx.to_csv(idx_path, index=False)
        print(f"\n全部完成! 输出目录: {args.out_root}", flush=True)


if __name__ == "__main__":
    main()
