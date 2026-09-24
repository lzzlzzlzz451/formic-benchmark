# -*- coding: utf-8 -*-
"""
轨迹预测基线实验 v2 (论文实验节):
  任务: 给定截至 t 帧的节点状态与交互图, 预测 t+h 帧位置 (h=1..20 滚动预测)
  模型: persistence / CV / ST-GNN (边特征注意力消息传递)
  v2 增强 (回应审稿):
    - ST-GNN 用 3 个训练种子, 报 mean +- std
    - 每个测试场景用 3 条独立种子轨迹评测 (canonical + 2 条额外 episode)
    - 深度超参扫描 (depth 1/3) 验证结论稳健性
  划分: within (canonical 前 900 帧训练, 三条轨迹的后段测试)
        transfer (8 训练场景 -> 4 代表场景)
输出: scenarios/analysis/baseline_results.csv
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "scenarios")
OUT = os.path.join(ROOT, "analysis")
os.makedirs(OUT, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
R_INT = 10.0
WORLD = 100.0
TRAIN_SEEDS = (0, 1, 2)
HORIZONS = [1, 5, 10, 20]
TEST_STEP = 3  # 测试起始帧抽样间隔

REP_SCEN = ["single_circle", "two_circles_slalom", "wall_with_door", "s_shape_maze"]
TRAIN_SCEN = ["mixed_shapes", "five_circles_scatter", "pillar_grid", "random_scatter",
              "diamond_center", "triangle_center", "ring_with_gaps", "side_food_detour"]


def load_arrays(nd):
    T, N = nd["frame"].nunique(), nd["agent_id"].nunique()
    pos = np.concatenate([nd["x"].values.reshape(T, N, 1) / WORLD,
                          nd["y"].values.reshape(T, N, 1) / WORLD], axis=2).astype(np.float32)
    st = nd["state"].values.reshape(T, N).astype(np.float32)
    return pos, st


def load_episodes(name):
    """[(episode_id, pos, st), ...]: canonical(0) + 额外种子轨迹"""
    eps = [(0, *load_arrays(pd.read_csv(os.path.join(ROOT, name, "ant_swarm_trajectory_data.csv"))))]
    ep_dir = os.path.join(ROOT, "eval_episodes")
    if os.path.isdir(ep_dir):
        for f in sorted(os.listdir(ep_dir)):
            if f.startswith(name + "_s") and f.endswith(".csv"):
                seed = int(re.search(r"_s(\d+)\.csv$", f).group(1))
                eps.append((seed, *load_arrays(pd.read_csv(os.path.join(ep_dir, f)))))
    return eps


class EdgeAttnLayer(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.edge_mlp = nn.Sequential(nn.Linear(3, 32), nn.GELU(), nn.Linear(32, heads))
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, h, pos, mask):
        B, N, D = h.shape
        rel = pos[:, :, None, :] - pos[:, None, :, :]
        dist = rel.norm(dim=-1, keepdim=True)
        e = self.edge_mlp(torch.cat([rel, dist], dim=-1)).permute(0, 3, 1, 2)  # (B,H,N,N)
        q = self.q(h).view(B, N, self.h, -1).permute(0, 2, 1, 3)
        k = self.k(h).view(B, N, self.h, -1).permute(0, 2, 1, 3)
        v = self.v(h).view(B, N, self.h, -1).permute(0, 2, 1, 3)
        att = (q @ k.transpose(-1, -2)) / np.sqrt(q.shape[-1]) + e
        pair_ok = (mask[:, None, :, None] & mask[:, None, None, :]).squeeze(1)
        adj = (dist.squeeze(-1) <= R_INT / WORLD) & pair_ok
        adj = adj | torch.eye(N, device=h.device, dtype=torch.bool)[None]  # 自环防空行 NaN
        adj = adj[:, None, :, :].expand(-1, self.h, -1, -1)
        att = att.masked_fill(~adj, float("-inf")).softmax(dim=-1)
        out = (att @ v).permute(0, 2, 1, 3).reshape(B, N, D)
        return self.ff(self.norm(h + out)) + self.norm(h + out)


class STGNN(nn.Module):
    def __init__(self, din=7, dim=64, depth=2):
        super().__init__()
        self.enc = nn.Linear(din, dim)
        self.layers = nn.ModuleList([EdgeAttnLayer(dim) for _ in range(depth)])
        self.dec = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, feats, pos, mask):
        h = Fn.gelu(self.enc(feats))
        for lyr in self.layers:
            h = lyr(h, pos, mask)
        return self.dec(h)


def make_feats(pos_t, v1, v2, st_t):
    return np.concatenate([pos_t, v1, v2, st_t[:, None]], axis=1)


def rollout(model, pos, st, frames, horizons, mode):
    errs = {h: [] for h in horizons}
    for t in frames:
        x, x_prev, x_prev2 = pos[t].copy(), pos[t - 1].copy(), pos[t - 2].copy()
        v_cv = x - x_prev
        for h in horizons:
            if mode == "persistence":
                pred = pos[t]
            elif mode == "cv":
                pred = np.clip(x + h * v_cv, 0, 1)
            else:
                cur, prev, prev2 = x.copy(), x_prev.copy(), x_prev2.copy()
                for _ in range(h):
                    f = make_feats(cur, cur - prev, prev - prev2, st[t])
                    with torch.no_grad():
                        vh = model(torch.tensor(f[None], device=DEV),
                                   torch.tensor(cur[None], device=DEV),
                                   torch.ones(1, cur.shape[0], device=DEV, dtype=torch.bool))[0].cpu().numpy()
                    new = np.clip(cur + vh, 0, 1)
                    prev2, prev, cur = prev, cur, new
                pred = cur
            errs[h].append(np.linalg.norm(pred - pos[t + h], axis=1).mean())
    return {h: float(np.mean(v)) * WORLD for h, v in errs.items()}


def test_frames(step=TEST_STEP):
    return list(range(900, 1200 - max(HORIZONS), step))


def train_model(scen_list, tseed=0, depth=2, epochs=25, bs=32, log=True):
    torch.manual_seed(tseed)
    data = {s: load_arrays(pd.read_csv(os.path.join(ROOT, s, "ant_swarm_trajectory_data.csv")))
            for s in scen_list}
    model = STGNN(depth=depth).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    pool = [(s, t) for s, (pos, st) in data.items() for t in range(2, 899)]
    rng = np.random.default_rng(tseed)
    for ep in range(epochs):
        rng.shuffle(pool)
        tot, nb = 0.0, 0
        for i in range(0, len(pool) - bs, bs):
            batch = pool[i:i + bs]
            Nmax = max(data[s][0].shape[1] for s, _ in batch)
            feats, poss, masks, targ = [], [], [], []
            for s, t in batch:
                pos, st = data[s]
                N = pos.shape[1]
                f = make_feats(pos[t], pos[t] - pos[t - 1], pos[t - 1] - pos[t - 2], st[t])
                y = pos[t + 1] - pos[t]
                pad = Nmax - N
                if pad:
                    f = np.pad(f, ((0, pad), (0, 0)))
                    y = np.pad(y, ((0, pad), (0, 0)))
                    poss.append(np.pad(pos[t], ((0, pad), (0, 0))))
                    masks.append(np.pad(np.ones(N, bool), (0, pad)))
                else:
                    poss.append(pos[t]); masks.append(np.ones(N, bool))
                feats.append(f); targ.append(y)
            F0 = torch.tensor(np.stack(feats), device=DEV)
            P0 = torch.tensor(np.stack(poss), device=DEV)
            M0 = torch.tensor(np.stack(masks), device=DEV)
            Y = torch.tensor(np.stack(targ), device=DEV)
            vhat = model(F0, P0, M0)
            loss = (Fn.mse_loss(vhat, Y, reduction="none") * M0[..., None]).sum() / M0.sum() / 2
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if log and (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1}/{epochs} loss {tot/nb:.6f}", flush=True)
    return model


def eval_gnn(model, split, scen, eps, results, tseed):
    for ep_id, pos, st in eps:
        for h, ade in rollout(model, pos, st, test_frames(), HORIZONS, "gnn").items():
            results.append(dict(split=split, scenario=scen, model="ST-GNN", horizon=h,
                                ade=ade, train_seed=tseed, episode=ep_id))


def eval_simple(mode, split, scen, eps, results):
    for ep_id, pos, st in eps:
        for h, ade in rollout(None, pos, st, test_frames(), HORIZONS, mode).items():
            results.append(dict(split=split, scenario=scen, model=mode, horizon=h,
                                ade=ade, train_seed=-1, episode=ep_id))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sweep", action="store_true", help="追加 depth 1/3 超参扫描")
    ap.add_argument("--depth", type=int, default=2, help="ST-GNN 层数 (默认 2)")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 2

    results = []
    eps_map = {s: load_episodes(s) for s in REP_SCEN}
    print(f"每场景 episode 数: {[len(v) for v in eps_map.values()]}", flush=True)

    # ---------- within: 每场景 3 个训练种子, 测 3 条轨迹 ----------
    if not args.quick:
        for s in REP_SCEN:
            for tseed in TRAIN_SEEDS:
                model = train_model([s], tseed=tseed, depth=args.depth, epochs=args.epochs)
                eval_gnn(model, "within", s, eps_map[s], results, tseed)
                print(f"[within/{s} seed{tseed}] done", flush=True)
                del model
                torch.cuda.empty_cache()

    # ---------- transfer: 8 场景训练 x 3 种子, 测 3 条轨迹 ----------
    for tseed in (TRAIN_SEEDS[:1] if args.quick else TRAIN_SEEDS):
        model = train_model(TRAIN_SCEN, tseed=tseed, depth=args.depth, epochs=args.epochs)
        for s in REP_SCEN:
            eval_gnn(model, "transfer", s, eps_map[s], results, tseed)
        print(f"[transfer seed{tseed}] all scenarios done", flush=True)
        del model
        torch.cuda.empty_cache()

    # ---------- 确定性基线 ----------
    for split in (("transfer",) if args.quick else ("within", "transfer")):
        for s in REP_SCEN:
            eval_simple("cv", split, s, eps_map[s], results)
            eval_simple("persistence", split, s, eps_map[s], results)

    # ---------- 超参扫描 ----------
    if args.sweep:
        for depth in (1, 3):
            m = train_model(TRAIN_SCEN, tseed=0, depth=depth, epochs=args.epochs)
            for s in REP_SCEN:
                pos, st = eps_map[s][0][1], eps_map[s][0][2]
                for h, ade in rollout(m, pos, st, test_frames(6), HORIZONS, "gnn").items():
                    results.append(dict(split="sweep", scenario=s, model=f"ST-GNN-d{depth}",
                                        horizon=h, ade=ade, train_seed=0, episode=0))
            print(f"[sweep depth={depth}] done", flush=True)
            del m
            torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, "baseline_results.csv"), index=False)
    print(df.groupby(["split", "scenario", "model", "horizon"])["ade"]
          .agg(["mean", "std"]).round(3).to_string(), flush=True)
    print("saved -> analysis/baseline_results.csv", flush=True)


if __name__ == "__main__":
    main()
