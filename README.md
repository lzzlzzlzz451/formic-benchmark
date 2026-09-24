# Ant Swarm Foraging Benchmark

A configurable ant-foraging simulator, a 16-scenario benchmark suite, and a dynamic
interaction-network dataset for spatio-temporal graph learning.

- **73,982,309** interaction records + **4,740,000** node-state records (1.6 GB)
- Obstacles composed from disks and arbitrary convex polygons via one geometric query
- Three multi-seed-verified difficulty regimes: free foraging / bottleneck / maze
- Consistency property: every edge can be recomputed exactly from logged coordinates
- Trajectory-prediction baselines included (persistence / constant velocity / ST-GNN)

## Quick start

```bash
pip install -r requirements.txt
python src/generate_scenarios.py            # regenerate all 16 scenarios (~1 h, seeds fixed)
python src/analyze_scenarios.py             # metrics + route-choice statistics
python src/replicates.py                    # 10-seed replicate study + pheromone ablation
python src/route_choice.py                  # route-choice statistics artifact
python src/baselines.py                     # trajectory-prediction baselines (3 train seeds x 3 test episodes, needs torch)
python src/gen_eval_episodes.py             # extra test episodes for the baseline protocol
```

Each generated scenario directory contains `ant_swarm_trajectory_data.csv`
(`frame, agent_id, x, y, state`), `ant_swarm_dynamic_edges.csv`
(`frame, source, target, distance`), two GIF animations, and
`scenario_config.json` (full geometry + seed). See `scenarios/README.md` in the
generated output for schemas and the audit check.

## Adding a scenario

One entry in `build_scenarios()` in `src/generate_scenarios.py`, combining
`circle_obstacle / rect_obstacle(angle) / polygon_obstacle(n-gon)`. The seed makes
every run reproducible.

## Citation

Placeholder --- replace with the archived DOI citation at submission.
