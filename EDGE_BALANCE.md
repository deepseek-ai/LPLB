# Edge-Balance CUDA Solver

## Summary

Edge Balance is an experimental graph-aware replacement for LPLB's dense interior-point LP solver. It assigns replicated-expert workload by repeatedly equalizing the loads at the endpoints of graph edges.

Unlike the current LPLB solver, Edge Balance does not construct dense LP matrices or call cuBLASDx and cuSolverDx. It operates directly on the fixed replication graph, preserves primal feasibility by construction, and uses precomputed edge colors to update independent edges in parallel without atomics.

On the current 960-instance synthetic benchmark corpus, Edge Balance was faster, independently feasible for every problem, and effectively optimal relative to HiGHS.

## Optimization model

For each directed edge `e = (u, v)`:

- `w[e]` is the replicated workload that may be split between its endpoints.
- `f[e]` is the amount transferred from source `u` to destination `v`.
- `0 <= f[e] <= w[e]`.

Starting from node base loads `b`, the node loads are:

```text
load = b + A f
```

Increasing an edge flow by `delta` changes only its two endpoints:

```text
f[e]    += delta
load[u] -= delta
load[v] += delta
```

The objective is to minimize the maximum node load:

```text
minimize max_i load[i]
```

For one edge, unconstrained pairwise equalization uses:

```text
desired_delta = 0.5 * (load[u] - load[v])
```

The update is clipped to preserve edge-flow bounds:

```text
delta = clamp(desired_delta, -f[e], w[e] - f[e])
```

Allowing a negative delta is important because later sweeps can undo earlier routing decisions.

## CUDA design

The implementation uses:

- One CUDA block per independent LPLB group.
- One warp per group for the currently supported group sizes (`G <= 32`).
- Shared memory for node loads, edge capacities, and edge flows.
- A feasible half-capacity initialization.
- Host-precomputed greedy edge coloring.
- Gauss-Seidel updates across colors.
- Parallel updates within each color.
- No atomics for load updates because each color is a matching.
- A block-wide maximum-change convergence check after every sweep.

Edges are indexed by `(destination rank, replica slot)`. The existing `r2o` topology tensor supplies each edge's source rank.

The default stopping configuration is:

```text
maximum sweeps: 64
flow tolerance: 1e-5
```

The solver returns:

- Original-endpoint routing probabilities.
- Final node loads.
- Maximum load per group.
- Number of completed sweeps.
- Convergence status.

## Implementation locations

- CUDA kernel: `lplb/resources/csrc-tmpl/minilp.cu`
- C++ extension binding: `csrc/plugin.cpp`
- Planner interface: `lplb/planner.py`
- Edge coloring: `lplb/edge_coloring.py`
- CUDA tests: `tests/test_edge_balance.py`

The existing production `solve_probs()` IPM path remains available. The experimental interface is `Planner.edge_balance_probs()`.

## Benchmark methodology

Both Edge Balance and current LPLB were tested using:

- The same saved corpus of 960 problems.
- Five graph shapes: `4x1`, `4x2`, `8x1`, `8x2`, and `16x2`.
- Six workload families: balanced, Gaussian, Zipf, sparse, hotspot, and bimodal.
- 32 samples per shape and workload family.
- 1,000 timed repetitions per instance.
- CUDA events for GPU latency.
- SciPy HiGHS outside the timed region for independent LP verification.

Feasibility checks include equality residuals and routing-variable bounds. Objective gap is measured against the HiGHS optimal objective.

## Results

| Shape | Current LPLB | Edge Balance | Speedup | Edge feasible | Edge mean gap |
|---|---:|---:|---:|---:|---:|
| `4x1` | 18.21 us | 14.43 us | 1.26x | 100% | approximately 0% |
| `4x2` | 22.36 us | 13.70 us | 1.63x | 100% | approximately 0% |
| `8x1` | 25.59 us | 13.53 us | 1.89x | 100% | approximately 0% |
| `8x2` | 39.58 us | 15.13 us | 2.62x | 100% | approximately 0% |
| `16x2` | 104.05 us | 19.24 us | 5.41x | 100% | approximately 0% |

Overall matched-corpus speedup was `1.77x`.

| Shape | Median sweeps | Maximum objective gap |
|---|---:|---:|
| `4x1` | 2 | 0.00012% |
| `4x2` | 2 | 0.00021% |
| `8x1` | 3 | 0.00027% |
| `8x2` | 4 | 0.00034% |
| `16x2` | 9 | 0.00164% |

Current LPLB was independently feasible on 76.56% of the `8x2` problems and 41.67% of the `16x2` problems. Edge Balance was independently feasible on 100% of all tested problems.

Run the focused tests after an editable installation:

```bash
pytest -q tests/test_edge_balance.py
```

## Limitations and next steps

The current evidence is based on synthetic workloads and the reconstructed LPLB LP model. Before production use, the solver should also be evaluated with:

- Real expert-workload traces.
- Multiple independent groups per launch.
- Distributed workload synchronization.
- Warm starts between consecutive batches.
- More graph topologies and adversarial capacities.
- Different convergence tolerances and sweep limits.
- A primal-dual stopping certificate.
- Consumers that may require distinct output storage rather than reusable buffers.

The current convergence test detects a coordinatewise fixed point. HiGHS comparisons show near-optimal solutions on this corpus, but a general global-optimality guarantee should rely on a valid primal-dual certificate or a separate proof for the graph model.

## Research Status

The edge-balance implementation and accompanying technical blog were developed
by Bochuan Lyu. Bochuan Lyu and Zedong Peng are currently extending this work
through end-to-end evaluations in inference systems. The broader algorithmic
analysis and experimental results are intended for inclusion in a future
research paper.

If you build on the current implementation, please reference the upstream LPLB
project, the Edge Balance implementation branch, and the technical blog:

- Upstream project: [deepseek-ai/LPLB](https://github.com/deepseek-ai/LPLB)
- Edge Balance implementation: [BochuanBob/LPLB — feature/edge-balance-solver](https://github.com/BochuanBob/LPLB/tree/feature/edge-balance-solver)
- Technical blog: [A Graph-Native CUDA Solver for MoE Load Balancing](https://bochuanbob.github.io/lplb-edge-balance/)
