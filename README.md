# Linear-Programming-Based Load Balancer (LPLB)

LPLB is a parallel load balancer that leverages linear programming to optimize expert parallel workload distribution for MoE (Mixture-of-Experts) models. It dynamically reorders experts based on workload statistics, constructs replicas considering static topology, and solves optimal token assignments for each batch to achieve dynamic load balancing. The reordering process is facilitated by [EPLB](https://github.com/deepseek-ai/EPLB), and real-time workload statistics can be provided by the user, collected via `torch.distributed`, or obtained through the internal communicators of a [Deep-EP](https://github.com/deepseek-ai/DeepEP) buffer. Its embedded LP solver implements single-SM Interior Point Method (IPM) and leverages NVIDIA's [cuSolverDx](https://developer.nvidia.com/cusolverdx-downloads) and [cuBLASDx](https://developer.nvidia.com/cublasdx-downloads) libraries for efficient linear algebra operations.

LPLB is currently in the early research stage, and performance improvements are still under evaluation.

## Installation

**Prerequisites:**

- CUDA Toolkit >= 12.6.3 (with cuSolverDx dependencies).
- DeepEP is optional but **strongly recommended** for practical use.
- EPLB is embedded.

```bash
./download-mathdx.sh
# export NVSHMEM_DIR=...  # Optional
pip install --no-build-isolation .
```

For testing, an editable installation is recommended:

```bash
pip install --no-build-isolation --editable .
pytest tests
```

## Interface and Example

```python
# Global successes counter
avail_counter = torch.zeros(1, dtype=torch.int64, device="cuda")
# Define topology of redundant experts
r2o = torch.tensor(
    [
        [3, 0, 1, 2, 7, 4, 5, 6],
        [6, 7, 4, 5, 0, 1, 2, 3],
    ]
).T.int().cuda()

planner = Planner(
    r2o,
    n_logical_experts + n_redundants_per_rank * ep_size,
    n_logical_experts,
    group=ep_group,
)
# Initialize from a DeepEP `buffer` (optional)
# planner.init_from_deep_ep(buffer)

N_SMS = 100
# Logical expert indices selected by the model
indices = ...
# Planner returns physical expert indices
redirected_indices = planner.run(indices, avail_counter, N_SMS)
```

### Experimental edge-balance solver

`Planner.edge_balance_probs()` provides a graph-native experimental solver that
updates nonconflicting replica edges in parallel using a cached greedy edge
coloring. It currently supports groups of at most 32 ranks and standard
`torch.distributed` workload reduction, but not the DeepEP communication path.

```python
probs, final_load, max_load, sweeps, converged = planner.edge_balance_probs(
    current_workload,
    max_sweeps=64,
    tolerance=1e-5,
)
```

See [EDGE_BALANCE.md](EDGE_BALANCE.md) for the formulation, implementation, and
experimental benchmark results.

## How LPLB Works

LPLB extends **EPLB** (Expert Parallelism Load Balancer) to address **dynamic load imbalance** in Mixture-of-Experts (MoE) training. While EPLB handles static imbalances (e.g., consistently overloaded experts due to data distribution), LPLB targets per-batch fluctuations caused by small-batch randomness during training.

1. **Redundant Experts**: Each redundant expert is linked to an original expert, forming edges between GPUs.
2. **Edge Capacity**: The capacity of an edge is the number of tokens assigned to its redundant expert in the current batch, defining the maximum token flow for balancing.
3. **LP Optimization**: LPLB solves a linear programming (LP) problem to redistribute tokens along these edges, minimizing load imbalance within an expert-parallel (EP) group while respecting edge capacities.

Experts to be replicated are selected via EPLB (reordering only, no replication). The heaviest experts are then replicated based on the chosen LPLB topology. Real-time workload synchronization is optimized using NVLINK and NVSHMEM instead of `torch.distributed.allreduce`, reducing communication overhead. This requires DeepEP as a prerequisite.

### Limitations

- The current planner balances only total token count, not accounting for non-linearity in grouped matrix multiplication time costs, which may lead to suboptimal performance.
- The solver takes ~100 µs for intra-node optimization (longer for inter-node), which may be non-negligible for small batches.
- Under extreme global load imbalance, LPLB may perform worse than EPLB due to differences in assigning redundant experts (LPLB avoids assigning multiple replicas to the same original expert).

### Typical Topologies

- **Cube**: Replicates experts on a subset of GPUs, forming a cube graph with diagonal edges. Requires at least 2 experts per GPU. Ideal for balancing within an 8-GPU EP subgroup without sacrificing inter-node communication.
- **Hypercube**: Similar to Cube but excludes diagonal edges and requires 16 GPUs. Suitable for expert parallelism across 16 GPUs.
- **Torus**: Replicates one expert on a neighbor GPU in the same node and another on a neighbor node, forming a torus graph. Requires at least 2 experts per GPU. Effective for global balancing but less efficient due to intra-node communication than Cube.

Custom topologies can be explored by modifying the `r2o` matrix.
