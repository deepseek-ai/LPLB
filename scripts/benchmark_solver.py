#!/usr/bin/env python3
"""Single-GPU microbenchmark for LPLB's generated LP problems."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linprog

from lplb import Planner


@dataclass(frozen=True)
class Config:
    name: str
    group_size: int
    duplicates: int


CONFIGS = {
    '4x1': Config('4x1_ring', 4, 1),
    '4x2': Config('4x2', 4, 2),
    '8x1': Config('8x1_ring', 8, 1),
    '8x2': Config('8x2_cube', 8, 2),
    '16x2': Config('16x2_hypercube', 16, 2),
}
DISTRIBUTIONS = ('balanced', 'gaussian', 'zipf', 'sparse', 'hotspot', 'bimodal')


def make_topology(config: Config) -> torch.Tensor:
    n = config.group_size
    if config.name == '8x2_cube':
        return torch.tensor(
            [[3, 6], [0, 7], [1, 4], [2, 5], [7, 0], [4, 1], [5, 2], [6, 3]]
        )
    if config.name == '16x2_hypercube':
        return torch.tensor(
            [
                [3, 12], [0, 13], [1, 14], [2, 15],
                [7, 0], [4, 1], [5, 2], [6, 3],
                [11, 4], [8, 5], [9, 6], [10, 7],
                [15, 8], [12, 9], [13, 10], [14, 11],
            ]
        )
    return torch.tensor(
        [[(rank + offset + 1) % n for offset in range(config.duplicates)] for rank in range(n)]
    )


def make_workload(kind: str, size: int, generator: torch.Generator) -> torch.Tensor:
    if kind == 'balanced':
        values = torch.ones(size)
    elif kind == 'gaussian':
        values = (torch.randn(size, generator=generator) * 0.30 + 1.0).clamp_min(0)
    elif kind == 'zipf':
        values = 1.0 / torch.arange(1, size + 1, dtype=torch.float32).pow(1.15)
        values = values[torch.randperm(size, generator=generator)]
        values /= values.mean()
    elif kind == 'sparse':
        values = torch.rand(size, generator=generator) * 2.0
        values[torch.rand(size, generator=generator) < 0.60] = 0
        values /= values.mean().clamp_min(1e-6)
    elif kind == 'hotspot':
        values = torch.full((size,), 0.25)
        hot = torch.randperm(size, generator=generator)[: max(1, size // 16)]
        values[hot] = 12.0
        values /= values.mean()
    elif kind == 'bimodal':
        values = torch.where(
            torch.rand(size, generator=generator) < 0.5,
            torch.tensor(0.25),
            torch.tensor(1.75),
        )
        values /= values.mean()
    else:
        raise ValueError(f'unknown distribution: {kind}')
    return values.mul(4096).round().clamp(0, torch.iinfo(torch.int32).max).to(torch.int32)


def color_edges(topology: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy edge coloring for edges indexed by (destination, replica slot)."""
    group_size, duplicates = topology.shape
    colors: list[list[int]] = []
    for destination in range(group_size):
        for duplicate in range(duplicates):
            edge = destination * duplicates + duplicate
            source = int(topology[destination, duplicate])
            for color in colors:
                endpoints = {
                    endpoint
                    for other in color
                    for endpoint in (
                        int(topology[other // duplicates, other % duplicates]),
                        other // duplicates,
                    )
                }
                if source not in endpoints and destination not in endpoints:
                    color.append(edge)
                    break
            else:
                colors.append([edge])
    colored = torch.tensor([edge for color in colors for edge in color], dtype=torch.int32)
    offsets = [0]
    for color in colors:
        offsets.append(offsets[-1] + len(color))
    return colored.cuda(), torch.tensor(offsets, dtype=torch.int32, device='cuda')


def verify_with_highs(
    planner: Planner, workload: torch.Tensor, probabilities: torch.Tensor
) -> dict[str, float | bool]:
    if planner.n_group != 1:
        raise ValueError('HiGHS verification currently expects one topology group')
    group_size = planner.group_size
    duplicates = planner.num_redundants
    physical = workload[planner.phy2log].reshape(
        planner.n_group, planner.group_size, planner.n_local_routed_experts
    )
    variable_count = planner.combined_redundant_experts * duplicates
    duplicate = physical[:, :, :variable_count].reshape(
        planner.n_group,
        group_size,
        planner.combined_redundant_experts,
        duplicates,
    ).sum(2)[0].double().cpu().numpy()
    fixed = physical[:, :, variable_count:-variable_count].sum(2)[0].double().cpu().numpy()
    r2o = planner.r2o.cpu().numpy()

    # Variables are [original ratios, replica ratios, load slacks, maximum load].
    edge_count = group_size * duplicates
    variable_total = 2 * edge_count + group_size + 1
    constraint_total = group_size + edge_count
    a_eq = np.zeros((constraint_total, variable_total), dtype=np.float64)
    b_eq = np.zeros(constraint_total, dtype=np.float64)
    for rank in range(group_size):
        for duplicate_index in range(duplicates):
            edge = rank * duplicates + duplicate_index
            a_eq[rank, edge] = duplicate[rank, duplicate_index]
            source = int(r2o[rank, duplicate_index])
            source_edge = source * duplicates + duplicate_index
            a_eq[rank, edge_count + source_edge] = duplicate[source, duplicate_index]
        a_eq[rank, 2 * edge_count + rank] = 1.0
        a_eq[rank, -1] = -1.0
        b_eq[rank] = -fixed[rank]
    for edge in range(edge_count):
        row = group_size + edge
        a_eq[row, edge] = 1.0
        a_eq[row, edge_count + edge] = 1.0
        b_eq[row] = 1.0

    objective = np.zeros(variable_total, dtype=np.float64)
    objective[-1] = 1.0
    reference = linprog(objective, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    original = probabilities[0].double().cpu().numpy().reshape(-1)
    replica = 1.0 - original
    loads_without_slack = a_eq[:group_size, : 2 * edge_count] @ np.concatenate(
        [original, replica]
    ) + fixed
    lplb_objective = float(loads_without_slack.max())
    slack = lplb_objective - loads_without_slack
    candidate = np.concatenate([original, replica, slack, [lplb_objective]])
    residual = float(np.max(np.abs(a_eq @ candidate - b_eq)))
    bound_violation = float(max(0.0, -candidate.min()))
    independently_feasible = bool(
        np.isfinite(candidate).all() and residual <= 1e-5 and bound_violation <= 1e-6
    )
    highs_objective = float(reference.fun) if reference.success else math.nan
    absolute_gap = lplb_objective - highs_objective if reference.success else math.nan
    relative_gap = absolute_gap / max(abs(highs_objective), 1e-12) if reference.success else math.nan
    return {
        'balance_ratio': lplb_objective / max(float(loads_without_slack.mean()), 1e-12),
        'independently_feasible': independently_feasible,
        'equality_residual': residual,
        'bound_violation': bound_violation,
        'lplb_objective': lplb_objective,
        'highs_success': bool(reference.success),
        'highs_objective': highs_objective,
        'absolute_objective_gap': absolute_gap,
        'relative_objective_gap': relative_gap,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--configs', nargs='+', choices=CONFIGS, default=['4x1', '4x2', '8x1', '8x2', '16x2'])
    parser.add_argument('--algorithm', choices=('lplb', 'half', 'edge-balance'), default='lplb')
    parser.add_argument('--solver-id', help='name stored in output metadata; defaults to --algorithm')
    parser.add_argument('--corpus', type=Path, help='load byte-identical problems from a saved corpus.pt')
    parser.add_argument('--samples', type=int, default=32, help='samples per distribution and shape')
    parser.add_argument('--repeats', type=int, default=20, help='timed solves per sample')
    parser.add_argument('--max-sweeps', type=int, default=64)
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=20260808)
    parser.add_argument('--output', type=Path, default=Path('benchmark-results'))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    args.output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    records: list[dict[str, object]] = []
    solver_id = args.solver_id or args.algorithm
    if args.corpus:
        corpus = torch.load(args.corpus, map_location='cpu', weights_only=True)
        args.seed = int(corpus['seed'])
    else:
        corpus = {'seed': args.seed, 'problems': []}
        for key in args.configs:
            config = CONFIGS[key]
            topology = make_topology(config)
            logical_experts = config.group_size * 8
            for distribution in DISTRIBUTIONS:
                for sample in range(args.samples):
                    corpus['problems'].append(
                        {
                            'config': key,
                            'distribution': distribution,
                            'sample': sample,
                            'topology': topology,
                            'workload': make_workload(distribution, logical_experts, generator),
                        }
                    )

    selected_problems = [
        problem for problem in corpus['problems'] if str(problem['config']) in args.configs
    ]
    selected_keys = list(dict.fromkeys(str(problem['config']) for problem in selected_problems))
    for key in selected_keys:
        config = CONFIGS[key]
        config_problems = [problem for problem in selected_problems if problem['config'] == key]
        topology = config_problems[0]['topology']
        logical_experts = config.group_size * 8
        planner = Planner(
            topology,
            logical_experts + config.duplicates * config.group_size,
            logical_experts,
            config.group_size,
        )

        # Trigger compilation and stabilize clocks/caches before measuring.
        warmup_workload = torch.full((logical_experts,), 4096, dtype=torch.int32, device='cuda')
        half_probabilities = torch.full(
            (planner.n_group, config.group_size, config.duplicates),
            0.5,
            dtype=torch.float32,
            device='cuda',
        )

        def solve_half() -> torch.Tensor:
            return half_probabilities

        for _ in range(10):
            if args.algorithm == 'lplb':
                counter = torch.zeros((), dtype=torch.int32, device='cuda')
                planner.solve_probs(warmup_workload, counter)
            elif args.algorithm == 'edge-balance':
                planner.edge_balance_probs(
                    warmup_workload,
                    args.max_sweeps,
                    args.tolerance,
                )
            else:
                solve_half()
        torch.cuda.synchronize()

        for problem in config_problems:
            distribution = str(problem['distribution'])
            sample = int(problem['sample'])
            workload_cpu = problem['workload']
            workload = workload_cpu.cuda()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if args.algorithm == 'lplb':
                counter = torch.zeros((), dtype=torch.int32, device='cuda')
                start.record()
                host_start = time.perf_counter_ns()
                for _ in range(args.repeats):
                    probabilities, _ = planner.solve_probs(workload, counter)
                host_latency_us = (time.perf_counter_ns() - host_start) / 1000 / args.repeats
                end.record()
                end.synchronize()
                reported_feasible = int(counter) == planner.n_group * args.repeats
                diagnostic_sweeps = 5
            elif args.algorithm == 'edge-balance':
                start.record()
                host_start = time.perf_counter_ns()
                for _ in range(args.repeats):
                    probabilities, _loads, _maximum, sweeps, converged = (
                        planner.edge_balance_probs(
                            workload,
                            args.max_sweeps,
                            args.tolerance,
                        )
                    )
                host_latency_us = (time.perf_counter_ns() - host_start) / 1000 / args.repeats
                end.record()
                end.synchronize()
                reported_feasible = bool(torch.all(converged))
                diagnostic_sweeps = float(sweeps.float().mean())
            else:
                start.record()
                host_start = time.perf_counter_ns()
                for _ in range(args.repeats):
                    probabilities = solve_half()
                host_latency_us = (time.perf_counter_ns() - host_start) / 1000 / args.repeats
                end.record()
                end.synchronize()
                reported_feasible = True
                diagnostic_sweeps = 0
            latency_us = start.elapsed_time(end) * 1000 / args.repeats
            quality = verify_with_highs(planner, workload, probabilities)
            records.append(
                {
                    'solver_id': solver_id,
                    'algorithm': args.algorithm,
                    'config': config.name,
                    'group_size': config.group_size,
                    'duplicates': config.duplicates,
                    'constraints': config.group_size * (1 + config.duplicates),
                    'variables': 2 * config.group_size * config.duplicates + config.group_size + 2,
                    'distribution': distribution,
                    'sample': sample,
                    'latency_us': latency_us,
                    'host_latency_us': host_latency_us,
                    'reported_feasible': reported_feasible,
                    'num_sweeps': diagnostic_sweeps,
                    **quality,
                }
            )

    csv_path = args.output / 'raw.csv'
    with csv_path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    torch.save(corpus, args.output / 'corpus.pt')

    summary = []
    for config_name in sorted({str(row['config']) for row in records}):
        for distribution in DISTRIBUTIONS:
            rows = [
                row for row in records
                if row['config'] == config_name and row['distribution'] == distribution
            ]
            latencies = [float(row['latency_us']) for row in rows]
            host_latencies = [float(row['host_latency_us']) for row in rows]
            quality_rows = [
                row for row in rows
                if bool(row['independently_feasible']) and bool(row['highs_success'])
            ]
            quality_gaps = [float(row['relative_objective_gap']) for row in quality_rows]
            summary.append(
                {
                    'config': config_name,
                    'distribution': distribution,
                    'samples': len(rows),
                    'median_us': percentile(latencies, 0.5),
                    'p90_us': percentile(latencies, 0.9),
                    'p99_us': percentile(latencies, 0.99),
                    'median_host_us': percentile(host_latencies, 0.5),
                    'reported_feasible_rate': sum(bool(row['reported_feasible']) for row in rows) / len(rows),
                    'independent_feasible_rate': sum(bool(row['independently_feasible']) for row in rows) / len(rows),
                    'highs_success_rate': sum(bool(row['highs_success']) for row in rows) / len(rows),
                    'mean_balance_ratio': sum(float(row['balance_ratio']) for row in rows) / len(rows),
                    'mean_relative_objective_gap': sum(quality_gaps) / len(quality_gaps) if quality_gaps else math.nan,
                    'max_relative_objective_gap': max(quality_gaps) if quality_gaps else math.nan,
                    'max_bound_violation': max(float(row['bound_violation']) for row in rows),
                    'mean_sweeps': sum(float(row['num_sweeps']) for row in rows) / len(rows),
                }
            )
    metadata = {
        'device': torch.cuda.get_device_name(),
        'torch': torch.__version__,
        'solver_id': solver_id,
        'algorithm': args.algorithm,
        'corpus_source': str(args.corpus) if args.corpus else None,
        'seed': args.seed,
        'problem_count': len(records),
        'repeats': args.repeats,
        'max_sweeps': args.max_sweeps,
        'tolerance': args.tolerance,
        'configs': [asdict(CONFIGS[key]) for key in selected_keys],
        'summary': summary,
    }
    (args.output / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')

    print(f"{'config':<18} {'distribution':<10} {'median us':>10} {'reported':>10} {'LP feas':>10} {'obj gap':>10}")
    for row in summary:
        print(
            f"{row['config']:<18} {row['distribution']:<10} "
            f"{row['median_us']:>10.2f} {row['reported_feasible_rate']:>10.1%} "
            f"{row['independent_feasible_rate']:>10.1%} "
            f"{row['mean_relative_objective_gap']:>10.2%}"
        )
    print(f'Wrote {csv_path}, {args.output / "summary.json"}, and {args.output / "corpus.pt"}')


if __name__ == '__main__':
    main()
