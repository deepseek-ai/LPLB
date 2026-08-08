import pytest
import torch

from lplb import Planner
from lplb.edge_coloring import color_replica_graph
from tests.utils import CUBE_8P2E


def test_edge_coloring_is_matching() -> None:
    topology = CUBE_8P2E.int().cuda()
    colored_edges, offsets = color_replica_graph(topology)
    colored_edges = colored_edges.cpu().tolist()
    offsets = offsets.cpu().tolist()
    duplicates = topology.shape[1]

    assert sorted(colored_edges) == list(range(topology.numel()))
    for begin, end in zip(offsets, offsets[1:]):
        endpoints: set[int] = set()
        for edge in colored_edges[begin:end]:
            destination, slot = divmod(edge, duplicates)
            source = int(topology[destination, slot])
            assert source not in endpoints
            assert destination not in endpoints
            endpoints.update((source, destination))


def test_edge_coloring_rejects_nonpermutation() -> None:
    topology = torch.tensor([[1], [1], [3], [0]], dtype=torch.int32, device='cuda')
    with pytest.raises(ValueError, match='must map every source exactly once'):
        color_replica_graph(topology)


def test_edge_balance_outputs_feasible_probabilities_and_loads() -> None:
    group_size = 8
    duplicates = 2
    logical_experts = 64
    planner = Planner(
        CUBE_8P2E,
        logical_experts + group_size * duplicates,
        logical_experts,
        group_size,
    )
    generator = torch.Generator(device='cuda').manual_seed(42)
    workload = (
        torch.randn(logical_experts, generator=generator, device='cuda')
        .mul(0.3)
        .add(1)
        .clamp_min(0)
        .mul(2**12)
        .to(torch.int32)
    )

    probabilities, final_load, maximum, sweeps, converged = planner.edge_balance_probs(workload)

    assert bool(torch.all(probabilities >= 0))
    assert bool(torch.all(probabilities <= 1))
    assert bool(torch.all(converged))
    assert bool(torch.all((1 <= sweeps) & (sweeps <= 64)))
    assert torch.allclose(maximum, final_load.max(dim=1).values, rtol=1e-5, atol=1e-2)
    assert float(final_load.max() / final_load.mean()) < 1.02
