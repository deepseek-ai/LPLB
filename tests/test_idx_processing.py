import pytest
import torch

from lplb.planner import Planner
from tests.utils import CUBE_8P2E


@pytest.mark.parametrize(
    ('num_tokens', 'weight'),
    [
        (0, 0.5),
        (1, 0.5),
        (498, 0.5),
        (499, 0.0),
        (499, 0.25),
        (499, 0.5),
        (499, 0.75),
        (499, 1.0),
        (500, 0.5),
        (998, 0.5),
        (1497, 0.25),
        (1497, 0.5),
        (4_304_000, 0.5),
        (8_608_000, 0.5),
    ],
)
def test_weighted_select_target_preserves_requested_split(num_tokens: int, weight: float) -> None:
    planner = Planner(CUBE_8P2E, 272, 256, ep_size=8)
    _phy2log, log2phy, _logcnt = planner.update_redundancy_mapping()
    idx = torch.zeros(num_tokens, dtype=torch.int64, device='cuda')
    weights = torch.full((1, 8, 2), weight, dtype=torch.float32, device='cuda')

    _, workload_by_sm = planner.count_workload(idx, n_sms=4)
    mapped = planner.weighted_select_target(idx, weights, workload_by_sm, n_sms=4)

    # Expert 0 has one original and one replica.
    originals = (mapped == log2phy[0, 0]).sum().item()
    replicas = (mapped == log2phy[0, 1]).sum().item()
    expected_originals = int(num_tokens * weight)
    assert originals == expected_originals
    assert replicas == num_tokens - expected_originals


@pytest.mark.parametrize(
    ('r2o', 'n_logical_experts', 'ep_size', 'n_redundants_per_rank'),
    [(CUBE_8P2E, 256, 32, 2)],
)
def test_count_workload(
    r2o: torch.Tensor,
    n_logical_experts: int,
    ep_size: int,
    n_redundants_per_rank: int,
) -> None:
    r2o = r2o.int().cuda()
    planner = Planner(
        r2o,
        n_logical_experts + n_redundants_per_rank * ep_size,
        n_logical_experts,
        ep_size,
    )

    torch.cuda.random.manual_seed(42)
    idx = torch.randint(-1, n_logical_experts, (4096, 8), device='cuda')

    workload, _workload_by_sm = planner.count_workload(
        idx,
        torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
    )

    print(f'{workload=}')

    # Verify workload counts
    assert workload.shape == (n_logical_experts,)
    assert workload.min() >= 0
    assert workload.sum() == (idx != -1).sum()

    # Compare with torch implementation
    expected_workload = torch.zeros(n_logical_experts + 1, dtype=torch.long, device='cuda')
    expected_workload.scatter_add_(0, idx.view(-1).long() + 1, torch.ones_like(idx.view(-1)).long())
    expected_workload = expected_workload[1:]

    print(f'{expected_workload=}')

    assert torch.all(workload == expected_workload)


@pytest.mark.parametrize(
    ('r2o', 'n_logical_experts', 'ep_size', 'n_redundants_per_rank'),
    [(CUBE_8P2E, 256, 32, 2)],
)
def test_weighted_select_target(
    r2o: torch.Tensor,
    n_logical_experts: int,
    ep_size: int,
    n_redundants_per_rank: int,
) -> None:
    r2o = r2o.int().cuda()
    planner = Planner(
        r2o,
        n_logical_experts + n_redundants_per_rank * ep_size,
        n_logical_experts,
        ep_size,
    )

    torch.cuda.random.manual_seed(42)
    workload_history = torch.randn(n_logical_experts, device='cuda') * 0.15 + 1
    workload_history = workload_history.clamp_min(0).mul(2**20).long()
    phy2log, log2phy, logcnt = planner.update_redundancy_mapping(workload_history)

    idx = torch.randint(-1, n_logical_experts, (65536,), device='cuda')
    o_weight = torch.randn(
        planner.n_group,
        planner.group_size,
        planner.num_redundants,
        device='cuda',
    )
    o_weight = (o_weight / 3 + 0.5).clamp(0, 1)

    num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    _workload, workload_by_sm = planner.count_workload(idx, num_sms)
    print(f'{workload_by_sm=}')
    mapped_idx = planner.weighted_select_target(idx, o_weight, workload_by_sm, num_sms)

    assert mapped_idx.shape == idx.shape
    assert mapped_idx.min() >= -1
    assert mapped_idx.max() < planner.n_routed_experts

    assert torch.all((mapped_idx == -1) == (idx == -1))

    log2phy_padded = torch.cat([torch.full((1, 2), -1, device='cuda'), log2phy], dim=0)
    either_equal = (mapped_idx == log2phy_padded[idx + 1, 0]) | (
        mapped_idx == log2phy_padded[idx + 1, 1]
    )
    assert torch.all(either_equal)

    mapped_idx_freq = torch.zeros(planner.n_routed_experts + 1, dtype=torch.int32, device='cuda')
    mapped_idx_freq.scatter_add_(0, mapped_idx + 1, torch.ones_like(mapped_idx, dtype=torch.int32))
    mapped_workloads_per_rank = mapped_idx_freq[1:].reshape(planner.n_group, planner.group_size, -1)
    original_workloads = mapped_workloads_per_rank[
        :, :, : planner.num_redundants * planner.combined_redundant_experts
    ].reshape(
        planner.n_group,
        planner.group_size,
        planner.combined_redundant_experts,
        planner.num_redundants,
    )
    redundant_workloads = mapped_workloads_per_rank[
        :, :, -planner.num_redundants * planner.combined_redundant_experts :
    ].reshape(
        planner.n_group,
        planner.group_size,
        planner.combined_redundant_experts,
        planner.num_redundants,
    )
    redundant_workloads = redundant_workloads.gather(
        1,
        planner.o2r.unsqueeze(0)
        .unsqueeze(2)
        .expand(planner.n_group, -1, planner.combined_redundant_experts, -1)
        .long(),
    )
    original_workloads = original_workloads.sum(2)
    redundant_workloads = redundant_workloads.sum(2)
    print(f'{original_workloads=}')
    print(f'{redundant_workloads=}')
    expected_original_workloads = o_weight * (original_workloads + redundant_workloads)
    print(f'{expected_original_workloads=}')
    err = (
        (expected_original_workloads - original_workloads)
        .abs()
        .div(original_workloads + redundant_workloads)
        .max()
    )
    assert err < 5e-2
