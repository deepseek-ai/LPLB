import pytest
import torch

from lplb import Planner
from tests.utils import (
    CUBE_8P2E,
    HYPERCUBE_16P2E,
    RING_8P,
    torus_2d,
)


def test_planner_solve_zero_workload() -> None:
    planner = Planner(CUBE_8P2E, 272, 256, ep_size=8)
    workload = torch.zeros(256, dtype=torch.int32, device='cuda')
    avail_counter = torch.zeros((), dtype=torch.int32, device='cuda')

    probs, global_workload = planner.solve_probs(workload, avail_counter)

    assert torch.isfinite(probs).all()
    assert torch.isfinite(global_workload).all()
    assert torch.count_nonzero(global_workload) == 0


@pytest.mark.parametrize(
    ('r2o', 'n_logical_experts', 'ep_size', 'n_redundants_per_rank', 'with_reorder', 'tolerance'),
    [
        (CUBE_8P2E, 256, 32, 2, True, 1.07),
        (CUBE_8P2E, 256, 32, 4, False, 1.07),
        (CUBE_8P2E, 256, 64, 2, True, 1.1),
        (RING_8P, 256, 32, 1, True, 1.07),
        (HYPERCUBE_16P2E, 256, 16, 2, True, 1.03),
        (torus_2d(8, 4), 256, 32, 2, False, 1.01),
    ],
)
def test_planner_solve(
    r2o: torch.Tensor,
    n_logical_experts: int,
    ep_size: int,
    n_redundants_per_rank: int,
    with_reorder: bool,
    tolerance: float,
) -> None:
    r2o = r2o.int().cuda()
    planner = Planner(
        r2o,
        n_logical_experts + n_redundants_per_rank * ep_size,
        n_logical_experts,
        ep_size,
    )

    torch.cuda.random.manual_seed(42)
    if with_reorder:
        workload_history = torch.randn(n_logical_experts, device='cuda') * 0.15 + 1
        workload_history = workload_history.clamp_min(0).mul(2**20).long()
    else:
        workload_history = None
    phy2log, _log2phy, _logcnt = planner.update_redundancy_mapping(workload_history)

    current_workload = torch.randn(n_logical_experts, device='cuda') * 0.3 + 1
    current_workload = current_workload.clamp_min(0).mul(2**12).int()
    # print(f'{current_workload=}')
    # print(f'{phy2log.reshape(4, 8, 10)[0, :, :2]=}')
    # print(f'{current_workload[phy2log].reshape(4, 8, 10)[0, :, :2] / current_workload.max()=}')
    avail_counter = torch.zeros((), dtype=torch.int, device='cuda')
    probs, _global_workload = planner.solve_probs(current_workload, avail_counter)

    # print(f'{probs=}')

    assert avail_counter == planner.n_group

    phy_experts_workload = current_workload[phy2log].reshape(
        ep_size // r2o.shape[0], r2o.shape[0], -1
    )
    # print(f'{phy_experts_workload=}')

    dup_workload = (
        phy_experts_workload[:, :, : planner.combined_redundant_experts * r2o.shape[1]]
        .reshape(
            ep_size // r2o.shape[0], r2o.shape[0], planner.combined_redundant_experts, r2o.shape[1]
        )
        .sum(2)
    )
    # print(f'before balancing {dup_workload=}')
    dup_workload = dup_workload * probs + (dup_workload * (1 - probs)).gather(
        dim=1, index=r2o.expand(*probs.shape).long()
    )
    # print(f'after balancing {dup_workload=}')
    dup_workload = dup_workload.sum(2)
    fixed_workload = phy_experts_workload[
        :,
        :,
        planner.combined_redundant_experts * r2o.shape[1] : -planner.combined_redundant_experts
        * r2o.shape[1],
    ].sum(2)
    actual_workload = dup_workload + fixed_workload
    print(f'{actual_workload=}')

    balance = float(actual_workload.max() / actual_workload.mean())
    print(f'balance coefficient: {balance}')
    assert balance < tolerance
