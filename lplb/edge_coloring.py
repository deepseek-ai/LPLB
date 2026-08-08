from __future__ import annotations

import torch


def color_replica_graph(topology: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedily color replica edges and return GPU-ready color arrays.

    Edges are indexed by ``destination * duplicates + replica_slot``. Each
    color is a matching, so its edges can update endpoint loads concurrently.
    """
    if topology.dim() != 2:
        raise ValueError('topology must have shape [group_size, duplicates]')
    group_size, duplicates = topology.shape
    if not 0 < group_size <= 32:
        raise ValueError('edge balance requires a group size in [1, 32]')
    if duplicates == 0:
        raise ValueError('topology must contain at least one replica slot')

    topology_list = topology.detach().cpu().tolist()
    expected_sources = list(range(group_size))
    for slot in range(duplicates):
        sources = [int(topology_list[destination][slot]) for destination in range(group_size)]
        if sorted(sources) != expected_sources:
            raise ValueError(f'replica slot {slot} must map every source exactly once')

    colors: list[list[int]] = []
    color_endpoints: list[set[int]] = []
    for destination in range(group_size):
        for slot in range(duplicates):
            edge = destination * duplicates + slot
            source = int(topology_list[destination][slot])
            for color, endpoints in zip(colors, color_endpoints):
                if source not in endpoints and destination not in endpoints:
                    color.append(edge)
                    endpoints.update((source, destination))
                    break
            else:
                colors.append([edge])
                color_endpoints.append({source, destination})

    colored_edges = [edge for color in colors for edge in color]
    offsets = [0]
    for color in colors:
        offsets.append(offsets[-1] + len(color))
    device = topology.device
    return (
        torch.tensor(colored_edges, dtype=torch.int32, device=device),
        torch.tensor(offsets, dtype=torch.int32, device=device),
    )
