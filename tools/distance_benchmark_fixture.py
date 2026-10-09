"""Deterministic synthetic inputs, not a replay of a production trajectory batch."""
import torch

from constraint_projection import ConstraintProjection
from distance_kl import distance_generator, reference_from_data
from tools.ablation_common import generator


def fixture(device='cpu', batch=64, max_pois=16, candidates=64, seed=135398):
    if batch < 1 or max_pois < 2 or candidates < 1:
        raise ValueError('Invalid fixture dimensions')
    raw = dict(poi_gps={6 + i: [40.7 + (i % 8) * .002, -74 + (i // 8) * .002]
                       for i in range(candidates)},
               sequences=[dict(checkins=[6 + (i + j) % candidates for j in range(2 + i % max_pois)])
                          for i in range(64)])
    reference = reference_from_data(raw)
    rng = torch.Generator().manual_seed(seed)
    length, vocab = 2 * max_pois, candidates + 8
    y = torch.randn(batch, length, vocab, generator=rng) * .3
    cat = torch.zeros(batch, length, dtype=torch.bool)
    poi = torch.zeros_like(cat)
    active = torch.ones(batch, dtype=torch.bool)
    for b in range(batch):
        n = b % (max_pois + 1)
        cat[b, :2*n:2] = True
        poi[b, 1:2*n:2] = True
        active[b] = b % 7 != 0
    y = torch.where(cat[..., None], y - 5, y)
    y[:, :, 4:6] += cat[..., None] * 5
    y = torch.where(poi[..., None], y - 5, y)
    y[:, :, 6:-2] += poi[..., None] * 5
    return dict(reference=reference, y=y.to(device), cat=cat.to(device),
                poi=poi.to(device), active=active.to(device), seed=seed)


def make_projector(data, backend=None, stochastic=True, distance_weight=1., outer=10, inner=50):
    device = data['y'].device
    kwargs = {} if backend is None else dict(distance_backend=backend)
    p = ConstraintProjection(data['y'].shape[-1], 2, 4, device=str(device), verbose=False,
        tau=0, lambda_init=1, mu_init=1, mu_alpha=2, mu_max=1000, eta=1, delta_tol=1e-6,
        outer_iterations=outer, inner_iterations=inner, early_stop=False,
        projection_existence_weight=5, projection_order_weight=1, projection_kl_weight=1,
        projection_distance_kl_weight=distance_weight, distance_reference=data['reference'],
        use_gumbel_softmax=stochastic, gumbel_temperature=3, collect_diagnostics=True,
        generator=generator(data['seed'], 0, 'projection', device),
        distance_generator=distance_generator(data['seed'], device), **kwargs)
    constraints = [[([0], [1])] if enabled else [] for enabled in data['active'].tolist()]
    a, b, mask = p.compile_batched_constraints(constraints, device)
    return p, (data['y'].transpose(1, 2), a, b, data['cat'], mask)
