"""Small train-only CPU integration check; never loads a checkpoint or writes results."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from constraint_projection import ConstraintProjection
from distance_kl import load_distance_reference, distance_generator
from evaluations.statistical_metrics import Get_Statistical_Metrics


def validate(dataset, samples=4):
    torch.set_num_threads(1)
    path = ROOT / 'data' / dataset / f'{dataset}_train.pkl'
    raw = torch.load(path, map_location='cpu', weights_only=False)
    ref = load_distance_reference(path)
    selected = [s for s in raw['sequences'] if len(s['checkins']) >= 2][:samples]
    n, length = len(selected), max(len(s['checkins']) for s in selected)
    vocab = max(int(ref.poi_ids.max()) + 3, int(raw['num_marks']) + int(raw['num_pois']) + 6)
    logits = torch.full((n, vocab, 2 * length), -8.)
    cat = torch.zeros(n, 2 * length, dtype=torch.bool)
    poi = torch.zeros_like(cat)
    for b, seq in enumerate(selected):
        size = len(seq['checkins'])
        cat[b, :size] = True
        poi[b, length:length + size] = True
        for i, (mark, token) in enumerate(zip(seq['marks'], seq['checkins'])):
            logits[b, int(mark), i] = 1
            logits[b, int(token), length + i] = 1
            logits[b, ref.poi_ids[(b + i) % len(ref.poi_ids)], length + i] = 0
    p = ConstraintProjection(vocab, int(raw['num_marks']), 4, device='cpu', verbose=False,
        outer_iterations=2, inner_iterations=2, eta=.05,
        projection_distance_kl_weight=1, distance_reference=ref,
        generator=torch.Generator().manual_seed(17), distance_generator=distance_generator(17, 'cpu'))
    a, b, mask = p.compile_batched_constraints([[([0], [1])]] * n, 'cpu')
    before = torch.get_rng_state()
    output = p.project_with_matrices(logits, a, b, cat, mask, poi_mask=poi)
    assert torch.isfinite(output).all()
    assert torch.equal(before, torch.get_rng_state())
    change = (output.transpose(1, 2)[poi] - logits.transpose(1, 2)[poi]).norm()
    assert change > 0, 'Distance objective did not change POI logits'
    assert p.last_projection_stats['distance_poi_gradient_norm'] > 0
    detail = {}
    metrics = Get_Statistical_Metrics(selected, selected, diagnostics=detail)
    assert metrics['Category'] == 0 and metrics['CategoryTransition'] == 0
    return dict(dataset=dataset, device='cpu', samples=n, reference_routes=ref.route_count,
                poi_logit_change_norm=float(change), metrics_version=metrics['evaluation_version'],
                global_rng_unchanged=True, projection=p.last_projection_stats)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', default='NewYork_PO1')
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset), ensure_ascii=False, indent=2, allow_nan=False))
