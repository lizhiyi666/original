"""Opt-in evaluation v3: deterministic top-k POI membership, legacy v2 untouched."""
from collections import Counter
import hashlib
import json
import math
from numbers import Integral

import numpy as np

from evaluations.statistical_metrics import JSD, STATISTICAL_NAMES
from tools.ablation_common import evaluate as evaluate_v2

EVALUATION_VERSION = 3
REVISION = 'grank-count-desc-poi-id-asc-v3'
TOP = 1000


def ranked_counts(events, top=TOP):
    """Use integer counts, then ascending integer POI ID; never float tie sorting."""
    if isinstance(top, bool) or not isinstance(top, Integral) or top <= 0:
        raise ValueError('top must be a positive integer')
    values = list(events)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in values):
        raise ValueError('G-RANK requires integer POI IDs')
    counts = Counter(int(v) for v in values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:int(top)], counts


def _diagnostic(selected, counts):
    boundary = selected[-1][1] if selected else None
    encoded = json.dumps(selected, separators=(',', ':')).encode('ascii')
    return dict(events=sum(counts.values()), unique_pois=len(counts), selected_pois=len(selected),
                boundary_count=boundary, boundary_tied=sum(v == boundary for v in counts.values()),
                boundary_selected=sum(v == boundary for _, v in selected),
                ranked_counts_sha256=hashlib.sha256(encoded).hexdigest())


def grank_jsd(generated, reference, top=TOP, *, diagnostics=None):
    selected_gen, counts_gen = ranked_counts(generated, top)
    selected_ref, counts_ref = ranked_counts(reference, top)
    if diagnostics is not None:
        diagnostics.update(revision=REVISION, top=int(top),
                           generated=_diagnostic(selected_gen, counts_gen),
                           reference=_diagnostic(selected_ref, counts_ref))
    if not selected_gen or not selected_ref:
        return float('nan')
    left, right = dict(selected_gen), dict(selected_ref)
    keys = sorted(set(left) | set(right))
    # Preserve historical probability normalization/epsilon, but fix reduction order.
    total_gen, total_ref = sum(counts_gen.values()), sum(counts_ref.values())
    p = np.array([left.get(k, 0) / total_gen for k in keys], dtype=np.float64)
    q = np.array([right.get(k, 0) / total_ref for k in keys], dtype=np.float64)
    return float(JSD(p, q))


def evaluate(refs, generated, poi_category, indices, *, diagnostics=None):
    """Reuse every v2 metric/constraint/filter; replace only G-RANK and its total."""
    metrics, rows = evaluate_v2(refs, generated, poi_category, indices)
    # Exactly v2's GPS-length filter; Distance's endpoint filter does not apply here.
    gen = [p for record in generated if len(record.get('gps', [])) > 1 for p in record['checkins']]
    real = [p for record in refs if len(record.get('gps', [])) > 1 for p in record['checkins']]
    detail = {}
    value = grank_jsd(gen, real, diagnostics=detail)
    metrics['G-RANK'] = value if math.isfinite(value) else None
    finite = [metrics[k] for k in STATISTICAL_NAMES if metrics[k] is not None]
    metrics['totalJSD'] = sum(finite) if finite else None
    metrics['evaluation_version'] = EVALUATION_VERSION
    if diagnostics is not None:
        diagnostics.update(detail)
    return metrics, rows
