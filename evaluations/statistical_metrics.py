import numpy as np
from collections import Counter, defaultdict

# Numeric revision keeps metric payloads compatible with scalar JSON/CSV consumers.
EVALUATION_VERSION = 2
STATISTICAL_NAMES = ('Distance', 'Radius', 'CategoryTransition', 'DailyLoc', 'Category', 'G-RANK')


def require_evaluation_version(metrics):
    if metrics.get('evaluation_version') != EVALUATION_VERSION or 'Interval' in metrics:
        raise ValueError('Evaluation version mismatch: recompute legacy metrics into a new v2 output directory')


def counter_jsd(left, right, both_empty=float('nan')):
    if not left and not right:
        return both_empty
    if not left or not right:
        return float(np.log(2))
    keys = sorted(set(left) | set(right))
    return float(JSD(np.array([left[k] for k in keys]), np.array([right[k] for k in keys])))


def _category_events(seqs):
    hourly = [Counter() for _ in range(24)]
    transitions = defaultdict(Counter)
    for seq in seqs:
        marks = seq['marks']
        times = np.asarray(seq['arrival_times'], dtype=float)
        if len(times) != len(marks) or times.ndim != 1 or not np.isfinite(times).all():
            raise ValueError('Category events require aligned finite arrival times and marks')
        if np.any(times < 0) or np.any(times >= 24) or np.any(np.diff(times) < 0):
            raise ValueError('Category event times must be ordered and in [0,24)')
        if any(c is None or not np.isfinite(c) or int(c) != c for c in marks):
            raise ValueError('Category events contain invalid category tokens')
        marks = [int(c) for c in marks]
        for h, c in zip(times.astype(int), marks):
            hourly[h][c] += 1
        for left, right in zip(marks[:-1], marks[1:]):
            transitions[left][right] += 1
    return hourly, transitions


def temporal_category_metrics(real_data, generated_data, diagnostics=None):
    real_hours, real_edges = _category_events(real_data)
    gen_hours, gen_edges = _category_events(generated_data)
    hourly = [counter_jsd(r, g, both_empty=0.0) for r, g in zip(real_hours, gen_hours)]
    has_events = any(real_hours) or any(gen_hours)
    sources = sorted(set(real_edges) | set(gen_edges))
    rows = {str(c): counter_jsd(real_edges[c], gen_edges[c]) for c in sources}
    if diagnostics is not None:
        diagnostics.update(evaluation_version=EVALUATION_VERSION,
            category_hourly=[dict(hour=h, jsd=hourly[h], real_count=sum(real_hours[h].values()),
                                 generated_count=sum(gen_hours[h].values())) for h in range(24)],
            category_transition_rows=rows,
            category_transition_counts={str(c): dict(real=sum(real_edges[c].values()),
                                                    generated=sum(gen_edges[c].values())) for c in sources})
    return dict(Category=float(np.mean(hourly)) if has_events else float('nan'),
                CategoryTransition=float(np.mean(list(rows.values()))) if rows else float('nan'))

def distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

def travel_distance(geo):
    return np.sum(distance(geo[:-1, 0], geo[:-1, 1], geo[1:, 0], geo[1:, 1]))


def radius(geo):
    center = np.mean(geo, axis = 0)
    return np.sqrt(np.mean(distance(geo[:, 0], geo[:, 1], center[0], center[1])))


def JSD(P_A, P_B):
    epsilon = 1e-14
    if P_A.sum() == 0 or P_B.sum() == 0:
        return float('nan')
    P_A = (P_A / P_A.sum() + epsilon)
    P_B = (P_B / P_B.sum() + epsilon)
    P_merged = 0.5 * (P_A + P_B)
    
    kl_PA_PM = np.sum(P_A * np.log(P_A / P_merged))
    kl_PB_PM = np.sum(P_B * np.log(P_B / P_merged))
    
    jsd = 0.5 * (kl_PA_PM + kl_PB_PM)
    return jsd

def arr_to_distribution(arr, min, max, bins):
    distribution, base = np.histogram(
        arr, np.arange(
            min, max, float(
                max - min) / bins))
    return distribution

def compute_probability_distribution(data):
    unique_elements, counts = np.unique(data, return_counts=True)
    total_counts = np.sum(counts)
    probabilities = counts / total_counts
    return unique_elements, probabilities

def category_jsd(generated_category, real_category):
    gen_category, prob_gen = compute_probability_distribution(generated_category)
    real_category, prob_real = compute_probability_distribution(real_category)

    p,q = (list(zip(gen_category, prob_gen)), list(zip(real_category, prob_real)))

    p = np.asarray(p)
    q = np.asarray(q)

    all_elements = set(p[:, 0]).union(set(q[:, 0]))
    p_probs = {element: 0.0 for element in all_elements}
    q_probs = {element: 0.0 for element in all_elements}
    
    for element, prob in p:
        p_probs[element] = prob
    
    for element, prob in q:
        q_probs[element] = prob

    jsd_value = JSD(np.array(list(p_probs.values())),np.array(list(q_probs.values())))
    return jsd_value

def grank_jsd(generated_category, real_category,top=1000):
    gen_category, prob_gen = compute_probability_distribution(generated_category)
    real_category, prob_real = compute_probability_distribution(real_category)
    sorted_indices = np.argsort(-prob_gen)
    gen_category = gen_category[sorted_indices]
    prob_gen = prob_gen[sorted_indices]
    
    sorted_indices = np.argsort(-prob_real)
    real_category = real_category[sorted_indices]
    prob_real = prob_real[sorted_indices]
    
    tt=top
    gen_category=gen_category[:tt]
    prob_gen=prob_gen[:tt]
    real_category=real_category[:tt]
    prob_real=prob_real[:tt]
    p,q = (list(zip(gen_category, prob_gen)), list(zip(real_category, prob_real)))

    p = np.asarray(p)
    q = np.asarray(q)

    all_elements = set(p[:, 0]).union(set(q[:, 0]))
    p_probs = {element: 0.0 for element in all_elements}
    q_probs = {element: 0.0 for element in all_elements}
    
    for element, prob in p:
        p_probs[element] = prob
    
    for element, prob in q:
        q_probs[element] = prob

    jsd_value = JSD(np.array(list(p_probs.values())),np.array(list(q_probs.values())))
    return jsd_value

def evaluation(generated, original):
    generated = np.array(generated)
    original = np.array(original)
    assert len(generated) > 0
    assert len(original) > 0
    max = np.max(generated) if np.max(generated) > np.max(original) else np.max(original)
    if max == 0:
        return 0.0
    p_gen = arr_to_distribution(generated, 0, max, 100)
    p_real = arr_to_distribution(original, 0, max, 100)
    jsd = JSD(p_gen, p_real)
    return jsd

def get_visits(trajs,max_locs):
    visits = np.zeros(shape=(max_locs), dtype=float)
    for t in trajs:
        visits[t] += 1
    visits = visits / np.sum(visits)
    return visits

def get_topk_visits(visits, K):
    locs_visits = [[i, visits[i]] for i in range(visits.shape[0])]
    locs_visits.sort(reverse=True, key=lambda d: d[1])
    topk_locs = [locs_visits[i][0] for i in range(K)]
    topk_probs = [locs_visits[i][1] for i in range(K)]
    return np.array(topk_probs), topk_locs

def Get_Statistical_Metrics(real_data, generated_data, min_seq_len=1, top=1000, *, diagnostics=None):
    if len(real_data) != len(generated_data):
        raise ValueError('Statistical evaluation requires aligned real/generated sequences')
    result = {name: float('nan') for name in STATISTICAL_NAMES}
    result.update(temporal_category_metrics(real_data, generated_data, diagnostics))
    statistics = [{k: [] for k in ('Distance', 'Radius', 'DailyLoc', 'G-RANK')} for _ in range(2)]
    # Preserve historical filtering of the other four metrics, including Distance's endpoints.
    for index, seqs in enumerate((generated_data, real_data)):
        for i, seq in enumerate(seqs):
            if len(seq.get('gps', [])) > min_seq_len:
                gps = np.asarray(seq['gps'])
                gen, real = generated_data[i], real_data[i]
                if len(gen['marks']) and len(real['marks']) and gen['marks'][0] == real['marks'][0] and gen['marks'][-1] == real['marks'][-1]:
                    statistics[index]['Distance'].append(travel_distance(gps))
                statistics[index]['Radius'].append(radius(gps))
                statistics[index]['DailyLoc'].append(len(set(seq['checkins'])))
                statistics[index]['G-RANK'].extend(seq['checkins'])
    for name in statistics[0]:
        gen, real = statistics[0][name], statistics[1][name]
        if gen and real:
            result[name] = grank_jsd(gen, real, top) if name == 'G-RANK' else evaluation(gen, real)
    valid = [result[name] for name in STATISTICAL_NAMES if np.isfinite(result[name])]
    result['totalJSD'] = sum(valid) if valid else float('nan')
    result['evaluation_version'] = EVALUATION_VERSION
    return result
