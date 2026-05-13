import os
import csv
import argparse
import numpy as np
import torch


def load_split(data_root: str, dataset: str, split: str):
    path = os.path.join(data_root, dataset, f"{dataset}_{split}.pkl")
    obj = torch.load(path, weights_only=False, map_location="cpu")
    return obj


def ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def po_edges_from_matrix(po_matrix, threshold=0.5):
    pm = ensure_numpy(po_matrix).astype(np.float32)
    C0, C1 = pm.shape
    assert C0 == C1
    edges = []
    for i in range(C0):
        for j in range(C0):
            if i != j and pm[i, j] > threshold:
                edges.append((i, j, float(pm[i, j])))
    return edges


def export_dataset(data_root: str, dataset: str, out_dir: str, threshold: float = 0.5):
    os.makedirs(out_dir, exist_ok=True)

    # output files
    events_csv = os.path.join(out_dir, f"{dataset}_events.csv")
    seq_csv = os.path.join(out_dir, f"{dataset}_seq_summary.csv")
    edges_csv = os.path.join(out_dir, f"{dataset}_po_edges.csv")

    # CSV headers
    events_header = [
        "dataset", "split", "seq_id", "event_idx",
        "arrival_time", "inter_time",
        "poi_id", "cat_id",
        "lat", "lon",
        "condition1", "condition2", "condition3", "condition4", "condition5", "condition6",
    ]

    seq_header = [
        "dataset", "split", "seq_id",
        "length",
        "t_min", "t_max", "duration",
        "dt_min", "dt_median", "dt_mean", "dt_max",
        "num_unique_pois", "num_unique_cats",
        "po_matrix_size", "po_edges_count", "po_density",
    ]

    edges_header = [
        "dataset", "split", "seq_id",
        "edge_from_cat", "edge_to_cat", "edge_value",
    ]

    with open(events_csv, "w", newline="", encoding="utf-8") as f_ev, \
         open(seq_csv, "w", newline="", encoding="utf-8") as f_seq, \
         open(edges_csv, "w", newline="", encoding="utf-8") as f_ed:

        w_ev = csv.writer(f_ev)
        w_seq = csv.writer(f_seq)
        w_ed = csv.writer(f_ed)

        w_ev.writerow(events_header)
        w_seq.writerow(seq_header)
        w_ed.writerow(edges_header)

        seq_global_id = 0

        for split in ["train", "test"]:
            data = load_split(data_root, dataset, split)
            sequences = data.get("sequences", [])

            for local_seq_id, seq in enumerate(sequences):
                seq_id = seq_global_id
                seq_global_id += 1

                arrival_times = ensure_numpy(seq.get("arrival_times", []))
                checkins = ensure_numpy(seq.get("checkins", []))
                marks = ensure_numpy(seq.get("marks", []))

                gps = seq.get("gps", None)
                if gps is None:
                    gps = [[np.nan, np.nan] for _ in range(len(arrival_times))]

                conds = {}
                for k in ["condition1", "condition2", "condition3", "condition4", "condition5", "condition6"]:
                    v = seq.get(k, None)
                    if v is None:
                        conds[k] = np.full(len(arrival_times), np.nan)
                    else:
                        conds[k] = ensure_numpy(v)

                L = len(arrival_times)

                # sequence-level stats
                if L >= 1:
                    t_min = float(np.min(arrival_times))
                    t_max = float(np.max(arrival_times))
                    duration = float(t_max - t_min)
                else:
                    t_min = t_max = duration = np.nan

                if L >= 2:
                    dts = np.diff(arrival_times.astype(np.float64))
                    dt_min = float(np.min(dts))
                    dt_med = float(np.median(dts))
                    dt_mean = float(np.mean(dts))
                    dt_max = float(np.max(dts))
                else:
                    dt_min = dt_med = dt_mean = dt_max = np.nan

                num_unique_pois = int(len(set(int(x) for x in checkins.tolist()))) if L > 0 else 0
                num_unique_cats = int(len(set(int(x) for x in marks.tolist()))) if L > 0 else 0

                pm = seq.get("po_matrix", None)
                if pm is None:
                    pm_size = 0
                    po_edges = []
                    po_edges_count = 0
                    po_density = np.nan
                else:
                    pm_np = ensure_numpy(pm)
                    pm_size = int(pm_np.shape[0])
                    po_edges = po_edges_from_matrix(pm_np, threshold=threshold)
                    po_edges_count = int(len(po_edges))
                    po_density = float((pm_np > threshold).mean())

                w_seq.writerow([
                    dataset, split, seq_id,
                    L,
                    t_min, t_max, duration,
                    dt_min, dt_med, dt_mean, dt_max,
                    num_unique_pois, num_unique_cats,
                    pm_size, po_edges_count, po_density,
                ])

                # edges export
                for (a, b, val) in po_edges:
                    w_ed.writerow([dataset, split, seq_id, a, b, val])

                # events export
                for i in range(L):
                    at = float(arrival_times[i])
                    inter = float(arrival_times[i] - arrival_times[i - 1]) if i > 0 else np.nan

                    poi = int(checkins[i]) if L > 0 else np.nan
                    cat = int(marks[i]) if L > 0 else np.nan

                    lat, lon = gps[i] if i < len(gps) else (np.nan, np.nan)
                    w_ev.writerow([
                        dataset, split, seq_id, i,
                        at, inter,
                        poi, cat,
                        float(lat), float(lon),
                        int(conds["condition1"][i]) if not np.isnan(conds["condition1"][i]) else "",
                        int(conds["condition2"][i]) if not np.isnan(conds["condition2"][i]) else "",
                        int(conds["condition3"][i]) if not np.isnan(conds["condition3"][i]) else "",
                        int(conds["condition4"][i]) if not np.isnan(conds["condition4"][i]) else "",
                        int(conds["condition5"][i]) if not np.isnan(conds["condition5"][i]) else "",
                        int(conds["condition6"][i]) if not np.isnan(conds["condition6"][i]) else "",
                    ])

    print(f"[OK] Exported:\n  {events_csv}\n  {seq_csv}\n  {edges_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data", help="Root directory containing dataset folders")
    parser.add_argument("--datasets", type=str, nargs="+", default=["Istanbul_PO1_OOD", "NewYork_PO1_OOD"])
    parser.add_argument("--out-dir", type=str, default="./csv_export")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for po_matrix edges")
    args = parser.parse_args()

    for ds in args.datasets:
        export_dataset(args.data_root, ds, args.out_dir, threshold=args.threshold)


if __name__ == "__main__":
    main()