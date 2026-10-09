"""Prepare the TSMC2014 New York data for the Marionette/PO1 pipeline.

The source file contains absolute UTC timestamps, venue metadata and user
check-ins.  The model expects one-day trajectories with a shared discrete
vocabulary, six contextual conditions and per-sequence partial-order data.
"""

from __future__ import annotations

import argparse
import copy
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


SPECIAL_TOKEN_COUNT = 4
CATEGORY_TOKEN_OFFSET = 4
POI_TOKEN_OFFSET = 13  # 4 special tokens + 9 category tokens
NUM_CATEGORIES = 9
PO_ENCODING_DIM = 32


def _torch_save(obj: object, path: Path) -> None:
    """Serialize ``obj`` to ``path`` without handing a path string to torch.

    ``torch.save`` forwards ``str`` paths to the C++ zip writer, which probes the
    parent directory through the narrow-character Windows API.  When the project
    lives under a directory with non-ASCII characters the probe fails and torch
    reports ``Parent directory ... does not exist`` even though it does exist.
    Passing an already-opened file object keeps path resolution inside Python,
    which uses the wide-character API, so the encoding round-trip is skipped.
    """
    with open(path, "wb") as handle:
        torch.save(obj, handle)


def map_to_root_category(category_name: object) -> str:
    """Map raw Foursquare category names to the nine project categories."""
    name = str(category_name).lower()
    if any(w in name for w in ["home", "residential", "neighborhood", "housing", "apartment", "condo"]):
        return "Residence"
    if any(w in name for w in [
        "subway", "train", "bus", "airport", "road", "bridge", "ferry", "light rail",
        "travel", "taxi", "parking", "gas station", "rest area", "hotel", "moving target",
    ]):
        return "Travel & Transport"
    if any(w in name for w in ["bar", "nightlife", "club", "pub", "brewery", "beer"]):
        return "Nightlife"
    if any(w in name for w in [
        "gym", "fitness", "park", "outdoors", "plaza", "athletic", "beach", "playground",
        "scenic", "garden", "pool", "campground", "cemetery", "harbor", "river", "ski",
    ]):
        return "Outdoors & Recreation"
    if any(w in name for w in [
        "theater", "music", "entertainment", "stadium", "art", "museum", "arcade", "casino",
        "comedy", "zoo", "aquarium", "cinema", "concert", "racetrack",
    ]):
        return "Arts & Entertainment"
    if any(w in name for w in [
        "office", "college", "medical", "building", "bank", "government", "school", "university",
        "student", "library", "studio", "factory", "community", "embassy", "post office",
    ]):
        return "Professional & Education"
    if any(w in name for w in [
        "restaurant", "coffee", "food", "pizza", "burger", "sandwich", "bakery", "caf", "diner",
        "snack", "joint", "steakhouse", "dessert", "breakfast", "salad", "soup", "ice cream",
        "bagel", "donut", "taco", "burrito", "mac & cheese", "wings", "tea",
    ]):
        return "Food"
    if any(w in name for w in [
        "shop", "store", "deli", "pharmacy", "drugstore", "mall", "laundry", "salon", "market",
        "spa", "garage", "bodega", "service", "car wash", "dealership",
    ]):
        return "Shop & Service"
    return "Other"


def season_token(month: int) -> int:
    if month in (3, 4, 5):
        return 34
    if month in (6, 7, 8):
        return 35
    if month in (9, 10, 11):
        return 36
    return 37


def _parse_source(path: Path) -> pd.DataFrame:
    columns = [
        "User_ID", "Venue_ID", "Venue_Category_ID", "Venue_Category_Name",
        "Latitude", "Longitude", "Timezone_Offset", "UTC_Time",
    ]
    frame = pd.read_csv(path, sep="\t", header=None, names=columns, encoding="latin-1")
    frame["UTC_Time"] = pd.to_datetime(
        frame["UTC_Time"], format="%a %b %d %H:%M:%S +0000 %Y", utc=True, errors="raise"
    )
    frame["Local_Time"] = frame["UTC_Time"] + pd.to_timedelta(frame["Timezone_Offset"], unit="m")
    frame = frame.sort_values(["User_ID", "Local_Time"], kind="mergesort")
    frame = frame.drop_duplicates(["User_ID", "Local_Time"], keep="first")
    frame["Root_Category"] = frame["Venue_Category_Name"].map(map_to_root_category)
    frame["Local_Date"] = frame["Local_Time"].dt.date
    return frame.reset_index(drop=True)


def _filter_pois(frame: pd.DataFrame, min_poi_frequency: int) -> pd.DataFrame:
    if min_poi_frequency <= 1:
        return frame
    counts = frame["Venue_ID"].value_counts()
    keep = set(counts[counts >= min_poi_frequency].index)
    return frame[frame["Venue_ID"].isin(keep)].copy()


def _build_daily_sequences(frame: pd.DataFrame, min_sequence_length: int) -> list[dict]:
    calendar = USFederalHolidayCalendar()
    local_times_naive = frame["Local_Time"].dt.tz_localize(None)
    holiday_dates = set(calendar.holidays(
        start=local_times_naive.min().normalize(),
        end=local_times_naive.max().normalize(),
    ).date)

    sequences: list[dict] = []
    for (user_id, local_date), group in frame.groupby(["User_ID", "Local_Date"], sort=True):
        group = group.sort_values("Local_Time", kind="mergesort")
        if len(group) < min_sequence_length:
            continue

        # The category is derived from the same row as the POI.  This avoids
        # a second, potentially inconsistent POI/category mapping step.
        category_names = group["Root_Category"].tolist()
        sequences.append({
            "_user_id": user_id,
            "_local_date": local_date,
            "_category_names": category_names,
            "_local_times": group["Local_Time"].tolist(),
            "_venue_ids": group["Venue_ID"].tolist(),
            "_lat": group["Latitude"].astype(float).tolist(),
            "_lon": group["Longitude"].astype(float).tolist(),
            "condition1_value": 25 + group["Local_Time"].iloc[0].weekday(),
            "condition2_value": 33 if local_date in holiday_dates else 32,
            "condition3_value": season_token(group["Local_Time"].iloc[0].month),
            "condition4_value": 38,
            "condition5_value": 39,
            "condition6_value": 40,
        })
    return sequences


def _make_po_matrix(category_tokens: np.ndarray, category_mapping: dict[int, int]) -> np.ndarray:
    matrix = np.zeros((NUM_CATEGORIES, NUM_CATEGORIES), dtype=np.float32)
    first = np.full(NUM_CATEGORIES, np.inf)
    last = np.full(NUM_CATEGORIES, -np.inf)
    for position, token in enumerate(category_tokens.tolist()):
        category_index = category_mapping[int(token)]
        first[category_index] = min(first[category_index], position)
        last[category_index] = max(last[category_index], position)
    for source in range(NUM_CATEGORIES):
        for target in range(NUM_CATEGORIES):
            if source != target and last[source] < first[target] and np.isfinite(first[target]):
                matrix[source, target] = 1.0
    return matrix


def _fit_po_encoder(train_sequences: list[dict]) -> tuple[StandardScaler, TruncatedSVD]:
    flattened = np.stack([seq["po_matrix"].reshape(-1) for seq in train_sequences])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(flattened)
    n_components = min(PO_ENCODING_DIM, scaled.shape[0] - 1, scaled.shape[1] - 1)
    if n_components < 1:
        raise ValueError("Not enough training sequences to fit a PO encoder")
    svd = TruncatedSVD(n_components=n_components, random_state=135398)
    svd.fit(scaled)
    return scaler, svd


def _materialize_sequences(
    raw_sequences: list[dict],
    poi_token_by_raw: dict[object, int],
    category_token_by_name: dict[str, int],
    poi_category: dict[int, int],
    category_mapping: dict[int, int],
    scaler: StandardScaler,
    svd: TruncatedSVD,
) -> list[dict]:
    output = []
    for seq_idx, raw in enumerate(raw_sequences):
        checkins = np.asarray([poi_token_by_raw[x] for x in raw["_venue_ids"]], dtype=np.int64)
        marks = np.asarray([category_token_by_name[x] for x in raw["_category_names"]], dtype=np.int64)
        # Enforce the invariant used by evaluation and by the diffusion model.
        marks = np.asarray([poi_category[int(poi)] for poi in checkins], dtype=np.int64)
        times = np.asarray([
            value.hour + value.minute / 60.0 + value.second / 3600.0 + value.microsecond / 3.6e9
            for value in raw["_local_times"]
        ], dtype=np.float32)
        if len(times) < 2 or np.any(np.diff(times) <= 0):
            # Cross-midnight events are split by date; repeated local times
            # were removed before grouping.  This guard catches DST anomalies.
            continue

        po_matrix = _make_po_matrix(marks, category_mapping)
        po_flat = po_matrix.reshape(1, -1)
        po_encoding = svd.transform(scaler.transform(po_flat))[0].astype(np.float32)
        n = len(checkins)
        cond_values = [
            raw["condition1_value"], raw["condition2_value"], raw["condition3_value"],
            raw["condition4_value"], raw["condition5_value"], raw["condition6_value"],
        ]
        seq = {
            "arrival_times": times,
            "marks": marks,
            "checkins": checkins,
            "gps": np.asarray(list(zip(raw["_lat"], raw["_lon"])), dtype=np.float64),
            "condition1": np.full(n, cond_values[0], dtype=np.int64),
            "condition2": np.full(n, cond_values[1], dtype=np.int64),
            "condition3": np.full(n, cond_values[2], dtype=np.int64),
            "condition4": np.full(n, cond_values[3], dtype=np.int64),
            "condition5": np.full(n, cond_values[4], dtype=np.int64),
            "condition6": np.full(n, cond_values[5], dtype=np.int64),
            "condition1_indicator": np.full(24, cond_values[0], dtype=np.int64),
            "condition2_indicator": np.full(24, cond_values[1], dtype=np.int64),
            "condition3_indicator": np.full(24, cond_values[2], dtype=np.int64),
            "condition4_indicator": np.full(24, cond_values[3], dtype=np.int64),
            "condition5_indicator": np.full(24, cond_values[4], dtype=np.int64),
            "condition6_indicator": np.full(24, cond_values[5], dtype=np.int64),
            "po_matrix": po_matrix,
            "po_encoding": po_encoding,
            "seq_idx": seq_idx,
            "source_user_id": raw["_user_id"],
            "source_local_date": str(raw["_local_date"]),
        }
        output.append(seq)
    return output


def _validate_dataset(data: dict, min_sequence_length: int) -> None:
    assert float(data["t_max"]) == 24.0
    assert data["num_marks"] == NUM_CATEGORIES
    assert set(data["poi_gps"]) == set(data["poi_category"])
    assert data["svd_components"].shape == (PO_ENCODING_DIM, NUM_CATEGORIES * NUM_CATEGORIES)
    for index, seq in enumerate(data["sequences"]):
        n = len(seq["arrival_times"])
        assert n >= min_sequence_length, (index, n)
        assert len(seq["marks"]) == len(seq["checkins"]) == n, index
        assert np.all(seq["arrival_times"] >= 0) and np.all(seq["arrival_times"] < 24), index
        assert np.all(np.diff(seq["arrival_times"]) > 0), index
        assert np.all((seq["marks"] >= 4) & (seq["marks"] < 13)), index
        assert np.all((seq["checkins"] >= POI_TOKEN_OFFSET) & (seq["checkins"] < POI_TOKEN_OFFSET + data["num_pois"])), index
        assert seq["po_matrix"].shape == (9, 9), index
        assert seq["po_encoding"].shape == (PO_ENCODING_DIM,), index
        for condition in range(1, 7):
            assert len(seq[f"condition{condition}_indicator"]) == 24, index
        for poi, category in zip(seq["checkins"], seq["marks"]):
            assert data["poi_category"][int(poi)] == int(category), (index, int(poi))


def prepare_dataset(
    source_path: Path,
    output_dir: Path,
    min_sequence_length: int = 5,
    min_poi_frequency: int = 10,
    train_fraction: float = 0.6,
    seed: int = 42,
) -> tuple[Path, Path]:
    source_path = Path(source_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"New York source file does not exist: {source_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[prepare_newyork_po1] source: {source_path}")
    print(f"[prepare_newyork_po1] output: {output_dir}")
    frame = _filter_pois(_parse_source(source_path), min_poi_frequency)
    raw_sequences = _build_daily_sequences(frame, min_sequence_length)
    if not raw_sequences:
        raise ValueError("No daily sequences satisfy the minimum length")

    # One shared vocabulary is built before the split, while PO encoding is
    # fitted only on the training portion.
    category_names = sorted({name for seq in raw_sequences for name in seq["_category_names"]})
    if len(category_names) != NUM_CATEGORIES:
        raise ValueError(f"Expected {NUM_CATEGORIES} categories, got {len(category_names)}: {category_names}")
    category_token_by_name = {name: CATEGORY_TOKEN_OFFSET + i for i, name in enumerate(category_names)}
    category_mapping = {token: i for i, token in enumerate(category_token_by_name.values())}

    raw_pois = sorted({poi for seq in raw_sequences for poi in seq["_venue_ids"]}, key=str)
    poi_token_by_raw = {poi: POI_TOKEN_OFFSET + i for i, poi in enumerate(raw_pois)}
    poi_gps = {}
    poi_category = {}
    poi_name_to_category = {}
    for _, row in frame.drop_duplicates("Venue_ID").iterrows():
        raw_poi = row["Venue_ID"]
        if raw_poi not in poi_token_by_raw:
            continue
        poi_token = poi_token_by_raw[raw_poi]
        category_token = category_token_by_name[row["Root_Category"]]
        poi_gps[poi_token] = f"{float(row['Latitude'])},{float(row['Longitude'])}"
        poi_category[poi_token] = category_token
        poi_name_to_category[raw_poi] = category_token

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(raw_sequences))
    split = int(len(order) * train_fraction)
    if split <= 0 or split >= len(order):
        raise ValueError("train_fraction must leave non-empty train and test splits")
    train_raw = [raw_sequences[i] for i in order[:split]]
    test_raw = [raw_sequences[i] for i in order[split:]]

    # Build temporary tokenized sequences to derive matrices for SVD fitting.
    def tokenized_for_po(items):
        result = []
        for raw in items:
            marks = np.asarray([category_token_by_name[x] for x in raw["_category_names"]], dtype=np.int64)
            result.append({"po_matrix": _make_po_matrix(marks, category_mapping)})
        return result

    train_po = tokenized_for_po(train_raw)
    test_po = tokenized_for_po(test_raw)
    scaler, svd = _fit_po_encoder(train_po)
    train_sequences = _materialize_sequences(train_raw, poi_token_by_raw, category_token_by_name, poi_category, category_mapping, scaler, svd)
    test_sequences = _materialize_sequences(test_raw, poi_token_by_raw, category_token_by_name, poi_category, category_mapping, scaler, svd)

    common = {
        "t_max": 24.0,
        "num_marks": NUM_CATEGORIES,
        "num_pois": len(poi_gps),
        "poi_gps": poi_gps,
        "poi_category": poi_category,
        "category_mapping": category_mapping,
        "num_categories": NUM_CATEGORIES,
        "po_encoding_dim": PO_ENCODING_DIM,
        "svd_components": torch.tensor(svd.components_, dtype=torch.float32),
        "svd_mean": torch.tensor(scaler.mean_, dtype=torch.float32),
        "svd_scale": torch.tensor(scaler.scale_, dtype=torch.float32),
        "split_seed": seed,
        "train_fraction": train_fraction,
        "min_sequence_length": min_sequence_length,
        "min_poi_frequency": min_poi_frequency,
        "source_file": str(source_path),
    }
    train_data = {**common, "sequences": train_sequences, "num_seqs": len(train_sequences)}
    test_data = {**common, "sequences": test_sequences, "num_seqs": len(test_sequences)}
    _validate_dataset(train_data, min_sequence_length)
    _validate_dataset(test_data, min_sequence_length)

    train_path = output_dir / "NewYork_PO1_train.pkl"
    test_path = output_dir / "NewYork_PO1_test.pkl"
    _torch_save(train_data, train_path)
    _torch_save(test_data, test_path)
    return train_path, test_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--source", type=Path, default=project_root.parent / "dataset_tsmc2014" / "dataset_TSMC2014_NYC.txt")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data" / "NewYork_PO1")
    parser.add_argument("--min-sequence-length", type=int, default=5)
    parser.add_argument("--min-poi-frequency", type=int, default=10)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_path, test_path = prepare_dataset(
        args.source,
        args.output_dir,
        min_sequence_length=args.min_sequence_length,
        min_poi_frequency=args.min_poi_frequency,
        train_fraction=args.train_fraction,
        seed=args.seed,
    )
    print(f"Saved {train_path}")
    print(f"Saved {test_path}")


if __name__ == "__main__":
    main()
