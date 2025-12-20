#!/usr/bin/env python3
# tools/ovr_csv_inspect.py
import csv
from collections import defaultdict
import numpy as np

outdir = "./ovr_diag_v2"  # 修改为你实际输出目录
# 1) top categories with low presence ratio
cat_file = f"{outdir}/category_stats.csv"
cats = []
with open(cat_file) as f:
    r = csv.DictReader(f)
    for row in r:
        cat = row['category']
        test_occ = float(row['test_total_occurrences']) if row['test_total_occurrences']!='' else 0.0
        gen_occ = float(row['gen_total_occurrences']) if row['gen_total_occurrences']!='' else 0.0
        test_pres = float(row['test_seq_presence']) if row['test_seq_presence']!='' else 0.0
        gen_pres = float(row['gen_seq_presence']) if row['gen_seq_presence']!='' else 0.0
        occ_ratio = float(row['occurrence_ratio_gen_to_test']) if row['occurrence_ratio_gen_to_test']!='' else np.nan
        pres_ratio = float(row['presence_ratio_gen_to_test']) if row['presence_ratio_gen_to_test']!='' else np.nan
        cats.append((cat, test_occ, gen_occ, test_pres, gen_pres, occ_ratio, pres_ratio))
cats_sorted = sorted(cats, key=lambda x: (0 if np.isnan(x[6]) else x[6]))
print("Top 20 categories by lowest presence_ratio (gen/test):")
for t in cats_sorted[:20]:
    print(t)

# 2) length stats summary
def read_len_csv(p):
    d = {}
    with open(p) as f:
        rows = list(csv.reader(f))
    for r in rows[1:]:
        d[r[0]] = float(r[1])
    return d

test_len = read_len_csv(f"{outdir}/seq_length_stats_test.csv")
gen_len = read_len_csv(f"{outdir}/seq_length_stats_gen.csv")
print("\nSequence length summary (test vs gen):")
for k in ['seq_length_mean','seq_length_median','seq_length_p25','seq_length_p75','seq_length_min','seq_length_max']:
    print(k, "test=", test_len.get(k), "gen=", gen_len.get(k))

# 3) top pairs with highest test support but low coverage (from pair_stats.csv)
pair_file = f"{outdir}/pair_stats.csv"
pairs = []
with open(pair_file) as f:
    r = csv.DictReader(f)
    for row in r:
        A = row['A']; B = row['B']
        support = int(row['support_in_test'])
        gen_support = int(row['gen_support_seq_count'])
        coverage = float(row['coverage']) if row['coverage']!='' else 0.0
        pairs.append((A,B,support,gen_support,coverage))
pairs.sort(key=lambda x: (-x[2], x[4]))  # high support, low coverage
print("\nTop 20 high-support pairs with lowest coverage:")
for p in pairs[:20]:
    print(p)