# save as tools/check_dataset_nan.py
import os, torch, numpy as np

def check(name, root="./data"):
    for split in ["train", "test"]:
        p = os.path.join(root, name, f"{name}_{split}.pkl")
        d = torch.load(p, weights_only=False)
        bad = 0
        for i, s in enumerate(d["sequences"]):
            t = np.array(s["arrival_times"], dtype=np.float64)
            if np.any(~np.isfinite(t)):
                print("NaN/Inf time:", split, i); bad += 1; continue
            if len(t) < 2:
                continue
            dt = np.diff(t)
            if np.any(dt < 0):
                print("Non-monotonic time:", split, i, "min_dt=", dt.min()); bad += 1
            if np.any(dt == 0):
                # 这条很关键：很多 intensity 实现对 dt==0 会直接炸成 nan
                print("Zero dt:", split, i); bad += 1
        print(name, split, "bad_seqs=", bad)

if __name__ == "__main__":
    check("NewYork")