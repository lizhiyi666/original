import torch, os, numpy as np

root="./data/NewYork_PO1"
name="NewYork_PO1"
train=torch.load(os.path.join(root,f"{name}_train.pkl"), weights_only=False)
test=torch.load(os.path.join(root,f"{name}_test.pkl"), weights_only=False)
seqs=train["sequences"]+test["sequences"]

from collections import Counter
cnt=Counter()
for s in seqs:
    pm=s.get("po_matrix")
    if pm is None: 
        continue
    pm=np.array(pm)
    idx=np.argwhere(pm>0.5)
    edges=set((int(i),int(j)) for i,j in idx if i!=j)
    for e in edges:
        cnt[e]+=1

supports=sorted(cnt.values())
print("num_edges", len(cnt))
print("min/median/max support", supports[0], supports[len(supports)//2], supports[-1])