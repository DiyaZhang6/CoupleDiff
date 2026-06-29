# pre_check.py 快速过滤脚本
import torch
from pathlib import Path
import pandas as pd

pdb_ids = pd.read_csv("/home/zdy/Project2/data/train.csv", header=None)[0].tolist()
valid_ids = []
g_base = Path("/home/zdy/Project2/data/processed_data/graph")
l_base = Path("/workspace/guest/zdy/Project2/data/processed_data/label")

for pid in pdb_ids:
    p_bb = g_base / pid / f"{pid}_backbone.pt"
    p_lbl = l_base / pid / f"{pid}_labels.pt"
    if p_bb.exists() and p_lbl.exists():
        valid_ids.append(pid)

pd.DataFrame(valid_ids).to_csv("train_split_cleaned.csv", index=False, header=False)