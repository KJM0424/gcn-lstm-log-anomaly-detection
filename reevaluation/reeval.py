"""GCN-LSTM 로그 이상 탐지 재평가.

원본 노트북(Graph_LSTM.ipynb 등)의 모델 구조와 하이퍼파라미터를 그대로 두고,
데이터를 나누는 방식만 바꿔 다시 학습·평가한다.

- random: 원본 방식. 블록을 무작위로 나누고, GCN과 LSTM이 서로 다른 분할을 쓴다.
- group : 같은 이벤트 시퀀스는 학습과 테스트 중 한쪽에만 들어간다. 두 단계가 같은 분할을 쓴다.

같은 (시퀀스, 라벨, 유형)을 가진 블록은 한 단위로 묶고 개수를 가중치로 쓴다.
학습 때는 개수에 비례해 뽑으므로 블록을 하나씩 학습하는 것과 기댓값이 같다.

사용:
    python reevaluation/reeval.py --protocol group --seed 0 --variants orig,fixed,seqlstm --out reevaluation/results/group_seed0.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# Loghub HDFS_v1의 전처리 파일(Event_traces.csv, anomaly_label.csv, HDFS.log_templates.csv)이 있는 폴더.
# 기본값은 저장소의 data/ 이고 --data로 바꿀 수 있다. 데이터는 용량 때문에 저장소에 넣지 않았다.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_SEQ_LEN = 100
BATCH = 32
EVENT_RE = re.compile(r"E\d+")


# ---------- 데이터 ----------

def load_units():
    """블록을 (시퀀스, 라벨, 유형) 단위로 묶는다. 원본처럼 이벤트 1개 이하 블록은 뺀다."""
    templates = pd.read_csv(DATA_DIR / "HDFS.log_templates.csv")
    event_ids = list(templates["EventId"].unique())
    idx = {e: i for i, e in enumerate(event_ids)}
    traces = pd.read_csv(DATA_DIR / "Event_traces.csv", usecols=["BlockId", "Type", "Features"])
    labels = pd.read_csv(DATA_DIR / "anomaly_label.csv")
    labels["y"] = labels["Label"].map({"Normal": 0, "Anomaly": 1})
    df = traces.merge(labels[["BlockId", "y"]], on="BlockId", how="inner")
    df["Type"] = df["Type"].fillna(-1).astype(int)

    unit_of: dict[tuple, int] = {}
    units: list[tuple] = []
    block_unit = np.empty(len(df), dtype=np.int64)
    keep = np.ones(len(df), dtype=bool)
    for i, (feat, y, t) in enumerate(zip(df["Features"], df["y"], df["Type"])):
        seq = tuple(idx[e] for e in EVENT_RE.findall(feat) if e in idx)
        if len(seq) <= 1:
            keep[i] = False
            continue
        key = (seq, int(y), int(t))
        u = unit_of.get(key)
        if u is None:
            u = len(units)
            unit_of[key] = u
            units.append(key)
        block_unit[i] = u
    return units, block_unit[keep], len(event_ids)


def split_counts(units, block_unit, protocol, seed):
    """단위별 학습·테스트 블록 수. (gcn_train, gcn_test, lstm_train, lstm_test)"""
    n_units = len(units)
    rng = np.random.default_rng(seed)
    if protocol == "random":
        out = []
        for _ in range(2):  # GCN과 LSTM이 서로 다른 무작위 분할을 쓴다 (원본과 같음)
            perm = rng.permutation(len(block_unit))
            cut = int(0.8 * len(block_unit))
            tr = np.bincount(block_unit[perm[:cut]], minlength=n_units)
            te = np.bincount(block_unit[perm[cut:]], minlength=n_units)
            out += [tr, te]
        return tuple(out)
    # group: 시퀀스 단위로 나눈다. 이상이 한 번이라도 있는 시퀀스와 아닌 시퀀스를 따로 섞어 비율을 맞춘다.
    total = np.bincount(block_unit, minlength=n_units)
    seq_units = defaultdict(list)
    for u, (seq, _, _) in enumerate(units):
        seq_units[seq].append(u)
    seqs = list(seq_units)
    has_anom = np.array([any(units[u][1] == 1 for u in seq_units[s]) for s in seqs])
    test_seq = np.zeros(len(seqs), dtype=bool)
    for flag in (True, False):
        ids = np.where(has_anom == flag)[0]
        ids = rng.permutation(ids)
        test_seq[ids[: int(round(0.2 * len(ids)))]] = True
    in_test = np.zeros(n_units, dtype=bool)
    for s, is_test in zip(seqs, test_seq):
        if is_test:
            in_test[seq_units[s]] = True
    tr = np.where(in_test, 0, total)
    te = np.where(in_test, total, 0)
    return tr, te, tr, te


# ---------- 모델 (원본 노트북과 같은 구조) ----------

class GNN(nn.Module):
    def __init__(self, n_feat, hidden=64, dropout=0.6):
        super().__init__()
        self.conv1 = GCNConv(n_feat, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(dropout)

    def node_embeddings(self, x, edge_index):
        x = self.drop(F.relu(self.conv1(x, edge_index)))
        x = self.drop(F.relu(self.conv2(x, edge_index)))
        return x

    def forward(self, x, edge_index, batch):
        x = self.node_embeddings(x, edge_index)
        return self.lin(self.drop(global_mean_pool(x, batch)))


class LSTMModel(nn.Module):
    def __init__(self, input_size, n_type, hidden=128, layers=2, dropout=0.2, embed_events=0):
        super().__init__()
        # embed_events > 0이면 GCN 임베딩 대신 이벤트 번호를 직접 임베딩한다 (GCN 없는 순서 모델)
        self.embed = nn.Embedding(embed_events + 1, input_size, padding_idx=0) if embed_events else None
        self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=dropout)
        self.fc_label = nn.Linear(hidden, 2)
        self.fc_type = nn.Linear(hidden, n_type)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, lengths, read):
        if self.embed is not None:
            x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.drop(out)
        if read == "last_padded":  # 원본: 패딩까지 지난 마지막 시점
            out = out[:, -1, :]
        else:  # 실제 마지막 이벤트의 시점
            out = out[torch.arange(out.size(0)), lengths - 1]
        return self.fc_label(out), self.fc_type(out)


# ---------- 평가 ----------

def prf(pred, y, w):
    pred, y, w = np.asarray(pred), np.asarray(y), np.asarray(w, dtype=float)
    tp = w[(pred == 1) & (y == 1)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    fn = w[(pred == 0) & (y == 1)].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn)}


def report(pred, y, test_count):
    """블록 기준(개수 가중)과 시퀀스 기준(단위마다 1)을 함께 낸다."""
    mask = test_count > 0
    pred, y, w = np.asarray(pred)[mask], np.asarray(y)[mask], test_count[mask]
    return {"block": prf(pred, y, w), "unit": prf(pred, y, np.ones(len(w)))}


def make_graph(seq, n_events):
    src, dst = list(seq[:-1]), list(seq[1:])
    return torch.eye(n_events), torch.tensor([src, dst], dtype=torch.long)


def sampler(weights, n, seed):
    g = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), int(n), replacement=True, generator=g)


def train_gcn(units, n_events, train_count, epochs, seed, max_steps, log):
    graphs = []
    for seq, y, _ in units:
        x, ei = make_graph(seq, n_events)
        graphs.append(Data(x=x, edge_index=ei, y=torch.tensor([y])))
    train_ids = np.where(train_count > 0)[0]
    data = [graphs[i] for i in train_ids]
    loader = GeoLoader(data, batch_size=BATCH, sampler=sampler(train_count[train_ids], train_count.sum(), seed))
    model = GNN(n_events)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for ep in range(epochs):
        model.train()
        tot, steps = 0.0, 0
        for b in loader:
            opt.zero_grad()
            loss = F.cross_entropy(model(b.x, b.edge_index, b.batch), b.y)
            loss.backward()
            opt.step()
            tot += loss.item()
            steps += 1
            if max_steps and steps >= max_steps:
                break
        log(f"GCN epoch {ep + 1}/{epochs} loss {tot / steps:.4f}")
    model.eval()
    preds, embs = [], []
    with torch.no_grad():
        for g, (seq, _, _) in zip(graphs, units):
            node = model.node_embeddings(g.x, g.edge_index)
            pooled = node.mean(dim=0, keepdim=True)
            preds.append(int(model.lin(pooled).argmax(1)))
            embs.append(node[list(seq[:MAX_SEQ_LEN])].clone())
    return np.array(preds), embs


class SeqSet(Dataset):
    def __init__(self, ids, seqs, units):
        self.ids, self.seqs, self.units = ids, seqs, units

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        u = self.ids[i]
        return self.seqs[u], self.units[u][1], self.units[u][2], u


def collate(batch):
    seqs = [b[0] for b in batch]
    lengths = torch.tensor([len(s) for s in seqs])
    return (pad_sequence(seqs, batch_first=True), lengths,
            torch.tensor([b[1] for b in batch]), torch.tensor([b[2] for b in batch]),
            torch.tensor([b[3] for b in batch]))


def train_lstm(units, seqs, train_count, test_count, read, epochs, seed, max_steps, log, embed_events=0, name="LSTM"):
    n_type = max(t for _, _, t in units) + 1
    input_size = 64
    torch.manual_seed(seed)
    model = LSTMModel(input_size, n_type, embed_events=embed_events)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_ids = np.where(train_count > 0)[0]
    loader = DataLoader(SeqSet(train_ids, seqs, units), batch_size=BATCH, collate_fn=collate,
                        sampler=sampler(train_count[train_ids], train_count.sum(), seed + 1))
    for ep in range(epochs):
        model.train()
        tot, steps = 0.0, 0
        for x, lengths, y, t, _ in loader:
            opt.zero_grad()
            out_y, out_t = model(x, lengths, read)
            loss = F.cross_entropy(out_y, y)
            valid = t != -1
            if valid.any():
                loss = loss + F.cross_entropy(out_t[valid], t[valid])
            loss.backward()
            opt.step()
            tot += loss.item()
            steps += 1
            if max_steps and steps >= max_steps:
                break
        log(f"{name} epoch {ep + 1}/{epochs} loss {tot / steps:.4f}")
    model.eval()
    test_ids = np.where(test_count > 0)[0]
    pred = np.zeros(len(units), dtype=np.int64)
    with torch.no_grad():
        for x, lengths, _, _, u in DataLoader(SeqSet(test_ids, seqs, units), batch_size=BATCH, collate_fn=collate):
            pred[u.numpy()] = model(x, lengths, read)[0].argmax(1).numpy()
    return pred


# ---------- 기준선 ----------

def lookup_baseline(units, train_count):
    """학습셋에서 본 시퀀스의 다수 라벨로 답한다. 처음 보는 시퀀스는 정상으로 답한다."""
    votes = defaultdict(Counter)
    for (seq, y, _), c in zip(units, train_count):
        if c:
            votes[seq][y] += int(c)
    return np.array([votes[seq].most_common(1)[0][0] if seq in votes else 0 for seq, _, _ in units])


def count_logreg(units, n_events, train_count):
    """이벤트 발생 횟수만 쓰는 로지스틱 회귀."""
    X = np.zeros((len(units), n_events))
    for i, (seq, _, _) in enumerate(units):
        for e in seq:
            X[i, e] += 1
    y = np.array([u[1] for u in units])
    m = train_count > 0
    clf = LogisticRegression(max_iter=2000)
    clf.fit(np.log1p(X[m]), y[m], sample_weight=train_count[m])
    return clf.predict(np.log1p(X))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["random", "group"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=0, help="에폭당 스텝 상한 (동작 확인용)")
    ap.add_argument("--variants", default="orig", help="쉼표로 구분: orig,fixed,seqlstm")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=None, help="전처리 CSV가 있는 폴더 (기본: 저장소의 data/)")
    a = ap.parse_args()
    if a.data:
        global DATA_DIR
        DATA_DIR = Path(a.data)
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.0f}s] {msg}", flush=True)

    units, block_unit, n_events = load_units()
    y = np.array([u[1] for u in units])
    total = np.bincount(block_unit, minlength=len(units))
    n_seq = len({u[0] for u in units})
    log(f"블록 {len(block_unit):,} / 단위 {len(units):,} / 고유 시퀀스 {n_seq:,} / 이상 블록 {int(total[y == 1].sum()):,}")

    g_tr, g_te, l_tr, l_te = split_counts(units, block_unit, a.protocol, a.seed)
    seen = {u[0] for u, c in zip(units, l_tr) if c}
    overlap = sum(int(c) for u, c in zip(units, l_te) if c and u[0] in seen) / max(int(l_te.sum()), 1)
    res = {
        "protocol": a.protocol, "seed": a.seed, "epochs": a.epochs, "max_steps": a.max_steps,
        "data": {"blocks": int(len(block_unit)), "units": len(units), "unique_sequences": n_seq,
                 "anomaly_blocks": int(total[y == 1].sum()),
                 "test_blocks": int(l_te.sum()), "test_anomaly_blocks": int(l_te[y == 1].sum()),
                 "test_units": int((l_te > 0).sum()),
                 "test_blocks_whose_sequence_is_in_train": round(overlap, 4)},
        "models": {},
    }
    log(f"분할 {a.protocol}: 테스트 블록 {int(l_te.sum()):,} (이상 {int(l_te[y == 1].sum()):,}), 학습셋과 같은 시퀀스 비율 {overlap:.4f}")

    res["models"]["조회표"] = report(lookup_baseline(units, l_tr), y, l_te)
    res["models"]["횟수 로지스틱 회귀"] = report(count_logreg(units, n_events, l_tr), y, l_te)
    log(f"조회표 {res['models']['조회표']['block']} / 로지스틱 {res['models']['횟수 로지스틱 회귀']['block']}")

    gcn_pred, embs = train_gcn(units, n_events, g_tr, a.epochs, a.seed, a.max_steps, log)
    res["models"]["GCN 단독"] = report(gcn_pred, y, g_te)
    log(f"GCN 단독 {res['models']['GCN 단독']['block']}")

    variants = a.variants.split(",")
    if "orig" in variants:
        p = train_lstm(units, embs, l_tr, l_te, "last_padded", a.epochs, a.seed, a.max_steps, log, name="GCN-LSTM(원본)")
        res["models"]["GCN-LSTM (원본 방식)"] = report(p, y, l_te)
        log(f"GCN-LSTM 원본 {res['models']['GCN-LSTM (원본 방식)']['block']}")
    if "fixed" in variants:
        p = train_lstm(units, embs, l_tr, l_te, "last_valid", a.epochs, a.seed, a.max_steps, log, name="GCN-LSTM(패딩 수정)")
        res["models"]["GCN-LSTM (패딩 수정)"] = report(p, y, l_te)
        log(f"GCN-LSTM 패딩 수정 {res['models']['GCN-LSTM (패딩 수정)']['block']}")
    if "seqlstm" in variants:
        ids = [torch.tensor([e + 1 for e in seq[:MAX_SEQ_LEN]]) for seq, _, _ in units]
        p = train_lstm(units, ids, l_tr, l_te, "last_valid", a.epochs, a.seed, a.max_steps, log, embed_events=n_events, name="LSTM(GCN 없음)")
        res["models"]["LSTM (GCN 없음, 순서 입력)"] = report(p, y, l_te)
        log(f"LSTM GCN 없음 {res['models']['LSTM (GCN 없음, 순서 입력)']['block']}")

    res["seconds"] = round(time.time() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"저장 {a.out}")


if __name__ == "__main__":
    main()
