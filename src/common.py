"""Shared helpers: loading derived EATD features, participant-level representations, metrics."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20261106
QUESTIONS = ["positive", "neutral", "negative"]
AUDIO_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
TEXT_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]
A_COLS = [f"audio_p_{l}" for l in AUDIO_LABELS]
T_COLS = [f"text_p_{l}" for l in TEXT_LABELS]


def load_derived(states_csv, egemaps_csv=None):
    """Return one row per answer with audio/text probability vectors (and eGeMAPS if given)."""
    S = pd.read_csv(states_csv)
    S["a_prob"] = list(S[A_COLS].to_numpy(dtype=float))
    S["t_prob"] = list(S[T_COLS].to_numpy(dtype=float))
    if egemaps_csv:
        G = pd.read_csv(egemaps_csv)
        gcols = [c for c in G.columns if c not in ("participant", "question")]
        G["egemaps"] = list(G[gcols].to_numpy(dtype=float))
        S = S.merge(G[["participant", "question", "egemaps"]], on=["participant", "question"], how="left")
    S = S.rename(columns={"participant": "pid", "question": "cond", "depressed": "y"})
    return S


def entropy(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def participant_matrix(df, mods, agg, use_entropy=False):
    """mods: string of A (audio probs), T (text probs), G (raw eGeMAPS).
    agg: 'mean' | 'meansd' | 'cond' (keep the three answers separate)."""
    rows = []
    for _, g in df.groupby("pid"):
        g = g.set_index("cond").reindex(QUESTIONS)
        feats = []
        for m in mods:
            key = {"A": "a_prob", "T": "t_prob", "G": "egemaps"}[m]
            M = np.vstack(g[key].values).astype(float)
            if agg == "cond":
                feats.append(M.reshape(-1))
            else:
                feats.append(np.nanmean(M, 0))
                if agg == "meansd":
                    feats.append(np.nanstd(M, 0))
            if use_entropy and m in "AT":
                feats.append(np.array([entropy(v) if np.isfinite(v).all() else np.nan for v in M]))
        rows.append(np.concatenate(feats))
    lab = df.groupby("pid")[["y", "split", "sds_standard"]].first()
    return np.nan_to_num(np.vstack(rows)), lab


def make_model(kind):
    if kind == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    return RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=3, random_state=SEED, n_jobs=-1)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    se, sp = tp / (tp + fn), tn / (tn + fp)
    return dict(AUC=roc_auc_score(y, p), bal_acc=(se + sp) / 2, sens=se, spec=sp,
                F1_dep=f1_score(y, pred, zero_division=0), macro_F1=f1_score(y, pred, average="macro"),
                n=len(y), n_dep=int(y.sum()))


def bootstrap_auc_ci(y, p, B=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p[i]))
    return np.percentile(aucs, [2.5, 97.5])


def cv_probs(X, y, kind="LR", reps=20):
    """Out-of-fold probabilities averaged over repeated stratified 5-fold CV."""
    if kind == "RF":
        reps = 5
    P = np.zeros((reps, len(y)))
    for r in range(reps):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(X, y):
            P[r, te] = make_model(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return P.mean(0)


def permutation_p(X, y, observed_auc, n=500, kind="LR"):
    rng = np.random.default_rng(SEED)
    count = 0
    for i in range(n):
        yp = rng.permutation(y)
        p = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + i).split(X, yp):
            p[te] = make_model(kind).fit(X[tr], yp[tr]).predict_proba(X[te])[:, 1]
        count += roc_auc_score(yp, p) >= observed_auc
    return (count + 1) / (n + 1)


def fdr_bh(p):
    p = np.asarray(p)
    order = np.argsort(p)[::-1]
    m = len(p)
    q = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        q[i] = prev
    return q
