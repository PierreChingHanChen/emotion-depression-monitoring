"""All depression-detection analyses reported in the paper.

Runs from the derived features in data/derived/ alone. The raw-text TF-IDF comparator
additionally needs the EATD transcripts (--eatd_dir); it is skipped otherwise.

Usage: python src/05_run_experiments.py --out results [--eatd_dir EATD-Corpus] [--n_perm 500]
"""
import argparse
import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.dirname(__file__))
from common import (AUDIO_LABELS, QUESTIONS, SEED, TEXT_LABELS, bootstrap_auc_ci, cv_probs, fdr_bh,  # noqa: E402
                    load_derived, make_model, metrics, participant_matrix, permutation_p)

SPECS = {
    "Audio emotion (mean)": ("A", "mean", False),
    "Text emotion (mean)": ("T", "mean", False),
    "Audio+Text emotion (mean)": ("AT", "mean", False),
    "Audio+Text emotion (mean+SD)": ("AT", "meansd", False),
    "Audio+Text emotion (condition-specific)": ("AT", "cond", False),
    "Audio+Text emotion (condition-specific) + uncertainty": ("AT", "cond", True),
    "Raw eGeMAPS (88, mean)": ("G", "mean", False),
    "Raw eGeMAPS + Audio+Text emotion (condition-specific)": ("GAT", "cond", False),
}
PERMUTED = ["Audio emotion (mean)", "Text emotion (mean)", "Audio+Text emotion (mean+SD)",
            "Audio+Text emotion (condition-specific)", "Raw eGeMAPS (88, mean)"]


def tfidf_model():
    return make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True),
                         LogisticRegression(C=1, class_weight="balanced", max_iter=5000))


def read_transcripts(eatd_dir, pids):
    docs = []
    for pid in pids:
        parts = []
        for q in QUESTIONS:
            for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
                try:
                    parts.append(open(os.path.join(eatd_dir, pid, f"{q}.txt"), encoding=enc).read().strip())
                    break
                except (UnicodeDecodeError, FileNotFoundError):
                    continue
        docs.append(" ".join(parts))
    return np.array(docs, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="data/derived/eatd_affective_states.csv")
    ap.add_argument("--egemaps", default="data/derived/eatd_egemaps.csv")
    ap.add_argument("--eatd_dir", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--n_perm", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = load_derived(a.states, a.egemaps)
    per = df.groupby("pid")[["y", "split", "sds_standard"]].first()

    desc = dict(participants=len(per), depressed=int(per.y.sum()),
                train=int((per.split == "train").sum()), train_depressed=int(per[per.split == "train"].y.sum()),
                val=int((per.split == "val").sum()), val_depressed=int(per[per.split == "val"].y.sum()),
                sds_depressed_mean=per[per.y == 1].sds_standard.mean(), sds_depressed_sd=per[per.y == 1].sds_standard.std(),
                sds_control_mean=per[per.y == 0].sds_standard.mean(), sds_control_sd=per[per.y == 0].sds_standard.std(),
                total_audio_h=df.duration_s.sum() / 3600, mean_answer_s=df.duration_s.mean())
    json.dump(desc, open(os.path.join(a.out, "descriptives.json"), "w"), indent=1)

    # ---- Table 4: participant-level detection (repeated CV and official split)
    rows, perm_rows, store = [], [], {}
    for name, (mods, agg, ent) in SPECS.items():
        X, lab = participant_matrix(df, mods, agg, ent)
        y = lab.y.values
        train = (lab.split == "train").values
        for kind in ["LR", "RF"]:
            p = cv_probs(X, y, kind)
            lo, hi = bootstrap_auc_ci(y, p)
            rows.append(dict(features=name, dim=X.shape[1], model=kind, protocol="20x5-fold CV",
                             **metrics(y, p), AUC_lo=lo, AUC_hi=hi))
            if kind == "LR":
                store[name] = (X, y, p)
            pv = make_model(kind).fit(X[train], y[train]).predict_proba(X[~train])[:, 1]
            lo, hi = bootstrap_auc_ci(y[~train], pv)
            rows.append(dict(features=name, dim=X.shape[1], model=kind, protocol="official split",
                             **metrics(y[~train], pv), AUC_lo=lo, AUC_hi=hi))
    for name in PERMUTED:
        X, y, p = store[name]
        auc = roc_auc_score(y, p)
        perm_rows.append(dict(features=name, AUC=auc, p_perm=permutation_p(X, y, auc, n=a.n_perm), n_perm=a.n_perm))

    if a.eatd_dir:  # raw-text comparator (needs transcripts)
        docs, y = read_transcripts(a.eatd_dir, per.index), per.y.values
        P = np.zeros((20, len(y)))
        for r in range(20):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED + r).split(docs, y):
                P[r, te] = tfidf_model().fit(docs[tr], y[tr]).predict_proba(docs[te])[:, 1]
        p = P.mean(0)
        lo, hi = bootstrap_auc_ci(y, p)
        rows.append(dict(features="Raw text TF-IDF (char 1-3 gram)", model="LR", protocol="20x5-fold CV",
                         **metrics(y, p), AUC_lo=lo, AUC_hi=hi))
        train = (per.split == "train").values
        pv = tfidf_model().fit(docs[train], y[train]).predict_proba(docs[~train])[:, 1]
        lo, hi = bootstrap_auc_ci(y[~train], pv)
        rows.append(dict(features="Raw text TF-IDF (char 1-3 gram)", model="LR", protocol="official split",
                         **metrics(y[~train], pv), AUC_lo=lo, AUC_hi=hi))
        n_text_perm = min(a.n_perm, 200)
        rng, cnt, obs = np.random.default_rng(1), 0, roc_auc_score(y, p)
        for i in range(n_text_perm):
            yp, pp = rng.permutation(y), np.zeros(len(y))
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=i).split(docs, yp):
                pp[te] = tfidf_model().fit(docs[tr], yp[tr]).predict_proba(docs[te])[:, 1]
            cnt += roc_auc_score(yp, pp) >= obs
        perm_rows.append(dict(features="Raw text TF-IDF (char 1-3 gram)", AUC=obs,
                              p_perm=(cnt + 1) / (n_text_perm + 1), n_perm=n_text_perm))
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "E1_detection.csv"), index=False)
    pd.DataFrame(perm_rows).to_csv(os.path.join(a.out, "E1_permutation.csv"), index=False)

    # ---- Table 5: single answers vs aggregated answers (participant-grouped CV)
    Xr = np.nan_to_num(np.vstack([np.concatenate([pa, pt]) for pa, pt in zip(df.a_prob, df.t_prob)]))
    yr, gr = df.y.values, df.pid.values
    P = np.zeros((20, len(yr)))
    for r in range(20):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED + r).split(Xr, yr, gr):
            P[r, te] = make_model("LR").fit(Xr[tr], yr[tr]).predict_proba(Xr[te])[:, 1]
    df["p"] = P.mean(0)
    e2 = [dict(evidence="single answer", **metrics(yr, df.p.values))]
    for k in (2, 3):
        ys, ps = [], []
        for _, g in df.groupby("pid"):
            for comb in combinations(range(len(g)), k):
                ys.append(g.y.iloc[0])
                ps.append(g.p.iloc[list(comb)].mean())
        e2.append(dict(evidence=f"mean of {k} answers", **metrics(np.array(ys), np.array(ps))))
    for q in QUESTIONS:
        m = (df.cond == q).values
        e2.append(dict(evidence=f"single answer: {q} question", **metrics(yr[m], df.p.values[m])))
    pd.DataFrame(e2).to_csv(os.path.join(a.out, "E2_evidence_accumulation.csv"), index=False)

    # ---- Figure 3: group differences in emotion probabilities
    e3 = []
    for key, labels, mod in [("a_prob", AUDIO_LABELS, "audio"), ("t_prob", TEXT_LABELS, "text")]:
        for q in QUESTIONS:
            g = df[df.cond == q]
            ok = np.array([np.isfinite(v).all() for v in g[key]])
            g = g[ok]
            M = np.vstack(g[key].values)
            for j, lab_name in enumerate(labels):
                dep, ctl = M[g.y.values == 1, j], M[g.y.values == 0, j]
                u = mannwhitneyu(dep, ctl)
                e3.append(dict(modality=mod, condition=q, emotion=lab_name, mean_dep=dep.mean(), mean_ctrl=ctl.mean(),
                               r_rb=1 - 2 * u.statistic / (len(dep) * len(ctl)), p=u.pvalue,
                               n_dep=len(dep), n_ctrl=len(ctl)))
    E3 = pd.DataFrame(e3)
    E3["q_fdr"] = fdr_bh(E3.p.values)
    E3.to_csv(os.path.join(a.out, "E3_affective_profiles.csv"), index=False)

    print(pd.DataFrame(rows).round(3).to_string())
    print(pd.DataFrame(perm_rows).round(4).to_string())
    print(pd.DataFrame(e2).round(3).to_string())


if __name__ == "__main__":
    main()
