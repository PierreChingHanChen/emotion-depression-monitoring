"""Train the Chinese text emotion encoder (character n-gram TF-IDF + logistic regression)
on SMP2020-EWECT (general-topic data) and evaluate it on the official labelled test set.

Usage: python src/03_train_text_encoder.py --ewect_dir SMP2020-EWECT/raw \
          --model_out models/text_emotion_lr.joblib --eval_out results/text_encoder_eval.json
The folder must contain usual_train.txt, usual_eval_labeled.txt and usual_test_labeled.txt (JSON lists).
"""
import argparse
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline


def load(path):
    return json.load(open(path, encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ewect_dir", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--eval_out", required=True)
    a = ap.parse_args()

    train = load(os.path.join(a.ewect_dir, "usual_train.txt")) + load(os.path.join(a.ewect_dir, "usual_eval_labeled.txt"))
    test = load(os.path.join(a.ewect_dir, "usual_test_labeled.txt"))
    X, y = [d["content"] for d in train], [d["label"] for d in train]
    Xt, yt = [d["content"] for d in test], [d["label"] for d in test]

    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), min_df=2, sublinear_tf=True, max_features=200000),
        LogisticRegression(C=4, max_iter=3000, class_weight="balanced"))
    model.fit(X, y)
    pred = model.predict(Xt)
    result = dict(accuracy=accuracy_score(yt, pred), macro_F1=f1_score(yt, pred, average="macro"),
                  n_train=len(X), n_test=len(Xt), report=classification_report(yt, pred, output_dict=True))
    print(classification_report(yt, pred, digits=3))
    for path in (a.model_out, a.eval_out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, a.model_out)
    json.dump(result, open(a.eval_out, "w"), indent=1)


if __name__ == "__main__":
    main()
