"""Train the audio emotion encoder (eGeMAPS + RBF-SVM) on CREMA-D and evaluate it
with speaker-grouped 5-fold cross-validation.

Usage: python src/02_train_audio_encoder.py --features work/cremad_egemaps.csv \
          --model_out models/audio_emotion_svm.joblib --eval_out results/audio_encoder_eval.json
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model_out", required=True)
    ap.add_argument("--eval_out", required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    df["speaker"] = df.file.str[:4]
    df["emotion"] = df.file.str.split("_").str[2]
    X = df.drop(columns=["file", "speaker", "emotion"]).values
    y, groups = df.emotion.values, df.speaker.values

    cv_model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced"))
    pred = cross_val_predict(cv_model, X, y, groups=groups, cv=GroupKFold(5), n_jobs=-1)
    result = dict(accuracy=accuracy_score(y, pred), macro_F1=f1_score(y, pred, average="macro"),
                  n_clips=len(y), n_speakers=int(df.speaker.nunique()),
                  report=classification_report(y, pred, output_dict=True))
    print(classification_report(y, pred, digits=3))

    final = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=3, gamma="scale", probability=True,
                                                class_weight="balanced", random_state=0))
    final.fit(X, y)
    for path in (a.model_out, a.eval_out):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(final, a.model_out)
    json.dump(result, open(a.eval_out, "w"), indent=1)


if __name__ == "__main__":
    main()
