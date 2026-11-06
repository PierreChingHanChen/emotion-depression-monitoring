"""Apply the two emotion encoders to EATD-Corpus and write the derived feature files
contained in data/derived/ (no audio or transcripts are written).

Usage: python src/04_extract_eatd_features.py --eatd_dir EATD-Corpus \
          --audio_model models/audio_emotion_svm.joblib --text_model models/text_emotion_lr.joblib \
          --out_dir data/derived
EATD-Corpus layout: <eatd_dir>/{t_1..t_83, v_1..v_79}/{positive,neutral,negative}_out.wav,
{positive,neutral,negative}.txt and new_label.txt (SDS standard score).
"""
import argparse
import os
import re

import joblib
import numpy as np
import opensmile
import pandas as pd

QUESTIONS = ["positive", "neutral", "negative"]
AUDIO_LABELS = ["ANG", "DIS", "FEA", "HAP", "NEU", "SAD"]
TEXT_LABELS = ["angry", "fear", "happy", "neutral", "sad", "surprise"]


def read_text(path):
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return open(path, encoding=enc).read().strip()
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eatd_dir", required=True)
    ap.add_argument("--audio_model", required=True)
    ap.add_argument("--text_model", required=True)
    ap.add_argument("--out_dir", default="data/derived")
    a = ap.parse_args()

    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    audio_model, text_model = joblib.load(a.audio_model), joblib.load(a.text_model)
    assert list(audio_model.classes_) == AUDIO_LABELS and list(text_model.classes_) == TEXT_LABELS

    states, egemaps = [], []
    for pid in sorted(d for d in os.listdir(a.eatd_dir) if re.match(r"^[tv]_\d+$", d)):
        folder = os.path.join(a.eatd_dir, pid)
        sds = float(read_text(os.path.join(folder, "new_label.txt")).split()[0])
        for q in QUESTIONS:
            wav = os.path.join(folder, f"{q}_out.wav")
            row = dict(participant=pid, split="train" if pid.startswith("t") else "val",
                       sds_standard=sds, depressed=int(sds >= 53), question=q)
            try:
                res = smile.process_file(wav)
                g = res.iloc[0].to_numpy(dtype=float)
                start, end = res.index.get_level_values("start")[0], res.index.get_level_values("end")[0]
                row["duration_s"] = round((end - start).total_seconds(), 3)
                pa = audio_model.predict_proba(g.reshape(1, -1))[0]
            except Exception as err:  # empty audio after silence removal
                print("audio unavailable:", pid, q, err)
                g, pa, row["duration_s"] = np.full(88, np.nan), np.full(6, np.nan), 0.0
            text = read_text(os.path.join(folder, f"{q}.txt")) or "。"
            pt = text_model.predict_proba([text])[0]
            row.update({f"audio_p_{l}": v for l, v in zip(AUDIO_LABELS, pa)})
            row.update({f"text_p_{l}": v for l, v in zip(TEXT_LABELS, pt)})
            states.append(row)
            egemaps.append(dict(participant=pid, question=q, **dict(zip(smile.feature_names, g))))

    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(states).round(6).to_csv(os.path.join(a.out_dir, "eatd_affective_states.csv"), index=False)
    pd.DataFrame(egemaps).to_csv(os.path.join(a.out_dir, "eatd_egemaps.csv"), index=False, float_format="%.6g")
    print(f"{len(states)} answers from {len(states) // 3} participants written to {a.out_dir}")


if __name__ == "__main__":
    main()
