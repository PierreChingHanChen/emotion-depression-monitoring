"""Extract eGeMAPS v02 functionals (88) for every CREMA-D clip.

Usage: python src/01_extract_cremad_egemaps.py --cremad_dir CREMA-D/AudioWAV --out work/cremad_egemaps.csv
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import opensmile
import pandas as pd


def work(args):
    folder, files = args
    sm = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                         feature_level=opensmile.FeatureLevel.Functionals)
    rows = []
    for f in files:
        x = sm.process_file(os.path.join(folder, f)).iloc[0]
        x["file"] = f
        rows.append(x)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cremad_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.cremad_dir) if f.endswith(".wav"))
    chunks = [(a.cremad_dir, files[i::a.workers]) for i in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as ex:
        df = pd.concat(ex.map(work, chunks))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.shape)


if __name__ == "__main__":
    main()
