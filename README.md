# emotion-depression-monitoring

Code, derived features and results for the manuscript

> **An emotion-conditioned architecture for multimodal depression risk monitoring with large language models: design and evaluation on public Mandarin interview data**
> Ching-Han Chen and Chun-Yi Hsu, Department of Computer Science and Information Engineering, National Central University, Taiwan.
> Submitted to *BMC Artificial Intelligence* (Collection: AI in mental health monitoring).

The study evaluates whether a compact, interpretable **affective state** (emotion probabilities from speech and text) carries information about depression, as the input layer for an emotion-conditioned large language model (LLM) care agent. All components are trained and evaluated on public data.

| Component | Data | Method | Held-out performance |
|---|---|---|---|
| Audio emotion encoder | CREMA-D (7442 clips, 91 speakers, 6 emotions) | eGeMAPS v02 (88 functionals) + RBF-SVM | Accuracy 0.53, macro F1 0.52 (speaker-grouped 5-fold CV) |
| Text emotion encoder | SMP2020-EWECT general-topic Weibo posts (6 emotions) | Character 1–3-gram TF-IDF + logistic regression | Accuracy 0.74, macro F1 0.71 (official test set, n = 5000) |
| Depression detection | EATD-Corpus (162 participants, 30 with SDS standard score ≥ 53) | Participant-level logistic regression / random forest | See `results/` |

Main result (20 × 5-fold CV, logistic regression): affective state AUC 0.60 (audio), 0.60 (text), 0.65 (audio + text, mean + SD; permutation p = 0.038); raw eGeMAPS 0.68; raw text TF-IDF 0.72.

## Repository layout

```
src/
  common.py                      shared helpers (representations, metrics, CV, permutation tests)
  01_extract_cremad_egemaps.py   eGeMAPS features for CREMA-D
  02_train_audio_encoder.py      audio emotion encoder + speaker-independent evaluation
  03_train_text_encoder.py       Chinese text emotion encoder + test-set evaluation
  04_extract_eatd_features.py    apply both encoders to EATD-Corpus -> data/derived/
  05_run_experiments.py          all depression analyses (Tables 4-5, Fig. 3 statistics)
  06_make_figures.py             Figures 2 and 3
data/derived/
  eatd_affective_states.csv      per answer: participant, split, SDS standard score, label, question,
                                 duration, 6 audio + 6 text emotion probabilities (486 rows)
  eatd_egemaps.csv               per answer: 88 eGeMAPS functionals (486 rows)
results/                         CSV/JSON outputs of the reported run, Additional file 1
figures/                         Figures 2 and 3 (PNG, PDF)
```

The derived files contain only numerical features computed from the recordings; they contain **no audio and no transcripts**. Participant identifiers are those of EATD-Corpus (`t_*` official training split, `v_*` official validation split). One answer (`v_79`, positive question) has no audio after silence removal; its audio probabilities are empty.

## Quick start: reproduce the depression analyses from the derived features

```bash
pip install -r requirements.txt
python src/05_run_experiments.py --out results          # Tables 4-5 and Fig. 3 statistics
python src/06_make_figures.py --results results --out figures
```

The raw-text TF-IDF comparator requires the EATD transcripts; add `--eatd_dir path/to/EATD-Corpus` to include it.

## Full pipeline from the original datasets

The original datasets are **not redistributed** here; obtain them from their providers and respect their terms of use.

| Dataset | Source |
|---|---|
| CREMA-D | https://github.com/CheyneyComputerScience/CREMA-D (`AudioWAV/`) |
| SMP2020-EWECT | e.g. https://github.com/BrownSweater/BERT_SMP2020-EWECT (`data/raw/usual_*.txt`) |
| EATD-Corpus | https://github.com/speechandlanguageprocessing/ICASSP2022-Depression |

```bash
python src/01_extract_cremad_egemaps.py --cremad_dir CREMA-D/AudioWAV --out work/cremad_egemaps.csv
python src/02_train_audio_encoder.py --features work/cremad_egemaps.csv --model_out models/audio_emotion_svm.joblib --eval_out results/audio_encoder_eval.json
python src/03_train_text_encoder.py --ewect_dir SMP2020-EWECT/raw --model_out models/text_emotion_lr.joblib --eval_out results/text_encoder_eval.json
python src/04_extract_eatd_features.py --eatd_dir EATD-Corpus --audio_model models/audio_emotion_svm.joblib --text_model models/text_emotion_lr.joblib --out_dir data/derived
python src/05_run_experiments.py --eatd_dir EATD-Corpus --out results
python src/06_make_figures.py --results results --out figures
```

All random seeds are fixed (`SEED = 20261106` in `src/common.py`). The full pipeline runs on a CPU-only computer in under an hour.

## Citation

If you use this code, please cite the paper (citation will be added on publication) and the archived release on Zenodo (DOI to be added).

## License

Code: MIT License (see `LICENSE`). Derived features are provided for research use; the underlying datasets remain subject to the terms of their original providers.

## Acknowledgements

This work was supported by the National Science and Technology Council, Taiwan, R.O.C. (Grant No. NSTC 115-2221-E-008-074). We thank the creators of EATD-Corpus, CREMA-D and SMP2020-EWECT for making their data publicly available.
