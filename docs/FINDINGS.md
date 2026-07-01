# End-to-End Review & Enhancements

A full pass over the project (code, model, data, and the report/paper claims),
the problems found, and what was changed to make the project honest and working.

## What was wrong

1. **The shipped model didn't match its own labels.**
   The old `CNN-model/` outputs **15 classes**, and `app.py` listed 15 species —
   but the pickled training data (`X_Augumented_Grayscale`) contained only
   **150 images from 2 classes** (75 each). The model had never seen the other
   13 species, so it mapped *every* input (including non-leaves) onto one of the
   two it knew. That is why a random image returned "Acer, 69%".

2. **No out-of-distribution rejection.**
   A softmax classifier always returns one of its trained classes; it cannot say
   "this isn't a leaf." There was no guard in front of it.

3. **Report vs. reality mismatch.**
   The report/paper describe **SVM + GLCM/HOG/LBP + PCA/RFE** reaching ~96.8%,
   compared across SVM/RF/KNN/Naïve Bayes. None of that exists in the code — the
   implementation is a CNN. The quoted accuracies were not produced by any script
   in the repo (the training notebook never evaluated a held-out test set).

4. **Only 4 of 15 dataset classes are present** on disk (`leaf1, leaf2, leaf5,
   leaf7`, 75 images each = 300 total). A faithful 15-class system is not
   reproducible from this repo.

5. **Won't install / run for others.** Dependencies were pinned to
   `tensorflow==2.5.0rc0` (a release candidate) and the model the app loaded
   (`CNN-model`, hyphen) was never committed — a fresh clone failed.

## What was changed

- **Trained a real model** (`scripts/train_model.py` → `leaf_cnn/`) on the 4 classes that
  actually exist, with a stratified **train/val/test split (192/48/60)**, data
  augmentation (rotation/flip/zoom/shift), dropout, and early stopping.
  **Held-out test accuracy: 98.3%** — see `reports/`.
- **Honest evaluation artifacts** in `reports/`: `metrics.md` (accuracy +
  per-class precision/recall/F1), `confusion_matrix.png`, `accuracy_loss.png`,
  `history.json`. These are the Chapter-6 figures the report promised, now real.
- **Out-of-distribution guard** in `app.py` (`looks_like_leaf_scan`): rejects
  images that aren't a single leaf on a near-white background (the model's domain)
  before classifying, plus a confidence floor so low-confidence inputs say
  "couldn't identify" instead of guessing.
- **`app.py` now loads `leaf_cnn/`** with the correct 4-class label list and a
  fixed 128×128 grayscale preprocessing path; deprecated `grayscale=True` replaced
  with `color_mode="grayscale"`.
- **Repo made clone-and-run**: installable `requirements.txt`, `.gitignore`,
  `samples/` images, and the trained model committed.

## Scaling to the full 15 species

The original project targeted **15 species**, but only 4 class folders were on
disk. `scripts/fetch_full_dataset.py` downloads the remaining 11 classes from
Linköping University and compacts them, restoring the full 1,125-image dataset.

A from-scratch grayscale CNN on 15 classes only reached **68%** — the elms and
willows (visually similar) confused each other badly. Switching to **transfer
learning** (MobileNetV2 ImageNet features + a small trainable head, using leaf
colour) lifted the held-out test accuracy to **98.7%**, with most species at
~1.00 F1. That model (`artifacts/model/`, ~10 MB) is what ships — see
[artifacts/reports/metrics.md](../artifacts/reports/metrics.md) for the full per-class breakdown.

## Honest scope

The model is trained on clean white-background scans, so real-world photos on busy
backgrounds remain out of domain — the guard rejects them rather than guessing.
The report's SVM/GLCM narrative still doesn't match the code; align it to the
MobileNetV2 pipeline and the metric in `artifacts/reports/metrics.md`.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/train_model.py   # retrains artifacts/model/ and regenerates artifacts/reports/
streamlit run app.py            # launches the admin/retrain UI
```
