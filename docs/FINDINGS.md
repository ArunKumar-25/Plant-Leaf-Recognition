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

## 2026-07-12: active-learning pipeline had never once produced an accepted retrain

A second full pass, this time on the self-retraining loop described in
[docs/ARCHITECTURE.md](ARCHITECTURE.md)'s "Active Learning & Self-Retraining"
section — it existed in full and ran on schedule, but had never once produced
an accepted retrain since it was built.

### What was wrong

1. **GBIF reinforcement fetch pulled field photos, not specimen scans.**
   `scripts/fetch_species_dataset.py` queried GBIF's occurrence search with no
   filter on how a photo was taken. The model is trained on scanned single
   leaves against a plain background; the fetcher was returning hand-holding-a-branch
   field photos, whole trees against the sky, and similar — a domain mismatch
   from the very images meant to reinforce the model.
2. **The regression gate rejected every retrain, always on a different class,
   on pure sampling noise.** With ~15 test images per class, one flipped
   prediction swings a class's measured recall by ~6.7% — enough to blow past
   even an already-widened flat percentage-point tolerance, with no real
   quality regression underneath. Confirmed live: Issues #14, #15, #16, #18
   each rejected a working retrain on a different class every time.
3. **A rejected retrain leaves its staged data in place "for retry" — so
   nothing was ever cleared.** Every daily-gather run since the project's
   start staged images to the `contributions` branch; since every weekly
   retrain before this pass was rejected, the backlog was never cleared and
   kept accumulating for weeks, including pre-fix field photos.
4. **`artifacts/reports/metrics.md` accuracy (100%) had drifted from what the
   website and README claimed (98.7%)** — a later retrain improved the
   committed model without the public-facing figures being updated to match.
5. **Every "View source on GitHub" link, social share URL, `robots.txt`
   sitemap, and the GBIF fetch User-Agent pointed at
   `github.com/ArunKumar-25/Plantify` — a repo that doesn't exist.** The
   actual repo is `ArunKumar-25/Plant-Leaf-Recognition`. Confirmed via a live
   404 on the old URL.
6. **Two frontend bugs, both pre-existing:** the "Try the demo" nav button had
   invisible dark-on-dark text specifically on `identify.html` (a CSS
   specificity collision between its own button styling and the active-nav-link
   styling, since that link's `data-nav` happens to match that page). And
   cross-page anchor links (e.g. "About" → `index.html#about-us`) landed with
   an instant, jarring jump instead of the smooth scroll same-page anchor
   clicks get, made worse by the `@view-transition` CSS rule silently
   swallowing an attempted JS-driven smooth-scroll fix.

### What was changed

- `fetch_species_dataset.py` now filters GBIF results to
  `basisOfRecord=PRESERVED_SPECIMEN` (herbarium scans, not field photos) plus
  a post-download near-white-background check and a minimum-resolution
  floor — none of which GBIF's metadata exposes up front.
- `regression_gate.py` now runs a one-sided two-proportion significance test
  on each class's raw correct/total counts (`per_class_support`, added to
  `evaluate_model.py`'s output) instead of a flat tolerance, when both sides
  have counts to compare — only rejects a class when the drop is
  statistically unlikely (p < 0.05 by default) to be sampling noise. Falls
  back to the previous flat-tolerance behavior otherwise (fully backward
  compatible — verified against the exact Issue #18 shape).
  - First real-world result: a retrain finally cleared the gate (baseline
    96.20% → new 95.78%) and opened PR #19 — the first accepted retrain in
    the project's history.
- PR #19 itself was closed unmerged after inspection showed it swept in the
  entire multi-week contaminated backlog (55 images, 11 classes), not just a
  fresh batch — confirmed by directly viewing staged images (e.g. maple seed
  pods against a blurry forest in `leaf2/Acer`). The stale backlog on
  `contributions` was cleared (turned out to already be auto-cleared by the
  workflow's own accept-path cleanup step) so future cycles rebuild it
  cleanly with the fixed filter.
- `web/index.html` and `README.md` accuracy figures updated from 98.7% to
  100%, matching `artifacts/reports/metrics.md`.
- Every `Plantify`-repo URL corrected to `Plant-Leaf-Recognition` across
  `README.md`, `web/*.html`, `web/partials/*.html`, `web/robots.txt`,
  `web/sitemap.xml`, and `fetch_species_dataset.py`'s User-Agent string.
- Nav-CTA specificity fixed (`.nav-links a.nav-cta.active` now wins over the
  generic active-link color rule); cross-page anchor scroll now animates
  correctly via a `load`-time `scrollIntoView`, which required removing the
  `@view-transition: navigation: auto` CSS rule entirely once testing showed
  it reliably swallows any scroll made shortly after a cross-document
  navigation.
- Added `web/robots.txt` and `web/sitemap.xml` (both previously missing).
- Added automatic `prefers-color-scheme: dark` support, then reverted it —
  the original light palette is the intended look regardless of OS setting,
  confirmed by direct feedback after shipping it.

None of this needed real user traffic to find — it surfaced from actually
running the pipeline end-to-end (rebuilding a working local environment,
triggering the real GitHub Actions workflows, and reading the resulting PR
diff and logs) rather than trusting that "the workflow exists and runs on
schedule" meant it worked.
