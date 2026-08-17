"""
Sensitivity analysis one-radiograph-per-patient
================================================================

Here it's perfomed an analysis to verify if patients with many radiographs can dominate
the estimates of the metrics.  
This script answers it *evaluation-only*: the downstream models
are NOT refitted and the training partitions are NOT touched.  

Three evaluations, all computed from the same cached test-set predictions:

  per_radiograph    w = 1 for every radiograph.  This is the published analysis
                    and is used as a QA GATE: it must reproduce
                    fig/<ds>/<emb>/predict-<task>/result_test.json.  A mismatch
                    means the patient-ID alignment is wrong -> abort.
  one_per_patient   HEADLINE.  One radiograph drawn uniformly at random per
                    patient, R = 200 repetitions, metrics averaged over draws.
                    The same draw indices are reused across all 10 fold-models
                    (and across tasks/classifiers of the same dataset+embedding)
                    so that fold-to-fold differences are not sampling noise.
  patient_balanced  ADDITIONAL CHECK.  Radiograph-level metrics with each
                    radiograph of patient i weighted by 1/N_i (inverse
                    cluster-size weighting).  NOT patient-level aggregation:
                    predictions are never combined into one per patient.

Fold aggregation uses the paper's own formula, mean +- t_{0.975,9} * sd/sqrt(10)
(replicated from utils.compute_confidence_interval, app/utils.py:922-946), so
the new numbers are directly comparable with the published tables.

Run from the REPOSITORY ROOT (all data paths resolve relative to CWD):

    python one-radiographs-per-patient/sensitivity_patient_level.py --scope full


Scopes
    table    MLP x cxr-foundation x {mimic, chexpert}          ~110 model-predicts
    ranking  MLP x 3 embeddings   x {mimic, chexpert}          ~330 model-predicts
    full     4 classifiers x 3 embeddings x 2 datasets         ~1320 model-predicts
"""

import argparse
import csv
import gc
import json
import os
import pickle
import sys
import time

import numpy as np
from scipy.stats import t as student_t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core.multiarray"] = np.core.multiarray
    sys.modules["numpy._core.umath"] = np.core.umath
    sys.modules["numpy._core.numeric"] = np.core.numeric


# ----------------------------------------------------------------------------- config

SEED = 42

DATASETS = ["mimic", "chexpert"]
EMBEDDINGS = ["cxr-foundation", "medclip", "biomedclip"]
CLASSIFIERS = ["logistic_regression", "random_forest", "xgboost", "mlp"]
MODES = ["per_radiograph", "one_per_patient", "patient_balanced"]
METRICS = ["accuracy", "precision", "recall", "f1", "roc"]
N_FOLDS = 10

TASKS = {
    "mimic": ["race-bin", "race-cls", "gender", "age_decile", "insurance", "disease"],
    "chexpert": ["gender", "age_decile", "race-bin", "race-cls", "disease"],
}

DEMOGRAPHIC_TASKS = {"gender", "age_decile", "race-bin", "race-cls", "insurance"}

SCOPES = {
    "table": {"embeddings": ["cxr-foundation"], "classifiers": ["mlp"]},
    "ranking": {"embeddings": EMBEDDINGS, "classifiers": ["mlp"]},
    "full": {"embeddings": EMBEDDINGS, "classifiers": CLASSIFIERS},
}

EXPECTED_SHAPE = {
    ("mimic", "cxr-foundation"): (39485, 9238),
    ("mimic", "medclip"): (39483, 9238),
    ("mimic", "biomedclip"): (39483, 9238),
    ("chexpert", "cxr-foundation"): (39503, 11454),
    ("chexpert", "medclip"): (43428, 11454),
    ("chexpert", "biomedclip"): (43428, 11454),
}

# Labels joined once per patient 
PATIENT_INVARIANT = {
    "mimic": ["gender", "race-bin", "race-cls", "insurance", "age_decile"],
    "chexpert": ["gender", "race-bin", "race-cls"],
}

DISPLAY_TASK = {
    "gender": "Sex",
    "age_decile": "Age",
    "race-bin": "Eth. B",
    "race-cls": "Eth. MC",
    "insurance": "Insurance",
    "disease": "No Finding",
}
LATEX_TASK_ORDER = ["gender", "age_decile", "race-bin", "race-cls", "insurance", "disease"]
DISPLAY_DATASET = {"mimic": "MIMIC-CXR", "chexpert": "CheXpert"}
DISPLAY_MODE = {
    "per_radiograph": "per radiograph",
    "one_per_patient": "one radiograph/patient",
    "patient_balanced": "patient-balanced (1/$N_i$)",
}

RESULTS_DIR = os.path.join("metrics", "sensitivity_patient_level")


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------- paths / small io

def data_dir(dataset, embedding, task=None):
    p = os.path.join("data", dataset, embedding)
    return p if task is None else os.path.join(p, task)


def models_dir(dataset, embedding, task):
    return os.path.join("models", dataset, embedding, task)


def metrics_dir(dataset, embedding, task):
    """Metric JSONs live under metrics/, never under fig/ (which holds images only)."""
    return os.path.join("metrics", dataset, embedding, f"predict-{task}")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def check_repo_root():
    missing = [d for d in ("data", "models", "metrics", "app") if not os.path.isdir(d)]
    if missing:
        raise SystemExit(
            "ERROR: run this script from the repository root -- all data paths resolve "
            f"relative to the current working directory. Missing here: {missing}\n"
            "    cd /home/prometeus/Scrivania/aquarta/fairness-ai && python app/sensitivity_patient_level.py ..."
        )


# --------------------------------------------------------- CI (paper's exact formula)

def compute_confidence_interval(data, confidence=0.95):
    """Verbatim replication of utils.compute_confidence_interval (app/utils.py:922-946).
    """
    n = len(data)
    if n < 2:
        return float(np.mean(data)), (None, None)
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / np.sqrt(n)
    t_critical = student_t.ppf((1 + confidence) / 2, df=n - 1)
    return float(mean), (float(mean - t_critical * std_err), float(mean + t_critical * std_err))


# ------------------------------------------------------- stage 1: patient-ID recovery

def _extract_path(p):
    """Helper reused verbatim from app/train_*_chexpert.py."""
    if isinstance(p, np.ndarray):
        p = p.tolist()
    if isinstance(p, list) and len(p) > 0:
        return _extract_path(p[0].replace("-small", ""))
    return str(p).replace("-small", "")


def _ids_mimic_cxr_foundation():
    # test_all_ids is already the per-row patient-ID vector aligned with X_test.
    _, _, _, test_all_ids = load_pickle(data_dir("mimic", "cxr-foundation") + "/train_test_idx.pkl")
    return np.asarray(test_all_ids).ravel()


def _ids_mimic_medclip_like(embedding):
    _, test_indices, _, _ = load_pickle(data_dir("mimic", "cxr-foundation") + "/train_test_idx.pkl")
    fname = "embedding_from_medclip.pkl" if embedding == "medclip" else "embedding_from_biomedclip.pkl"
    tup = load_pickle(os.path.join(data_dir("mimic", embedding), fname))
    _, _, valid_id, ids = tup
    ids = np.asarray(ids).ravel()
    valid = np.zeros(len(ids), dtype=bool)
    valid[np.asarray(valid_id)] = True
    te_i = np.asarray(test_indices)
    te_i = te_i[valid[te_i]]
    out = ids[te_i]
    del tup, ids, valid
    gc.collect()
    return out


def _ids_chexpert_from_split():
    # Byte-identical to the reconstructed per-row vector for both medclip and
    # biomedclip (verified with np.array_equal); the two share the split by design.
    _, test_all_ids = load_pickle(data_dir("chexpert", "medclip") + "/train_test_ids.pkl")
    return np.asarray(test_all_ids).ravel()


def _ids_chexpert_cxr_foundation():
    import pandas as pd

    log("      rebuilding CheXpert / cxr-foundation row order (loads embed.pkl, ~2.4 GB)...")
    df_meta = pd.read_csv(os.path.join("data", "chexpert", "metadata.csv"))

    df_embed = load_pickle(os.path.join(data_dir("chexpert", "cxr-foundation"), "embed.pkl"))
    paths_cxr = df_embed["Path"].tolist()  # keep only Path: the merge order does not
    del df_embed                           # depend on the embeddings, and this avoids
    gc.collect()                           # materialising them twice

    medclip_tuple = load_pickle(os.path.join(data_dir("chexpert", "medclip"), "embedding_from_medclip.pkl"))
    medclip_paths = set(_extract_path(p) for p in medclip_tuple[3])
    del medclip_tuple
    gc.collect()

    dm = pd.merge(df_meta, pd.DataFrame({"Path": paths_cxr}), on="Path", how="inner")
    del df_meta, paths_cxr
    gc.collect()
    dm = dm[dm["Path"].isin(medclip_paths)]
    log(f"      cohort after MedCLIP filtering: {len(dm)} rows (paper: 198,836)")

    pid = dm["PATIENT"].str.replace("patient", "", regex=False).astype(int).values
    del dm
    gc.collect()

    _, test_all_ids = load_pickle(data_dir("chexpert", "medclip") + "/train_test_ids.pkl")
    mask = np.isin(pid, np.unique(np.asarray(test_all_ids)))
    return pid[mask]


def get_test_patient_ids(dataset, embedding, force=False):
    """Per-row patient IDs aligned with the saved X_test.  Cached to disk."""
    cache = os.path.join(data_dir(dataset, embedding), "test_patient_ids.pkl")
    if os.path.exists(cache) and not force:
        ids = np.asarray(load_pickle(cache)).ravel()
        log(f"      patient IDs from cache: {cache}")
    else:
        if (dataset, embedding) == ("mimic", "cxr-foundation"):
            ids = _ids_mimic_cxr_foundation()
        elif dataset == "mimic":
            ids = _ids_mimic_medclip_like(embedding)
        elif (dataset, embedding) == ("chexpert", "cxr-foundation"):
            ids = _ids_chexpert_cxr_foundation()
        else:
            ids = _ids_chexpert_from_split()
        with open(cache, "wb") as f:
            pickle.dump(ids, f)
        log(f"      patient IDs reconstructed and cached -> {cache}")

    exp = EXPECTED_SHAPE.get((dataset, embedding))
    got = (len(ids), len(np.unique(ids)))
    if exp is not None and got != exp:
        raise SystemExit(
            f"ERROR: reconstructed patient IDs for {dataset}/{embedding} have "
            f"(rows, patients) = {got}, expected {exp}. Refusing to continue -- "
            "delete the cache and investigate the reconstruction."
        )
    log(f"      {got[0]} radiographs / {got[1]} patients")
    return ids


# ------------------------------------------------------- stage 1b: draw index matrix

class PatientIndex:
    """Groups test rows by patient and holds the R x P matrix of draw indices.

    The draws depend only on (patient_ids, seed, R), so every task, classifier and
    fold-model of a given dataset+embedding sees the *same* selected radiographs.
    """

    def __init__(self, patient_ids, repetitions, seed):
        self.patient_ids = np.asarray(patient_ids).ravel()
        self.n_rows = len(self.patient_ids)
        uniq, counts = np.unique(self.patient_ids, return_counts=True)
        self.unique_ids = uniq
        self.counts = counts
        self.n_patients = len(uniq)

        order = np.argsort(self.patient_ids, kind="stable")
        starts = np.cumsum(counts) - counts

        rng = np.random.RandomState(seed)
        offsets = np.floor(rng.random_sample((repetitions, self.n_patients)) * counts).astype(np.int64)
        # guard against u*count rounding up to count for the largest clusters
        np.minimum(offsets, counts - 1, out=offsets)
        self.draws = order[starts[None, :] + offsets]

        # 1/N_i weights, one entry per row, aligned with X_test
        inv = 1.0 / counts.astype(np.float64)
        pos = np.searchsorted(uniq, self.patient_ids)
        self.weights = inv[pos]
        self.repetitions = repetitions

    def label_variability(self, y_true):
        """Fraction of patients with more than one distinct label value."""
        order = np.argsort(self.patient_ids, kind="stable")
        pid_s = self.patient_ids[order]
        y_s = np.asarray(y_true).ravel()[order]
        new_patient = np.r_[True, pid_s[1:] != pid_s[:-1]]
        changed = np.r_[False, y_s[1:] != y_s[:-1]] & ~new_patient
        grp = np.cumsum(new_patient) - 1
        varying = np.zeros(self.n_patients, dtype=bool)
        np.logical_or.at(varying, grp, changed)
        return float(varying.sum()) / self.n_patients


# ------------------------------------------------------------- stage 2: predictions

def _reduce_predictions(prob, classifier, n_class):
    """Raw model output -> (hard labels, score) exactly as utils.evaluate_models does."""
    prob = np.asarray(prob)
    if n_class == 2:
        if classifier == "mlp":
            y_pred = (prob > 0.5).astype(int).ravel()
            score = prob.ravel()
        else:
            y_pred = (prob[:, 1] > 0.5).astype(int)
            score = prob[:, 1]
    else:
        y_pred = np.argmax(prob, axis=1)
        score = prob
    return y_pred, score


def _as_int_labels(y):
    y = np.asarray(y).ravel()
    if y.dtype.kind in "fc":
        yi = y.astype(np.int64)
        if not np.all(yi == y):
            raise SystemExit("ERROR: non-integral class labels in y_test.")
        y = yi
    return y.astype(np.int64)


MLP_LOADER_USED = {"strategy": None}


def _snake(name):
    """'Dense' -> 'dense', 'InputLayer' -> 'input_layer' (Keras' own convention)."""
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1].isupper():
            out.append("_")
        out.append(ch.lower())
    return "".join(out)


def _load_mlp_weights_from_archive(model, path):
    import io
    import zipfile

    import h5py

    with zipfile.ZipFile(path) as z:
        blob = z.read("model.weights.h5")
    seen = {}
    with h5py.File(io.BytesIO(blob), "r") as h5:
        if "layers" not in h5:
            raise RuntimeError(f"{path}: unexpected archive layout (no 'layers' group)")
        for layer in model.layers:
            base = _snake(type(layer).__name__)
            k = seen.get(base, 0)
            seen[base] = k + 1
            name = base if k == 0 else f"{base}_{k}"
            expected = layer.get_weights()
            if not expected:
                continue
            group = h5["layers"].get(name)
            if group is None or "vars" not in group:
                raise RuntimeError(f"{path}: no stored variables for layer '{name}'")
            values = [np.asarray(group["vars"][str(j)]) for j in range(len(expected))]
            for got, exp in zip(values, expected):
                if got.shape != exp.shape:
                    raise RuntimeError(f"{path}: shape mismatch on layer '{name}': "
                                       f"archive {got.shape} vs model {exp.shape}")
            layer.set_weights(values)
    return model


def _load_mlp_fold(path, input_dim, n_class, strategy=None):
    """Load one published MLP fold-model.  Returns (model, strategy_used).
    """
    from models import create_mlp

    order = [strategy] if strategy else ["load_weights", "load_model", "h5_direct"]
    errors = []
    for strat in order:
        try:
            if strat == "load_weights":
                model = create_mlp(input_dim, n_class=n_class)
                model.load_weights(path)
            elif strat == "load_model":
                import tensorflow as tf
                model = tf.keras.models.load_model(path)
            elif strat == "h5_direct":
                model = create_mlp(input_dim, n_class=n_class)
                _load_mlp_weights_from_archive(model, path)
            else:
                raise ValueError(f"unknown MLP loader '{strat}'")
            return model, strat
        except Exception as exc:  # report every strategy, then abort
            errors.append(f"    {strat:13s} {type(exc).__name__}: {str(exc)[:150]}")
    raise SystemExit(
        f"\nERROR: cannot load {path}.\n" + "\n".join(errors) +
        "\n  The published archives were written by Keras 3.4.1; check which conda\n"
        "  environment is active (see the notes at the top of this file)."
    )


def predict_folds(dataset, embedding, task, classifier, cache_dir, X_test=None,
                  y_test=None, force=False):
    """Predicted probabilities of the 10 fold-models, cached to .npz.

    One model is loaded, used and freed at a time: the published test_*.py keeps
    all 40 models in RAM simultaneously, which is not necessary here.
    """
    path = os.path.join(cache_dir, f"{dataset}__{embedding}__{task}__{classifier}.npz")
    if os.path.exists(path) and not force:
        z = np.load(path)
        probs = [z[f"fold_{i}"] for i in range(1, N_FOLDS + 1)]
        y_true = _as_int_labels(z["y_true"])
        n_class = int(z["n_class"])
        z.close()
        return probs, y_true, n_class, True

    if X_test is None:
        raise RuntimeError("X_test is required when predictions are not cached")

    y_true = _as_int_labels(y_test)
    n_class = len(np.unique(y_true))
    folder = models_dir(dataset, embedding, task)
    probs = []
    t0 = time.time()

    if classifier == "mlp":
        import tensorflow as tf

        strategy = MLP_LOADER_USED["strategy"]
        for i in range(1, N_FOLDS + 1):
            model, strategy = _load_mlp_fold(os.path.join(folder, f"mlp_fold_{i}.keras"),
                                             X_test.shape[1], n_class, strategy)
            if MLP_LOADER_USED["strategy"] is None:
                MLP_LOADER_USED["strategy"] = strategy
                note = "" if strategy == "load_weights" else (
                    "  <-- the published create_mlp+load_weights path does not work "
                    "in this environment")
                log(f"        MLP loader: {strategy}{note}")
            # default batch_size, as in utils.evaluate_models, to keep the
            # arithmetic bit-comparable with the published numbers
            probs.append(np.asarray(model.predict(X_test, verbose=0)))
            del model
            tf.keras.backend.clear_session()
            gc.collect()
    else:
        import joblib

        for i in range(1, N_FOLDS + 1):
            model = joblib.load(os.path.join(folder, f"{classifier}_fold_{i}.pkl"))
            probs.append(np.asarray(model.predict_proba(X_test)))
            del model
            gc.collect()

    os.makedirs(cache_dir, exist_ok=True)
    payload = {f"fold_{i + 1}": p for i, p in enumerate(probs)}
    payload["y_true"] = y_true
    payload["n_class"] = np.asarray(n_class)
    np.savez(path, **payload)
    log(f"        predicted 10x {classifier} in {time.time() - t0:.1f}s -> cached")
    return probs, y_true, n_class, False


# ------------------------------------------------------------------ metric machinery

def sklearn_metrics(y_true, y_pred, score, n_class, sample_weight=None):
    """Reference implementation: identical calls to utils.evaluate_models."""
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)

    out = {
        "accuracy": float(accuracy_score(y_true, y_pred, sample_weight=sample_weight)),
        "precision": float(precision_score(y_true, y_pred, average="macro",
                                           zero_division=0, sample_weight=sample_weight)),
        "recall": float(recall_score(y_true, y_pred, average="macro",
                                     zero_division=0, sample_weight=sample_weight)),
        "f1": float(f1_score(y_true, y_pred, average="macro",
                             zero_division=0, sample_weight=sample_weight)),
    }
    if n_class == 2:
        out["roc"] = float(roc_auc_score(y_true, score, sample_weight=sample_weight))
    else:
        # 'ovo' raises when sample_weight is given -- always use 'ovr'
        out["roc"] = float(roc_auc_score(y_true, score, multi_class="ovr",
                                         average="macro", sample_weight=sample_weight))
    return out


def _metrics_from_cm(cm):
    """Macro accuracy/precision/recall/F1 from a confusion matrix (rows=true).

    Mirrors sklearn's semantics with zero_division=0: a class whose denominator is
    zero contributes 0 and is still included in the macro average.  F1 is formed
    as 2PR/(P+R), the same expression sklearn evaluates.
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    total = cm.sum()

    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    return {
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "precision": float(prec.mean()),
        "recall": float(rec.mean()),
        "f1": float(f1.mean()),
    }


def _auc_from_sorted(y_bin_sorted, score_sorted):
    """Tie-aware Mann-Whitney AUC on an already score-sorted subset."""
    n = len(y_bin_sorted)
    n_pos = int(y_bin_sorted.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    pos = np.arange(1, n + 1, dtype=np.float64)
    first = np.r_[True, score_sorted[1:] != score_sorted[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], n]
    csum = np.r_[0.0, np.cumsum(pos)]
    grp_mean_rank = (csum[ends] - csum[starts]) / (ends - starts)
    ranks = np.repeat(grp_mean_rank, ends - starts)
    return float((ranks[y_bin_sorted].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


class FoldEvaluator:
    """Per-fold precomputation so each of the R draws costs O(n_rows) boolean work.

    The score order is computed once; a draw is then applied as a boolean mask over
    the already-sorted array, which yields the sorted subset without re-sorting.
    Draw indices are unique by construction (one radiograph per patient).
    """

    def __init__(self, y_true, y_pred, score, n_class):
        self.n_class = n_class
        self.n_rows = len(y_true)
        self.code = y_true * n_class + y_pred
        self.cols = []
        if n_class == 2:
            pairs = [(np.asarray(score, dtype=np.float64), y_true == 1)]
        else:
            pairs = [(np.asarray(score[:, c], dtype=np.float64), y_true == c)
                     for c in range(n_class)]
        for s, y_bin in pairs:
            order = np.argsort(s, kind="mergesort")
            self.cols.append((order, s[order], y_bin[order]))

    def metrics_for_subset(self, idx):
        cm = np.bincount(self.code[idx], minlength=self.n_class ** 2)
        out = _metrics_from_cm(cm.reshape(self.n_class, self.n_class))
        mask = np.zeros(self.n_rows, dtype=bool)
        mask[idx] = True
        aucs = []
        for order, s_sorted, y_bin_sorted in self.cols:
            sel = mask[order]
            aucs.append(_auc_from_sorted(y_bin_sorted[sel], s_sorted[sel]))
        aucs = np.asarray(aucs, dtype=np.float64)
        finite = np.isfinite(aucs)
        out["roc"] = float(aucs[finite].mean()) if finite.any() else float("nan")
        out["_missing_class"] = int((~finite).sum())
        return out

    def metrics_full(self):
        return self.metrics_for_subset(np.arange(self.n_rows))


# ------------------------------------------------------------------------- QA gates

def qa_against_published(dataset, embedding, task, classifier, per_fold, atol):
    """per_radiograph must reproduce the published result_test.json."""
    path = os.path.join(metrics_dir(dataset, embedding, task), "result_test.json")
    if not os.path.exists(path):
        log(f"        QA: WARNING no published file at {path} -- gate skipped")
        return {"checked": False}

    with open(path) as f:
        published = json.load(f)
    if classifier not in published:
        log(f"        QA: WARNING {classifier} absent from {path} -- gate skipped")
        return {"checked": False}

    worst, worst_where = 0.0, None
    for metric in METRICS:
        mean, ci = compute_confidence_interval(per_fold[metric])
        ref = published[classifier][metric]
        for name, ours, theirs in (("mean", mean, ref["mean"]),
                                   ("ci_low", ci[0], ref["confidence_interval"][0]),
                                   ("ci_high", ci[1], ref["confidence_interval"][1])):
            if theirs is None or ours is None:
                continue
            d = abs(ours - theirs)
            if d > worst:
                worst, worst_where = d, f"{metric}.{name} (ours {ours!r} vs published {theirs!r})"

    if worst > atol:
        
        scale = ("alignment/reduction error -- the patient-ID vector or the "
                 "prediction reduction is wrong" if worst > 1e-4 else
                 "float drift, not a logic error -- almost certainly the wrong "
                 "environment (see the notes at the top of this file)")
        raise SystemExit(
            f"\nQA GATE FAILED for {dataset}/{embedding}/{task}/{classifier}\n"
            f"  per_radiograph does not reproduce {path}\n"
            f"  max |deviation| = {worst:.3e} > atol {atol:.1e} at {worst_where}\n"
            f"  diagnosis: {scale}\n"
            f"  environment: {environment_info()}\n"
            "  Aborting on purpose. Do not raise --qa-atol to paper over the first case."
        )
    log(f"        QA: reproduces published result_test.json (max dev {worst:.2e})")
    return {"checked": True, "max_abs_deviation": worst}


def qa_fast_path(evaluator, y_true, y_pred, score, n_class, atol):
    """The fast confusion-matrix/rank path must agree with sklearn on the full set."""
    fast = evaluator.metrics_full()
    ref = sklearn_metrics(y_true, y_pred, score, n_class)
    worst, where = 0.0, None
    for metric in METRICS:
        d = abs(fast[metric] - ref[metric])
        if d > worst:
            worst, where = d, metric
    if worst > atol:
        raise SystemExit(
            f"\nQA GATE FAILED: the fast metric path disagrees with sklearn by "
            f"{worst:.3e} (> {atol:.1e}) on {where}. Aborting."
        )
    return worst


# ------------------------------------------------------------------ the three modes

def evaluate_combination(dataset, embedding, task, classifier, probs, y_true, n_class,
                         pidx, modes, args):
    """Returns {mode: {metric: {mean, confidence_interval, fold_sd, draw_sd}}}."""
    per_fold = {m: {k: [] for k in METRICS} for m in modes}
    draw_matrix = {k: np.full((N_FOLDS, pidx.repetitions), np.nan) for k in METRICS}
    fast_dev, missing_class = 0.0, 0
    prevalence = {}

    for fold, prob in enumerate(probs):
        y_pred, score = _reduce_predictions(prob, classifier, n_class)
        evaluator = FoldEvaluator(y_true, y_pred, score, n_class)

        if fold == 0:
            fast_dev = qa_fast_path(evaluator, y_true, y_pred, score, n_class, args.qa_atol)

        if "per_radiograph" in modes:
            # sample_weight is left as None so this is the exact published call
            vals = sklearn_metrics(y_true, y_pred, score, n_class)
            for k in METRICS:
                per_fold["per_radiograph"][k].append(vals[k])

        if "patient_balanced" in modes:
            vals = sklearn_metrics(y_true, y_pred, score, n_class,
                                   sample_weight=pidx.weights)
            for k in METRICS:
                per_fold["patient_balanced"][k].append(vals[k])

        if "one_per_patient" in modes:
            acc = {k: np.empty(pidx.repetitions) for k in METRICS}
            for r in range(pidx.repetitions):
                vals = evaluator.metrics_for_subset(pidx.draws[r])
                missing_class += vals.pop("_missing_class")
                for k in METRICS:
                    acc[k][r] = vals[k]
            for k in METRICS:
                draw_matrix[k][fold] = acc[k]
                per_fold["one_per_patient"][k].append(float(np.nanmean(acc[k])))

        del evaluator
    del probs
    gc.collect()

    if missing_class:
        log(f"        WARNING: {missing_class} one-vs-rest AUCs skipped across draws "
            "(a class was absent from the sampled subset)")

    counts = np.bincount(y_true, minlength=n_class).astype(np.float64)
    prevalence["per_radiograph"] = (counts / counts.sum()).tolist()
    if "patient_balanced" in modes:
        wc = np.bincount(y_true, weights=pidx.weights, minlength=n_class)
        prevalence["patient_balanced"] = (wc / wc.sum()).tolist()
    if "one_per_patient" in modes:
        sub = np.zeros(n_class, dtype=np.float64)
        for r in range(pidx.repetitions):
            sub += np.bincount(y_true[pidx.draws[r]], minlength=n_class)
        prevalence["one_per_patient"] = (sub / sub.sum()).tolist()

    results = {}
    for mode in modes:
        results[mode] = {}
        for k in METRICS:
            scores = per_fold[mode][k]
            mean, ci = compute_confidence_interval(scores)
            entry = {
                "mean": mean,
                "confidence_interval": list(ci),
                "fold_sd": float(np.std(scores, ddof=1)) if len(scores) > 1 else None,
            }
            if mode == "one_per_patient":
                # variability of a *single* random draw, at the 10-fold ensemble level
                per_draw = np.nanmean(draw_matrix[k], axis=0)
                entry["draw_sd"] = float(np.nanstd(per_draw, ddof=1))
                entry["draw_min"] = float(np.nanmin(per_draw))
                entry["draw_max"] = float(np.nanmax(per_draw))
            results[mode][k] = entry

    qa = qa_against_published(dataset, embedding, task, classifier,
                              per_fold["per_radiograph"], args.qa_atol) \
        if ("per_radiograph" in modes and not args.no_qa) else {"checked": False}
    qa["fast_path_max_deviation"] = fast_dev

    return results, prevalence, qa


# ------------------------------------------------------------------------- reporting

def environment_info():
    """Versions of the libraries actually in play, recorded with every result.
    """
    info = {"python": sys.version.split()[0], "numpy": np.__version__}
    for name in ("scipy", "sklearn", "pandas", "keras", "tensorflow", "xgboost", "joblib"):
        module = sys.modules.get(name)
        info[name] = getattr(module, "__version__", None) if module is not None else None
    info["mlp_loader"] = MLP_LOADER_USED["strategy"]
    return info


def write_task_json(dataset, embedding, task, classifier, results, prevalence, qa,
                    pidx, args):
    """Merge into metrics/.../result_test_patient_level.json (never result_test.json)."""
    folder = metrics_dir(dataset, embedding, task)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "result_test_patient_level.json")
    doc = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                doc = json.load(f)
        except (ValueError, OSError):
            doc = {}
    doc["meta"] = {
        "generated_by": "app/sensitivity_patient_level.py",
        "dataset": dataset,
        "embedding": embedding,
        "task": task,
        "repetitions": pidx.repetitions,
        "seed": args.seed,
        "n_radiographs": int(pidx.n_rows),
        "n_patients": int(pidx.n_patients),
        "aggregation": "mean +- t_{0.975,9} * sd/sqrt(10) over the 10 CV fold-models",
        "models_refitted": False,
        "environment": environment_info(),
    }
    doc["prevalence"] = prevalence
    entry = doc.get(classifier, {})
    entry.update(results)
    entry["qa"] = qa
    doc[classifier] = entry
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return path


def write_summary_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["dataset", "embedding", "classifier", "task", "task_group", "mode", "metric",
              "mean", "ci_low", "ci_high", "fold_sd", "draw_sd",
              "delta_vs_per_radiograph_pp", "n_radiographs", "n_patients"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    log(f"  wrote {path} ({len(rows)} rows)")


def write_prevalence_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["dataset", "embedding", "task", "mode", "class", "share"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log(f"  wrote {path} ({len(rows)} rows)")


def _fmt_cell(entry, pct=True):
    if entry is None or entry.get("mean") is None:
        return "--"
    scale = 100.0 if pct else 1.0
    mean = entry["mean"] * scale
    lo, hi = entry["confidence_interval"]
    if lo is None:
        return f"{mean:.2f}"
    return f"{mean:.2f} \\scriptsize{{({lo * scale:.2f}, {hi * scale:.2f})}}"


def write_latex_tables(store, args):
    """One classifier x one embedding x both datasets x all tasks, per the agreed layout.

    Defaults to MLP + CXR-Foundation, the example agreed for the Supplementary Material.
    """
    emb, clf = args.latex_embedding, args.latex_classifier
    modes = [m for m in ("per_radiograph", "one_per_patient") if m in args.modes]
    if len(modes) < 2:
        log("  LaTeX tables skipped (need both per_radiograph and one_per_patient)")
        return
    have = [(d, tk) for d in DATASETS for tk in LATEX_TASK_ORDER
            if (d, emb, clf, tk) in store]
    if not have:
        log(f"  LaTeX tables skipped ({clf} x {emb} not in this scope)")
        return

    full = [
        f"% Table R1 -- response to Reviewer Comment 6. {clf} + {emb}.",
        f"% R = {args.repetitions} random draws, seed {args.seed}. Models not refitted.",
        "% Generated by app/sensitivity_patient_level.py -- do not edit by hand.",
        "\\begin{tabular}{lllccccc}",
        "\\toprule",
        "Dataset & Task & Analysis & Accuracy & Precision & Recall & F1 & ROC-AUC \\\\",
        "\\midrule",
    ]
    delta = [
        f"% Compact supplementary variant, {clf} + {emb}: difference",
        "% (one radiograph/patient - per radiograph) in percentage points.",
        "% Generated by app/sensitivity_patient_level.py -- do not edit by hand.",
        "\\begin{tabular}{llccccc}",
        "\\toprule",
        "Dataset & Task & $\\Delta$Acc & $\\Delta$Prec & $\\Delta$Rec & $\\Delta$F1 & $\\Delta$AUC \\\\",
        "\\midrule",
    ]

    last_ds = None
    for dataset in DATASETS:
        tasks = [tk for tk in LATEX_TASK_ORDER if (dataset, emb, clf, tk) in store]
        if not tasks:
            continue
        if last_ds is not None:
            full.append("\\midrule")
            delta.append("\\midrule")
        last_ds = dataset
        for ti, task in enumerate(tasks):
            res = store[(dataset, emb, clf, task)]
            ds_cell = DISPLAY_DATASET[dataset] if ti == 0 else ""
            for mi, mode in enumerate(modes):
                cells = [_fmt_cell(res[mode][k]) for k in METRICS]
                full.append(" & ".join(
                    [ds_cell if mi == 0 else "",
                     DISPLAY_TASK[task] if mi == 0 else "",
                     DISPLAY_MODE[mode]] + cells) + " \\\\")
            d = [100.0 * (res["one_per_patient"][k]["mean"] - res["per_radiograph"][k]["mean"])
                 for k in METRICS]
            delta.append(" & ".join([ds_cell, DISPLAY_TASK[task]]
                                    + [f"{v:+.2f}" for v in d]) + " \\\\")

    for buf, name in ((full, "table_r1.tex"), (delta, "table_supp_delta.tex")):
        buf += ["\\bottomrule", "\\end{tabular}"]
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            f.write("\n".join(buf) + "\n")
        log(f"  wrote {path}")


def write_placeholders(store, args):
    """[X], [Y], [Z], [W] for the rebuttal, demographic tasks separated from disease."""
    def deltas(mode, group, metrics=METRICS):
        out = []
        for (d, e, c, tk), res in store.items():
            if mode not in res or "per_radiograph" not in res:
                continue
            if (tk in DEMOGRAPHIC_TASKS) != (group == "demographic"):
                continue
            for k in metrics:
                a, b = res[mode][k]["mean"], res["per_radiograph"][k]["mean"]
                if a is None or b is None:
                    continue
                out.append({"dataset": d, "embedding": e, "classifier": c, "task": tk,
                            "metric": k, "delta_pp": 100.0 * (a - b)})
        return out

    def stats(items, key="delta_pp"):
        if not items:
            return None
        v = np.abs([i[key] for i in items])
        arg = int(np.argmax(v))
        return {
            "n": len(items),
            "median_abs_pp": float(np.median(v)),
            "mean_abs_pp": float(np.mean(v)),
            "max_abs_pp": float(v[arg]),
            "max_at": {k: items[arg][k] for k in ("dataset", "embedding", "classifier", "task", "metric")},
        }

    ph = {"scope": args.scope, "repetitions": args.repetitions, "seed": args.seed,
          "combinations_evaluated": len(store)}

    if "one_per_patient" in args.modes:
        demo = deltas("one_per_patient", "demographic")
        dis = deltas("one_per_patient", "disease")
        # [X] = median_abs_pp and [Y] = max_abs_pp of the same comparison set
        ph["XY_demographic"] = stats(demo)
        ph["disease_all_metrics"] = stats(dis)
        ph["Z_disease_auc_max_abs_pp"] = stats([i for i in dis if i["metric"] == "roc"])
        ph["disease_accuracy"] = stats([i for i in dis if i["metric"] == "accuracy"])
        ph["per_metric_demographic"] = {
            k: stats([i for i in demo if i["metric"] == k]) for k in METRICS}

    if "patient_balanced" in args.modes and "one_per_patient" in args.modes:
        agree = []
        for (d, e, c, tk), res in store.items():
            for k in METRICS:
                a, b = res["patient_balanced"][k]["mean"], res["one_per_patient"][k]["mean"]
                if a is None or b is None:
                    continue
                agree.append({"dataset": d, "embedding": e, "classifier": c, "task": tk,
                              "metric": k, "delta_pp": 100.0 * (a - b)})
        ph["W_weighted_vs_sampled_max_abs_pp"] = stats(agree)

    path = os.path.join(args.out_dir, "rebuttal_placeholders.json")
    with open(path, "w") as f:
        json.dump(ph, f, indent=1)
    log(f"  wrote {path}")

    # human-readable companion
    md = ["# Sensitivity analysis -- Reviewer Comment 6", "",
          f"- scope: `{args.scope}`  |  repetitions: {args.repetitions}  |  seed: {args.seed}",
          f"- combinations (dataset x embedding x classifier x task): {len(store)}",
          f"- models refitted: **no**; training partitions: **untouched**", ""]

    def block(title, s, note=""):
        if not s:
            return
        md.extend([f"## {title}", ""])
        if note:
            md.extend([note, ""])
        w = s["max_at"]
        md.extend([
            f"- values compared: {s['n']}",
            f"- **median |delta| = {s['median_abs_pp']:.2f} pp**",
            f"- mean |delta| = {s['mean_abs_pp']:.2f} pp",
            f"- **max |delta| = {s['max_abs_pp']:.2f} pp** "
            f"({w['dataset']} / {w['embedding']} / {w['classifier']} / {w['task']} / {w['metric']})",
            "",
        ])

    if "one_per_patient" in args.modes:
        block("[X] and [Y] -- demographic tasks, one radiograph per patient vs per radiograph",
              ph.get("XY_demographic"),
              "[X] = median |delta|, [Y] = max |delta|, over Sex / Age / Eth. B / Eth. MC / "
              "Insurance x all five metrics x every evaluated model, embedding and dataset.")
        block("[Z] -- No Finding, ROC-AUC only", ph.get("Z_disease_auc_max_abs_pp"),
              "[Z] = max |delta| in ROC-AUC for the disease proxy. Threshold-dependent "
              "metrics for this task also absorb the prevalence shift and are reported "
              "separately below.")
        block("No Finding -- accuracy (prevalence effect, must be explained, not hidden)",
              ph.get("disease_accuracy"))
        block("No Finding -- all metrics", ph.get("disease_all_metrics"))
        md.extend(["## Demographic tasks, per metric", "",
                   "| metric | median abs delta (pp) | max abs delta (pp) |",
                   "|---|---|---|"])
        for k in METRICS:
            s = (ph.get("per_metric_demographic") or {}).get(k)
            if s:
                md.append(f"| {k} | {s['median_abs_pp']:.2f} | {s['max_abs_pp']:.2f} |")
        md.append("")
    if ph.get("W_weighted_vs_sampled_max_abs_pp"):
        block("[W] -- patient-balanced (1/N_i) vs repeated random selection",
              ph["W_weighted_vs_sampled_max_abs_pp"],
              "[W] = max |delta| between the two estimators, across all tasks and metrics.")

    md.extend(["## Prevalence shift", "",
               "See `prevalence.csv`: class shares under each weighting, per dataset, "
               "embedding and task. The 'No Finding' rows are the ones quoted in the "
               "response letter.", ""])

    path = os.path.join(args.out_dir, "deltas_summary.md")
    with open(path, "w") as f:
        f.write("\n".join(md) + "\n")
    log(f"  wrote {path}")


# ------------------------------------------------------------------------------ main

def build_plan(args):
    plan = []
    for dataset in args.datasets:
        for embedding in args.embeddings:
            tasks = [tk for tk in TASKS[dataset] if tk in args.tasks]
            if not tasks:
                continue
            plan.append((dataset, embedding, tasks, args.classifiers))
    return plan


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Patient-level sensitivity analysis for Reviewer Comment 6 "
                    "(evaluation only; no model is refitted).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope", required=True, choices=sorted(SCOPES),
                   help="table = MLP x CXR-Foundation x 2 datasets; "
                        "ranking = MLP x 3 embeddings x 2 datasets; "
                        "full = 4 classifiers x 3 embeddings x 2 datasets")
    p.add_argument("--repetitions", type=int, default=200,
                   help="R, number of random one-per-patient draws (default 200)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--embeddings", default=None, help="override the scope's embeddings")
    p.add_argument("--classifiers", default=None, help="override the scope's classifiers")
    p.add_argument("--tasks", default=None, help="default: all tasks of each dataset")
    p.add_argument("--modes", default=",".join(MODES))
    p.add_argument("--qa-atol", type=float, default=1e-9,
                   help="tolerance of the QA gate against the published result_test.json")
    p.add_argument("--no-qa", action="store_true",
                   help="skip the comparison with result_test.json (not recommended)")
    p.add_argument("--force-predict", action="store_true", help="ignore the .npz prediction cache")
    p.add_argument("--force-ids", action="store_true", help="rebuild the patient-ID caches")
    p.add_argument("--no-id-check", action="store_true",
                   help="skip the within-patient label-invariance check")
    p.add_argument("--latex-classifier", default="mlp", choices=CLASSIFIERS,
                   help="classifier shown in the generated LaTeX tables (default mlp)")
    p.add_argument("--latex-embedding", default="cxr-foundation", choices=EMBEDDINGS,
                   help="embedding shown in the generated LaTeX tables (default cxr-foundation)")
    p.add_argument("--out-dir", default=RESULTS_DIR)
    p.add_argument("--cache-dir", default=None, help="default: <out-dir>/pred_cache")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    a = p.parse_args(argv)

    a.datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    a.embeddings = ([e.strip() for e in a.embeddings.split(",") if e.strip()]
                    if a.embeddings else SCOPES[a.scope]["embeddings"])
    a.classifiers = ([c.strip() for c in a.classifiers.split(",") if c.strip()]
                     if a.classifiers else SCOPES[a.scope]["classifiers"])
    a.tasks = ([t.strip() for t in a.tasks.split(",") if t.strip()]
               if a.tasks else sorted(set(TASKS["mimic"]) | set(TASKS["chexpert"])))
    a.modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    a.cache_dir = a.cache_dir or os.path.join(a.out_dir, "pred_cache")

    for name, values, allowed in (("datasets", a.datasets, DATASETS),
                                  ("embeddings", a.embeddings, EMBEDDINGS),
                                  ("classifiers", a.classifiers, CLASSIFIERS),
                                  ("modes", a.modes, MODES)):
        bad = [v for v in values if v not in allowed]
        if bad:
            p.error(f"unknown {name}: {bad} (allowed: {allowed})")
    if "per_radiograph" not in a.modes and not a.no_qa:
        p.error("the QA gate needs per_radiograph; add it to --modes or pass --no-qa")
    return a


def main(argv=None):
    args = parse_args(argv)
    check_repo_root()
    plan = build_plan(args)
    n_predicts = sum(len(tasks) * len(clfs) * N_FOLDS for _, _, tasks, clfs in plan)
    log("=" * 78)
    log(f"Patient-level sensitivity analysis -- scope '{args.scope}'")
    log(f"  datasets    : {args.datasets}")
    log(f"  embeddings  : {args.embeddings}")
    log(f"  classifiers : {args.classifiers}")
    log(f"  modes       : {args.modes}")
    log(f"  R           : {args.repetitions}   seed: {args.seed}")
    log(f"  model-predicts to run (if nothing is cached): {n_predicts}")
    log(f"  prediction cache: {args.cache_dir}")
    log("  models are NOT refitted; result_test.json is NEVER overwritten")
    log("=" * 78)
    if args.dry_run:
        for dataset, embedding, tasks, clfs in plan:
            log(f"  {dataset}/{embedding}: {len(tasks)} tasks x {len(clfs)} classifiers -> {tasks}")
        log("  (dry run: nothing was written)")
        return 0

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    store, csv_rows, prev_rows = {}, [], []
    t_start = time.time()

    for dataset, embedding, tasks, clfs in plan:
        log(f"\n[{dataset} / {embedding}]")
        patient_ids = get_test_patient_ids(dataset, embedding, force=args.force_ids)
        pidx = PatientIndex(patient_ids, args.repetitions, args.seed)

        for task in tasks:
            log(f"    task '{task}'")
            y_true, n_class, X_test, y_test = None, None, None, None
            id_checked = False

            # load the test pickle at most once per task, and only if some
            # classifier still needs predicting
            needed = [c for c in clfs if args.force_predict or not os.path.exists(
                os.path.join(args.cache_dir, f"{dataset}__{embedding}__{task}__{c}.npz"))]
            if needed:
                path = os.path.join(data_dir(dataset, embedding, task), "test_dataset_std.pkl")
                if not os.path.exists(path):
                    log(f"      SKIP: {path} not found")
                    continue
                X_test, y_test = load_pickle(path)
                if X_test.shape[0] != pidx.n_rows:
                    raise SystemExit(
                        f"ERROR: X_test has {X_test.shape[0]} rows but the patient-ID vector "
                        f"has {pidx.n_rows} for {dataset}/{embedding}/{task}. Alignment is broken."
                    )
                log(f"      X_test {X_test.shape}, models needing prediction: {needed}")

            for classifier in clfs:
                if not os.path.isdir(models_dir(dataset, embedding, task)):
                    log(f"      SKIP {classifier}: no models at {models_dir(dataset, embedding, task)}")
                    continue
                log(f"      -- {classifier}")
                t0 = time.time()
                probs, y_true, n_class, was_cached = predict_folds(
                    dataset, embedding, task, classifier, args.cache_dir,
                    X_test=X_test, y_test=y_test, force=args.force_predict)
                if len(y_true) != pidx.n_rows:
                    raise SystemExit(
                        f"ERROR: cached y_true has {len(y_true)} rows, patient-ID vector has "
                        f"{pidx.n_rows} for {dataset}/{embedding}/{task}. Delete the cache."
                    )

                if not args.no_id_check and not id_checked:
                    id_checked = True
                    var = pidx.label_variability(y_true)
                    if task in PATIENT_INVARIANT[dataset]:
                        if var > 0:
                            raise SystemExit(
                                f"\nQA GATE FAILED: '{task}' is joined once per patient and must be "
                                f"constant within patient, but {100 * var:.2f}% of patients show more "
                                f"than one value in {dataset}/{embedding}. The patient-ID alignment "
                                "is wrong. Aborting."
                            )
                        log(f"        ID check: label constant within patient (0.00%)")
                    else:
                        log(f"        within-patient label variability: {100 * var:.2f}% of patients")

                results, prevalence, qa = evaluate_combination(
                    dataset, embedding, task, classifier, probs, y_true, n_class,
                    pidx, args.modes, args)
                del probs
                gc.collect()

                out = write_task_json(dataset, embedding, task, classifier, results,
                                      prevalence, qa, pidx, args)
                store[(dataset, embedding, classifier, task)] = results

                base = results.get("per_radiograph")
                for mode in args.modes:
                    for metric in METRICS:
                        e = results[mode][metric]
                        csv_rows.append({
                            "dataset": dataset, "embedding": embedding,
                            "classifier": classifier, "task": task,
                            "task_group": "demographic" if task in DEMOGRAPHIC_TASKS else "disease",
                            "mode": mode, "metric": metric,
                            "mean": e["mean"],
                            "ci_low": e["confidence_interval"][0],
                            "ci_high": e["confidence_interval"][1],
                            "fold_sd": e.get("fold_sd"),
                            "draw_sd": e.get("draw_sd"),
                            "delta_vs_per_radiograph_pp":
                                None if base is None else 100.0 * (e["mean"] - base[metric]["mean"]),
                            "n_radiographs": int(pidx.n_rows),
                            "n_patients": int(pidx.n_patients),
                        })

                for mode, shares in prevalence.items():
                    for cls, share in enumerate(shares):
                        prev_rows.append({"dataset": dataset, "embedding": embedding,
                                          "task": task, "mode": mode, "class": cls,
                                          "share": share})

                log(f"        done in {time.time() - t0:.1f}s -> {out}")
                for mode in args.modes:
                    e = results[mode]
                    log("          {:<24s} acc {:.4f}  prec {:.4f}  rec {:.4f}  f1 {:.4f}  auc {:.4f}".format(
                        mode, e["accuracy"]["mean"], e["precision"]["mean"],
                        e["recall"]["mean"], e["f1"]["mean"], e["roc"]["mean"]))

            del X_test, y_test
            gc.collect()

        del pidx, patient_ids
        gc.collect()

    log("\n" + "=" * 78)
    log(f"Evaluated {len(store)} combinations in {(time.time() - t_start) / 60:.1f} min")
    if not store:
        log("Nothing was evaluated -- check the filters.")
        return 1

    # deduplicate prevalence rows (identical for every classifier of a task)
    seen, prev_unique = set(), []
    for r in prev_rows:
        key = (r["dataset"], r["embedding"], r["task"], r["mode"], r["class"])
        if key not in seen:
            seen.add(key)
            prev_unique.append(r)

    write_summary_csv(csv_rows, os.path.join(args.out_dir, "summary.csv"))
    write_prevalence_csv(prev_unique, os.path.join(args.out_dir, "prevalence.csv"))
    write_latex_tables(store, args)
    write_placeholders(store, args)
    log("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
