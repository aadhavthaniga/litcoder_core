"""ECoG through the existing ``unified.py`` CLI: parity with the Python API,
temporal semantics, and unchanged fMRI argument handling. Synthetic and offline."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import unified
from encoding.assembly import ECoGAssemblyGenerator
from encoding.downsample.downsampling import Downsampler
from encoding.features.embeddings import StaticEmbeddingFeatureExtractor
from encoding.features.FIR_expander import FIR
from encoding.features.simple_features import WordRateFeatureExtractor
from encoding.models.nested_cv import NestedCVModel
from encoding.trainer import AbstractTrainer
from encoding.utils import ModelSaver

REPO = Path(__file__).resolve().parents[1]
RATE, N = 10.0, 240
ALPHAS = [0.1, 1.0, 10.0]
BLOCKED = {
    "delays_s": [0, 0.1, 0.2], "protocol": "blocked", "embargo_s": 0.3,
    "memory_budget_bytes": 200_000_000,
    "feature_specs": [
        {"method": "words_per_second", "support_s": [0, 0]},
        {"method": "word_sum", "support_s": [0, 0]},
    ],
}
DIAGNOSTIC = {**{k: v for k, v in BLOCKED.items() if k != "embargo_s"},
              "protocol": "diagnostic_in_sample"}
# Effective config of the pre-change CLI for this command (snapshot of HEAD 8a9ca6f).
FMRI_ARGV = ("--dataset_type lebel --data_dir d --subject UTS03 --tr 2 --ndelays 4 "
             "--lookback 256 --cache_dir c --modality wordrate --model_name wordrate").split()
FMRI_SNAPSHOT = {
    'dataset_type': 'lebel', 'data_dir': 'd', 'subject': 'UTS03', 'tr': 2.0,
    'context_type': 'fullcontext', 'use_volume': False, 'story_idx': None,
    'modality': 'wordrate', 'modalities': None, 'model_name': 'wordrate', 'model_names': None,
    'layer_idx': 9, 'last_token': False, 'n_outer_folds': 5, 'n_inner_folds': 5,
    'folding_type': 'chunked', 'chunk_length': 20, 'singcutoff': 1e-10,
    'downsample_method': 'lanczos', 'lanczos_cutoff_mult': 1.0, 'lanczos_window': 3,
    'normalize_features': False, 'normalize_targets': False, 'ndelays': 4, 'lookback': 256,
    'use_gpu': False, 'cache_dir': 'c', 'results_dir': 'results', 'logger_backend': 'wandb',
    'wandb_project_name': 'lit-encoding', 'vector_path': None, 'binary': False,
    'lowercase': False, 'chunk_size': 0.1, 'context_size': 16.0, 'story_order': None,
    **{f"{p}{d}_{e}": None for p in ("", "train_", "test_")
       for d in ("features", "targets") for e in ("start", "end")},
}


def write_inputs(root, config):
    """Two recordings with a declared neural gap, written as an ecog.json manifest."""
    data = root / "data"
    data.mkdir()
    rng = np.random.default_rng(7)
    records = []
    for run in range(2):
        name, origin = f"run-{run}", 100.0 * run
        times = origin + np.arange(N) / RATE
        words, onsets, signs = [], [], np.zeros(N)
        for i in range(N):
            for k in range(rng.integers(0, 3)):
                word = "plus" if rng.random() > .5 else "minus"
                words.append(word)
                onsets.append(times[i] + k / (4 * RATE))
                signs[i] += 1 if word == "plus" else -1
        counts = np.bincount(np.searchsorted(times, onsets, side="right") - 1, minlength=N)
        brain = np.zeros((N, 2))
        brain[1:, 0] += 0.5 * counts[:-1]
        brain[2:, 1] += signs[:-2]
        brain += 0.3 * rng.normal(size=brain.shape)
        valid = np.ones(N, dtype=bool)
        valid[50:53] = False
        brain[~valid] = np.nan
        arrays = {"brain_data": brain.astype(np.float32), "neural_times": times,
                  "stimulus_times": np.asarray(onsets), "valid_samples": valid}
        record = {
            "recording_id": name, "subject_id": "S1", "run_id": name,
            "sampling_rate_hz": RATE, "neural_time_unit": "s", "stimulus_time_unit": "s",
            "neural_clock": "rec", "stimulus_clock": "rec",
            "words": words, "stimuli": words, "word_ids": [f"{name}:w{j}" for j in range(len(words))],
            "sample_ids": [f"{name}:s{i}" for i in range(N)],
            "electrodes": [{"electrode_id": "E2"}, {"electrode_id": "E1"}],
            "signal_units": "synthetic",
            "invalid_intervals": [{"start": times[50], "stop": times[53], "reason": "gap"}],
            "provenance": {"source": "synthetic CLI fixture",
                           "preprocessing": {"description": "none", "support_s": [0, 0]},
                           "stimulus_context": {"policy": "single words"}},
        }
        for field, value in arrays.items():
            np.save(data / f"{name}_{field}.npy", value)
            record[field] = {"npy": f"{name}_{field}.npy"}
        records.append((record, times))
    (data / "ecog.json").write_text(json.dumps(
        {"schema_version": 1, "recordings": [r for r, _ in records]}))
    vectors = root / "tiny.w2v.txt"
    vectors.write_text("2 1\nplus 1\nminus -1\n")
    cfg = root / "ecog_cfg.json"
    cfg.write_text(json.dumps(config))
    return data, cfg, vectors, {r["recording_id"]: t for r, t in records}


def cli_argv(root, data, cfg, vectors, *extra):
    return ["--dataset_type", "ecog", "--data_dir", str(data), "--subject", "S1",
            "--modalities", "wordrate", "embeddings", "--model_names", "wordrate", "tiny",
            "--vector_path", str(vectors), "--ecog_config", str(cfg),
            "--cache_dir", str(root / "cache"), "--results_dir", str(root / "cli"),
            "--logger_backend", "tensorboard", "--normalize_features", "--normalize_targets",
            *extra]


def api_run(root, data, vectors, config, **fit):
    """The documented Python API, built independently of the CLI helpers."""
    assembly = ECoGAssemblyGenerator(data).generate_assembly("S1")
    static = StaticEmbeddingFeatureExtractor(dict(
        vector_path=str(vectors), binary=False, lowercase=False,
        oov_handling="copy_prev", use_tqdm=True))
    trainer = AbstractTrainer(
        assembly, [WordRateFeatureExtractor({}), static], None,
        NestedCVModel("ridge_regression"), None, {}, layer_idx=9, lookback=256,
        dataset_type="ecog", logger_backend="tensorboard", results_dir=str(root / "api"),
        ecog_temporal_config=config)
    trainer.train(use_gpu=False, single_alpha=True, normalpha=True, use_corr=True,
                  normalize_features=True, normalize_targets=True, singcutoff=1e-10, **fit)
    return load(trainer.last_run_dir_)


def load(run_dir):
    return ModelSaver(Path(run_dir).parent).load_encoding_result(run_dir)


def forbid_fmri_preparation(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("ECoG entered fMRI preparation")
    for owner, name in [(Downsampler, "downsample"), (FIR, "make_delayed"),
                        (AbstractTrainer, "extract_and_downsample_features"),
                        (AbstractTrainer, "apply_fir_delays"), (AbstractTrainer, "structure_data"),
                        (unified.UnifiedTrainer, "__init__"), (unified.UnifiedTrainer, "prepare_data")]:
        monkeypatch.setattr(owner, name, fail)


# --- argument acceptance / rejection ---------------------------------------

def test_fmri_effective_defaults_unchanged():
    assert vars(unified.parse_args(FMRI_ARGV)) == FMRI_SNAPSHOT
    assert list(vars(unified.parse_args(FMRI_ARGV))) == list(FMRI_SNAPSHOT)


@pytest.mark.parametrize("drop", ["--tr", "--ndelays", "--lookback"])
def test_fmri_required_options_still_exit_2(drop):
    i = FMRI_ARGV.index(drop)
    with pytest.raises(SystemExit) as error:
        unified.parse_args(FMRI_ARGV[:i] + FMRI_ARGV[i + 2:])
    assert error.value.code == 2


@pytest.mark.parametrize("extra", [["--ecog_config", "x.json"], ["--alphas", "1"]])
def test_ecog_only_options_rejected_for_fmri(extra):
    with pytest.raises(SystemExit) as error:
        unified.parse_args(FMRI_ARGV + extra)
    assert error.value.code == 2


ECOG_BASE = "--dataset_type ecog --data_dir d --subject S1 --cache_dir c --modality wordrate --model_name wordrate".split()


def test_ecog_minimal_arguments_accepted():
    args = vars(unified.parse_args(ECOG_BASE + ["--ecog_config", "cfg.json", "--alphas", "1", "2"]))
    assert args["lookback"] == 256 and args["alphas"] == [1.0, 2.0]
    assert all(args[name] is None for name in unified.ECOG_REJECTED)


@pytest.mark.parametrize("extra", [
    [], ["--tr", "2"], ["--ndelays", "4"], ["--features_start", "3"], ["--use_volume"],
    ["--folding_type", "kfold"], ["--chunk_length", "5"], ["--downsample_method", "average"],
    ["--lanczos_window", "5"], ["--story_idx", "1"], ["--story_order", "a"],
])
def test_ecog_rejects_missing_config_and_fmri_only_options(extra, capsys):
    config = [] if not extra else ["--ecog_config", "cfg.json"]
    with pytest.raises(SystemExit) as error:
        unified.parse_args(ECOG_BASE + config + extra)
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert (extra[0] if extra else "--ecog_config") in message


def test_fmri_main_still_uses_unified_trainer(monkeypatch):
    seen = {}

    class Fake:
        DATASET_CONFIGS = unified.UnifiedTrainer.DATASET_CONFIGS

        def __init__(self, config):
            seen.update(config)

        def train(self):
            return {"median_score": 0.0}

    monkeypatch.setattr(unified, "UnifiedTrainer", Fake)
    monkeypatch.setattr(unified, "run_ecog", lambda c: pytest.fail("fMRI routed to ECoG"))
    unified.main(FMRI_ARGV)
    assert seen == {**FMRI_SNAPSHOT, "modalities": ["wordrate"], "model_names": ["wordrate"]}


# --- ECoG CLI behaviour -------------------------------------------------------

def test_blocked_cli_matches_python_api(tmp_path, monkeypatch):
    data, cfg, vectors, times = write_inputs(tmp_path, BLOCKED)
    with monkeypatch.context() as patch:
        forbid_fmri_preparation(patch)
        trainer = unified.main(cli_argv(tmp_path, data, cfg, vectors, "--n_outer_folds", "3",
                                        "--n_inner_folds", "2", "--alphas", *map(str, ALPHAS)))
    cli = load(trainer.last_run_dir_)
    api = api_run(tmp_path, data, vectors, BLOCKED, alphas=ALPHAS, n_outer_folds=3, n_inner_folds=2)

    # Scores, predictions, splits and fitted state are identical.
    np.testing.assert_array_equal(cli["metrics"]["correlations"], api["metrics"]["correlations"])
    assert cli["metrics"]["inference_status"] == "not_assessed"
    assert len(cli["results"]["folds"]) == 3
    for a, b in zip(cli["results"]["folds"], api["results"]["folds"]):
        for key in ("train_indices", "test_indices", "predictions", "targets",
                    "weights", "selected_alphas"):
            np.testing.assert_array_equal(a[key], b[key])
        for ia, ib in zip(a["inner_splits"], b["inner_splits"]):
            np.testing.assert_array_equal(ia["train_indices"], ib["train_indices"])
            np.testing.assert_array_equal(ia["validation_indices"], ib["validation_indices"])
    assert cli["hyperparams"] == api["hyperparams"]
    ca, cb = cli["results"]["configuration"], api["results"]["configuration"]
    assert ca.pop("alphas").tolist() == cb.pop("alphas").tolist() == ALPHAS
    assert ca == cb and ca["folding_type"] == "blocked" and ca["n_outer_folds"] == 3

    # Aligned rows, exclusions, support bounds and delay records are identical.
    meta, api_meta = cli["results"]["metadata"], api["results"]["metadata"]
    temporal = meta["temporal"]
    for key in ("rows", "exclusions", "support_bounds_s", "feature_specs", "feature_widths",
                "protocol", "embargo_s", "design_order", "recordings"):
        assert repr(temporal[key]) == repr(api_meta["temporal"][key]), key
    assert meta["sample_ids"] == api_meta["sample_ids"]

    # Sample timing and delays survive untouched.
    for row in temporal["rows"] + temporal["exclusions"]:
        assert row["time_s"] == times[row["recording_id"]][row["source_index"]]
        assert row["sample_id"] == f"{row['recording_id']}:s{row['source_index']}"
    assert not any(r["source_index"] in (50, 51, 52) for r in temporal["rows"])
    for record in temporal["recordings"].values():
        assert record["delays"]["requested_s"] == [0, 0.1, 0.2]
        assert record["delays"]["sample_offsets"] == [0, 1, 2]
        assert record["sampling_rate_hz"] == RATE
    assert temporal["embargo_s"] == 0.3 and tuple(temporal["feature_widths"]) == (1, 1)

    # Blocked CV: contiguous test blocks covering every row once, and no retained
    # training row whose support overlaps an embargo-dilated test row.
    bounds = np.asarray(temporal["support_bounds_s"])
    recordings = np.array([r["recording_id"] for r in temporal["rows"]])
    tested = np.concatenate([f["test_indices"] for f in cli["results"]["folds"]])
    np.testing.assert_array_equal(np.sort(tested), np.arange(len(recordings)))
    for fold in cli["results"]["folds"]:
        train, test = fold["train_indices"], fold["test_indices"]
        np.testing.assert_array_equal(test, np.arange(test[0], test[-1] + 1))
        assert not set(train) & set(test)
        assert len(train) < len(recordings) - len(test)  # the embargo purged rows
        for rec in np.unique(recordings[test]):
            held = bounds[test[recordings[test] == rec]] + [-0.3, 0.3]
            for lo, hi in bounds[train[recordings[train] == rec]]:
                assert not ((held[:, 0] <= hi) & (lo <= held[:, 1])).any()

    # Reproducibility record: effective config, hashes, no absolute paths.
    run = meta["run_config"]
    assert run["ecog_temporal_config"] == BLOCKED
    assert run["ecog_config_sha256"] == hashlib.sha256(cfg.read_bytes()).hexdigest()
    assert run["manifest_sha256"] == hashlib.sha256((data / "ecog.json").read_bytes()).hexdigest()
    assert (run["subject"], run["modalities"], run["model_names"]) == (
        "S1", ["wordrate", "embeddings"], ["wordrate", "tiny"])
    assert run["fit_options"]["n_outer_folds"] == 3 and run["encoding_model"] == "ridge_regression"
    assert str(tmp_path) not in json.dumps(run, default=str)
    assert "run_config" not in api_meta


def test_diagnostic_in_sample_protocol(tmp_path, monkeypatch):
    data, cfg, vectors, _ = write_inputs(tmp_path, DIAGNOSTIC)
    with monkeypatch.context() as patch:
        forbid_fmri_preparation(patch)
        trainer = unified.main(cli_argv(tmp_path, data, cfg, vectors, "--alphas", "1.0"))
    cli = load(trainer.last_run_dir_)
    api = api_run(tmp_path, data, vectors, DIAGNOSTIC, alphas=[1.0])
    assert cli["metrics"]["evaluation"] == "in_sample_diagnostic"
    [fold] = cli["results"]["folds"]
    assert fold["model_role"] == "in_sample_diagnostic"
    np.testing.assert_array_equal(fold["predictions"], api["results"]["folds"][0]["predictions"])
    temporal = cli["results"]["metadata"]["temporal"]
    assert temporal["protocol"] == "diagnostic_in_sample"
    assert temporal["validity_policy"] == "allow_finite_unknown_in_sample"
    assert repr(temporal["rows"]) == repr(api["results"]["metadata"]["temporal"]["rows"])


def test_diagnostic_semantics_are_enforced_through_cli(tmp_path):
    data, cfg, vectors, _ = write_inputs(tmp_path, DIAGNOSTIC)
    with pytest.raises(ValueError, match="do not use CV fold counts"):
        unified.main(cli_argv(tmp_path, data, cfg, vectors, "--alphas", "1", "--n_outer_folds", "3"))
    with pytest.raises(ValueError, match="one fixed positive"):
        unified.main(cli_argv(tmp_path, data, cfg, vectors))


def test_cli_subprocess_smoke(tmp_path):
    data, cfg, vectors, _ = write_inputs(tmp_path, BLOCKED)
    env = {**os.environ, "PYTHONPATH": str(REPO), "WANDB_MODE": "disabled",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MPLBACKEND": "Agg"}
    # Real reporting path (no conftest stubs); wandb in disabled mode stays offline.
    argv = cli_argv(tmp_path, data, cfg, vectors, "--n_outer_folds", "3", "--n_inner_folds", "2",
                    "--logger_backend", "wandb")
    done = subprocess.run([sys.executable, str(REPO / "unified.py"), *argv], cwd=tmp_path,
                          env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-3000:]
    [run] = [p for p in (tmp_path / "cli").iterdir() if p.name.startswith("run_")]
    result = load(run)
    assert result["results"]["metadata"]["temporal"]["protocol"] == "blocked"
    assert np.isfinite(result["metrics"]["correlations"]).all()
