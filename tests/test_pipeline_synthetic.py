"""End-to-end tests for the encoding pipeline against a synthetic assembly.

Three groups:

* **Structural** -- the assembly obeys the same conventions as the real LeBel
  assembly, and each stage transforms shapes the way the next stage expects.
* **Recovery** -- a signal planted at FIR lags 2 and 3 in vertices 0..9 is
  recovered by the real pipeline, at the right lags and in the right vertices.
* **Leakage / ordering** -- the held-out story cannot reach the training matrix,
  the FIR expansion is causal, and the recovered correlation genuinely depends on
  temporal alignment rather than on some artefact of the plumbing.

Everything runs offline: no wandb run, no fsaverage download, Agg backend. See
``conftest._no_external_side_effects``.
"""

import numpy as np
import pytest

from conftest import (
    BRAIN_TRIM_HEAD,
    BRAIN_TRIM_TAIL,
    BRAIN_TRIM_TOTAL,
    FIR_DELAYS,
    N_FEATURES,
    N_SIGNAL_VERTICES,
    N_VERTICES,
    PLANTED_LAGS,
    TR,
    TRIMMING_CONFIG,
    SyntheticTokenFeatureExtractor,
    column_correlations,
    reference_legacy_average,
    reference_make_delayed,
    ridge_fit_predict,
    synthetic_assembly,
    _NullBrainPlotter,
    _NullLogger,
)

from encoding.downsample.downsampling import Downsampler
from encoding.features.FIR_expander import FIR
from encoding.models.nested_cv import NestedCVModel
from encoding.trainer import AbstractTrainer

# Keep the nested CV cheap; the point is correctness, not hyperparameter search.
MODEL_KWARGS = dict(alphas=[1.0, 10.0, 100.0, 1000.0], n_inner_folds=2, use_gpu=False)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_trainer(assembly, results_dir, **overrides):
    """Build an AbstractTrainer wired the way train_simple.py wires it."""
    kwargs = dict(
        assembly=assembly,
        feature_extractors=[SyntheticTokenFeatureExtractor()],
        downsampler=Downsampler(),
        model=NestedCVModel(model_name="ridge_regression"),
        fir_delays=FIR_DELAYS,
        trimming_config=dict(TRIMMING_CONFIG),
        use_train_test_split=True,
        logger_backend="wandb",
        dataset_type="synthetic",
        results_dir=str(results_dir),
        downsample_config={"method": "legacy_average"},
    )
    kwargs.update(overrides)
    return AbstractTrainer(**kwargs)


@pytest.fixture(scope="module")
def pipeline_data(assembly, tmp_path_factory):
    """Run extract -> downsample -> FIR -> structure once and reuse it.

    Module-scoped, so the function-scoped autouse patching in conftest is not yet
    active; patch explicitly for the duration of trainer construction.
    """
    results_dir = tmp_path_factory.mktemp("results")
    with pytest.MonkeyPatch.context() as mp:
        import encoding.trainer as trainer_mod
        import wandb

        mp.setattr(trainer_mod, "WandBLogger", _NullLogger)
        mp.setattr(trainer_mod, "BrainPlotter", _NullBrainPlotter)
        mp.setattr(wandb, "init", lambda *a, **k: None, raising=False)

        trainer = make_trainer(assembly, results_dir)
        features = trainer.extract_and_downsample_features()
        delayed = trainer.apply_fir_delays(features)
        data = trainer.structure_data(delayed)

    return {
        "trainer": trainer,
        "features": features,
        "delayed": delayed,
        "data": data,
        "results_dir": results_dir,
    }


@pytest.fixture(scope="module")
def fitted(pipeline_data):
    """The nested-CV ridge fit. One fit shared by the recovery tests."""
    data = pipeline_data["data"]
    model = pipeline_data["trainer"].model
    metrics, weights, alphas = model.fit_predict(
        features=data["Rstim"],
        targets=data["Rresp"],
        X_test=data["Pstim"],
        y_test=data["Presp"],
        **MODEL_KWARGS,
    )
    return {"metrics": metrics, "weights": weights, "alphas": alphas}


# ==========================================================================
# Structural
# ==========================================================================

def test_assembly_top_level_structure(assembly):
    """Top-level attributes mirror SimpleNeuroidAssembly's contract."""
    assert assembly.stories == ["synthstory0", "synthstory1", "synthstory2"]
    assert assembly.dims == ("presentation", "neuroid")
    assert assembly.get_validation_method() == "outer"
    assert set(assembly.story_data) == set(assembly.stories)

    expected_rows = sum(sd.brain_data.shape[0] for sd in assembly.story_data.values())
    assert assembly.shape == (expected_rows, N_VERTICES)
    assert assembly.data.shape == assembly.shape


def test_story_field_shapes_match_lebel_conventions(assembly, truth):
    """Word-clock fields agree with each other; TR-clock fields agree with each other."""
    for name in assembly.stories:
        sd = assembly.story_data[name]
        st = truth.stories[name]

        # Word clock: one entry per word.
        assert len(sd.words) == st.n_words
        assert len(sd.stimuli) == st.n_words
        assert sd.data_times.shape == (st.n_words,)

        # TR clock: one entry per TR.
        assert sd.tr_times.shape == (st.n_trs,)
        assert sd.word_rates.shape == (st.n_trs, 1)

        # Brain data is 15 TRs shorter: 10 trimmed at the head, 5 at the tail.
        assert sd.brain_data.shape == (st.n_trs - BRAIN_TRIM_TOTAL, N_VERTICES)

        # Optional fields, as in the packaged LeBel assembly.
        assert sd.temporal_baseline is None
        assert sd.mask_indices is None

        # Word onsets are sorted and land inside the run.
        assert np.all(np.diff(sd.data_times) > 0)
        assert sd.data_times.min() >= sd.tr_times[0]
        assert sd.data_times.max() < sd.tr_times[-1] + TR


def test_split_indices_are_legacy_cut_points(assembly, truth):
    """split_indices are np.split cut points into the word axis, not per-word labels.

    This is the invariant that holds for all 25 stories of the real assembly:
    len(split_indices) + 1 == len(tr_times).
    """
    for name in assembly.stories:
        sd = assembly.story_data[name]
        st = truth.stories[name]
        si = np.asarray(sd.split_indices)

        assert si.size + 1 == len(sd.tr_times)
        assert np.all(np.diff(si) >= 0), "cut points must be non-decreasing"
        assert si.min() >= 0
        assert si.max() <= st.n_words
        # Empty TRs show up as repeated cut points, not as missing entries.
        assert np.unique(si).size < si.size


def test_legacy_average_matches_independent_reference(assembly, truth):
    """LegacyAverageDownsampler == averaging each TR's words, computed from counts."""
    ds = Downsampler()
    for name in assembly.stories:
        sd = assembly.story_data[name]
        st = truth.stories[name]

        out = ds.downsample(
            data=st.word_features,
            data_times=sd.data_times,
            tr_times=sd.tr_times,
            method="legacy_average",
            split_indices=sd.split_indices,
        )
        assert out.shape == (st.n_trs, N_FEATURES)
        np.testing.assert_allclose(out, reference_legacy_average(st.word_features, st.counts))


def test_legacy_average_leaves_empty_trs_as_zero_rows(assembly, truth):
    """TRs with no words come back as exact zero rows, not NaN and not interpolated.

    Worth pinning: a silent TR is indistinguishable from a genuinely zero feature,
    so downstream code must not treat "all zero" as missing data.
    """
    ds = Downsampler()
    name = assembly.stories[0]
    sd = assembly.story_data[name]
    st = truth.stories[name]

    out = ds.downsample(
        data=st.word_features,
        data_times=sd.data_times,
        tr_times=sd.tr_times,
        method="legacy_average",
        split_indices=sd.split_indices,
    )

    assert st.empty_trs.size > 0, "fixture should contain some silent TRs"
    assert np.all(out[st.empty_trs] == 0.0)
    assert np.isfinite(out).all()

    occupied = np.setdiff1d(np.arange(st.n_trs), st.empty_trs)
    assert np.any(out[occupied] != 0.0)


@pytest.mark.parametrize("method", ["legacy_average", "legacy_sum", "legacy_last"])
def test_legacy_downsamplers_return_one_row_per_tr(assembly, truth, method):
    """All three legacy methods use np.split, so all produce len(split_indices)+1 rows."""
    ds = Downsampler()
    name = assembly.stories[0]
    sd = assembly.story_data[name]
    st = truth.stories[name]

    out = ds.downsample(
        data=st.word_features,
        data_times=sd.data_times,
        tr_times=sd.tr_times,
        method=method,
        split_indices=sd.split_indices,
    )
    assert out.shape == (st.n_trs, N_FEATURES)
    assert out.shape[0] == len(sd.split_indices) + 1


@pytest.mark.parametrize("method", ["average", "sum", "last"])
def test_nonlegacy_downsamplers_misread_split_indices(assembly, truth, method):
    """Pin a known bug: the non-legacy methods read split_indices as per-word TR labels.

    They compute ``max(split_indices) + 1`` rows, which for this convention is
    roughly the word count rather than the TR count. Only the ``legacy_*`` methods
    are usable with these assemblies. If this test ever fails because the output is
    now (n_trs, ...), the bug has been fixed and this test should become the
    positive assertion instead.
    """
    ds = Downsampler()
    name = assembly.stories[0]
    sd = assembly.story_data[name]
    st = truth.stories[name]

    out = ds.downsample(
        data=st.word_features,
        data_times=sd.data_times,
        tr_times=sd.tr_times,
        method=method,
        split_indices=sd.split_indices,
    )
    assert out.shape[0] == max(sd.split_indices) + 1
    assert out.shape[0] != st.n_trs
    assert out.shape[0] > st.n_trs


@pytest.mark.parametrize(
    "method", ["legacy_average", "legacy_sum", "legacy_last", "average", "sum", "last"]
)
def test_downsamplers_require_split_indices(assembly, truth, method):
    """Omitting split_indices is a loud error, not a silent fallback."""
    ds = Downsampler()
    sd = assembly.story_data[assembly.stories[0]]
    st = truth.stories[assembly.stories[0]]

    with pytest.raises(ValueError, match="split_indices"):
        ds.downsample(
            data=st.word_features,
            data_times=sd.data_times,
            tr_times=sd.tr_times,
            method=method,
        )


def test_fir_expansion_matches_reference(assembly, truth):
    """FIR.make_delayed stacks one block per delay, in the order given."""
    st = truth.stories[assembly.stories[0]]
    delayed = FIR.make_delayed(st.tr_features, FIR_DELAYS)

    assert delayed.shape == (st.n_trs, N_FEATURES * len(FIR_DELAYS))
    np.testing.assert_allclose(delayed, reference_make_delayed(st.tr_features, FIR_DELAYS))

    # Block d holds features shifted forward by d, zero-padded at the front.
    for block, d in enumerate(FIR_DELAYS):
        cols = slice(block * N_FEATURES, (block + 1) * N_FEATURES)
        assert np.all(delayed[:d, cols] == 0.0)
        np.testing.assert_allclose(delayed[d:, cols], st.tr_features[:-d])


def test_structure_data_shapes_and_story_split(pipeline_data, assembly, truth):
    """Trimming pairs features with targets row-for-row, and the last story is held out."""
    features = pipeline_data["features"]
    delayed = pipeline_data["delayed"]
    data = pipeline_data["data"]

    for name in assembly.stories:
        st = truth.stories[name]
        assert features[name].shape == (st.n_trs, N_FEATURES)
        assert delayed[name].shape == (st.n_trs, N_FEATURES * len(FIR_DELAYS))

    train_stories = assembly.stories[:-1]
    test_story = assembly.stories[-1]

    expected_train = sum(truth.stories[s].n_trs - BRAIN_TRIM_TOTAL for s in train_stories)
    expected_test = truth.stories[test_story].n_trs - (
        TRIMMING_CONFIG["test_features_start"] + BRAIN_TRIM_TAIL
    )

    assert data["Rstim"].shape == (expected_train, N_FEATURES * len(FIR_DELAYS))
    assert data["Rresp"].shape == (expected_train, N_VERTICES)
    assert data["Pstim"].shape == (expected_test, N_FEATURES * len(FIR_DELAYS))
    assert data["Presp"].shape == (expected_test, N_VERTICES)

    # Features and targets must line up row-for-row, which nothing in
    # _create_train_test_split actually checks.
    assert data["Rstim"].shape[0] == data["Rresp"].shape[0]
    assert data["Pstim"].shape[0] == data["Presp"].shape[0]
    assert np.isfinite(data["Rstim"]).all()
    assert np.isfinite(data["Pstim"]).all()


# ==========================================================================
# Recovery
# ==========================================================================

def test_recovers_planted_signal_vertices(fitted, truth):
    """Vertices 0..9 carry the planted signal and should be predicted well."""
    correlations = np.asarray(fitted["metrics"]["correlations"])
    assert correlations.shape == (N_VERTICES,)

    signal_r = correlations[truth.signal_vertices]
    # Ceiling is ~0.89 at the fixture's default noise_ratio of 0.5.
    assert np.median(signal_r) > 0.75
    assert signal_r.min() > 0.6


def test_noise_vertices_are_not_predicted(fitted, truth):
    """Everything past vertex 9 is noise and must not be predictable."""
    correlations = np.asarray(fitted["metrics"]["correlations"])
    noise_r = np.abs(correlations[truth.noise_vertices])

    assert np.median(noise_r) < 0.12
    assert np.percentile(noise_r, 99) < 0.35
    # And the signal vertices must stand clearly apart from the noise floor.
    signal_r = correlations[truth.signal_vertices]
    assert signal_r.min() > np.percentile(noise_r, 99.9)


def test_fdr_selects_the_planted_vertices(fitted, truth):
    """FDR correction should flag the planted vertices and few others."""
    significant = np.asarray(fitted["metrics"]["significant_mask"], dtype=bool)
    assert significant[truth.signal_vertices].all()
    # Allow a handful of false positives, but nothing like a whole-brain flag.
    assert significant.sum() < 5 * N_SIGNAL_VERTICES


def test_recovered_kernel_peaks_at_planted_lags(fitted, truth):
    """The learned FIR kernel bumps at lags 2 and 3, and is flat at lags 1 and 4."""
    weights = fitted["weights"]
    assert weights.shape == (N_FEATURES * len(FIR_DELAYS), N_VERTICES)

    # (n_delays, n_features, n_vertices) -- make_delayed hstacks blocks in delay order.
    by_lag = weights.reshape(len(FIR_DELAYS), N_FEATURES, N_VERTICES)
    signal_norms = {
        delay: float(np.linalg.norm(by_lag[i][:, truth.signal_vertices]))
        for i, delay in enumerate(FIR_DELAYS)
    }

    planted = [signal_norms[d] for d in PLANTED_LAGS]
    empty = [signal_norms[d] for d in FIR_DELAYS if d not in PLANTED_LAGS]

    assert min(planted) > 3 * max(empty), (
        f"kernel did not peak at lags {PLANTED_LAGS}: {signal_norms}"
    )

    # Noise vertices should have no structured kernel at all.
    noise_norm = float(np.linalg.norm(by_lag[:, :, truth.noise_vertices]))
    per_vertex_noise = noise_norm / np.sqrt(len(truth.noise_vertices))
    per_vertex_signal = min(planted) / np.sqrt(N_SIGNAL_VERTICES)
    assert per_vertex_signal > per_vertex_noise


def test_trainer_train_runs_offline_and_reports_metrics(assembly, tmp_path):
    """trainer.train() end to end, with logging and brain plotting stubbed out."""
    trainer = make_trainer(assembly, tmp_path)
    assert isinstance(trainer.experiment_logger, _NullLogger)
    assert isinstance(trainer.brain_plotter, _NullBrainPlotter)

    metrics = trainer.train(**MODEL_KWARGS)

    correlations = np.asarray(metrics["correlations"])
    assert np.median(correlations[:N_SIGNAL_VERTICES]) > 0.75
    assert metrics["median_score"] == pytest.approx(float(np.median(correlations)))
    assert 0 < metrics["n_significant"] < 5 * N_SIGNAL_VERTICES

    # Metrics reached the logger and the plotter was invoked but drew nothing.
    assert "median_correlation" in trainer.experiment_logger.scalars
    assert trainer.brain_plotter.calls == 1

    # ModelSaver wrote under the temp results dir, not into the repo.
    assert any(tmp_path.rglob("hyperparams.json"))


# ==========================================================================
# Leakage and ordering
# ==========================================================================

def test_fir_is_causal(assembly, truth):
    """No delay is negative, and perturbing TR t never changes rows before t."""
    assert all(d > 0 for d in FIR_DELAYS), "a delay <= 0 would expose the present/future"

    st = truth.stories[assembly.stories[0]]
    base = FIR.make_delayed(st.tr_features, FIR_DELAYS)

    perturb_at = st.n_trs // 2
    bumped = st.tr_features.copy()
    bumped[perturb_at] += 1000.0
    after = FIR.make_delayed(bumped, FIR_DELAYS)

    changed_rows = np.flatnonzero(np.any(base != after, axis=1))
    assert changed_rows.size > 0
    assert changed_rows.min() > perturb_at, (
        "a future timepoint leaked into an earlier delayed row"
    )
    # Only the planted delays' worth of rows should move.
    assert set(changed_rows.tolist()) == {perturb_at + d for d in FIR_DELAYS}


def test_held_out_story_cannot_influence_training_matrix(pipeline_data, assembly):
    """Corrupting the held-out story leaves the training matrices bit-identical.

    Poisoning is a sharper check than comparing row counts: if any test-story value
    reached X_train or Y_train, the training matrices would move. Both the features
    and the brain data of the held-out story are corrupted, so the check covers
    predictors and targets.
    """
    trainer = pipeline_data["trainer"]
    baseline = pipeline_data["data"]
    test_story = assembly.stories[-1]

    poisoned = {k: v.copy() for k, v in pipeline_data["delayed"].items()}
    poisoned[test_story][:] = np.nan

    story = assembly.story_data[test_story]
    clean_brain = story.brain_data
    try:
        story.brain_data = np.full_like(clean_brain, np.nan)
        data = trainer.structure_data(poisoned)
    finally:
        story.brain_data = clean_brain

    assert np.isfinite(data["Rstim"]).all()
    assert np.isfinite(data["Rresp"]).all()
    np.testing.assert_array_equal(data["Rstim"], baseline["Rstim"])
    np.testing.assert_array_equal(data["Rresp"], baseline["Rresp"])

    # The poison did land on the test side, so the check above is not vacuous.
    assert not np.array_equal(data["Pstim"], baseline["Pstim"])
    assert np.isnan(data["Presp"]).all()


def test_nan_features_are_silently_zeroed(pipeline_data, assembly):
    """Pin a silent-failure path: NaN *features* are laundered by np.nan_to_num.

    _create_train_test_split wraps X_train/X_test in np.nan_to_num but leaves
    Y_train/Y_test alone. A feature extractor that starts emitting NaN therefore
    produces an all-zero predictor block and a plausible-looking fit instead of an
    error, while the same corruption in the targets propagates loudly.
    """
    trainer = pipeline_data["trainer"]
    test_story = assembly.stories[-1]

    poisoned = {k: v.copy() for k, v in pipeline_data["delayed"].items()}
    poisoned[test_story][:] = np.nan

    data = trainer.structure_data(poisoned)

    assert np.all(data["Pstim"] == 0.0), "NaN features are converted to zeros, not raised"
    assert np.isfinite(data["Presp"]).all()


def test_train_and_test_draw_from_disjoint_stories(pipeline_data, assembly, truth):
    """The split is by story: the last one is test, everything before it is train."""
    trainer = pipeline_data["trainer"]
    stories = list(pipeline_data["features"].keys())

    assert stories == assembly.stories
    train_stories, test_stories = stories[:-1], stories[-1:]
    assert set(train_stories).isdisjoint(test_stories)
    assert len(test_stories) == 1

    # Row budget accounts for every train row, leaving no room for test rows.
    expected = sum(truth.stories[s].n_trs - BRAIN_TRIM_TOTAL for s in train_stories)
    assert pipeline_data["data"]["Rstim"].shape[0] == expected


def test_each_split_is_zscored_independently(pipeline_data, assembly):
    """z-scoring happens per story within a split; train data cannot shift the test set.

    Note this also documents that the *test* split is standardised with its own
    statistics rather than the training set's -- standard in this literature, but
    it is test-set information entering the test features, so it is pinned here
    deliberately rather than by accident.
    """
    trainer = pipeline_data["trainer"]
    delayed = pipeline_data["delayed"]
    baseline = pipeline_data["data"]

    scaled = {k: v.copy() for k, v in delayed.items()}
    for story in assembly.stories[:-1]:
        scaled[story] = scaled[story] * 50.0 + 7.0

    data = trainer.structure_data(scaled)

    # Test split untouched by the train-side rescaling.
    np.testing.assert_allclose(data["Pstim"], baseline["Pstim"])
    # And z-scoring makes the train split invariant to affine rescaling.
    np.testing.assert_allclose(data["Rstim"], baseline["Rstim"], atol=1e-8)

    # Each split is standardised with its own statistics.
    assert np.allclose(baseline["Pstim"].mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(baseline["Pstim"].std(axis=0), 1.0, atol=1e-6)


def test_alignment_is_load_bearing(pipeline_data, truth):
    """Break the temporal alignment and the recovered correlation collapses.

    Guards against a result that looks good for the wrong reason: if prediction
    survived an arbitrary time shift, the correlation would be coming from
    something other than the planted lag structure.
    """
    data = pipeline_data["data"]
    signal = truth.signal_vertices

    aligned = ridge_fit_predict(data["Rstim"], data["Rresp"][:, signal], data["Pstim"])
    aligned_r = column_correlations(aligned, data["Presp"][:, signal])
    assert np.median(aligned_r) > 0.7

    # A shift well outside the delay set, so no lag can absorb it.
    shift = 37
    rolled_train = np.roll(data["Rstim"], shift, axis=0)
    rolled_test = np.roll(data["Pstim"], shift, axis=0)
    rolled = ridge_fit_predict(rolled_train, data["Rresp"][:, signal], rolled_test)
    rolled_r = column_correlations(rolled, data["Presp"][:, signal])

    assert np.median(np.abs(rolled_r)) < 0.2
    assert np.median(aligned_r) > 3 * np.median(np.abs(rolled_r))


def test_pipeline_order_is_downsample_then_fir(pipeline_data, assembly, truth):
    """Downsampling must precede FIR expansion; the row counts pin the order.

    Expanding word-level features first would give one row per word and silently
    misalign everything downstream, so the two orders are not interchangeable.
    """
    features = pipeline_data["features"]
    delayed = pipeline_data["delayed"]

    for name in assembly.stories:
        st = truth.stories[name]

        assert features[name].shape[0] == st.n_trs
        assert features[name].shape[0] != st.n_words

        assert delayed[name].shape == (st.n_trs, N_FEATURES * len(FIR_DELAYS))

        # The other order produces a word-length matrix that cannot pair with brain data.
        fir_first = FIR.make_delayed(st.word_features, FIR_DELAYS)
        assert fir_first.shape[0] == st.n_words
        assert fir_first.shape[0] != assembly.story_data[name].brain_data.shape[0]


def test_trimming_config_pairs_features_with_past_tr_only(truth):
    """The trimming offsets keep every predictor at or before its target's TR.

    Brain row j is paired with feature row j + 10, whose planted lags point at TRs
    j + 8 and j + 7. Both precede the feature TR, so no target is predicted from
    its own future.
    """
    feature_offset = TRIMMING_CONFIG["train_features_start"]
    target_offset = TRIMMING_CONFIG["train_targets_start"]
    assert feature_offset - target_offset == BRAIN_TRIM_HEAD

    for lag in PLANTED_LAGS:
        source_tr = feature_offset - lag
        assert 0 <= source_tr < feature_offset, "predictor must come from a past TR"

    # The test split uses different offsets but must preserve the same relationship.
    assert (
        TRIMMING_CONFIG["test_features_start"] - TRIMMING_CONFIG["test_targets_start"]
        == BRAIN_TRIM_HEAD
    )
    assert TRIMMING_CONFIG["train_features_end"] == -BRAIN_TRIM_TAIL
    assert TRIMMING_CONFIG["test_features_end"] == -BRAIN_TRIM_TAIL


def test_determinism(pipeline_data, monkeypatch):
    """The same seed reproduces the correlation vector; a different fold seed does not.

    The second half is what stops this passing vacuously. Note it has to compare the
    *correlation arrays*: changing the fold seed reshuffles which alpha each voxel
    gets, but on this fixture it moves the median by only ~5e-6 and leaves
    n_significant unchanged, so asserting on either of those would pass no matter
    what the pipeline did.

    ``fit_predict`` exposes no seed argument -- ``_find_best_alphas`` calls
    ``create_folds`` without one, so the fold seed is always the default 0. The
    alternate seed is injected by patching that call site.
    """
    data = pipeline_data["data"]
    model = pipeline_data["trainer"].model

    def fit():
        metrics, _, alphas = model.fit_predict(
            features=data["Rstim"],
            targets=data["Rresp"],
            X_test=data["Pstim"],
            y_test=data["Presp"],
            **MODEL_KWARGS,
        )
        return np.asarray(metrics["correlations"]), np.asarray(alphas)

    first, first_alphas = fit()
    second, second_alphas = fit()

    # Same seed, same everything: reproducible.
    assert np.allclose(first, second, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(first_alphas, second_alphas)

    # Different fold seed: a genuinely different model.
    import encoding.models.nested_cv as nested_cv_mod
    from encoding.models.folding import create_folds as real_create_folds

    def seeded_create_folds(n_samples, fold_type, n_folds, chunk_length=None,
                            trim_size=None, groups=None, seed=0):
        return real_create_folds(
            n_samples, fold_type, n_folds, chunk_length, trim_size, groups, seed=7
        )

    monkeypatch.setattr(nested_cv_mod, "create_folds", seeded_create_folds)
    other, other_alphas = fit()

    assert not np.allclose(first, other, rtol=0, atol=1e-12), (
        "changing the inner-CV fold seed produced an identical correlation vector, "
        "which means the seed is not reaching create_folds and the first half of "
        "this test proves nothing"
    )
    assert (first_alphas != other_alphas).any(), "no voxel changed its selected alpha"


def test_fixture_is_deterministic():
    """Same seed, same assembly -- so a failure is reproducible."""
    a = synthetic_assembly(n_stories=2, n_trs=(60, 60), n_vertices=64, seed=3)
    b = synthetic_assembly(n_stories=2, n_trs=(60, 60), n_vertices=64, seed=3)

    for name in a.stories:
        np.testing.assert_array_equal(
            a.story_data[name].brain_data, b.story_data[name].brain_data
        )
        assert a.story_data[name].split_indices == b.story_data[name].split_indices
