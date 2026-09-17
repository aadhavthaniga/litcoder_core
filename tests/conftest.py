"""Shared fixtures for the synthetic end-to-end pipeline tests.

The real LeBel assembly is ~3 GB and needs a GPU-ish machine to be useful. These
tests instead build a small assembly with the *same structural conventions* and a
signal planted at known FIR lags, so the whole extract -> downsample -> FIR ->
trim -> ridge path can be checked for correctness in seconds.

Conventions copied from ``data/assembly_lebel_uts03.pkl`` (verified against all 25
stories of that file):

* ``len(split_indices) + 1 == len(tr_times)`` -- split_indices are *cut points*
  into the word axis, consumed by ``np.split``, not a per-word TR label.
* ``brain_data.shape[0] == len(tr_times) - 15`` -- 10 TRs trimmed at the head and
  5 at the tail upstream of the assembly.
* ``len(words) == len(stimuli) == len(data_times) == n_words``
* ``word_rates.shape == (n_trs, 1)``
"""

import os

# Must happen before anything imports matplotlib/wandb (encoding.trainer pulls in
# encoding.plotting.plotting_utils, which imports matplotlib and nilearn at module
# scope). Belt and braces with the monkeypatching in _no_external_side_effects.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_DISABLED", "true")

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from encoding.assembly.assemblies import SimpleNeuroidAssembly
from encoding.assembly.story_data import StoryData
from encoding.features.base import BaseFeatureExtractor
from encoding.features.embeddings import StaticEmbeddingFeatureExtractor

# --------------------------------------------------------------------------
# Constants mirroring the real LeBel assembly / train_simple.py
# --------------------------------------------------------------------------

N_VERTICES = 20484          # fsaverage5, both hemispheres
N_FEATURES = 4              # synthetic embedding dimensionality
N_SIGNAL_VERTICES = 10      # vertices 0..9 carry the planted signal
PLANTED_LAGS = (2, 3)       # the "bump" in the planted FIR kernel
FIR_DELAYS = [1, 2, 3, 4]   # delays the pipeline is run with
TR = 1.5
BRAIN_TRIM_HEAD = 10        # TRs already removed from brain_data at the front
BRAIN_TRIM_TAIL = 5         # ... and at the back
BRAIN_TRIM_TOTAL = BRAIN_TRIM_HEAD + BRAIN_TRIM_TAIL

# Same trimming used in train_simple.py for the LeBel dataset.
TRIMMING_CONFIG = {
    "train_features_start": 10,
    "train_features_end": -5,
    "train_targets_start": 0,
    "train_targets_end": None,
    "test_features_start": 50,
    "test_features_end": -5,
    "test_targets_start": 40,
    "test_targets_end": None,
}


# --------------------------------------------------------------------------
# Deterministic token -> vector map (a stand-in for a static embedding model)
# --------------------------------------------------------------------------

def token_vector(token: str, dim: int = N_FEATURES) -> np.ndarray:
    """Map a token to a fixed pseudo-random vector.

    Uses blake2b rather than ``hash()`` so the mapping is stable across processes
    (Python salts str hashes per interpreter run).
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(digest, "little")
    return np.random.default_rng(seed).standard_normal(dim)


class SyntheticTokenFeatureExtractor(StaticEmbeddingFeatureExtractor):
    """Word-level feature extractor that plugs into the real factory dispatch.

    Subclasses ``StaticEmbeddingFeatureExtractor`` so that
    ``FeatureExtractorFactory._get_modality_from_extractor`` reports "embeddings"
    and the factory feeds it ``assembly.get_words()[idx]`` (one entry per word).
    The parent ``__init__`` is deliberately bypassed: it requires a gensim vector
    file on disk, which is exactly what we are replacing.

    The class name must not contain "wordrate", or ``AbstractTrainer._should_downsample``
    would skip downsampling for it.
    """

    def __init__(self, dim: int = N_FEATURES):
        BaseFeatureExtractor.__init__(self, {"model_name": "synthetic-tokens"})
        self.dim = dim

    def extract_features(self, stimuli, **kwargs) -> np.ndarray:
        if isinstance(stimuli, str):
            stimuli = stimuli.split()
        return np.vstack([token_vector(tok, self.dim) for tok in stimuli])


# --------------------------------------------------------------------------
# Reference implementations -- deliberately independent of encoding/
# --------------------------------------------------------------------------

def reference_legacy_average(word_features: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Average word features per TR, given the number of words in each TR.

    Written from ``counts`` rather than from ``split_indices``/``np.split`` so it
    is an independent oracle for LegacyAverageDownsampler rather than a restatement
    of it. TRs with no words stay as zero rows, matching the library.
    """
    out = np.zeros((len(counts), word_features.shape[1]))
    pos = 0
    for i, c in enumerate(counts):
        if c:
            out[i] = word_features[pos:pos + c].mean(axis=0)
        pos += c
    return out


def reference_make_delayed(features: np.ndarray, delays) -> np.ndarray:
    """Causal FIR expansion: block d of row t holds features[t - d], else zeros."""
    n_t, n_d = features.shape
    blocks = []
    for d in delays:
        block = np.zeros((n_t, n_d))
        if d > 0:
            block[d:] = features[:-d]
        elif d == 0:
            block = features.copy()
        else:
            block[:d] = features[-d:]
        blocks.append(block)
    return np.hstack(blocks)


def column_correlations(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pearson r between matching columns of two (n_samples, n_targets) matrices."""
    az = a - a.mean(axis=0)
    bz = b - b.mean(axis=0)
    denom = np.linalg.norm(az, axis=0) * np.linalg.norm(bz, axis=0)
    out = np.zeros(a.shape[1])
    nz = denom > 0
    out[nz] = (az[:, nz] * bz[:, nz]).sum(axis=0) / denom[nz]
    return out


def ridge_fit_predict(X_train, Y_train, X_test, alpha: float = 100.0) -> np.ndarray:
    """Plain closed-form ridge; a fast stand-in for NestedCVModel in ablations."""
    n_feat = X_train.shape[1]
    gram = X_train.T @ X_train + alpha * np.eye(n_feat)
    weights = np.linalg.solve(gram, X_train.T @ Y_train)
    return X_test @ weights


# --------------------------------------------------------------------------
# Ground truth bundled alongside the assembly
# --------------------------------------------------------------------------

@dataclass
class StoryTruth:
    """Everything a test needs to know about how one synthetic story was built."""

    name: str
    n_trs: int
    n_words: int
    counts: np.ndarray                 # words per TR, length n_trs
    word_features: np.ndarray          # (n_words, N_FEATURES)
    tr_features: np.ndarray            # (n_trs, N_FEATURES), the correct downsampling
    empty_trs: np.ndarray              # indices of TRs containing no words


@dataclass
class GroundTruth:
    """Ground truth for the whole synthetic assembly."""

    stories: Dict[str, StoryTruth] = field(default_factory=dict)
    lag_weights: Dict[int, np.ndarray] = field(default_factory=dict)  # lag -> (N_FEATURES, N_SIGNAL)
    signal_vertices: np.ndarray = field(default_factory=lambda: np.arange(N_SIGNAL_VERTICES))
    noise_vertices: np.ndarray = field(default_factory=lambda: np.arange(N_SIGNAL_VERTICES, N_VERTICES))
    planted_lags: Tuple[int, ...] = PLANTED_LAGS
    fir_delays: Tuple[int, ...] = tuple(FIR_DELAYS)
    trimming_config: Dict = field(default_factory=lambda: dict(TRIMMING_CONFIG))


# --------------------------------------------------------------------------
# The fixture builder
# --------------------------------------------------------------------------

def synthetic_assembly(
    n_stories: int = 3,
    n_trs: Tuple[int, ...] = (200, 185, 210),
    n_vertices: int = N_VERTICES,
    n_features: int = N_FEATURES,
    n_signal_vertices: int = N_SIGNAL_VERTICES,
    planted_lags: Tuple[int, ...] = PLANTED_LAGS,
    noise_ratio: float = 0.5,
    max_words_per_tr: int = 12,
    seed: int = 0,
) -> SimpleNeuroidAssembly:
    """Build a small assembly with a signal planted at known FIR lags.

    The returned object is a real ``SimpleNeuroidAssembly`` and can be handed to
    ``AbstractTrainer`` anywhere the LeBel assembly would go. Ground truth is
    attached as ``assembly.truth`` (a :class:`GroundTruth`).

    How the signal is planted
    -------------------------
    Per story, word-level features ``W`` are averaged into TR-level features ``F``
    using the legacy convention. Brain data is then

        brain[j, v] = sum_{d in planted_lags} F[j + BRAIN_TRIM_HEAD - d] @ lag_weights[d][:, v]

    for ``v`` in the first ``n_signal_vertices``, and pure noise elsewhere.

    The ``+ BRAIN_TRIM_HEAD`` offset is what makes the planting consistent with
    ``TRIMMING_CONFIG``: the trainer pairs ``features[10:-5]`` with ``targets[0:]``,
    so brain row ``j`` is aligned with feature row ``j + 10``, whose FIR block for
    delay ``d`` holds ``F[j + 10 - d]``. Only *past* TRs are ever referenced, so the
    construction is causal by design.

    Args:
        n_stories: Number of stories. The last one becomes the held-out test story.
        n_trs: Per-story TR counts (cycled if shorter than ``n_stories``).
        noise_ratio: Noise std as a fraction of each signal vertex's own std.
            0.5 gives an achievable ceiling of r ~= 0.89.
        max_words_per_tr: Words per TR are drawn uniformly from [0, this), so some
            TRs are empty -- which is what exercises the zero-row path in the
            legacy downsamplers.
    """
    if n_stories < 2:
        raise ValueError("need at least 2 stories for a train/test split")
    if max(planted_lags) >= BRAIN_TRIM_HEAD:
        raise ValueError("planted lags must be < BRAIN_TRIM_HEAD to stay in range")

    rng = np.random.default_rng(seed)

    # One shared set of lag weights, so the planted kernel is identical across stories.
    lag_weights = {
        lag: rng.standard_normal((n_features, n_signal_vertices))
        for lag in planted_lags
    }

    truth = GroundTruth(
        lag_weights=lag_weights,
        signal_vertices=np.arange(n_signal_vertices),
        noise_vertices=np.arange(n_signal_vertices, n_vertices),
        planted_lags=tuple(planted_lags),
    )

    story_data_list: List[StoryData] = []

    for s in range(n_stories):
        story_rng = np.random.default_rng(seed + 1000 + s)
        name = f"synthstory{s}"
        n_tr = n_trs[s % len(n_trs)]

        # --- word/TR bookkeeping, legacy split_indices convention -------------
        counts = story_rng.integers(0, max_words_per_tr, size=n_tr)
        # Mirror the real data: silence at the very start and end of a run.
        counts[:3] = 0
        counts[-2:] = 0
        n_words = int(counts.sum())

        # Cut points into the word axis: cumsum minus the final total, giving
        # exactly n_tr - 1 entries so that np.split yields n_tr chunks.
        split_indices = np.cumsum(counts)[:-1].astype(np.int64).tolist()

        words = [f"w{s}_{i}" for i in range(n_words)]
        # Cumulative-context stimuli, one per word, as in the real assembly.
        stimuli = [" ".join(words[max(0, i - 5):i + 1]) for i in range(n_words)]

        tr_times = np.arange(n_tr, dtype=np.float64) * TR

        # Word onsets spread inside their own TR, so data_times stays sorted.
        data_times = np.empty(n_words, dtype=np.float64)
        pos = 0
        for i, c in enumerate(counts):
            if c:
                offsets = np.sort(story_rng.uniform(0, TR, size=c))
                data_times[pos:pos + c] = tr_times[i] + offsets
                pos += c

        word_features = np.vstack([token_vector(w, n_features) for w in words])
        tr_features = reference_legacy_average(word_features, counts)

        # --- plant the FIR signal --------------------------------------------
        n_brain = n_tr - BRAIN_TRIM_TOTAL
        j = np.arange(n_brain)
        signal = np.zeros((n_brain, n_signal_vertices))
        for lag in planted_lags:
            signal += tr_features[j + BRAIN_TRIM_HEAD - lag] @ lag_weights[lag]

        brain = np.empty((n_brain, n_vertices), dtype=np.float64)
        signal_std = signal.std(axis=0)
        brain[:, :n_signal_vertices] = signal + story_rng.standard_normal(
            (n_brain, n_signal_vertices)
        ) * (signal_std * noise_ratio)
        # Noise vertices get a comparable scale so nothing is separable by variance alone.
        brain[:, n_signal_vertices:] = story_rng.standard_normal(
            (n_brain, n_vertices - n_signal_vertices)
        ) * float(np.median(signal_std))

        truth.stories[name] = StoryTruth(
            name=name,
            n_trs=n_tr,
            n_words=n_words,
            counts=counts,
            word_features=word_features,
            tr_features=tr_features,
            empty_trs=np.flatnonzero(counts == 0),
        )

        story_data_list.append(
            StoryData(
                name=name,
                brain_data=brain,
                stimuli=stimuli,
                split_indices=split_indices,
                tr_times=tr_times,
                data_times=data_times,
                temporal_baseline=None,
                word_rates=counts.reshape(-1, 1).astype(np.float64),
                words=words,
                mask_indices=None,
                audio_path=f"/nonexistent/audio/{name}.wav",
            )
        )

    assembly = SimpleNeuroidAssembly(story_data_list, validation_method="outer")
    assembly.truth = truth
    return assembly


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

class _NullLogger:
    """Stand-in for WandBLogger/TensorBoardLogger that records calls in memory."""

    def __init__(self, *args, **kwargs):
        self.scalars: Dict[str, float] = {}
        self.images: List[str] = []

    def log_scalar(self, name, value, step=None):
        self.scalars[name] = value

    def log_image(self, name, figure, step=None):
        self.images.append(name)

    def log_histogram(self, name, values, step=None):
        pass


class _NullBrainPlotter:
    """Stand-in for BrainPlotter; the real one downloads fsaverage and renders 3D."""

    def __init__(self, logger=None):
        self.logger = logger
        self.calls = 0

    def log_plots(self, *args, **kwargs):
        self.calls += 1


@pytest.fixture(autouse=True)
def _no_external_side_effects(monkeypatch):
    """Keep tests offline: no wandb run, no nilearn surface download, no GUI backend."""
    import matplotlib
    matplotlib.use("Agg", force=True)

    import encoding.trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "WandBLogger", _NullLogger)
    monkeypatch.setattr(trainer_mod, "TensorBoardLogger", _NullLogger)
    monkeypatch.setattr(trainer_mod, "BrainPlotter", _NullBrainPlotter)

    # setup_logger calls wandb.init() directly before constructing WandBLogger.
    import wandb
    monkeypatch.setattr(wandb, "init", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(wandb, "log", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(wandb, "finish", lambda *a, **k: None, raising=False)

    # Last line of defence if a test reaches the real plotting utilities.
    import encoding.plotting.plotting_utils as plotting_utils
    monkeypatch.setattr(
        plotting_utils.datasets,
        "fetch_surf_fsaverage",
        lambda *a, **k: pytest.fail("test tried to download fsaverage"),
        raising=False,
    )


@pytest.fixture(scope="module")
def assembly() -> SimpleNeuroidAssembly:
    """The synthetic assembly, built once per test module."""
    return synthetic_assembly()


@pytest.fixture(scope="module")
def truth(assembly) -> GroundTruth:
    return assembly.truth


@pytest.fixture
def extractor() -> SyntheticTokenFeatureExtractor:
    return SyntheticTokenFeatureExtractor()


@pytest.fixture
def downsample_config() -> Dict:
    """legacy_average is the only averaging method that reads split_indices correctly."""
    return {"method": "legacy_average"}
