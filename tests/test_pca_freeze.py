"""Ràng buộc #6: PCA fit một lần rồi freeze, không refit ngầm."""

from __future__ import annotations

import numpy as np
import pytest

from worker.dedup import Deduplicator, FaissStore
from worker.pca import FrozenPCA, PCAFrozenError, fit_pca, load_or_raise


@pytest.fixture
def vectors():
    rng = np.random.default_rng(7)
    return rng.normal(size=(3000, 64)).astype(np.float32)


def test_fit_refuses_tiny_sample():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="cần >="):
        fit_pca(rng.normal(size=(50, 64)).astype(np.float32), "pca_v1", "hist", min_samples=2000)


def test_save_refuses_overwrite(tmp_path, vectors):
    p = fit_pca(vectors, "pca_v1", "hist", n_components=16, min_samples=100)
    path = tmp_path / "pca_v1.pkl"
    p.save(path)
    with pytest.raises(PCAFrozenError, match="đã freeze"):
        p.save(path)  # refit đè lên là thứ làm hỏng toàn bộ index cũ


def test_transform_is_stable_across_reload(tmp_path, vectors):
    p = fit_pca(vectors, "pca_v1", "hist", n_components=16, min_samples=100)
    path = tmp_path / "pca_v1.pkl"
    p.save(path)

    x = vectors[:5]
    before = p.transform(x)
    after = FrozenPCA.load(path).transform(x)
    np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)


def test_version_mismatch_is_rejected(tmp_path, vectors):
    p = fit_pca(vectors, "pca_v1", "hist", n_components=16, min_samples=100)
    path = tmp_path / "pca_v1.pkl"
    p.save(path)
    with pytest.raises(PCAFrozenError, match="pca_v2"):
        load_or_raise(path, "pca_v2")


def test_missing_pca_tells_you_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="fit_pca"):
        load_or_raise(tmp_path / "khong_ton_tai.pkl", "pca_v1")


def test_output_is_normalised_for_cosine(tmp_path, vectors):
    p = fit_pca(vectors, "pca_v1", "hist", n_components=16, min_samples=100)
    z = p.transform(vectors[:100])
    np.testing.assert_allclose(np.linalg.norm(z, axis=1), 1.0, atol=1e-5)


def test_index_refuses_mixing_pca_versions(tmp_path):
    store = FaissStore(tmp_path / "idx.index", 8, "pca_v1")
    v = np.ones((1, 8), dtype=np.float32) / np.sqrt(8)
    store.add(v, "f1")
    store.save()
    with pytest.raises(RuntimeError, match="reindex"):
        FaissStore(tmp_path / "idx.index", 8, "pca_v2")


def test_dedup_theta_looser_for_failure_source(tmp_path):
    store = FaissStore(tmp_path / "idx.index", 8, "pca_v1")
    d = Deduplicator(store, theta=0.90, theta_failure=0.99)

    a = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    b = np.array([0.96, 0.28, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    b /= np.linalg.norm(b)
    store.add(a, "f1")

    # Cùng một cặp vector: nguồn A coi là trùng, nguồn B vẫn giữ lại.
    assert d.check(b, "uncertainty").duplicate
    assert not d.check(b, "failure_mining").duplicate


def test_index_searches_whole_dataset_not_just_batch(tmp_path):
    """Index nạp lại từ đĩa vẫn thấy vector của hôm qua."""
    store = FaissStore(tmp_path / "idx.index", 4, "pca_v1")
    v = np.array([1, 0, 0, 0], dtype=np.float32)
    store.add(v, "hom_qua")
    store.save()

    store2 = FaissStore(tmp_path / "idx.index", 4, "pca_v1")
    sim, nid = store2.search(v)
    assert nid == "hom_qua"
    assert sim > 0.99
