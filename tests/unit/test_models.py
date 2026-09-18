"""Tests for the model registry and pinned-weights handling. No network, no torch."""

from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import pytest

from gpu_benchlab.core.errors import BackendError, ConfigurationError, UnavailableError
from gpu_benchlab.models.registry import (
    RANDOM_WEIGHTS,
    ModelSpec,
    WeightsSpec,
    get_model,
    list_models,
    register_model,
    unregister_model,
)
from gpu_benchlab.models.weights import ensure_weights, sha256_of

PAYLOAD = b"pretend these are weights"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


class TestResnet50Spec:
    def test_published_values(self) -> None:
        """Values from docs/models.md; a silent edit here would mislabel results."""
        spec = get_model("resnet50")
        assert spec.expected_parameters == 25_557_032
        assert spec.architecture == "ResNet"
        assert spec.input_shape == (3, 224, 224)
        assert spec.output_shape == (1000,)
        assert spec.default_weights == "IMAGENET1K_V2"

    def test_weights_urls_are_pinned_https_with_hash_prefix(self) -> None:
        spec = get_model("resnet50")
        for weights in spec.weights.values():
            assert weights.url.startswith("https://download.pytorch.org/models/")
            assert weights.sha256_prefix in weights.filename

    def test_default_resolves_to_explicit_pin_not_library_default(self) -> None:
        assert get_model("resnet50").resolve_weights(None) == "IMAGENET1K_V2"

    def test_random_is_always_valid(self) -> None:
        assert get_model("resnet50").resolve_weights("random") == RANDOM_WEIGHTS

    def test_unknown_weights_rejected(self) -> None:
        with pytest.raises(ConfigurationError, match="IMAGENET1K_V1"):
            get_model("resnet50").resolve_weights("DEFAULT")


class TestRegistry:
    def test_unknown_model_lists_alternatives(self) -> None:
        with pytest.raises(ConfigurationError, match="resnet50"):
            get_model("resnet51")

    def test_duplicate_registration_rejected(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_model(get_model("resnet50"))

    def test_test_only_models_are_hidden_from_listing(self) -> None:
        spec = ModelSpec(
            name="hidden-test-model",
            family="vision",
            architecture="X",
            build=lambda: None,
            input_shape=(1,),
            output_shape=(1,),
            expected_parameters=1,
            default_weights=RANDOM_WEIGHTS,
            test_only=True,
        )
        register_model(spec)
        try:
            assert spec not in list_models()
            assert spec in list_models(include_test_only=True)
        finally:
            unregister_model("hidden-test-model")

    def test_registry_import_does_not_need_torch(self) -> None:
        """Builders import lazily; the registry itself must work without torch."""
        assert callable(get_model("resnet50").build)


def spec_for(prefix: str, url: str = "https://example.invalid/w/weights-abc.pth") -> WeightsSpec:
    return WeightsSpec(url=url, sha256_prefix=prefix)


class TestEnsureWeights:
    def test_cached_file_is_verified_and_hashed(self, tmp_path: Path) -> None:
        spec = spec_for(PAYLOAD_SHA[:8])
        (tmp_path / spec.filename).write_bytes(PAYLOAD)
        resolved = ensure_weights(spec, cache_dir=tmp_path)
        assert resolved.sha256 == PAYLOAD_SHA
        assert resolved.downloaded is False

    def test_hash_mismatch_is_refused(self, tmp_path: Path) -> None:
        spec = spec_for("deadbeef")
        (tmp_path / spec.filename).write_bytes(PAYLOAD)
        with pytest.raises(BackendError, match="does not match"):
            ensure_weights(spec, cache_dir=tmp_path)

    def test_missing_with_downloads_disabled_is_unavailable(self, tmp_path: Path) -> None:
        with pytest.raises(UnavailableError, match="not cached"):
            ensure_weights(spec_for("abc"), cache_dir=tmp_path, allow_download=False)

    def test_non_https_url_refused(self, tmp_path: Path) -> None:
        with pytest.raises(BackendError, match="non-HTTPS"):
            ensure_weights(spec_for("abc", url="http://example.invalid/w.pth"), cache_dir=tmp_path)

    def test_download_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(PAYLOAD))
        spec = spec_for(PAYLOAD_SHA[:8])
        resolved = ensure_weights(spec, cache_dir=tmp_path)
        assert resolved.downloaded is True
        assert resolved.path.read_bytes() == PAYLOAD
        assert sha256_of(resolved.path) == PAYLOAD_SHA
        assert not list(tmp_path.glob("*.part")), "no partial file left behind"

    def test_corrupt_download_is_discarded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"junk"))
        spec = spec_for(PAYLOAD_SHA[:8])
        with pytest.raises(BackendError):
            ensure_weights(spec, cache_dir=tmp_path)
        assert not (tmp_path / spec.filename).exists()
        assert not list(tmp_path.glob("*.part"))

    def test_network_failure_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def offline(url: str, timeout: float) -> io.BytesIO:
            raise OSError("network unreachable")

        monkeypatch.setattr(urllib.request, "urlopen", offline)
        with pytest.raises(UnavailableError, match="Could not download"):
            ensure_weights(spec_for("abc"), cache_dir=tmp_path)
        assert not list(tmp_path.glob("*.part"))
