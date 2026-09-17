"""Known-answer tests for the compute-capability -> precision matrix.

These use real, documented GPU compute capabilities. If a threshold in
`capability.py` is changed carelessly, these break — which is the point: the
compatibility matrix decides which experiments are even attempted, so a silent
error here would produce a suite that skips valid work or attempts impossible work.
"""

from __future__ import annotations

import pytest

from gpu_benchlab.hardware.capability import (
    Precision,
    architecture_name,
    precision_support,
    supported_precisions,
)


def support_map(major: int, minor: int) -> dict[Precision, tuple[bool, bool]]:
    """precision -> (supported, tensor_core)."""
    return {p.precision: (p.supported, p.tensor_core) for p in precision_support(major, minor)}


class TestArchitectureName:
    @pytest.mark.parametrize(
        ("major", "minor", "expected"),
        [
            (6, 1, "Pascal"),  # GTX 1080
            (7, 0, "Volta"),  # V100
            (7, 5, "Turing"),  # RTX 2080 / T4
            (8, 0, "Ampere"),  # A100
            (8, 6, "Ampere"),  # RTX 3090
            (8, 9, "Ada Lovelace"),  # RTX 4090 / L4
            (9, 0, "Hopper"),  # H100
            (10, 0, "Blackwell"),  # B200
            (12, 0, "Blackwell"),  # RTX 5090
        ],
    )
    def test_known_architectures(self, major: int, minor: int, expected: str) -> None:
        assert architecture_name(major, minor) == expected

    def test_unknown_minor_degrades_to_family(self) -> None:
        # A hypothetical future SM 9.7 part must still be describable, not crash.
        name = architecture_name(9, 7)
        assert "Hopper" in name
        assert "9.7" in name

    def test_entirely_unknown_capability_degrades(self) -> None:
        name = architecture_name(99, 3)
        assert "Unknown" in name
        assert "99.3" in name


class TestPrecisionSupport:
    def test_every_precision_is_reported(self) -> None:
        """Unsupported precisions must still appear, so we can explain *why*."""
        reported = {p.precision for p in precision_support(6, 1)}
        assert reported == set(Precision)

    def test_fp32_always_supported(self) -> None:
        for cc in [(3, 5), (6, 1), (7, 5), (8, 9), (12, 0)]:
            assert support_map(*cc)[Precision.FP32][0] is True

    def test_pascal_gtx1080_sm61(self) -> None:
        """Pascal: INT8 via DP4A but no tensor cores of any kind."""
        s = support_map(6, 1)
        assert s[Precision.INT8] == (True, False), "DP4A INT8 without tensor cores"
        assert s[Precision.FP16][1] is False, "Pascal has no FP16 tensor cores"
        assert s[Precision.TF32] == (False, False)
        assert s[Precision.BF16] == (False, False)
        assert s[Precision.FP8] == (False, False)

    def test_volta_v100_sm70(self) -> None:
        """Volta introduced FP16 tensor cores but not INT8 tensor cores."""
        s = support_map(7, 0)
        assert s[Precision.FP16] == (True, True)
        assert s[Precision.INT8] == (True, False), "INT8 tensor cores arrive at SM 7.5"
        assert s[Precision.BF16] == (False, False)

    def test_turing_t4_sm75(self) -> None:
        s = support_map(7, 5)
        assert s[Precision.FP16] == (True, True)
        assert s[Precision.INT8] == (True, True)
        assert s[Precision.INT4] == (True, True)
        assert s[Precision.BF16] == (False, False), "BF16 arrives with Ampere"
        assert s[Precision.FP8] == (False, False)

    def test_ampere_a100_sm80(self) -> None:
        s = support_map(8, 0)
        assert s[Precision.TF32] == (True, True)
        assert s[Precision.BF16] == (True, True)
        assert s[Precision.FP8] == (False, False), "FP8 needs Ada 8.9 or Hopper 9.0"

    def test_ada_rtx4090_sm89(self) -> None:
        s = support_map(8, 9)
        assert s[Precision.FP8] == (True, True)
        assert s[Precision.FP4] == (False, False)

    def test_hopper_h100_sm90(self) -> None:
        s = support_map(9, 0)
        assert s[Precision.FP8] == (True, True)
        assert s[Precision.BF16] == (True, True)
        assert s[Precision.FP4] == (False, False)

    def test_blackwell_sm100_has_fp4(self) -> None:
        s = support_map(10, 0)
        assert s[Precision.FP4] == (True, True)
        assert s[Precision.FP8] == (True, True)

    def test_fp8_boundary_is_exactly_8_9(self) -> None:
        """8.6 (RTX 3090) must not claim FP8; 8.9 (RTX 4090) must."""
        assert support_map(8, 6)[Precision.FP8][0] is False
        assert support_map(8, 9)[Precision.FP8][0] is True

    def test_notes_are_non_empty(self) -> None:
        """Every entry must explain itself; the CLI surfaces these verbatim."""
        for ps in precision_support(8, 9):
            assert ps.note.strip(), f"{ps.precision} has an empty note"

    def test_tensor_core_implies_supported(self) -> None:
        """A precision can never be tensor-core accelerated but unsupported."""
        for major, minor in [(6, 1), (7, 5), (8, 9), (9, 0), (12, 0)]:
            for ps in precision_support(major, minor):
                if ps.tensor_core:
                    assert ps.supported, f"SM {major}.{minor} {ps.precision}: TC without support"


class TestSupportedPrecisions:
    def test_returns_only_supported(self) -> None:
        result = supported_precisions(7, 5)
        assert Precision.FP16 in result
        assert Precision.INT8 in result
        assert Precision.BF16 not in result
        assert Precision.FP8 not in result

    def test_ordering_is_stable(self) -> None:
        assert supported_precisions(8, 9) == supported_precisions(8, 9)
