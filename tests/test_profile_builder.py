"""Tests for the layer that chooses a route and records what it chose.

The point of these is not the numbers -- those are tested where they are
computed -- but the decisions: that each of the three routes to the limit is
taken in the situation it exists for, that a forced route is honoured, and that
the result says which route produced it.
"""

from fractions import Fraction

import numpy as np
import pytest

from qstr_dataset.profile_builder import (
    magnitude_profile,
    magnitude_profiles,
)

#: No off-diagonal zero, so A(infinity) is nonsingular and the cheap route wins.
NO_ZEROS = np.array([[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]], dtype=float)

#: A(infinity) has rank 4 of 5, so the cheap route cannot apply and the limit is
#: 7/3, reachable either symbolically or from the plateau.
SINGULAR_AT_INFINITY = np.array([[0, 1, 0, 1, 2], [1, 0, 2, 0, 1], [0, 2, 0, 1, 0],
                                 [1, 0, 1, 0, 2], [2, 1, 0, 2, 0]], dtype=float)

SHORT_GRID = np.logspace(-1, 1.5, 24)


def build(matrix, **options):
    options.setdefault("t_grid", SHORT_GRID)
    options.setdefault("with_poles", False)
    return magnitude_profile(matrix, **options)


class TestLimitRoutes:
    def test_zero_pattern_is_taken_when_a_infinity_is_nonsingular(self):
        result = build(NO_ZEROS)
        assert result.limit_method == "zero_pattern"
        assert result.limit_is_proved
        assert result.limit == 4
        assert result.rank_at_infinity == result.size == 4

    def test_symbolic_is_taken_when_the_cheap_route_cannot_apply(self):
        result = build(SINGULAR_AT_INFINITY, symbolic_timeout_sec=60)
        assert result.limit_method == "symbolic"
        assert result.limit_is_proved
        assert result.limit == Fraction(7, 3)
        # The symbolic route also yields where double precision lands, which
        # nothing else provides.
        assert result.limit_at_infinity == 2

    def test_the_plateau_is_taken_when_the_symbolic_route_is_skipped(self):
        result = build(SINGULAR_AT_INFINITY, symbolic_timeout_sec=0)
        assert result.limit_method == "arbitrary_precision"
        assert not result.limit_is_proved
        assert result.limit == Fraction(7, 3)

    def test_the_symbolic_route_is_not_attempted_once_the_cheap_one_answered(self):
        result = build(NO_ZEROS, symbolic_timeout_sec=60)
        assert all("symbolic" not in note for note in result.notes)


class TestCurveRoutes:
    def test_the_symbolic_curve_is_used_when_the_symbolic_route_ran(self):
        result = build(SINGULAR_AT_INFINITY, symbolic_timeout_sec=60)
        assert result.curve_method == "symbolic"

    def test_double_precision_can_be_forced(self):
        result = build(NO_ZEROS, curve_method="double_precision")
        assert result.curve_method == "double_precision"
        assert result.curve["genmag"].tolist() == result.curve["genmag_double"].tolist()

    def test_arbitrary_precision_can_be_forced_and_differs_in_the_tail(self):
        grid = np.array([0.5, 5.0, 40.0])
        certified = magnitude_profile(SINGULAR_AT_INFINITY, grid, with_poles=False,
                                      curve_method="arbitrary_precision",
                                      symbolic_timeout_sec=0)
        assert certified.curve_method == "arbitrary_precision"
        # The certified curve tends to the limit; double precision does not.
        assert certified.curve["genmag"].iloc[-1] == pytest.approx(7 / 3, rel=1e-9)
        assert certified.curve["genmag_double"].iloc[-1] == pytest.approx(2.0, rel=1e-6)

    def test_mixed_precision_recomputes_only_the_untrusted_points(self):
        grid = np.array([0.5, 5.0, 40.0])
        result = magnitude_profile(SINGULAR_AT_INFINITY, grid, with_poles=False,
                                   curve_method="mixed_precision",
                                   symbolic_timeout_sec=0)
        recomputed = result.curve["precision_bits"] > 53
        assert recomputed.iloc[-1]          # the tail is not trustworthy in float64
        assert not recomputed.iloc[0]       # the head is


class TestTheReturnedFrame:
    def test_carries_the_columns_the_figure_needs(self):
        result = build(SINGULAR_AT_INFINITY, symbolic_timeout_sec=0)
        assert result.curve.index.name == "t"
        for column in ("genmag", "genmag_double", "usable_digits"):
            assert column in result.curve.columns
        assert len(result.curve) == len(SHORT_GRID)

    def test_usable_digits_falls_as_t_grows(self):
        result = build(SINGULAR_AT_INFINITY, symbolic_timeout_sec=0)
        usable = result.curve["usable_digits"]
        assert usable.iloc[0] > usable.iloc[-1]

    def test_poles_are_found_when_asked_for(self):
        result = magnitude_profile(SINGULAR_AT_INFINITY, SHORT_GRID,
                                   symbolic_timeout_sec=0, with_poles=True)
        assert result.poles_method == "sign_change"


class TestArgumentChecking:
    def test_an_unknown_curve_method_is_rejected(self):
        with pytest.raises(ValueError, match="curve_method"):
            build(NO_ZEROS, curve_method="wishful")

    def test_an_unknown_limit_method_is_rejected(self):
        with pytest.raises(ValueError, match="limit_method"):
            build(NO_ZEROS, limit_method="wishful")

    def test_a_nonzero_diagonal_is_rejected(self):
        matrix = NO_ZEROS.copy()
        matrix[2, 2] = 1.0
        with pytest.raises(ValueError, match="diagonal"):
            build(matrix)


class TestBatch:
    def test_results_survive_the_cache_unchanged(self, tmp_path):
        dissims = {"a": NO_ZEROS, "b": SINGULAR_AT_INFINITY}
        options = dict(t_grid=SHORT_GRID, with_poles=False, symbolic_timeout_sec=0)
        first, results = magnitude_profiles(dissims, "test", tmp_path,
                                            progress=False, **options)
        second, reloaded = magnitude_profiles(dissims, "test", tmp_path,
                                              progress=False, **options)
        assert list(first["limit"]) == list(second["limit"])
        assert np.allclose(results["b"].curve["genmag"], reloaded["b"].curve["genmag"])
        assert reloaded["b"].limit == Fraction(7, 3)
        assert reloaded["a"].limit_method == "zero_pattern"

    def test_a_changed_matrix_is_recomputed(self, tmp_path):
        options = dict(t_grid=SHORT_GRID, with_poles=False, symbolic_timeout_sec=0)
        magnitude_profiles({"a": NO_ZEROS}, "test", tmp_path, progress=False, **options)
        changed = NO_ZEROS.copy()
        changed[0, 1] = changed[1, 0] = 5.0
        summary, results = magnitude_profiles({"a": changed}, "test", tmp_path,
                                              progress=False, **options)
        assert summary.loc["a", "status"] == "ok"
        assert results["a"].limit == 4      # still four points, recomputed not reused

    def test_a_failing_matrix_costs_only_itself(self, tmp_path):
        broken = NO_ZEROS.copy()
        broken[1, 1] = 1.0
        dissims = {"good": NO_ZEROS, "bad": broken}
        options = dict(t_grid=SHORT_GRID, with_poles=False, symbolic_timeout_sec=0)
        summary, results = magnitude_profiles(dissims, "test", tmp_path,
                                              progress=False, **options)
        assert summary.loc["good", "status"] == "ok"
        assert summary.loc["bad", "status"] == "error"
        assert "diagonal" in summary.loc["bad", "error"]
        assert set(results) == {"good"}
