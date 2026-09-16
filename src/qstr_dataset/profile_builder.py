"""One entry point for the magnitude profile, choosing its own route.

The library side keeps one function per method and no branching, so the methods
stay comparable against each other. This is the layer allowed to decide, and
every decision it takes is recorded in the result: which route produced the
curve, which produced the limit, whether that limit is proved or identified
from digits, and what was tried and abandoned on the way.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from genmag_exact import arbitrary_precision, limit_shortcut
from genmag_exact.genmag_profile import GenmagProfile
from qstr_diversity import core

from . import paths

CURVE_METHODS = ("symbolic", "arbitrary_precision", "mixed_precision", "double_precision")
LIMIT_METHODS = ("zero_pattern", "symbolic", "arbitrary_precision", "double_precision")


@dataclass(frozen=True)
class MagnitudeProfileResult:
    """Everything the figure needs, plus a record of how it was obtained."""

    curve: pd.DataFrame          # index t; the chosen curve beside the double precision one
    curve_method: str
    limit: Fraction | None       # the t -> infinity value, when a rational was obtained
    limit_decimal: str | None    # its decimal, valid even when no rational was identified
    limit_method: str
    limit_is_proved: bool        # True for zero_pattern and symbolic, False for digits
    limit_at_infinity: Fraction | None   # 1^T A(inf)^+ 1, where double precision lands
    poles: list[float]
    poles_method: str
    blind_t: list[float]         # t where the pole scan could not settle the sign
    rank_at_infinity: int
    size: int
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# running the symbolic route under a budget
# --------------------------------------------------------------------------

def symbolic_profile_as_dictionary(matrix_as_lists):
    """The symbolic profile, as the plain dictionary that crosses a process pipe.

    Public because profiles_batch runs the same computation under the same
    budget: a process target has to be importable by name, and there should be
    one definition of what "build the symbolic profile" means.
    """
    return GenmagProfile.from_distance_matrix(
        np.asarray(matrix_as_lists, dtype=float)).to_dict()


def symbolic_profile_poles(profile_dictionary):
    """Exact pole locations from an already built profile, as plain floats."""
    return [float(pole) for pole in GenmagProfile.from_dict(profile_dictionary).poles()]


def _worker(queue, function, arguments):
    try:
        queue.put(("ok", function(*arguments)))
    except BaseException as exc:                      # noqa: BLE001 - reported, not raised
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_with_timeout(function, arguments, seconds: int):
    """Run function(*arguments) in a child process, killing it if it overruns.

    Purpose: whether the symbolic route finishes cannot be predicted from the
    size of the matrix -- 26 points took 42 s and 28 points did not finish in
    35 minutes -- so the honest test is to try it under a budget.

    A child process rather than signal.alarm, which profiles_batch uses and
    which no longer works here: with python-flint installed, sympy switches its
    ground types to flint and spends its time inside C, where a Python signal
    handler cannot run. A 3 second alarm failed to stop a 17 second computation
    even after 200 seconds. Terminating a process is not subject to that.

    Returns (value, seconds_spent, status) with status one of ok, timeout, error.
    """
    context = multiprocessing.get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_worker, args=(queue, function, arguments))
    started = time.perf_counter()
    process.start()
    process.join(seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return None, time.perf_counter() - started, "timeout"
    status, payload = queue.get()
    return (payload if status == "ok" else None), time.perf_counter() - started, status


# --------------------------------------------------------------------------
# the double precision side
# --------------------------------------------------------------------------

def default_profile_t_grid(points: int = 260) -> np.ndarray:
    """A log grid over the range where the magnitude actually moves.

    Purpose: a separate grid from backends.default_t_grid, which runs out to
    t = 1e8. Double precision reaches that cheaply, by settling on the wrong
    plateau; ball arithmetic would need roughly t/ln(10) digits to reach it
    honestly, which is tens of millions. Nothing happens out there anyway --
    the curve is flat past t = 100 -- so the certified routes stop at 200.
    """
    return np.logspace(-2, 2.3, points)


def _as_matrix(dissim) -> np.ndarray:
    if isinstance(dissim, pd.DataFrame):
        return dissim.to_numpy(dtype=float)
    return np.asarray(dissim, dtype=float)


def _double_precision_columns(matrix: np.ndarray, t_values) -> dict[str, list[float]]:
    """The double precision curve and, per point, how far it can be trusted.

    Purpose: this is the curve the numeric backend has always produced, kept
    beside the chosen one so the figure can show where the two part.
    usable_digits is 16 minus the digits the conditioning of A(t) destroys, and
    it is what decides which points mixed_precision recomputes. On subject 004
    it drops below 8 at the same t where the double precision curve steps onto
    the wrong plateau, which is the evidence that it measures the right thing.
    """
    labels = range(len(matrix))
    genmag, usable = [], []
    for t in t_values:
        similarity = pd.DataFrame(np.exp(-float(t) * matrix), index=labels, columns=labels)
        genmag.append(float(core.cal_sim2genmag(similarity)))
        usable.append(core.cal_sim2usable_digits(similarity))
    return {"genmag_double": genmag, "usable_digits": usable}


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def magnitude_profile(
    dissim,
    t_grid: Sequence[float] | None = None,
    *,
    curve_method: str = "auto",
    limit_method: str = "auto",
    symbolic_timeout_sec: int = 60,
    pole_timeout_sec: int = 60,
    curve_budget_sec: float = 120.0,
    curve_digits: int = 15,
    limit_digits: int = 30,
    trust_digits: float = 8.0,
    with_poles: bool = True,
    progress: bool = False,
) -> MagnitudeProfileResult:
    """The magnitude profile of one dissimilarity matrix, by whichever route fits.

    Purpose: the single call the notebooks make. It picks a route, records it,
    and returns a frame the plotting side can draw without knowing any of this.

    The limit is tried in this order, which is not the order LIMIT_METHODS
    lists because the cheapest route is also one of the rounding-free ones:

      zero_pattern         milliseconds, whenever A(infinity) is nonsingular
      symbolic             under symbolic_timeout_sec; set that to 0 to skip it
      arbitrary_precision  the plateau, certified to limit_digits

    The curve descends CURVE_METHODS in order, most rigorous first:

      symbolic             the rational function, when the symbolic route ran
      arbitrary_precision  every point certified, when it fits curve_budget_sec
      mixed_precision      double precision, recomputing only untrusted points
      double_precision     what the numeric backend has always done

    The symbolic route is attempted only when zero_pattern did not settle the
    limit, since otherwise it costs tens of seconds to learn nothing new. On a
    matrix where it will not finish, the attempt still costs its full budget
    every call: pass symbolic_timeout_sec=0 once that is known.

    Pass curve_method or limit_method explicitly to force a route, which is
    what comparing the backends against each other needs.
    """
    if curve_method not in CURVE_METHODS + ("auto",):
        raise ValueError(f"curve_method must be 'auto' or one of {CURVE_METHODS}")
    if limit_method not in LIMIT_METHODS + ("auto",):
        raise ValueError(f"limit_method must be 'auto' or one of {LIMIT_METHODS}")

    matrix = _as_matrix(dissim)
    as_lists = matrix.tolist()
    limit_shortcut.validate_distance_matrix(as_lists)
    size = len(as_lists)
    t_values = np.asarray(default_profile_t_grid() if t_grid is None else t_grid, dtype=float)
    notes: list[str] = []

    # ---- the limit -------------------------------------------------------
    zero_pattern = limit_shortcut.limit_from_zero_pattern(as_lists, validate=False)
    rank_at_infinity = zero_pattern.rank_at_infinity
    limit: Fraction | None = None
    limit_decimal: str | None = None
    limit_is_proved = False
    limit_at_infinity: Fraction | None = None
    chosen_limit_method = limit_method
    profile: GenmagProfile | None = None

    if limit_method in ("auto", "zero_pattern"):
        if zero_pattern.is_applicable:
            limit, limit_is_proved = zero_pattern.limit, True
            limit_decimal, chosen_limit_method = f"{float(limit):.15g}", "zero_pattern"
            notes.append(f"A(infinity) is nonsingular (rank {rank_at_infinity}/{size}), "
                         "so the limit needed no approximation")
        else:
            notes.append(f"zero_pattern does not apply: "
                         f"rank A(infinity) = {rank_at_infinity}/{size}")

    if limit is None and limit_method in ("auto", "symbolic") and symbolic_timeout_sec > 0:
        if progress:
            print(f"trying the symbolic route, up to {symbolic_timeout_sec}s ...")
        built, spent, status = run_with_timeout(
            symbolic_profile_as_dictionary, (as_lists,), symbolic_timeout_sec)
        if built is not None:
            profile = GenmagProfile.from_dict(built)
            limit, limit_is_proved = profile.limit, True
            limit_decimal = None if limit is None else f"{float(limit):.15g}"
            limit_at_infinity = profile.m_at_infinity
            chosen_limit_method = "symbolic"
            notes.append(f"the symbolic route finished in {spent:.1f}s")
        else:
            notes.append(f"the symbolic route ended in {status} after {spent:.0f}s")

    if limit is None and limit_method in ("auto", "arbitrary_precision"):
        plateau = arbitrary_precision.limit_from_plateau(
            as_lists, wanted_digits=limit_digits)
        limit, limit_decimal = plateau.limit, plateau.decimal
        limit_is_proved, chosen_limit_method = False, "arbitrary_precision"
        notes.append(
            f"the limit was read off the plateau at t = {plateau.t_confirm:.0f}, "
            f"{plateau.agreeing_digits:.0f} digits confirmed, denominators up to "
            f"{plateau.denominator_bound:.1e} distinguishable")

    # ---- the curve -------------------------------------------------------
    columns: dict[str, Any] = _double_precision_columns(matrix, t_values)

    if limit_method == "double_precision":
        # The plateau the double precision curve settles on, for comparison. It
        # is not the limit: on subject 004 it is 36.47 against the true 32.43.
        limit, limit_is_proved = None, False
        limit_decimal = f"{columns['genmag_double'][-1]:.10f}"

    chosen_curve_method = curve_method
    if curve_method == "auto":
        if profile is not None:
            chosen_curve_method = "symbolic"
        else:
            # Time one real evaluation at the largest t, where the precision
            # demand is highest, and scale it by the number of points. That
            # over-estimates, which errs towards the cheaper route.
            started = time.perf_counter()
            arbitrary_precision.genmag_at(as_lists, float(t_values.max()),
                                          wanted_digits=curve_digits, validate=False)
            estimate = (time.perf_counter() - started) * len(t_values)
            chosen_curve_method = ("arbitrary_precision"
                                   if estimate <= curve_budget_sec else "mixed_precision")
            notes.append(f"a fully certified curve was estimated at {estimate:.0f}s "
                         f"against a {curve_budget_sec:.0f}s budget")

    if chosen_curve_method == "symbolic":
        if profile is None:
            built, spent, status = run_with_timeout(
                symbolic_profile_as_dictionary, (as_lists,), symbolic_timeout_sec)
            if built is None:
                raise TimeoutError(
                    f"the symbolic route ended in {status} after {spent:.0f}s; "
                    "pass curve_method='arbitrary_precision' instead")
            profile = GenmagProfile.from_dict(built)
        values, _, _ = profile.genmag_curve(t_values)
        columns["genmag"] = values
    elif chosen_curve_method in ("arbitrary_precision", "mixed_precision"):
        if chosen_curve_method == "arbitrary_precision":
            needs_precision = np.ones(len(t_values), dtype=bool)
        else:
            needs_precision = np.asarray(columns["usable_digits"]) < trust_digits
            notes.append(f"{int(needs_precision.sum())} of {len(t_values)} points "
                         "were recomputed in ball arithmetic")
        certified: dict[int, arbitrary_precision.CertifiedValue] = {}
        precision_bits = arbitrary_precision.DEFAULT_STARTING_PRECISION_BITS
        indices = np.flatnonzero(needs_precision)
        for done, index in enumerate(indices, start=1):
            value = arbitrary_precision.genmag_at(
                as_lists, float(t_values[index]), wanted_digits=curve_digits,
                starting_precision_bits=precision_bits, validate=False)
            certified[int(index)] = value
            precision_bits = max(arbitrary_precision.DEFAULT_STARTING_PRECISION_BITS,
                                 value.precision_bits // 2)
            if progress:
                print(f"\r[{done}/{len(indices)}] t={t_values[index]:.4g}".ljust(50),
                      end="", flush=True)
        if progress and len(indices):
            print()
        columns["genmag"] = [certified[i].value if i in certified
                             else columns["genmag_double"][i]
                             for i in range(len(t_values))]
        columns["radius"] = [certified[i].radius if i in certified else float("nan")
                             for i in range(len(t_values))]
        columns["accurate_digits"] = [certified[i].accurate_digits if i in certified
                                      else float("nan") for i in range(len(t_values))]
        columns["precision_bits"] = [certified[i].precision_bits if i in certified else 53
                                     for i in range(len(t_values))]
        columns["is_resolved"] = [certified[i].is_resolved if i in certified else True
                                  for i in range(len(t_values))]
    else:
        columns["genmag"] = list(columns["genmag_double"])

    curve = pd.DataFrame(columns, index=pd.Index(t_values, name="t"))

    # ---- the poles -------------------------------------------------------
    poles: list[float] = []
    blind_t: list[float] = []
    poles_method = "skipped"
    if with_poles:
        if profile is not None:
            found, spent, status = run_with_timeout(
                symbolic_profile_poles, (profile.to_dict(),), pole_timeout_sec)
            if found is not None:
                poles, poles_method = found, "symbolic"
            else:
                notes.append(f"exact root isolation ended in {status} after {spent:.0f}s")
        if poles_method == "skipped":
            poles, blind_t = arbitrary_precision.poles_by_sign_change(
                as_lists, t_values, progress=progress)
            poles_method = "sign_change"
            if blind_t:
                notes.append(f"the pole scan could not settle the sign at "
                             f"{len(blind_t)} points, from t = {min(blind_t):.3g}")

    return MagnitudeProfileResult(
        curve=curve, curve_method=chosen_curve_method,
        limit=limit, limit_decimal=limit_decimal, limit_method=chosen_limit_method,
        limit_is_proved=limit_is_proved, limit_at_infinity=limit_at_infinity,
        poles=poles, poles_method=poles_method, blind_t=blind_t,
        rank_at_infinity=rank_at_infinity, size=size, notes=notes,
    )

# --------------------------------------------------------------------------
# many matrices at once, with a cache on disk
# --------------------------------------------------------------------------

#: Bump this whenever the meaning of a stored field changes. A record stamped
#: with an older version is ignored and recomputed, so there is no cache folder
#: to move aside by hand.
CACHE_VERSION: int = 1


def _matrix_fingerprint(matrix: np.ndarray) -> str:
    """A short hash of the matrix, so a changed input invalidates its cache."""
    return hashlib.sha1(
        np.ascontiguousarray(matrix, dtype=float).tobytes()).hexdigest()[:16]


def _fraction_to_pair(value: Fraction | None):
    return None if value is None else [value.numerator, value.denominator]


def _pair_to_fraction(pair):
    return None if pair is None else Fraction(pair[0], pair[1])


def _result_to_dictionary(result: MagnitudeProfileResult) -> dict:
    """A JSON-safe form of the result, so a finished profile survives the session.

    Purpose: building one profile takes tens of seconds, and the figures get
    redrawn far more often than the numbers change. Fractions are stored as
    numerator/denominator pairs and the curve column by column, so nothing
    depends on a pickle protocol or on the classes keeping their current shape.
    """
    return {
        "curve": {"t": [float(t) for t in result.curve.index],
                  "columns": {name: result.curve[name].tolist()
                              for name in result.curve.columns}},
        "curve_method": result.curve_method,
        "limit": _fraction_to_pair(result.limit),
        "limit_decimal": result.limit_decimal,
        "limit_method": result.limit_method,
        "limit_is_proved": result.limit_is_proved,
        "limit_at_infinity": _fraction_to_pair(result.limit_at_infinity),
        "poles": list(result.poles),
        "poles_method": result.poles_method,
        "blind_t": list(result.blind_t),
        "rank_at_infinity": result.rank_at_infinity,
        "size": result.size,
        "notes": list(result.notes),
    }


def _result_from_dictionary(data: dict) -> MagnitudeProfileResult:
    """Rebuild a result stored by _result_to_dictionary."""
    stored = data["curve"]
    curve = pd.DataFrame(stored["columns"],
                         index=pd.Index(stored["t"], name="t", dtype=float))
    return MagnitudeProfileResult(
        curve=curve,
        curve_method=data["curve_method"],
        limit=_pair_to_fraction(data["limit"]),
        limit_decimal=data["limit_decimal"],
        limit_method=data["limit_method"],
        limit_is_proved=data["limit_is_proved"],
        limit_at_infinity=_pair_to_fraction(data["limit_at_infinity"]),
        poles=list(data["poles"]),
        poles_method=data["poles_method"],
        blind_t=list(data["blind_t"]),
        rank_at_infinity=data["rank_at_infinity"],
        size=data["size"],
        notes=list(data["notes"]),
    )


def _profile_one(key, matrix_as_lists, options: dict) -> dict:
    """Worker: one subject, returning JSON-safe data and never raising.

    A matrix that fails should cost that subject and nothing else, so the
    exception is reported as a field rather than propagated out of the batch.
    """
    started = time.perf_counter()
    try:
        result = magnitude_profile(np.asarray(matrix_as_lists, dtype=float), **options)
        return {"subject": key, "status": "ok", "error": None,
                "sec": time.perf_counter() - started,
                "result": _result_to_dictionary(result)}
    except BaseException as exc:                  # noqa: BLE001 - reported, not raised
        return {"subject": key, "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "sec": time.perf_counter() - started, "result": None}


def magnitude_profiles(
    dissims: Mapping[Any, Any],
    name: str,
    cache_dir: Path | str | None = None,
    *,
    recompute: bool = False,
    max_workers: int | None = None,
    progress: bool = True,
    **profile_options,
) -> tuple[pd.DataFrame, dict[Any, MagnitudeProfileResult]]:
    """Build a magnitude profile for every matrix in dissims, with a cache.

    Purpose: the whole-dataset version of magnitude_profile. One 93x93 subject
    takes around half a minute, so an eleven subject run is minutes rather than
    seconds; every finished subject is written to disk, and a second run reuses
    it. A record is reused only when this code version produced it from this
    exact matrix, so editing a matrix recomputes just that subject.

    dissims  : subject -> dissimilarity matrix (DataFrame or array)
    name     : prefix for the cache files, e.g. "togashi"
    cache_dir: defaults to data/interim/magnitude_profiles

    Any further keyword argument is passed straight to magnitude_profile, so
    the batch shares its ladder and its budgets. On a dataset where the
    symbolic route cannot finish, pass symbolic_timeout_sec=0 and save that
    budget on every subject.

    Returns (summary, results): one row per subject, and the results that
    finished, keyed by subject.
    """
    cache_dir = Path(paths.interim("magnitude_profiles")
                     if cache_dir is None else cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fingerprints: dict[Any, str] = {}
    cached: list[dict] = []
    todo: list[tuple[Any, list]] = []
    stale = 0
    for key, dissim in dissims.items():
        matrix = _as_matrix(dissim)
        fingerprints[key] = _matrix_fingerprint(matrix)
        path = cache_dir / f"{name}_{key}.json"
        if path.exists() and not recompute:
            record = json.loads(path.read_text())
            if (record.get("cache_version") == CACHE_VERSION
                    and record.get("fingerprint") == fingerprints[key]):
                cached.append(record)
                continue
            stale += 1
        todo.append((key, matrix.tolist()))

    if progress:
        print(f"{len(cached)} cached, {len(todo)} to compute"
              f"{f' ({stale} stale)' if stale else ''}")

    records = list(cached)
    started = time.perf_counter()

    def keep(record: dict) -> None:
        record["cache_version"] = CACHE_VERSION
        record["fingerprint"] = fingerprints[record["subject"]]
        (cache_dir / f"{name}_{record['subject']}.json").write_text(json.dumps(record))
        records.append(record)

    if todo and max_workers in (None, 1):
        for done, (key, matrix_as_lists) in enumerate(todo, start=1):
            record = _profile_one(key, matrix_as_lists, profile_options)
            keep(record)
            if progress:
                print(f"[{done}/{len(todo)}] subject={key} {record['status']} "
                      f"({record['sec']:.1f}s)")
    elif todo:
        context = multiprocessing.get_context("fork")
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
            futures = {pool.submit(_profile_one, key, matrix_as_lists, profile_options): key
                       for key, matrix_as_lists in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                keep(record)
                if progress:
                    print(f"[{done}/{len(todo)}] subject={record['subject']} "
                          f"{record['status']} ({record['sec']:.1f}s)")
    if todo and progress:
        print(f"computed {len(todo)} in {time.perf_counter() - started:.1f}s")

    results: dict[Any, MagnitudeProfileResult] = {}
    rows = []
    for record in records:
        row = {"subject": record["subject"], "status": record["status"],
               "sec": record["sec"], "error": record["error"]}
        if record["status"] == "ok":
            result = _result_from_dictionary(record["result"])
            results[record["subject"]] = result
            row.update({
                "size": result.size,
                "rank_at_infinity": result.rank_at_infinity,
                "limit": str(result.limit),
                "limit_float": (None if result.limit is None else float(result.limit)),
                "limit_decimal": result.limit_decimal,
                "limit_method": result.limit_method,
                "limit_is_proved": result.limit_is_proved,
                "curve_method": result.curve_method,
                "poles_method": result.poles_method,
                "n_poles": len(result.poles),
                "n_blind": len(result.blind_t),
            })
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("subject").sort_index()
    return summary, results
