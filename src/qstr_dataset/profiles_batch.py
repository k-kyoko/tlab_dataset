"""Run the exact backend over many subjects at once, in parallel, with a cache.

One subject takes roughly a minute on a 23x23 matrix, so 120 of them is
around two hours in series. Splitting them across cores brings that down to
minutes. Every finished profile is written to disk as JSON, so an interrupted
run resumes instead of starting over.

Nothing sympy-shaped crosses a process boundary: GenmagProfile stores plain
Fractions, and the workers hand back the dictionary that to_dict produces.

The time limits are enforced by terminating a child process, not by SIGALRM.
With python-flint installed, sympy switches its ground types to flint and
spends its time inside C, where a Python signal handler cannot run: a 3 second
alarm failed to stop a 17 second computation even after 200 seconds. The
budget is therefore applied from outside, by profile_builder.run_with_timeout.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

import mpmath
import numpy as np
import pandas as pd
from genmag_exact.genmag_profile import GenmagProfile

from .profile_builder import (
    run_with_timeout,
    symbolic_profile_as_dictionary,
    symbolic_profile_poles,
)

#: Building one profile took a median of 20 s and a maximum of 65 s on the Amy
#: matrices, so five minutes is a generous ceiling. A subject that hits it is
#: cheap to lose when the work is spread over several cores.
DEFAULT_PROFILE_TIMEOUT_SEC: int = 300

#: Finding the poles is capped separately, so a slow root search does not throw
#: away a profile that already succeeded.
DEFAULT_POLE_TIMEOUT_SEC: int = 300

#: Digits kept for the pole locations. Evaluating the generalized magnitude at
#: a pole needs the location to this accuracy, so float would not do.
POLE_DPS: int = 50

#: Bump this whenever the maths in GenmagProfile changes. Cached records
#: stamped with an older version are ignored and recomputed, so there is no
#: need to move the cache folder aside by hand.
CACHE_VERSION: int = 2


def _matrix_fingerprint(matrix: np.ndarray) -> str:
    """A short hash of the matrix, so a changed input invalidates its cache."""
    contiguous = np.ascontiguousarray(matrix, dtype=float)
    return hashlib.sha1(contiguous.tobytes()).hexdigest()[:16]


def _profile_one(key: Any, matrix: np.ndarray, timeout_sec: int,
                 pole_timeout_sec: int, with_poles: bool) -> dict[str, Any]:
    """Build one profile under a time budget, returning JSON-safe data only.

    The profile and the poles get their own limits. Finding the poles is the
    cheaper half but the more variable one, and losing an expensive profile
    because its poles ran long would be a waste, so a pole timeout leaves the
    profile intact and records the poles as unavailable.
    """
    started = time.perf_counter()

    profile_dictionary, _, status = run_with_timeout(
        symbolic_profile_as_dictionary, (matrix.tolist(),), timeout_sec)
    if profile_dictionary is None:
        return {"subject": key, "status": status, "profile": None, "poles": None,
                "poles_status": "not attempted",
                "sec": time.perf_counter() - started,
                "error": None if status == "timeout" else status}

    poles = None
    poles_status = "skipped"
    if with_poles:
        found, _, pole_status = run_with_timeout(
            symbolic_profile_poles, (profile_dictionary,), pole_timeout_sec)
        if found is None:
            poles_status = pole_status
        else:
            # Stored as decimal strings: mpf is not JSON-serialisable, and
            # float would throw away digits an evaluation at a pole needs.
            poles = [mpmath.nstr(mpmath.mpf(pole), POLE_DPS) for pole in found]
            poles_status = "ok"

    return {"subject": key, "status": "ok", "profile": profile_dictionary,
            "poles": poles, "poles_status": poles_status,
            "sec": time.perf_counter() - started, "error": None}


def _cache_path(cache_dir: Path, name: str, key: Any) -> Path:
    return cache_dir / f"{name}_{key}.json"


def genmag_profiles(
    dissims: Mapping[Any, Any],
    name: str,
    cache_dir: Path,
    max_workers: int | None = None,
    timeout_sec: int = DEFAULT_PROFILE_TIMEOUT_SEC,
    pole_timeout_sec: int = DEFAULT_POLE_TIMEOUT_SEC,
    with_poles: bool = True,
    recompute: bool = False,
    progress: bool = True,
) -> tuple[pd.DataFrame, dict[Any, GenmagProfile], dict[Any, list | None]]:
    """Exact genmag profiles for every subject in dissims.

    Subjects already present in cache_dir are loaded instead of recomputed,
    unless recompute is True, so an interrupted run picks up where it stopped.

    dissims  : subject -> dissimilarity matrix (DataFrame or array)
    name     : prefix for the cache files, e.g. "amy"
    cache_dir: where the per-subject JSON files go

    Returns (summary, profiles, poles): one row per subject, and two
    dictionaries keyed by subject holding the results that finished.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: list[dict[str, Any]] = []
    todo: list[tuple[Any, np.ndarray]] = []
    fingerprints: dict[Any, str] = {}
    stale = 0
    for key, dissim in dissims.items():
        matrix = np.asarray(dissim, dtype=float)
        fingerprints[key] = _matrix_fingerprint(matrix)
        path = _cache_path(cache_dir, name, key)
        if path.exists() and not recompute:
            record = json.loads(path.read_text())
            # A cached record is reused only when this code version produced it
            # from this exact matrix.
            if (record.get("cache_version") == CACHE_VERSION
                    and record.get("fingerprint") == fingerprints[key]):
                cached.append(record)
                continue
            stale += 1
        todo.append((key, matrix))

    if progress:
        note = f" ({stale} stale)" if stale else ""
        print(f"{len(cached)} cached, {len(todo)} to compute{note}")

    results = list(cached)
    started = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(
            max_workers=max_workers, mp_context=mp.get_context("fork")
        ) as pool:
            futures = {
                pool.submit(_profile_one, key, matrix, timeout_sec,
                            pole_timeout_sec, with_poles): key
                for key, matrix in todo
            }
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                result["cache_version"] = CACHE_VERSION
                result["fingerprint"] = fingerprints[result["subject"]]
                results.append(result)
                _cache_path(cache_dir, name, result["subject"]).write_text(
                    json.dumps(result))
                if progress:
                    print(f"\r[{done}/{len(todo)}] subject={result['subject']} "
                          f"{result['status']} ({result['sec']:.1f}s)".ljust(70),
                          end="", flush=True)
        if progress:
            print(f"\rdone: {len(todo)} computed in "
                  f"{time.perf_counter() - started:.1f}s".ljust(70))

    profiles: dict[Any, GenmagProfile] = {}
    poles_by_subject: dict[Any, list[float] | None] = {}
    rows = []
    for result in results:
        row = {"subject": result["subject"], "status": result["status"],
               "sec": result["sec"], "error": result["error"]}
        if result["status"] == "ok":
            profile = GenmagProfile.from_dict(result["profile"])
            profiles[result["subject"]] = profile
            poles_by_subject[result["subject"]] = (
                None if result["poles"] is None
                else [mpmath.mpf(text) for text in result["poles"]])
            row.update({
                "limit": str(profile.limit),
                "limit_float": (float(profile.limit)
                                if profile.limit is not None else None),
                "m_at_infinity": str(profile.m_at_infinity),
                "m_at_infinity_float": float(profile.m_at_infinity),
                "generic_rank": profile.generic_rank,
                "rank_at_infinity": profile.rank_at_infinity,
                "method": profile.method,
                "has_weighting_at_infinity": profile.has_weighting_at_infinity,
                "poles_status": result.get("poles_status"),
                "n_poles": (None if result["poles"] is None
                            else len(result["poles"])),
                "largest_pole_t": (None if not result["poles"] else
                                   float(max(mpmath.mpf(t)
                                             for t in result["poles"]))),
                "degree_den": len(profile.den) - 1,
            })
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("subject").sort_index()
    return summary, profiles, poles_by_subject