"""Draw the magnitude profile, and say in the figure how it was computed.

The point of the figure is not only the shape of the curve. It is where the
double precision curve and the certified one part company, and which of the two
horizontal levels the magnitude actually tends to. So both curves are drawn,
both levels are drawn, and the legend names the route each number came from.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from genmag_exact.genmag_profile import break_at_poles

#: Two series, two hues, assigned by identity and never cycled: the curve that
#: was chosen, and the double precision curve kept beside it for comparison.
#: Validated for colour vision deficiency separation against a light surface.
CHOSEN_CURVE_COLOR = "#2563eb"      # blue
DOUBLE_PRECISION_COLOR = "#ea580c"  # orange

#: Chrome. Guides and grid stay one step off the surface so the data is the only
#: loud thing; the two horizontal levels are told apart by dash pattern and by
#: the legend, not by colour, and no status hue is spent on them.
GUIDE_STRONG = "#6b7280"
GUIDE_FAINT = "#9ca3af"
GRID_COLOR = "#e5e7eb"
BLIND_COLOR = "#f3f4f6"
TEXT_COLOR = "#111827"

#: Plain words for the route names, for the line under the title. The figure
#: should say how its numbers were obtained without the reader having to know
#: the method strings the profile builder uses.
CURVE_METHOD_WORDS = {
    "symbolic": "algebraic (rational function)",
    "arbitrary_precision": "numerical, arbitrary precision",
    "mixed_precision": "numerical, double + arbitrary precision",
    "double_precision": "numerical, double precision",
}
LIMIT_METHOD_WORDS = {
    "zero_pattern": "algebraic (inverse of A(inf))",
    "symbolic": "algebraic (rational function)",
    "arbitrary_precision": "numerical, certified digits",
    "double_precision": "numerical, double precision",
}
POLE_METHOD_WORDS = {
    "symbolic": "algebraic (roots of the denominator)",
    "sign_change": "numerical (determinant sign scan)",
    "skipped": "not computed",
}


def _limit_label(result) -> str:
    """The legend text for the limit. The route behind it is in the subtitle."""
    if result.limit is not None:
        return f"limit = {result.limit} = {float(result.limit):.6g}"
    if result.limit_decimal is not None:
        return f"limit = {float(result.limit_decimal):.6g}"
    return "limit: not obtained"


def _method_subtitle(result) -> str:
    """One line naming the route behind each number in the figure.

    Purpose: the same figure can come from three quite different computations,
    and which one it was changes how much the plateau can be trusted. Saying so
    on the figure keeps that from living only in the notebook that made it.
    """
    parts = [
        f"curve: {CURVE_METHOD_WORDS.get(result.curve_method, result.curve_method)}",
        f"limit: {LIMIT_METHOD_WORDS.get(result.limit_method, result.limit_method)}",
        f"poles: {POLE_METHOD_WORDS.get(result.poles_method, result.poles_method)}",
    ]
    return "   |   ".join(parts)


def _double_precision_level(result) -> float:
    """Where the double precision curve settles, which is not the limit.

    Purpose: when the symbolic route ran, this is 1^T A(infinity)^+ 1 exactly.
    Otherwise it is read off the far end of the double precision curve, which
    lands on that same value because pinv truncates the vanishing directions.
    """
    if result.limit_at_infinity is not None:
        return float(result.limit_at_infinity)
    return float(result.curve["genmag_double"].iloc[-1])


def plot_magnitude_profile(result, title: str = "", ax=None, ylim=None,
                           trust_panel: bool = True, fontsize: int = 14):
    """Draw one magnitude profile from a MagnitudeProfileResult.

    Purpose: turn the result into the figure without recomputing anything. The
    profile builder has already chosen its routes and recorded them, so this
    function only reads.

    trust_panel adds a short panel underneath showing how many digits of the
    double precision curve survive the conditioning of A(t). It is what
    explains the step in the orange curve, so it is on by default. Passing ax
    turns it off, since a supplied axes has no room beneath it.

    Returns (main_axes, trust_axes or None).
    """
    curve = result.curve
    t_values = curve.index.to_numpy(dtype=float)
    poles = sorted(float(pole) for pole in result.poles)

    if ax is not None:
        main_axes, trust_axes, figure = ax, None, ax.figure
    elif trust_panel:
        figure, (main_axes, trust_axes) = plt.subplots(
            2, 1, figsize=(11.5, 8.0), sharex=True, layout="constrained",
            gridspec_kw={"height_ratios": [3.4, 1]})
    else:
        figure, main_axes = plt.subplots(figsize=(11.5, 6.4), layout="constrained")
        trust_axes = None

    # A rational function diverges at each pole, so the line must not be drawn
    # across one. break_at_poles inserts a nan at every pole it is given.
    t_broken, chosen = break_at_poles(t_values, curve["genmag"].tolist(), poles)
    _, double = break_at_poles(t_values, curve["genmag_double"].tolist(), poles)

    # The region the pole scan could not settle. Saying so is part of the
    # figure: no pole drawn there does not mean no pole there.
    if result.blind_t:
        main_axes.axvspan(min(result.blind_t), max(result.blind_t),
                          color=BLIND_COLOR, zorder=0,
                          label="the pole scan was blind here")

    for index, pole in enumerate(poles):
        main_axes.axvline(pole, color=GUIDE_FAINT, lw=0.8, ls=":", zorder=1,
                          label="pole" if index == 0 else None)

    limit_level = None if result.limit_decimal is None else float(result.limit_decimal)
    double_level = _double_precision_level(result)
    if limit_level is not None:
        main_axes.axhline(limit_level, color=GUIDE_STRONG, lw=1.2, ls="--", zorder=2,
                          label=_limit_label(result))
    if limit_level is None or abs(double_level - limit_level) > 1e-9:
        main_axes.axhline(double_level, color=GUIDE_FAINT, lw=1.2, ls=":", zorder=2,
                          label=f"where double precision lands = {double_level:.6g}")

    # The double precision curve goes underneath as a wash, so it reads as a
    # halo wherever the two agree and as a separate line wherever they do not.
    main_axes.plot(t_broken, double, lw=3.5, alpha=0.35, solid_capstyle="round",
                   color=DOUBLE_PRECISION_COLOR, zorder=3,
                   label="double precision (np.linalg.pinv)")
    main_axes.plot(t_broken, chosen, lw=2, solid_capstyle="round",
                   color=CHOSEN_CURVE_COLOR, zorder=4,
                   label=f"genmag ({result.curve_method})")

    if ylim is None:
        reference = limit_level if limit_level is not None else double_level
        ylim = (-0.12 * abs(reference), 1.75 * abs(reference))
    main_axes.set_ylim(*ylim)
    main_axes.set_xscale("log")
    main_axes.set_ylabel("genmag", fontsize=fontsize, color=TEXT_COLOR)

    # One direct label, on the level that matters. The two horizontal levels sit
    # close together and both curves end near one of them, so the label carries
    # an end marker on the chosen curve and a leader down to it: the value is
    # then attached to a mark whose colour says which curve it belongs to,
    # while the text itself stays in ink.
    if limit_level is not None:
        last_t = float(t_values[-1])
        last_value = float(curve["genmag"].iloc[-1])
        main_axes.plot([last_t], [last_value], marker="o", ms=9, zorder=5,
                       color=CHOSEN_CURVE_COLOR, mec="white", mew=2)
        main_axes.annotate(f"{limit_level:.4f}", xy=(last_t, last_value),
                           xytext=(-6, -34), textcoords="offset points",
                           ha="right", va="top", fontsize=fontsize,
                           color=TEXT_COLOR,
                           arrowprops=dict(arrowstyle="-", color=CHOSEN_CURVE_COLOR,
                                           lw=1.2, shrinkA=2, shrinkB=6))

    main_axes.set_title(title, fontsize=fontsize + 4, loc="left", color=TEXT_COLOR,
                        pad=26)
    main_axes.text(0.0, 1.012, _method_subtitle(result), transform=main_axes.transAxes,
                   fontsize=fontsize - 2, color=GUIDE_STRONG, va="bottom", ha="left")
    # The legend goes below the figure rather than into a corner: six entries of
    # this length collide with the curve wherever they are put inside the axes.
    if ax is None:
        figure.legend(*main_axes.get_legend_handles_labels(), frameon=False, ncols=2,
                      loc="outside lower center", fontsize=fontsize - 1,
                      labelcolor=TEXT_COLOR)
    else:
        main_axes.legend(frameon=False, loc="best", fontsize=fontsize - 1,
                         labelcolor=TEXT_COLOR)

    for axes in (main_axes, trust_axes):
        if axes is None:
            continue
        axes.grid(True, lw=0.6, color=GRID_COLOR, ls="-")
        axes.set_axisbelow(True)
        axes.tick_params(labelsize=fontsize - 1, colors=TEXT_COLOR)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(GRID_COLOR)

    if trust_axes is not None:
        trust_axes.plot(t_values, curve["usable_digits"].to_numpy(), lw=2,
                        color=DOUBLE_PRECISION_COLOR, solid_capstyle="round")
        trust_axes.set_xscale("log")
        trust_axes.set_ylabel("trusted\ndigits", fontsize=fontsize - 1, color=TEXT_COLOR)
        trust_axes.set_xlabel("t", fontsize=fontsize, color=TEXT_COLOR)
        trust_axes.set_ylim(bottom=0)
    else:
        main_axes.set_xlabel("t", fontsize=fontsize, color=TEXT_COLOR)

    return main_axes, trust_axes