#!/usr/bin/env python3
"""
map_style.py -- shared "GRASS documentation style" framing for map panels.

One function, style_map_panel(), that every map figure in this project should
call after plotting its own content (imshow layers, overlays, transects,
whatever) onto a matplotlib Axes. It adds the framing GRASS's own
documentation figures use:

  - a coordinate graticule (easting/northing grid + edge tick labels on all
    four sides), like d.grid
  - a north arrow + scale bar, grouped together, bottom-right, INSIDE the
    map frame
  - a separate horizontal colourbar strip below the map (only if you pass a
    mappable + a cax to draw it into), with a bracketed label
    ("Elevation change [m]") and 4-5 ticks
  - a thin border box around the map, opaque white background

WHY MATPLOTLIB, NOT grass.jupyter: every figure in this project composes a
map panel alongside non-map panels (elevation profiles, insets) into one
vector PDF with embedded Type 42 fonts at an exact mm size for the journal.
grass.jupyter's Map class renders GRASS displays to raster/HTML; it isn't
built to sit inside a multi-panel matplotlib figure with that level of
layout and font control, so the map content is still produced by reading
GRASS-exported GeoTIFFs, and this module only handles the shared framing.

WHY A DEDICATED COLOURBAR AXES (cax), NOT A CONJURED ONE: colourbar position
has to be exact and reproducible at a fixed mm size, not derived after the
fact from wherever matplotlib's layout engine happened to put the map axes.
Reserve the colourbar's row explicitly in the calling script's own GridSpec
(a thin row directly below the map row) and pass that Axes in as `cax`. This
keeps map_style.py layout-agnostic -- it works the same whether the map
panel is the whole figure or one of several.

Usage:
    from map_style import style_map_panel

    fig = plt.figure(figsize=(...))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.05], hspace=0.35)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[1, 0])
    ... plot your map content onto ax_map ...
    im = ax_map.imshow(dz, cmap=my_cmap, norm=my_norm, extent=extent)
    style_map_panel(ax_map, extent, cax=ax_cbar, mappable=im,
                    cbar_label="Elevation change", cbar_units="m")
"""

import numpy as np
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ==========================================================================
def _nice_step(span, target_n=5):
    """Pick a 'nice' graticule step (1/2/5 x 10^n) giving ~target_n lines
    across the given span."""
    if span <= 0:
        return 1.0
    raw = span / target_n
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if m * mag >= raw:
            return float(m * mag)
    return float(10 * mag)


# ==========================================================================
def add_graticule(ax, extent, step=None, line_color="0.35", line_width=0.35,
                  line_alpha=0.5, label_fmt="{:,.0f}", fontsize=6,
                  zorder=15):
    """Coordinate grid + edge tick labels on all four sides (GRASS d.grid
    style). `extent` is (left, right, bottom, top) in map units (metres)."""
    left, right, bottom, top = extent
    if step is None:
        step = _nice_step(min(right - left, top - bottom))

    x0 = np.ceil(left / step) * step
    xs = np.arange(x0, right, step)
    y0 = np.ceil(bottom / step) * step
    ys = np.arange(y0, top, step)

    for x in xs:
        ax.axvline(x, color=line_color, linewidth=line_width, alpha=line_alpha,
                  zorder=zorder)
    for y in ys:
        ax.axhline(y, color=line_color, linewidth=line_width, alpha=line_alpha,
                  zorder=zorder)

    # Labels on bottom+left only (GRASS d.grid default) -- mirroring them
    # onto the top/right edges too just collides at the corners once the
    # panel is narrow. Tick MARKS still appear on all four sides.
    ax.set_xticks(xs)
    ax.set_yticks(ys)
    ax.set_xticklabels([label_fmt.format(x) for x in xs], fontsize=fontsize,
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels([label_fmt.format(y) for y in ys], fontsize=fontsize,
                       rotation=90, va="center")
    ax.tick_params(axis="both", direction="out", length=2.5, width=0.5,
                  color="black", labelsize=fontsize, top=True, right=True,
                  labeltop=False, labelright=False, pad=2)


# ==========================================================================
def add_north_scale(ax, extent, avoid_bbox=None, scale_m=None,
                    corner="lower right", margin_frac=0.05, fontsize=6.5,
                    zorder=20, color="black", draw_scale=True, draw_north=True,
                    lw_scale=1.0, halo=None):
    """North arrow + scale bar, grouped together, inside the map frame --
    matches fig_transects_gdal_style.png (v3): plain line/arrow/text (BLACK
    by default), no stroke and no background box, positioned with real
    breathing room (a fraction of the whole map's size) rather than crammed
    into the very corner. Pass `color` to override for a panel whose
    underlying imagery makes black hard to read (e.g. white against a
    dark-canopy corner) -- the default stays black everywhere else.

    Two things were tried and reverted here: a white background box (wide
    enough to guarantee legibility, it covered real map content -- the B-B'
    label, basin interior); and white-with-dark-stroke glyphs sized/placed
    to fit a tight ~20 m corner strip outside the basin's bounding box
    (correct in principle, but the tight sizing this forced made the whole
    group read as small and congested against the frame edge). Since the
    exterior is light in this styling (see SCRIM_COLOR in
    make_transect_figure_v2.py), plain black reads clearly on its own, the
    same way it does in the reference figure -- no stroke or box needed.

    `avoid_bbox` is accepted for call-site compatibility but unused now;
    the fraction-based margin below is what the reference figure actually
    uses, and this project's basin shape leaves that corner clear anyway.

    `corner` is currently only implemented for "lower right" (what the GRASS
    doc-figure style uses and what every figure in this paper needs) but is
    kept as a parameter rather than hardcoded so a different corner is a
    one-line change here, not a rewrite at every call site.

    `draw_scale` / `draw_north` let a multi-panel figure place ONE scale bar
    and ONE north arrow instead of repeating them in every panel. Both
    default to True, so existing single-panel callers are unaffected. Use
    them only where the panels genuinely share a scale -- a scale bar that
    is right for one panel and wrong for its neighbour is worse than a
    repeated one.
    """
    if corner != "lower right":
        raise NotImplementedError('only corner="lower right" is implemented')

    # `halo`: outline every glyph and stroke in this colour. The default of
    # None keeps the original plain-black look, which relies on the map's
    # exterior being light. Pass e.g. "white" where the group has to sit on
    # imagery -- a clipped-to-basin panel puts it half on white margin and
    # half on dark terrain, where plain black is unreadable.
    fx = ([pe.withStroke(linewidth=2.6 * lw_scale, foreground=halo)]
          if halo else None)

    left, right, bottom, top = extent
    w, h = right - left, top - bottom
    if scale_m is None:
        scale_m = _nice_step(0.22 * w, target_n=1)

    mx, my = margin_frac * w, margin_frac * h
    x_scale_right = right - mx
    x_scale_left = x_scale_right - scale_m
    y_scale = bottom + my

    if draw_scale:
        ax.plot([x_scale_left, x_scale_right], [y_scale, y_scale], color=color,
                linewidth=1.8 * lw_scale, solid_capstyle="butt", zorder=zorder,
                path_effects=fx)
        # tick caps at each end, GRASS-style scale bar
        tick_h = 0.012 * h
        for xt in (x_scale_left, x_scale_right):
            ax.plot([xt, xt], [y_scale - tick_h, y_scale + tick_h], color=color,
                    linewidth=1.2 * lw_scale, zorder=zorder, path_effects=fx)
        ax.text((x_scale_left + x_scale_right) / 2, y_scale + 0.02 * h,
                f"{scale_m:,.0f} m", ha="center", va="bottom", fontsize=fontsize,
                color=color, zorder=zorder, path_effects=fx)

    # north arrow directly above the scale bar's right end, breathing room
    # above the label rather than immediately on top of it
    if draw_north:
        xn = x_scale_right
        yn0 = y_scale + (0.05 * h if draw_scale else 0.0)
        yn1 = yn0 + 0.09 * h
        ax.annotate("", xy=(xn, yn1), xytext=(xn, yn0),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    linewidth=1.2 * lw_scale,
                                    mutation_scale=10 * lw_scale),
                    zorder=zorder)
        ax.text(xn, yn1 + 0.015 * h, "N", ha="center", va="bottom",
                fontsize=fontsize + 1, fontweight="bold", color=color,
                zorder=zorder, path_effects=fx)



# ==========================================================================
def add_scale_north_inline(ax, scale_m, mm_per_m, axes_w_mm, color="black",
                           fontsize=6.5, zorder=5, lw_scale=1.0):
    """Scale bar + north arrow as FIGURE-level furniture, on the legend line.

    For a multi-panel figure whose panels all share one extent and scale, the
    scale bar belongs to the figure, not to any one panel. This draws it into
    its own small axes (typically beside the shared colourbar) instead of
    inside a map frame.

    The bar must still be geometrically true, so its length is not chosen
    freely: `mm_per_m` is the figure millimetres one map metre occupies in
    the panels, and the bar is drawn `scale_m * mm_per_m` millimetres long,
    expressed as a fraction of this axes' width (`axes_w_mm`). Taking the bar
    out of the map frame therefore does not decouple it from the map scale.
    Measure both from the rendered axes (get_window_extent / fig.dpi) rather
    than from nominal GridSpec sizes, which ignore the aspect-ratio shrink.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    bar_frac = (scale_m * mm_per_m) / axes_w_mm
    x0, y0 = 0.02, 0.42
    x1 = x0 + bar_frac

    ax.plot([x0, x1], [y0, y0], color=color, linewidth=1.8 * lw_scale,
            solid_capstyle="butt", zorder=zorder)
    for xt in (x0, x1):
        ax.plot([xt, xt], [y0 - 0.10, y0 + 0.10], color=color,
                linewidth=1.2 * lw_scale, zorder=zorder)
    ax.text((x0 + x1) / 2, y0 + 0.17, f"{scale_m:,.0f} m", ha="center",
            va="bottom", fontsize=fontsize, color=color, zorder=zorder)

    xn = x1 + 0.18
    ax.annotate("", xy=(xn, y0 + 0.42), xytext=(xn, y0 - 0.28),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                linewidth=1.2 * lw_scale,
                                mutation_scale=10 * lw_scale),
                zorder=zorder)
    ax.text(xn, y0 + 0.48, "N", ha="center", va="bottom", fontsize=fontsize + 1,
            fontweight="bold", color=color, zorder=zorder)
    return ax


# ==========================================================================
def add_colorbar_strip(ax, mappable, label, units=None, ticks=None,
                       n_ticks=5, fontsize=7, label_fontsize=7.5, cax=None,
                       size="4%", pad=0.55):
    """Horizontal colourbar directly below `ax`, titled "Label [units]"
    with 4-5 ticks -- the GRASS documentation-figure look.

    Creates its own axes via make_axes_locatable rather than accepting a
    pre-sized one: an aspect='equal' map axes gets SHRUNK inside whatever
    box a GridSpec allocated it (to preserve the data's true aspect ratio),
    so a colourbar axes placed by GridSpec geometry alone ends up detached
    from the map's actual rendered edge. make_axes_locatable derives the
    colourbar's position from the axes' real (post-aspect-shrink) bbox at
    draw time, so it stays snug against the map regardless of its aspect
    ratio. Pass `cax` only to override this with your own axes.
    """
    if cax is None:
        # aspect='equal' (and set_anchor) only take effect at draw time --
        # append_axes reads ax.get_position() immediately, so without
        # forcing a draw first it derives the colourbar's position from
        # the axes' pre-shrink (full-cell) box, not its true rendered one.
        ax.figure.canvas.draw()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size=size, pad=pad)
    fig = cax.figure
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    if ticks is None:
        vmin, vmax = mappable.get_clim()
        ticks = np.linspace(vmin, vmax, n_ticks)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=fontsize, length=2.5, width=0.5)
    title = f"{label} [{units}]" if units else label
    cbar.set_label(title, fontsize=label_fontsize)
    cbar.outline.set_linewidth(0.5)
    return cbar


# ==========================================================================
def finalize_frame(ax, extent):
    """Thin border box, opaque white background, no default tick marks
    (the graticule supplies its own), correct aspect and limits."""
    left, right, bottom, top = extent
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    # aspect='equal' shrinks the axes box to match the data's true aspect
    # ratio, and by default CENTRES that shrunk box within whatever cell it
    # was allocated. In a two-column layout where the map's column is much
    # taller than the map itself, that centring puts the map's colourbar
    # (attached just below it) at a figure height that can land on top of
    # unrelated content in the other column. Anchoring to the top pushes
    # all the leftover space to the bottom instead, where it's harmless.
    ax.set_anchor("N")
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")


# ==========================================================================
def style_map_panel(ax, extent, cax=None, mappable=None, cbar_label=None,
                    cbar_units=None, cbar_ticks=None, scale_m=None,
                    graticule_step=None, basin_label=None,
                    basin_label_pos=(0.5, 0.80), avoid_bbox=None,
                    scale_color="black", draw_scale=True, draw_north=True):
    """One call that applies all of the above to a map panel:
    graticule, north arrow + scale bar (bottom-right, inside the frame),
    thin border + white background, and (if cax/mappable given) a
    horizontal colourbar strip below.

    `avoid_bbox`: see add_north_scale -- pass the basin boundary's
    (min_easting, max_easting, min_northing, max_northing) so the north
    arrow/scale bar land in the map's exterior/buffer margin, clear of the
    basin interior and any labels near its edge, instead of a fixed
    fraction of the whole map that may or may not clear the boundary.

    `scale_color`: colour of the north arrow + scale bar (default black);
    pass e.g. "white" for a panel whose lower-right corner is too dark for
    black to read clearly.

    Call this AFTER plotting the map's own content (imshow layers,
    overlays, transects, boundary, labels) -- it only adds framing, it
    does not touch what's already drawn, other than setting ticks/limits.
    """
    finalize_frame(ax, extent)
    add_graticule(ax, extent, step=graticule_step)
    add_north_scale(ax, extent, avoid_bbox=avoid_bbox, scale_m=scale_m,
                    color=scale_color, draw_scale=draw_scale,
                    draw_north=draw_north)
    if basin_label:
        import matplotlib.patheffects as pe
        ax.text(*basin_label_pos, basin_label, transform=ax.transAxes,
               ha="center", va="top", fontsize=7.5, fontweight="bold",
               zorder=21,
               path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    if mappable is not None:
        add_colorbar_strip(ax, mappable, cbar_label, units=cbar_units,
                          ticks=cbar_ticks, cax=cax)


def _endpoint_label_pos(x_end, y_end, x_in, y_in, offset_m, side_m):
    """Offset point for one transect endpoint's label: outward along the
    line's own direction, plus a perpendicular component so the label
    doesn't sit ON the line's own extension."""
    dx, dy = x_end - x_in, y_end - y_in
    n = (dx**2 + dy**2) ** 0.5
    if n == 0:
        return [x_end, y_end]
    ux, uy = dx / n, dy / n
    px, py = -uy, ux
    return [x_end + ux * offset_m + px * side_m, y_end + uy * offset_m + py * side_m]


def draw_transects(ax, geom, specs, offset_m=20, side_m=15, min_sep_m=35,
                   color="black", halo_width=2.4, line_width=1.1, fontsize=7,
                   zorder=6):
    """Draw multiple transect lines and their endpoint labels TOGETHER, so
    labels from DIFFERENT transects that happen to sit close together (by
    design, e.g. two transects placed only ~20 m apart) can be separated
    from each other, not just from their own line.

    WHY THIS HAS TO SEE ALL TRANSECTS AT ONCE: an earlier per-transect
    offset (this function used to be called draw_transect and handle one
    line at a time) could not fix this -- confirmed by direct measurement:
    B-B's start label and C-C's end label offset to positions only ~23 m
    apart NO MATTER how large the per-transect offset was scaled, because
    both offsets happened to point in nearly the same direction, so scaling
    moved both labels together in lockstep instead of separating them. The
    only fix that actually works is comparing ALL label positions against
    each other and pushing apart any pair that ends up too close.

    `specs`: list of (transect_name, start_label, end_label) tuples.
    `geom`: the transect_geometry.csv DataFrame (columns transect,
    vertex_order, easting, northing).
    """
    entries = []
    for name, start_lab, end_lab in specs:
        sub = geom[geom["transect"] == name].sort_values("vertex_order")
        ax.plot(sub["easting"], sub["northing"], color="white",
               linewidth=halo_width, zorder=zorder, solid_capstyle="round")
        ax.plot(sub["easting"], sub["northing"], color=color,
               linewidth=line_width, zorder=zorder + 1, solid_capstyle="round")

        x0, y0 = sub.iloc[0][["easting", "northing"]]
        x1, y1 = sub.iloc[-1][["easting", "northing"]]
        x0p, y0p = sub.iloc[1][["easting", "northing"]] if len(sub) > 1 else (x1, y1)
        x1p, y1p = sub.iloc[-2][["easting", "northing"]] if len(sub) > 1 else (x0, y0)
        entries.append({"label": start_lab,
                        "pos": _endpoint_label_pos(x0, y0, x0p, y0p, offset_m, side_m)})
        entries.append({"label": end_lab,
                        "pos": _endpoint_label_pos(x1, y1, x1p, y1p, offset_m, side_m)})

    # pairwise repulsion: push apart any two labels closer than min_sep_m,
    # iterated so a chain of near-collisions settles into place.
    #
    # BUG FIXED HERE: the original condition was `0 < d < min_sep_m`, which
    # EXCLUDES d == 0 -- two labels landing at exactly the same point (a real
    # possibility: two transect endpoints offset outward can coincide, e.g.
    # if the offset geometry is symmetric) were then never separated at all,
    # left stacked with zero separation. Two glyphs drawn on top of each
    # other blend into an unreadable mess of overlapping strokes -- this is
    # almost certainly what earlier renders reported as a label coming out
    # as "G'" or "P" instead of the real text: not a font/encoding problem,
    # a silent repulsion no-op. Handling d == 0 with an arbitrary but
    # deterministic separation direction (rather than the undefined dx/d,
    # dy/d that would divide by zero) closes that gap.
    for _ in range(30):
        moved = False
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                xi, yi = entries[i]["pos"]
                xj, yj = entries[j]["pos"]
                dx, dy = xj - xi, yj - yi
                d = (dx**2 + dy**2) ** 0.5
                if d < min_sep_m:
                    if d == 0:
                        ux, uy = 1.0, 0.0  # arbitrary, deterministic
                        push = min_sep_m / 2
                    else:
                        push = (min_sep_m - d) / 2
                        ux, uy = dx / d, dy / d
                    entries[i]["pos"][0] -= ux * push
                    entries[i]["pos"][1] -= uy * push
                    entries[j]["pos"][0] += ux * push
                    entries[j]["pos"][1] += uy * push
                    moved = True
        if not moved:
            break

    for e in entries:
        ax.text(e["pos"][0], e["pos"][1], e["label"], fontsize=fontsize,
               fontweight="bold", color=color, ha="center", va="center",
               zorder=zorder + 2,
               path_effects=[pe.withStroke(linewidth=2, foreground="white")])
