"""Generate the four revised paper figures from the dynamic extinction runs."""

from __future__ import annotations

import os
import sys
import json
import csv
from contextlib import contextmanager

import matplotlib
FIGURE4_ONLY = "--figure4-only" in sys.argv
NO_SHOW_FIGURE4 = "--no-show-figure4" in sys.argv
FIGURE_S1_ONLY = "--figure-s1-only" in sys.argv
TABLE_S1_ONLY = "--table-s1-only" in sys.argv
NO_SHOW_FIGURE_S1 = "--no-show-figure-s1" in sys.argv
NO_SHOW_ALL = "--no-show" in sys.argv
matplotlib.use("Agg" if NO_SHOW_ALL else "TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text

from trophic_web_methods import (
    extinction_simulation,
    ksi_indexer,
    load_food_webs,
    robustness_50,
    trophic_level_indexer,
)


#ANT_COLOR = "#545cd6"
#ARC_COLOR = "#35bcdb"

ANT_COLOR = "blue"
ARC_COLOR = "red"
WEB_COLORS = {"Antarctic": ANT_COLOR, "Arctic": ARC_COLOR}
THRESHOLDS = (0.25, 0.50, 0.75)
CRITERIA = (
    ("trophic_level", "bottom", "Lowest"),
    ("trophic_level", "median", "Median"),
    ("trophic_level", "top", "Highest"),
    ("ksi", "bottom", "Lowest"),
    ("ksi", "median", "Median"),
    ("ksi", "top", "Highest"),
)
LINE_STYLES = {"top": "-", "median": "--", "bottom": ":"}
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


@contextmanager
def quiet_infomap():
    """Keep the repeated Infomap logs out of the figure-generation output."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        old_stdout = sys.stdout
        old_stdout.flush()
        saved_stdout_fd = os.dup(1)
        sys.stdout = sink
        os.dup2(sink.fileno(), 1)
        try:
            yield
        finally:
            sys.stdout.flush()
            os.dup2(saved_stdout_fd, 1)
            os.close(saved_stdout_fd)
            sys.stdout = old_stdout


def _simulate(graph, metric, criterion, threshold):
    """Return a full population/KSI trace and accumulated secondary losses."""
    initial_count = graph.vcount()
    current = graph.copy()
    counts = []
    ksi_by_species = {name: [] for name in graph.vs["name"]}
    secondary_losses = 0

    for _ in range(initial_count):
        if current.vcount():
            trophic_level_indexer(current)
            ksi_indexer(current)
            for vertex in current.vs:
                ksi_by_species[vertex["name"]].append(float(vertex["ksi"]))
        counts.append(current.vcount())
        before = current.vcount()
        current, primary_names = extinction_simulation(
            current,
            type_=criterion,
            metric=metric,
            percent=0,
            threshold=threshold,
        )
        secondary_losses += max(0, before - current.vcount() - len(primary_names))

    x_pct = np.arange(initial_count, dtype=float) / initial_count * 100
    y_pct = np.asarray(counts, dtype=float) / initial_count * 100
    return {
        "counts": counts,
        "x_pct": x_pct,
        "y_pct": y_pct,
        "ksi_by_species": ksi_by_species,
        "initial_count": initial_count,
        "secondary_count": secondary_losses,
        "secondary_pct": secondary_losses / initial_count * 100,
        "r50": robustness_50(counts, initial_count),
    }


def collect_all(antarctic, arctic):
    traces = {}
    webs = {"Antarctic": antarctic, "Arctic": arctic}
    total = len(webs) * len(THRESHOLDS) * len(CRITERIA)
    completed = 0
    for threshold in THRESHOLDS:
        for metric, criterion, _ in CRITERIA:
            for web_name, graph in webs.items():
                completed += 1
                print(
                    f"Simulation {completed}/{total}: {web_name}, {metric}/"
                    f"{criterion}, ET={threshold:.0%}"
                )
                with quiet_infomap():
                    trace = _simulate(graph, metric, criterion, threshold)
                traces[(web_name, metric, criterion, threshold)] = trace
    return traces


def collect_figure4(antarctic, arctic):
    """Collect the four ET=50% runs shown in Figure 4."""
    scenarios = (("trophic_level", "top"), ("ksi", "bottom"))
    traces = {}
    for metric, criterion in scenarios:
        for web, graph in (("Antarctic", antarctic), ("Arctic", arctic)):
            print(f"Simulation: {web}, {metric}/{criterion}, ET=50%")
            with quiet_infomap():
                traces[(web, metric, criterion, 0.50)] = _simulate(
                    graph, metric, criterion, 0.50
                )
    return traces


def collect_figure_s1(antarctic, arctic):
    """Collect highest-TL and lowest-KSI runs for ET=25% and ET=75%."""
    scenarios = (("trophic_level", "top"), ("ksi", "bottom"))
    traces = {}
    for threshold in (0.25, 0.75):
        for metric, criterion in scenarios:
            for web, graph in (("Antarctic", antarctic), ("Arctic", arctic)):
                print(
                    f"Simulation: {web}, {metric}/{criterion}, ET={threshold:.0%}"
                )
                with quiet_infomap():
                    traces[(web, metric, criterion, threshold)] = _simulate(
                        graph, metric, criterion, threshold
                    )
    return traces


def _species_metrics(graph):
    """Calculate the manuscript species-level metrics on a graph copy."""
    graph = graph.copy()
    trophic_level_indexer(graph)
    with quiet_infomap():
        ksi_indexer(graph)
    names = graph.vs["name"]
    metrics = {
        "name": names,
        "tl": np.asarray(graph.vs["trophic_level"], dtype=float),
        "degree": np.asarray(graph.degree(), dtype=int),
        "betweenness": np.asarray(graph.betweenness(), dtype=float),
        "pagerank": np.asarray(graph.pagerank(), dtype=float),
        "ksi": np.asarray(graph.vs["ksi"], dtype=float),
    }
    order = sorted(
        range(len(names)),
        key=lambda i: (
            not np.isfinite(metrics["tl"][i]),
            metrics["tl"][i] if np.isfinite(metrics["tl"][i]) else float("inf"),
            str(names[i]).casefold(),
        ),
    )
    return [{key: (value[i] if isinstance(value, np.ndarray) else value[i])
             for key, value in metrics.items()} for i in order]


def generate_table_s1(antarctic, arctic):
    """Write two side-by-side species metric lists, independently ordered by TL."""
    antarctic_rows = _species_metrics(antarctic)
    arctic_rows = _species_metrics(arctic)
    headers = (
        "Antarctic Species", "Antarctic TL", "Antarctic Deg", "Antarctic Btw",
        "Antarctic PR", "Antarctic KSI", "Arctic Species", "Arctic TL",
        "Arctic Deg", "Arctic Btw", "Arctic PR", "Arctic KSI",
    )
    output_path = os.path.join(OUTPUT_DIR, "Table_S1.csv")
    with open(output_path, "w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(headers)
        for index in range(max(len(antarctic_rows), len(arctic_rows))):
            row = []
            for rows in (antarctic_rows, arctic_rows):
                if index < len(rows):
                    item = rows[index]
                    row.extend([
                        item["name"],
                        f"{item['tl']:.9g}",
                        int(item["degree"]),
                        f"{item['betweenness']:.9g}",
                        f"{item['pagerank']:.9g}",
                        f"{item['ksi']:.9g}",
                    ])
                else:
                    row.extend([""] * 6)
            writer.writerow(row)
    print(f"Saved {output_path}")
    return output_path


def top_original_ksi_species(graph, count=3):
    """Return species with highest KSI in the unextinct original web."""
    graph = graph.copy()
    with quiet_infomap():
        ksi_indexer(graph)
    ranked = sorted(
        zip(graph.vs["ksi"], graph.vs["name"]),
        key=lambda item: (-float(item[0]), str(item[1]).casefold()),
    )
    return {name for _, name in ranked[:count]}


def trace_for(traces, web, metric, criterion, threshold):
    return traces[(web, metric, criterion, threshold)]


def figure1(traces):
    """Six population panels; each overlays all three ET trajectories."""
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9), sharex=True, sharey=True)
    rows = (CRITERIA[:3], CRITERIA[3:])
    et_alphas = {0.25: 0.42, 0.50: 0.70, 0.75: 1.0}
    for row, criteria in enumerate(rows):
        for col, (metric, criterion, label) in enumerate(criteria):
            ax = axes[row, col]
            for web, color in WEB_COLORS.items():
                for threshold in THRESHOLDS:
                    trace = trace_for(traces, web, metric, criterion, threshold)
                    ax.plot(
                        trace["x_pct"], trace["y_pct"], color=color,
                        alpha=et_alphas[threshold], linewidth=2.1,
                        linestyle={0.25: ":", 0.50: "--", 0.75: "-"}[threshold],
                        label=f"{web}, ET {threshold:.0%}",
                    )
            ax.set_title(label, fontsize=12)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 102)
            ax.grid(alpha=0.22)
            if col == 0:
                ax.set_ylabel("Population remaining (%)")
            if row == 1:
                ax.set_xlabel("Primary extinctions (% of original web)")
            if row == 0 and col == 2:
                ax.legend(fontsize=9.5, ncol=1, loc="upper right", frameon=True)
    fig.text(0.012, 0.72, "Trophic level", rotation=90, va="center", fontsize=13, weight="bold")
    fig.text(0.012, 0.28, "KSI", rotation=90, va="center", fontsize=13, weight="bold")
    fig.suptitle("Dynamic extinction trajectories across thresholds", y=0.99, fontsize=15)
    fig.tight_layout(rect=(0.04, 0.02, 1, 0.96))
    return fig


def figure2(traces):
    """Cumulative secondary losses at the end of each full dynamic run."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    for ax, (metric, panel_title) in zip(axes, (("trophic_level", "Trophic level"), ("ksi", "KSI"))):
        for web, color in WEB_COLORS.items():
            for criterion, criterion_label in (("top", "Highest"), ("median", "Median"), ("bottom", "Lowest")):
                y = [
                    trace_for(traces, web, metric, criterion, threshold)["secondary_pct"]
                    for threshold in THRESHOLDS
                ]
                ax.plot(
                    np.asarray(THRESHOLDS) * 100, y,
                    color=color, linestyle=LINE_STYLES[criterion], marker="o",
                    linewidth=2, markersize=5,
                    label=f"{web}, {criterion_label}",
                )
        ax.set_title(panel_title)
        ax.set_xlabel("Extinction threshold (%)")
        ax.set_xticks([25, 50, 75])
        ax.grid(alpha=0.25)
        if metric == "ksi":
            ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("Secondary extinctions (% of original web)")
    fig.suptitle("Cumulative secondary extinctions by threshold", fontsize=15)
    fig.tight_layout()
    return fig


def figure3(traces):
    """R50 winner/margin heatmap and aggregate Antarctic/Arctic win summaries."""
    r50_ant = np.full((len(THRESHOLDS), len(CRITERIA)), np.nan)
    r50_arc = np.full_like(r50_ant, np.nan)
    delta = np.full_like(r50_ant, np.nan)
    for i, threshold in enumerate(THRESHOLDS):
        for j, (metric, criterion, _) in enumerate(CRITERIA):
            a = trace_for(traces, "Antarctic", metric, criterion, threshold)["r50"]
            b = trace_for(traces, "Arctic", metric, criterion, threshold)["r50"]
            r50_ant[i, j], r50_arc[i, j], delta[i, j] = a, b, a - b

    fig = plt.figure(figsize=(15.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.65, 1), height_ratios=(1, 1))
    ax_heat = fig.add_subplot(grid[:, 0])
    ax_rate = fig.add_subplot(grid[0, 1])
    ax_margin = fig.add_subplot(grid[1, 1])
    max_abs = max(float(np.nanmax(np.abs(delta))), 1e-9)
    cmap = LinearSegmentedColormap.from_list("arctic_tie_antarctic", [ARC_COLOR, "#ffffff", ANT_COLOR])
    # A signed square-root scale makes modest wins visible while preserving sign.
    display_delta = np.sign(delta) * np.sqrt(np.abs(delta) / max_abs)
    image = ax_heat.imshow(display_delta, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax_heat.set_yticks(range(3), ["25%", "50%", "75%"])
    ax_heat.set_ylabel("Extinction threshold")
    ax_heat.set_xticks(range(6), [entry[2] for entry in CRITERIA])
    ax_heat.set_xlabel("Removal criterion", labelpad=30)
    ax_heat.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.5)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    ax_heat.axvline(2.5, color="#202020", linewidth=2.4)
    ax_heat.text(1, -0.1, "Trophic level", transform=ax_heat.get_xaxis_transform(),
                 ha="center", va="top", fontsize=10, weight="bold")
    ax_heat.text(4, -0.1, "KSI", transform=ax_heat.get_xaxis_transform(),
                 ha="center", va="top", fontsize=10, weight="bold")
    cbar = fig.colorbar(image, ax=ax_heat, fraction=0.045, pad=0.03)
    cbar.set_label("R50 difference (Antarctic − Arctic)")
    cbar_ticks = np.linspace(-1, 1, 5)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([
        f"{np.sign(value) * value**2 * max_abs:+.2f}" for value in cbar_ticks
    ])

    wins_a = int(np.sum(delta > 1e-12))
    wins_r = int(np.sum(delta < -1e-12))
    ties = int(delta.size - wins_a - wins_r)
    total = int(delta.size)
    bars = ax_rate.bar(
        ["Antarctic", "Arctic", "Tie"], [wins_a, wins_r, ties],
        color=[ANT_COLOR, ARC_COLOR, "#9ca3af"], edgecolor="#333333", linewidth=0.6,
    )
    ax_rate.set_title("Win frequency")
    ax_rate.set_ylabel("Scenarios won")
    ax_rate.set_ylim(0, max(wins_a, wins_r, ties, 1) * 1.3)
    ax_rate.grid(axis="y", alpha=0.22)
    for bar, count in zip(bars, (wins_a, wins_r, ties)):
        ax_rate.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                     f"{count}/{total} ({count / total:.0%})", ha="center", va="bottom", fontsize=9)

    margins_a = delta[delta > 1e-12]
    margins_r = -delta[delta < -1e-12]
    mean_a = float(np.mean(margins_a)) if margins_a.size else 0.0
    mean_r = float(np.mean(margins_r)) if margins_r.size else 0.0
    sd_a = float(np.std(margins_a)) if margins_a.size else 0.0
    sd_r = float(np.std(margins_r)) if margins_r.size else 0.0
    ax_margin.bar(
        ["Antarctic", "Arctic"], [mean_a, mean_r], yerr=[sd_a, sd_r], capsize=5,
        color=[ANT_COLOR, ARC_COLOR], edgecolor="#333333", linewidth=0.6,
    )
    ax_margin.set_title("Winning margin")
    ax_margin.set_ylabel("Mean R50 advantage (± SD)")
    ax_margin.grid(axis="y", alpha=0.22)
    return fig


def _top_ksi_names(trace, count=3):
    values = []
    for name, series in trace["ksi_by_species"].items():
        if series:
            values.append((series[0], name))
    return [name for _, name in sorted(values, reverse=True)[:count]]


def figure4(traces):
    """All species KSI trajectories, highlighting the three highest initial-KSI species per web."""
    scenarios = (
        ("trophic_level", "top", "A  Highest trophic level removal"),
        ("ksi", "bottom", "B  Lowest KSI removal"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    line_states = {}
    for ax, (metric, criterion, title) in zip(axes, scenarios):
        for web in WEB_COLORS:
            trace = trace_for(traces, web, metric, criterion, 0.50)
            selected = set(_top_ksi_names(trace))
            group_color = WEB_COLORS[web]
            for name, values in trace["ksi_by_species"].items():
                x = trace["x_pct"][:len(values)]
                is_selected = name in selected
                line, = ax.plot(
                    x,
                    values,
                    color=group_color,
                    alpha=0.16,
                    linewidth=0.8 if is_selected else 0.65,
                    zorder=1,
                    label=name,
                    picker=2 if is_selected else False,
                )
                if is_selected:
                    line_states[line] = {
                        "active": False,
                        "original_lw": line.get_linewidth(),
                        "original_color": line.get_color(),
                        "axes": ax,
                        "annotation": None,
                    }
        ax.set_title(title)
        ax.set_xlabel("Primary extinctions (% of original web)")
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.22)

    def toggle_line(line, x=None, y=None):
        state = line_states[line]
        state["active"] = not state["active"]
        if state["active"]:
            line.set_linewidth(2)
            line.set_alpha(1)
            line.set_color(state["original_color"])
            line.set_zorder(10)
            anchor_point = (
                (x, y) if x is not None and y is not None
                else (line.get_xdata()[0], line.get_ydata()[0])
            )
            if state["annotation"] is not None:
                state["annotation"].remove()
            label = state["axes"].annotate(
                line.get_label(),
                xy=anchor_point,
                xytext=(30, 30),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->", shrinkA=0, shrinkB=0, relpos=(0, 0)),
                fontsize=10,
                zorder=11,
            )
            state["annotation"] = label
            label.set_picker(True)
            label.draggable(True)
            print(f"Labeled {line.get_label()} at ({anchor_point[0]:.2f}, {anchor_point[1]:.3f})")
        else:
            line.set_linewidth(state["original_lw"])
            line.set_alpha(0.16)
            line.set_color(state["original_color"])
            line.set_zorder(1)
            if state["annotation"] is not None:
                state["annotation"].set_visible(False)
        fig.canvas.draw_idle()

    def label_selected_line(event):
        line = event.artist
        if line not in line_states:
            return
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        if x is None or y is None:
            return
        toggle_line(line, x, y)

    fig.canvas.mpl_connect("pick_event", label_selected_line)
    axes[0].set_ylabel("KSI of surviving species")
    fig.suptitle("KSI trajectories through dynamic extinction", fontsize=14)
    fig.tight_layout()
    return fig


def figure_s1(traces, original_top_ksi):
    """Plot ET=25% and 75% KSI trajectories with original-web leaders selectable."""
    thresholds = (0.25, 0.75)
    scenarios = (
        ("trophic_level", "top", "Highest trophic level removal"),
        ("ksi", "bottom", "Lowest KSI removal"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    line_states = {}
    for row, threshold in enumerate(thresholds):
        for col, (metric, criterion, scenario_title) in enumerate(scenarios):
            ax = axes[row, col]
            for web in WEB_COLORS:
                trace = trace_for(traces, web, metric, criterion, threshold)
                color = WEB_COLORS[web]
                selected = original_top_ksi[web]
                for name, values in trace["ksi_by_species"].items():
                    x = trace["x_pct"][:len(values)]
                    is_selected = name in selected
                    line, = ax.plot(
                        x,
                        values,
                        color=color,
                        alpha=0.32 if is_selected else 0.12,
                        linewidth=1.15 if is_selected else 0.55,
                        zorder=2 if is_selected else 1,
                        label=name,
                        picker=2 if is_selected else False,
                    )
                    if is_selected:
                        line_states[line] = {
                            "active": False,
                            "original_alpha": line.get_alpha(),
                            "original_lw": line.get_linewidth(),
                            "original_color": line.get_color(),
                            "axes": ax,
                            "annotation": None,
                        }
            panel_letter = "ABCD"[row * 2 + col]
            ax.set_title(f"{panel_letter}  ET={threshold:.0%} · {scenario_title}")
            ax.set_xlim(0, 100)
            ax.grid(alpha=0.22)
            ax.set_ylabel("KSI of surviving species")
            if row == 1:
                ax.set_xlabel("Primary extinctions (% of original web)")

    def toggle_line(line, x, y):
        state = line_states[line]
        state["active"] = not state["active"]
        if state["active"]:
            line.set_linewidth(2.0)
            line.set_alpha(1.0)
            line.set_color(state["original_color"])
            line.set_zorder(10)
            if state["annotation"] is not None:
                state["annotation"].remove()
            label = state["axes"].annotate(
                line.get_label(),
                xy=(x, y),
                xytext=(30, 30),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(
                    arrowstyle="->", shrinkA=0, shrinkB=0, relpos=(0, 0)
                ),
                fontsize=10,
                zorder=11,
            )
            state["annotation"] = label
            label.set_picker(True)
            label.draggable(True)
            print(f"Labeled {line.get_label()} at ({x:.2f}, {y:.3f})")
        else:
            line.set_linewidth(state["original_lw"])
            line.set_alpha(state["original_alpha"])
            line.set_color(state["original_color"])
            line.set_zorder(2)
            if state["annotation"] is not None:
                state["annotation"].set_visible(False)
        fig.canvas.draw_idle()

    def label_selected_line(event):
        line = event.artist
        if line not in line_states:
            return
        x, y = event.mouseevent.xdata, event.mouseevent.ydata
        if x is not None and y is not None:
            toggle_line(line, x, y)

    fig.canvas.mpl_connect("pick_event", label_selected_line)
    fig.suptitle("KSI trajectories through dynamic extinction", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")

    # Also export a text-free plotting layer and text layout metadata for the
    # editable PowerPoint version. All coordinates are stored as canvas ratios.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_width, canvas_height = fig.canvas.get_width_height()
    legend_text_ids = set()
    legends = []
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        legend_texts = legend.get_texts()
        legend_text_ids.update(id(text) for text in legend_texts)
        legend_box = legend.get_window_extent(renderer)
        frame = legend.get_frame()
        frame_face = frame.get_facecolor()
        frame_edge = frame.get_edgecolor()
        entries = []
        handles = legend.get_lines()
        for index, text in enumerate(legend_texts):
            bounds = text.get_window_extent(renderer)
            handle = handles[index] if index < len(handles) else None
            handle_bounds = handle.get_window_extent(renderer) if handle is not None else bounds
            entries.append({
                "text": text.get_text(),
                "text_bbox": [bounds.x0 / canvas_width, bounds.y0 / canvas_height,
                              bounds.x1 / canvas_width, bounds.y1 / canvas_height],
                "font_size_pt": text.get_fontproperties().get_size_in_points(),
                "bold": text.get_fontproperties().get_weight() in ("bold", "semibold", "heavy", 600, 700, 800, 900),
                "color": to_hex(text.get_color(), keep_alpha=False),
                "handle_bbox": [handle_bounds.x0 / canvas_width, handle_bounds.y0 / canvas_height,
                                handle_bounds.x1 / canvas_width, handle_bounds.y1 / canvas_height],
                "line_color": to_hex(handle.get_color(), keep_alpha=False) if handle is not None else "#000000",
                "line_width_pt": handle.get_linewidth() if handle is not None else 1.0,
                "line_style": handle.get_linestyle() if handle is not None else "-",
                "marker": handle.get_marker() if handle is not None else "None",
                "marker_size_pt": handle.get_markersize() if handle is not None else 0,
            })
        legends.append({
            "bbox": [legend_box.x0 / canvas_width, legend_box.y0 / canvas_height,
                     legend_box.x1 / canvas_width, legend_box.y1 / canvas_height],
            "face_color": to_hex(frame_face, keep_alpha=False),
            "edge_color": to_hex(frame_edge, keep_alpha=False),
            "entries": entries,
        })

    text_items = []
    for text in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        value = text.get_text()
        if not value or not value.strip() or id(text) in legend_text_ids or not text.get_visible():
            continue
        try:
            bbox_patch = text.get_bbox_patch() if isinstance(text, Annotation) else None
            bounds = (
                bbox_patch.get_window_extent(renderer)
                if bbox_patch is not None
                else text.get_window_extent(renderer)
            )
        except Exception:
            continue
        weight = text.get_fontproperties().get_weight()
        item = {
            "text": value,
            "bbox": [bounds.x0 / canvas_width, bounds.y0 / canvas_height,
                     bounds.x1 / canvas_width, bounds.y1 / canvas_height],
            "font_size_pt": text.get_fontproperties().get_size_in_points(),
            "bold": weight in ("bold", "semibold", "heavy", 600, 700, 800, 900),
            "color": to_hex(text.get_color(), keep_alpha=False),
            "rotation": float(text.get_rotation()),
            "ha": text.get_horizontalalignment(),
            "va": text.get_verticalalignment(),
        }
        if isinstance(text, Annotation):
            if bbox_patch is not None:
                item["bbox_face_color"] = to_hex(bbox_patch.get_facecolor(), keep_alpha=False)
                item["bbox_edge_color"] = to_hex(bbox_patch.get_edgecolor(), keep_alpha=False)
                item["bbox_line_width_pt"] = float(bbox_patch.get_linewidth())
            anchor_x, anchor_y = text.axes.transData.transform(text.xy)
            item["arrow_anchor_bbox"] = [anchor_x / canvas_width, anchor_y / canvas_height]
            if text.arrow_patch is not None:
                item["arrow_color"] = to_hex(text.arrow_patch.get_edgecolor(), keep_alpha=False)
                item["arrow_line_width_pt"] = float(text.arrow_patch.get_linewidth())
        text_items.append(item)

    base_path = os.path.join(OUTPUT_DIR, filename.replace(".png", "_editable_base.png"))
    if fig.get_constrained_layout():
        frozen_positions = [(ax, ax.get_position().frozen()) for ax in fig.axes]
        fig.set_constrained_layout(False)
        for ax, position in frozen_positions:
            ax.set_position(position)
    hidden = []
    for artist in fig.findobj(match=lambda artist: isinstance(artist, Text)):
        hidden.append((artist, artist.get_visible()))
        artist.set_visible(False)
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is not None:
            hidden.append((legend, legend.get_visible()))
            legend.set_visible(False)
    fig.savefig(base_path, dpi=300, facecolor="white")
    for artist, was_visible in hidden:
        artist.set_visible(was_visible)

    metadata_path = base_path.replace("_editable_base.png", "_editable_text.json")
    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump({
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "dpi": float(fig.dpi),
            "texts": text_items,
            "legends": legends,
        }, metadata_file, ensure_ascii=False, indent=2)
    plt.close(fig)
    print(f"Saved {path}")


def preview(fig, title, instructions=None):
    """Show one figure and wait for its window to close before continuing."""
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)
    if instructions:
        print(instructions)
    else:
        print(f"Previewing {title}. Close the figure window to continue and save it.")
    plt.show(block=True)


def main():
    antarctic, arctic = load_food_webs()
    if TABLE_S1_ONLY:
        generate_table_s1(antarctic, arctic)
        return
    if FIGURE_S1_ONLY:
        traces = collect_figure_s1(antarctic, arctic)
        original_top_ksi = {
            "Antarctic": top_original_ksi_species(antarctic),
            "Arctic": top_original_ksi_species(arctic),
        }
        fig_s1 = figure_s1(traces, original_top_ksi)
        if not NO_SHOW_ALL and not NO_SHOW_FIGURE_S1:
            print(
                "Click each highlighted species line to add its label arrow; "
                "drag the white boxes. Close Figure S1 to save."
            )
            fig_s1.canvas.manager.set_window_title("Figure S1")
            plt.show(block=True)
        save(fig_s1, "Figure_S1.png")
        return

    if FIGURE4_ONLY:
        traces = collect_figure4(antarctic, arctic)
    else:
        generate_table_s1(antarctic, arctic)
        traces = collect_all(antarctic, arctic)
        fig1 = figure1(traces)
        if not NO_SHOW_ALL:
            preview(fig1, "Figure 1")
        save(fig1, "Figure_1_new.png")

        fig2 = figure2(traces)
        if not NO_SHOW_ALL:
            preview(fig2, "Figure 2")
        save(fig2, "Figure_2_new.png")

        fig3 = figure3(traces)
        if not NO_SHOW_ALL:
            preview(fig3, "Figure 3")
        save(fig3, "Figure_3_new.png")

    fig4 = figure4(traces)
    if not NO_SHOW_ALL and not NO_SHOW_FIGURE4:
        preview(
            fig4,
            "Figure 4",
            "Click each selected species line to add its label arrow; "
            "drag the white boxes. Close the Figure 4 window to save.",
        )
    save(fig4, "Figure_4_new.png")

    if not FIGURE4_ONLY:
        original_top_ksi = {
            "Antarctic": top_original_ksi_species(antarctic),
            "Arctic": top_original_ksi_species(arctic),
        }
        s1_traces = {
            key: trace for key, trace in traces.items() if key[3] in (0.25, 0.75)
        }
        fig_s1 = figure_s1(s1_traces, original_top_ksi)
        if not NO_SHOW_ALL and not NO_SHOW_FIGURE_S1:
            preview(
                fig_s1,
                "Figure S1",
                "Click each highlighted species line to add its label arrow; "
                "drag the white boxes. Close Figure S1 to save.",
            )
        save(fig_s1, "Figure_S1.png")


if __name__ == "__main__":
    main()
