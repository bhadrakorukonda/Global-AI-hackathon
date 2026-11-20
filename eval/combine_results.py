"""Combine eval runs made on different days into one summary and chart.

Groq's free daily token quota can't judge every config in one day, so configs run separately
(`run_eval.py --configs ... --out eval/results/<name>`). This merges those folders into
eval/results/{results.jsonl, summary.md, chart-light.svg, chart-dark.svg}. No network, spends nothing.

    uv run python eval/combine_results.py
"""

import sys
from pathlib import Path

from staleguard.evaluation import (
    ConfigMetrics,
    ItemResult,
    compute_metrics,
    render_summary,
)

RESULTS = Path(__file__).parent / "results"
ORDER = ("verify_all", "guarded", "guarded_warm")
LABELS = {"verify_all": "Check every claim", "guarded": "StaleGuard", "guarded_warm": "StaleGuard, warm cache"}

# Reference palette (dataviz skill), validated for these 3 slots in both modes. Light slot 3 is < 3:1 on the
# surface, so every bar carries a value label (relief rule) and the README also has the table.
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "series": ("#2a78d6", "#eb6834", "#1baf7a")},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "series": ("#3987e5", "#d95926", "#199e70")},
}
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def load() -> list[ItemResult]:
    """Every results.jsonl under eval/results/<run>/ (one config per run folder; later files win per config)."""
    by_config: dict[str, list[ItemResult]] = {}
    for path in sorted(RESULTS.glob("*/results.jsonl")):
        rows = [ItemResult.model_validate_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        for cfg in {r.config for r in rows}:
            by_config[cfg] = [r for r in rows if r.config == cfg]
    return [r for cfg in ORDER for r in by_config.get(cfg, [])]


def _bar(x: float, y: float, w: float, h: float, fill: str, up: bool) -> str:
    """Bar with 4px rounded data end; the baseline end stays square."""
    r = min(4.0, w / 2, h / 2) if h > 0 and w > 0 else 0
    if r == 0:
        return ""
    if up:  # vertical bar, rounded top
        d = f"M{x},{y + h} V{y + r} Q{x},{y} {x + r},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h} Z"
    else:  # horizontal bar, rounded right end
        d = f"M{x},{y} H{x + w - r} Q{x + w},{y} {x + w},{y + r} V{y + h - r} Q{x + w},{y + h} {x + w - r},{y + h} H{x} Z"
    return f'<path d="{d}" fill="{fill}"/>'


def _text(x: float, y: float, s: str, fill: str, size: int = 13, anchor: str = "start", weight: int = 400) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;")
    return (f'<text x="{x:.1f}" y="{y:.1f}" fill="{fill}" font-size="{size}" font-weight="{weight}" '
            f'text-anchor="{anchor}" font-family="{FONT}">{s}</text>')


def chart(metrics: list[ConfigMetrics], theme: str) -> str:
    """Two panels: accuracy (grouped bars, one 0-100% axis) and cost (three small multiples, each its own axis)."""
    t = THEMES[theme]
    cfgs = [m.config for m in metrics]
    color = {c: t["series"][ORDER.index(c)] for c in cfgs}
    W, H = 920, 470
    out = [(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
            f'aria-label="StaleGuard eval: accuracy and cost per answer by configuration">'),
           f'<rect width="{W}" height="{H}" fill="{t["surface"]}"/>']
    # legend (always present for >= 2 series)
    lx = 24
    for c in cfgs:
        out.append(f'<rect x="{lx}" y="22" width="12" height="12" rx="3" fill="{color[c]}"/>')
        out.append(_text(lx + 18, 33, LABELS[c], t["ink2"]))
        lx += 26 + 8 * len(LABELS[c])

    # ---- panel A: accuracy
    ax, ay, aw, ah = 56, 96, 380, 300
    out.append(_text(24, 74, "Accuracy on 51 hand-checked answers", t["ink"], 15, weight=600))
    for pct in (0, 25, 50, 75, 100):
        y = ay + ah - ah * pct / 100
        out.append(f'<line x1="{ax}" x2="{ax + aw}" y1="{y}" y2="{y}" stroke="{t["grid" if pct else "axis"]}" stroke-width="1"/>')
        out.append(_text(ax - 8, y + 4, f"{pct}%", t["muted"], 11, "end"))
    groups = [("Stale answers caught", "recall", "higher is better"), ("Flags that were right", "precision", "higher is better"),
              ("False alarms", "false_alarm_rate", "lower is better")]
    gw = aw / len(groups)
    bw = min(30.0, (gw - 28) / len(cfgs) - 2)
    for gi, (name, field, hint) in enumerate(groups):
        gx = ax + gi * gw + (gw - len(cfgs) * (bw + 2) + 2) / 2
        for ci, c in enumerate(cfgs):
            v = getattr(metrics[ci], field) or 0.0
            h = ah * v
            x = gx + ci * (bw + 2)
            out.append(_bar(x, ay + ah - h, bw, h, color[c], up=True))
            out.append(_text(x + bw / 2, ay + ah - h - 6, f"{v * 100:.0f}%", t["ink2"], 11, "middle"))
        cx = ax + gi * gw + gw / 2
        out.append(_text(cx, ay + ah + 20, name, t["ink"], 12, "middle"))
        out.append(_text(cx, ay + ah + 36, hint, t["muted"], 11, "middle"))

    # ---- panel B: cost per answer, small multiples
    bx, bwid = 500, 396
    out.append(_text(bx, 74, "Cost per answer", t["ink"], 15, weight=600))
    panels = [("Tavily search credits", lambda m: m.tavily_calls_per_answer or 0.0, lambda v: f"{v:.2f}"),
              ("Verification tokens", lambda m: m.tokens_per_answer or 0.0, lambda v: f"{v:,.0f}"),
              ("Median added latency", lambda m: (m.median_latency_ms or 0.0) / 1000, lambda v: f"{v:.1f} s")]
    py, bar_h, label_w = 96, 16, 150
    for title, get, fmt in panels:
        out.append(_text(bx, py + 4, title, t["ink2"], 12, weight=600))
        vals = [get(m) for m in metrics]
        top = max(vals) or 1.0
        for ci, c in enumerate(cfgs):
            y = py + 14 + ci * (bar_h + 4)
            out.append(_text(bx, y + 12, LABELS[c], t["muted"], 11))
            w = (bwid - label_w - 60) * vals[ci] / top
            out.append(f'<line x1="{bx + label_w}" x2="{bx + label_w}" y1="{y - 2}" y2="{y + bar_h + 2}" stroke="{t["axis"]}"/>')
            out.append(_bar(bx + label_w, y, w, bar_h, color[c], up=False))
            out.append(_text(bx + label_w + w + 6, y + 12, fmt(vals[ci]), t["ink2"], 11))
        py += 14 + len(cfgs) * (bar_h + 4) + 26
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    results = load()
    cfgs = [c for c in ORDER if any(r.config == c for r in results)]
    if not cfgs:
        print("No results under", RESULTS)
        raise SystemExit(1)
    metrics = [compute_metrics(c, results) for c in cfgs]
    judges = {c: sorted({r.models["judge"] for r in results if r.config == c and "judge" in r.models}) for c in cfgs}
    meta = {
        "Runs combined": ", ".join(str(p.parent.name) for p in sorted(RESULTS.glob("*/results.jsonl"))),
        "Dataset": f"{metrics[0].n} items, hand-verified (eval/dataset.jsonl)",
        "Judge model(s) that answered, per config": "; ".join(f"{c}: {', '.join(j) or 'n/a'}" for c, j in judges.items()),
        "Stale threshold": "STALE claim with correction and confidence ≥ 0.7",
        "Tavily": "basic depth, max_results=5, include_published_date=True",
    }
    (RESULTS / "results.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in results), encoding="utf-8")
    (RESULTS / "summary.md").write_text(render_summary(metrics, results, meta), encoding="utf-8")
    for theme in THEMES:
        (RESULTS / f"chart-{theme}.svg").write_text(chart(metrics, theme), encoding="utf-8")
    print((RESULTS / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
