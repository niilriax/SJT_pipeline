from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "outputs" / "final_reports"
OUTPUT = REPORT_ROOT / "recent_three_runs_report.html"


RUN_DIRS = [
    REPORT_ROOT / "6a0f832c-cc2b-46ca-a074-e449abf9ba67" / "bank-v1-ebf4fddc463e",
    REPORT_ROOT / "6b94c430-8ed2-4ad4-bbc7-1916ff6731d0" / "bank-v3-7a37ddbdfe19",
    REPORT_ROOT / "c5895f6c-b092-442b-a337-4a081e3a6b32" / "bank-v2-1a478a657818",
]

# The existing Mussel comparison visualisation contains these persisted item-level
# values. They are kept here only to redraw that comparison with explicit axes.
MUSSEL_GREGARIOUSNESS = [
    (1, .57, .871, 2, [.57, 0, .43, 0]), (2, .27, .600, 4, [.19, .08, .50, .23]),
    (3, .58, .873, 3, [.58, 0, .23, .19]), (4, .54, .880, 2, [.54, 0, .04, .42]),
    (5, .57, .875, 2, [.01, .56, .40, .03]), (6, .58, .873, 2, [.58, 0, .38, .04]),
    (7, .55, .876, 3, [.55, 0, .15, .30]), (8, .39, .828, 4, [.06, .33, .06, .55]),
    (9, .55, .873, 3, [.55, 0, .06, .39]), (10, .58, .855, 3, [.30, .28, 0, .42]),
    (11, .23, .575, 3, [.23, 0, .42, .35]), (12, .63, .810, 4, [.09, .54, .20, .17]),
    (13, .55, .823, 3, [.48, .07, .43, .02]), (14, .34, .713, 3, [0, .34, .57, .09]),
    (15, .52, .812, 2, [.50, .02, .45, .03]), (16, .49, .828, 2, [.47, .02, .01, .50]),
    (17, .58, .873, 2, [0, .58, .40, .02]), (18, .45, .838, 3, [.45, 0, .46, .09]),
    (19, .05, .181, 2, [.05, 0, .04, .91]), (20, .34, .724, 3, [.32, .02, .41, .25]),
    (21, .37, .781, 3, [.34, .03, .53, .10]), (22, .54, .870, 3, [.02, .52, .11, .35]),
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (dict, list)):
        return html.escape(json.dumps(value, ensure_ascii=False, indent=2))
    return html.escape(str(value))


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def current_comparison_data(run: dict[str, Any]) -> list[dict[str, Any]]:
    data = []
    for index, item in enumerate(run["items"], 1):
        stats = item.get("_stats") or {}
        option_stats = stats.get("option_statistics") or {}
        data.append(
            {
                "id": f"{index:02d}",
                "x": stats.get("difficulty"),
                "y": item_citc(stats),
                "eff": stats.get("effective_option_count"),
                "rates": [float((option_stats.get(oid) or {}).get("rate") or 0) for oid in "ABCD"],
            }
        )
    return data


def svg_scatter(data: list[dict[str, Any]], title: str, series: str, css_color: str) -> str:
    W, H, x0, x1, y0, y1 = 560, 390, 74, 532, 38, 326
    plot_w, plot_h = x1 - x0, y1 - y0
    def px(x: float) -> float:
        return x0 + x * plot_w
    def py(y: float) -> float:
        return y1 - ((y + 1) / 2) * plot_h
    parts = [
        f"<svg class='comparison-chart' viewBox='0 0 {W} {H}' role='img' aria-label='{esc(title)}'>",
        f"<title>{esc(title)}</title>",
        f"<text class='chart-title' x='{W/2}' y='20' text-anchor='middle'>{esc(title)}</text>",
        f"<rect data-chart-frame='true' x='{x0}' y='{y0}' width='{plot_w}' height='{plot_h}' fill='none' stroke='var(--border)' />",
    ]
    for tick in [0, .25, .5, .75, 1]:
        x = px(tick)
        parts.append(f"<line class='chart-gridline' x1='{x}' y1='{y0}' x2='{x}' y2='{y1}' />")
        parts.append(f"<line class='chart-tick' x1='{x}' y1='{y1}' x2='{x}' y2='{y1+6}' />")
        parts.append(f"<text class='chart-tick-label' x='{x}' y='{y1+22}' text-anchor='middle'>{tick:g}</text>")
    for tick in [-1, -.5, 0, .5, 1]:
        y = py(tick)
        parts.append(f"<line class='chart-gridline' x1='{x0}' y1='{y}' x2='{x1}' y2='{y}' />")
        parts.append(f"<line class='chart-tick' x1='{x0-6}' y1='{y}' x2='{x0}' y2='{y}' />")
        parts.append(f"<text class='chart-tick-label' x='{x0-10}' y='{y+4}' text-anchor='end'>{tick:g}</text>")
    parts.append(f"<text class='chart-axis-title' data-axis='x' x='{(x0+x1)/2}' y='{H-16}' text-anchor='middle'>标准化难度（0–1）</text>")
    parts.append(f"<text class='chart-axis-title' data-axis='y' transform='rotate(-90)' x='{-((y0+y1)/2)}' y='18' text-anchor='middle'>校正题总相关 CITC（-1–1）</text>")
    for d in data:
        if not isinstance(d.get("x"), (int, float)) or not isinstance(d.get("y"), (int, float)):
            continue
        parts.append(
            f"<circle cx='{px(float(d['x'])):.2f}' cy='{py(float(d['y'])):.2f}' r='5' fill='{css_color}' fill-opacity='.78' stroke='var(--foreground)' stroke-width='.7'>"
            f"<title>{esc(series)} Row {esc(d['id'])}：难度={fmt(d['x'])}，CITC={fmt(d['y'])}，有效选项数={esc(d.get('eff'))}</title></circle>"
        )
    parts.append("</svg>")
    return "".join(parts)


def svg_effective_bars(current: list[dict[str, Any]], mussel: list[dict[str, Any]]) -> str:
    W, H, x0, x1, y0, y1 = 560, 390, 74, 532, 38, 326
    plot_w, plot_h = x1 - x0, y1 - y0
    counts = []
    for k in [1, 2, 3, 4]:
        counts.append((k, sum(d.get("eff") == k for d in current), sum(d[3] == k for d in mussel)))
    ymax = max(max(c[1], c[2]) for c in counts) or 1
    def py(v: float) -> float:
        return y1 - v / ymax * plot_h
    band = plot_w / 4
    parts = [
        f"<svg class='comparison-chart' viewBox='0 0 {W} {H}' role='img' aria-label='有效选项数分布对比'>",
        "<title>有效选项数分布对比</title>",
        f"<text class='chart-title' x='{W/2}' y='20' text-anchor='middle'>有效选项数分布（题目数量）</text>",
        f"<rect data-chart-frame='true' x='{x0}' y='{y0}' width='{plot_w}' height='{plot_h}' fill='none' stroke='var(--border)' />",
    ]
    for tick in range(0, ymax + 1, max(1, ymax // 5 or 1)):
        y = py(tick)
        parts.append(f"<line class='chart-gridline' x1='{x0}' y1='{y}' x2='{x1}' y2='{y}' />")
        parts.append(f"<text class='chart-tick-label' x='{x0-10}' y='{y+4}' text-anchor='end'>{tick}</text>")
    for i, k in enumerate([1, 2, 3, 4]):
        cx = x0 + band * (i + .5)
        parts.append(f"<text class='chart-tick-label' x='{cx}' y='{y1+22}' text-anchor='middle'>{k}</text>")
        bw = band * .27
        for offset, value, label, color in [(-bw, counts[i][1], '当前 SJT', 'var(--viz-series-1)'), (0, counts[i][2], 'Mussel', 'var(--viz-series-2)')]:
            x = cx + offset
            y = py(value)
            parts.append(f"<rect x='{x}' y='{y}' width='{bw}' height='{y1-y}' fill='{color}' fill-opacity='.78'><title>{label}：有效选项数={k}，题目数={value}</title></rect>")
            parts.append(f"<text class='chart-value-label' x='{x+bw/2}' y='{y-5}' text-anchor='middle'>{value}</text>")
    parts.append(f"<text class='chart-axis-title' data-axis='x' x='{(x0+x1)/2}' y='{H-16}' text-anchor='middle'>有效选项数（≥5%选择率的选项）</text>")
    parts.append(f"<text class='chart-axis-title' data-axis='y' transform='rotate(-90)' x='{-(y0+y1)/2}' y='18' text-anchor='middle'>题目数量</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_heatmap(data: list[dict[str, Any]], title: str, css_color: str) -> str:
    W, row_h, x0, x1, y0 = 560, 20, 74, 532, 42
    H = y0 + row_h * len(data) + 68
    cell_w = (x1 - x0) / 4
    parts = [
        f"<svg class='comparison-chart heatmap-chart' viewBox='0 0 {W} {H}' role='img' aria-label='{esc(title)}'>",
        f"<title>{esc(title)}</title>",
        f"<text class='chart-title' x='{W/2}' y='20' text-anchor='middle'>{esc(title)}</text>",
        f"<rect data-chart-frame='true' x='{x0}' y='{y0}' width='{x1-x0}' height='{row_h*len(data)}' fill='none' stroke='var(--border)' />",
    ]
    for i, oid in enumerate("ABCD"):
        x = x0 + cell_w * (i + .5)
        parts.append(f"<text class='chart-tick-label' x='{x}' y='{y0-10}' text-anchor='middle'>{oid}</text>")
    for r, d in enumerate(data):
        y = y0 + r * row_h
        parts.append(f"<text class='chart-tick-label' x='{x0-10}' y='{y+14}' text-anchor='end'>Row {esc(d.get('id'))}</text>")
        for i, value in enumerate(d.get("rates") or [0, 0, 0, 0]):
            x = x0 + i * cell_w
            opacity = .08 + .84 * max(0, min(1, float(value)))
            parts.append(f"<rect x='{x+1}' y='{y+1}' width='{cell_w-2}' height='{row_h-2}' fill='{css_color}' fill-opacity='{opacity:.3}'><title>Row {esc(d.get('id'))}，选项{'ABCD'[i]}：{float(value)*100:.1f}%</title></rect>")
            if value >= .10:
                parts.append(f"<text class='heat-label' x='{x+cell_w/2}' y='{y+14}' text-anchor='middle'>{float(value)*100:.0f}%</text>")
    parts.append(f"<text class='chart-axis-title' data-axis='x' x='{(x0+x1)/2}' y='{H-12}' text-anchor='middle'>选项</text>")
    parts.append(f"<text class='chart-axis-title' data-axis='y' transform='rotate(-90)' x='{-(y0+row_h*len(data)/2)}' y='18' text-anchor='middle'>题号</text>")
    parts.append("</svg>")
    return "".join(parts)


def comparison_charts_html(run: dict[str, Any]) -> str:
    current = current_comparison_data(run)
    mussel = [
        {"id": f"{row[0]:02d}", "x": row[1], "y": row[2], "eff": row[3], "rates": row[4]}
        for row in MUSSEL_GREGARIOUSNESS
    ]
    return (
        "<h2>三、Mussel Gregariousness 对比图（坐标修正版）</h2>"
        "<p class='muted'>当前 SJT 数据来自最近一次外向性—群居性运行的题项统计；Mussel 数据沿用原对比图中的22道题记录。横轴和纵轴均明确标注，点的标题提示包含题号和数值。</p>"
        "<div class='comparison-chart-grid'>"
        f"<div>{svg_scatter(current, '当前 SJT：难度—CITC', '当前 SJT', 'var(--viz-series-1)')}</div>"
        f"<div>{svg_scatter(mussel, 'Mussel：难度—CITC', 'Mussel', 'var(--viz-series-2)')}</div>"
        f"<div>{svg_effective_bars(current, MUSSEL_GREGARIOUSNESS)}</div>"
        f"<div>{svg_heatmap(current, '当前 SJT：选项选择率', 'var(--viz-series-1)')}</div>"
        f"<div>{svg_heatmap(mussel, 'Mussel：选项选择率', 'var(--viz-series-2)')}</div>"
        "</div>"
        "<div class='chart-legend'><span><i style='background:var(--viz-series-1)'></i>当前 SJT</span><span><i style='background:var(--viz-series-2)'></i>Mussel</span></div>"
    )


def item_citc(stats: dict[str, Any]) -> Any:
    for key in ("facet_corrected_item_total_correlation", "corrected_item_total_correlation"):
        block = stats.get(key)
        if isinstance(block, dict):
            return block.get("r")
    return None


def item_rec(stats: dict[str, Any]) -> Any:
    return (stats.get("quality_evaluation") or {}).get("recommendation")


def disposition_for(report: dict[str, Any], item_id: str) -> dict[str, Any]:
    return (report.get("item_final_dispositions") or {}).get(item_id, {})


def run_payload(root: Path) -> dict[str, Any]:
    technical = read_json(root / "technical_report.json")
    database = read_json(root / "item_database.json")
    virtual = read_json(root / "virtual_respondent_report.json")
    items = database.get("items", [])
    stats_map = technical.get("item_statistics") or {}
    for item in items:
        embedded = item.get("psychometric_statistics") or {}
        item["_stats"] = stats_map.get(item.get("item_id"), embedded)
    return {
        "root": root,
        "technical": technical,
        "database": database,
        "virtual": virtual,
        "items": items,
    }


def construct_html(run: dict[str, Any]) -> str:
    t = run["technical"]
    profile = t.get("construct_profile") or {}
    ref = t.get("construct_profile_ref") or {}
    facets = profile.get("facets") or []
    chunks = [
        "<h3>构念与分面</h3>",
        "<div class='construct-grid'>",
        f"<div><b>量表</b><br>{esc(ref.get('inventory_name'))}</div>",
        f"<div><b>选择层级</b><br>{esc(ref.get('selection_level'))}</div>",
        f"<div><b>域</b><br>{esc(ref.get('domain_name'))} ({esc(ref.get('domain_id'))})</div>",
        f"<div><b>目标人群</b><br>{esc((t.get('test_specification') or {}).get('target_population'))}</div>",
        "</div>",
    ]
    for facet in facets:
        evidence = facet.get("behavior_evidence") or []
        chunks.append("<section class='facet-card'>")
        chunks.append(
            f"<h4>{esc(facet.get('facet_name'))} <code>{esc(facet.get('facet_id'))}</code></h4>"
        )
        chunks.append(
            "<dl class='definition'><dt>定义</dt><dd>"
            + esc(facet.get("definition"))
            + "</dd><dt>高水平表现</dt><dd>"
            + esc(facet.get("high_behavior"))
            + "</dd><dt>低水平表现</dt><dd>"
            + esc(facet.get("low_behavior"))
            + "</dd></dl>"
        )
        if facet.get("common_confounds"):
            chunks.append("<p><b>常见混淆：</b>" + esc("；".join(facet["common_confounds"])) + "</p>")
        if facet.get("inappropriate_conditions"):
            chunks.append("<p><b>不适用情境：</b>" + esc("；".join(facet["inappropriate_conditions"])) + "</p>")
        if facet.get("forbidden_patterns"):
            chunks.append("<p><b>禁用模式：</b>" + esc("；".join(facet["forbidden_patterns"])) + "</p>")
        chunks.append("<h5>Behavior Evidence</h5><div class='evidence-grid'>")
        for ev in evidence:
            chunks.append(
                "<article class='evidence-card'>"
                f"<h6>{esc(ev.get('behavior_id'))} · {esc(ev.get('behavior_dimension'))}</h6>"
                f"<p><b>可观察行为：</b>{esc(ev.get('observable_behavior'))}</p>"
                f"<p><b>高水平：</b>{esc(ev.get('high_expression'))}</p>"
                f"<p><b>低水平：</b>{esc(ev.get('low_expression'))}</p>"
                f"<p><b>边界：</b>{esc(ev.get('boundary_condition'))}</p>"
                f"<p><b>来源条目：</b>{esc(', '.join(ev.get('source_item_ids') or []))}</p>"
                "</article>"
            )
        chunks.append("</div></section>")
    return "\n".join(chunks)


def item_table_html(run: dict[str, Any]) -> str:
    t = run["technical"]
    rows = []
    for index, item in enumerate(run["items"], 1):
        stats = item.get("_stats") or {}
        q = stats.get("quality_evaluation") or {}
        disp = disposition_for(t, item.get("item_id"))
        rows.append(
            "<tr>"
            f"<td>Row {index:02d}</td><td>{esc(item.get('target_dimension_id'))}</td>"
            f"<td>{fmt(item_citc(stats))}</td><td>{fmt(stats.get('difficulty'))}</td>"
            f"<td>{esc(stats.get('effective_option_count', q.get('effective_option_count')))}</td>"
            f"<td>{esc(item_rec(stats))}</td><td>{esc(disp.get('status') or item.get('final_status'))}</td>"
            f"<td>{esc(disp.get('warning_reason'))}</td>"
            "</tr>"
        )
    return (
        "<h3>题项测量摘要</h3>"
        "<p class='muted'>difficulty 为当前报告中的标准化平均得分；CITC优先读取分面内校正题总相关。第三次运行使用旧版统计字段，但本页按其真实字段读取。</p>"
        "<div class='table-wrap'><table><thead><tr><th>题号</th><th>目标维度</th><th>CITC</th><th>难度</th><th>有效选项数</th><th>统计建议</th><th>最终状态</th><th>警告</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def item_details_html(run: dict[str, Any]) -> str:
    t = run["technical"]
    chunks = ["<h3>完整题目与选项</h3>"]
    for index, item in enumerate(run["items"], 1):
        stats = item.get("_stats") or {}
        q = stats.get("quality_evaluation") or {}
        disp = disposition_for(t, item.get("item_id"))
        options = item.get("response_options") or []
        option_rows = []
        opt_stats = stats.get("option_statistics") or {}
        scores = item.get("scoring_key") or {}
        for option in options:
            oid = option.get("option_id")
            os = opt_stats.get(oid) or {}
            option_rows.append(
                "<tr>"
                f"<td>{esc(oid)}</td><td>{esc(option.get('behavioral_level'))}</td><td>{esc(scores.get(oid))}</td>"
                f"<td>{esc(option.get('text'))}</td><td>{fmt(os.get('count'), 0)}</td><td>{fmt(os.get('rate'))}</td>"
                "</tr>"
            )
        chunks.append(
            "<details class='item-card'><summary>"
            f"Row {index:02d} · {esc(item.get('target_dimension_id'))} · CITC {fmt(item_citc(stats))} · 难度 {fmt(stats.get('difficulty'))} · {esc(item_rec(stats))}"
            "</summary><div class='item-body'>"
            f"<p><b>item_id：</b><code>{esc(item.get('item_id'))}</code><br>"
            f"<b>版本：</b>{esc(item.get('version'))}　<b>最终状态：</b>{esc(disp.get('status') or item.get('final_status'))}　"
            f"<b>警告：</b>{esc(disp.get('warning_reason'))}</p>"
            f"<p><b>情境类别：</b>{esc(item.get('context_category'))}</p>"
            f"<p><b>情境：</b>{esc(item.get('scenario'))}</p>"
            f"<p><b>作答要求：</b>{esc(item.get('response_instruction'))}</p>"
            f"<p><b>构念理由：</b>{esc(item.get('construct_rationale'))}</p>"
            "<div class='table-wrap'><table><thead><tr><th>选项</th><th>行为等级</th><th>计分</th><th>文本</th><th>选择人数</th><th>选择率</th></tr></thead><tbody>"
            + "".join(option_rows)
            + "</tbody></table></div>"
            f"<p><b>统计：</b>样本量 {esc(stats.get('n'))}；均值 {fmt(stats.get('mean'))}；标准差 {fmt(stats.get('standard_deviation'))}；有效选项数 {esc(stats.get('effective_option_count', q.get('effective_option_count')))}；诊断标记 {esc('；'.join(q.get('diagnostic_flags') or []))}</p>"
            "<details><summary>展开该题完整统计 JSON</summary><pre class='raw-json'>"
            + esc(json.dumps(stats, ensure_ascii=False, indent=2))
            + "</pre></details>"
            "</div></details>"
        )
    return "\n".join(chunks)


def run_html(run: dict[str, Any], ordinal: int) -> str:
    t = run["technical"]
    v = run["virtual"]
    spec = t.get("test_specification") or {}
    ts = t.get("test_statistics") or {}
    ev = (ts.get("convergent_validity") or {})
    rho = ev.get("sjt_total_by_neo_dimension") or {}
    crit = (ev.get("criterion") or {}).get("neo_ffi_dimension")
    target = rho.get(crit, {}).get("rho") if crit else None
    non_targets = {k: val.get("rho") for k, val in rho.items() if k != crit and isinstance(val, dict) and isinstance(val.get("rho"), (int, float))}
    max_non = max(non_targets.items(), key=lambda x: abs(x[1])) if non_targets else (None, None)
    margin = (ts.get("measurement_evaluation") or {}).get("validity", {}).get("discriminant", {}).get("target_margin")
    recs = {}
    for item in run["items"]:
        rec = item_rec(item.get("_stats") or {}) or "unknown"
        recs[rec] = recs.get(rec, 0) + 1
    accepted = sum(1 for item in run["items"] if disposition_for(t, item.get("item_id")).get("status") == "accepted")
    warning = sum(1 for item in run["items"] if disposition_for(t, item.get("item_id")).get("status") == "accepted_with_warning")
    generated = t.get("generated_at") or ""
    return (
        f"<section class='run-section'><h2>运行 {ordinal}：{esc(t.get('run_id'))}</h2>"
        f"<p class='run-meta'><b>生成时间：</b>{esc(generated)}　<b>题库：</b>{esc(t.get('item_bank_id'))}　<b>版本：</b>{esc(t.get('item_bank_version'))}</p>"
        "<div class='metric-grid'>"
        f"<div><b>目标构念</b><br>{esc((t.get('construct_profile_ref') or {}).get('domain_name'))} / {esc(', '.join((t.get('construct_profile_ref') or {}).get('selected_facet_ids') or []))}</div>"
        f"<div><b>题目数</b><br>{esc(len(run['items']))}</div><div><b>虚拟样本</b><br>{esc(v.get('respondent_count'))}</div>"
        f"<div><b>Cronbach α</b><br>{fmt(ts.get('cronbach_alpha'))}</div><div><b>目标 Neo-FFI {esc(crit)} 相关</b><br>{fmt(target)}</div>"
        f"<div><b>最大非目标相关</b><br>{esc(max_non[0])}: {fmt(max_non[1])}</div><div><b>目标—非目标差值</b><br>{fmt(margin)}</div>"
        f"<div><b>统计建议</b><br>{esc(recs)}</div><div><b>最终状态</b><br>accepted {accepted}；warning {warning}</div>"
        "</div>"
        + construct_html(run)
        + item_table_html(run)
        + item_details_html(run)
        + "</section>"
    )


SUMMARY_PROMPT_ZH = """你需要根据虚拟被试逐题问卷作答，总结其反复出现的行为反应模式。只能使用提供的题目陈述和作答标签。请区分直接观察到的作答模式与更宽泛的推断，不得做出超出回答证据支持范围的推断。请描述该个体在什么情况下倾向于采取某种反应、不同情境之间有意义的张力，以及仍存在的不确定性。保留回答中的矛盾，不要强行塑造一个完全一致的人格画像。不要命名或复述潜在人格特质、分面、目标构念或笼统的人格标签；不要做价值评价。不要虚构人口统计信息、个人经历、动机、能力、资源、诊断、生活事件或因果解释。不要提及问卷名称、题目代码、数字分数、计分方向或心理测量术语。使用“倾向于”“可能”“在某些情况下”等校准性表达。用简体中文写一个100—180字的紧凑单段落。只返回JSON：{\"summary\":\"...\"}."""


def real_summary_examples(runs: list[dict[str, Any]]) -> str:
    chunks = []
    for run in runs[:1]:
        virtual = run["virtual"]
        path_text = ((virtual.get("response_summary") or {}).get("persona_summary_path"))
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= 2:
                break
        chunks.append(f"运行 {run['technical'].get('run_id')}：{path_text}")
        for record in records:
            chunks.append(
                f"被试ID：{record.get('respondent_id')}\n"
                f"模型：{record.get('model_id')}；提示词版本：{record.get('prompt_version')}；"
                f"来源：{record.get('response_source', '本次生成')}\n"
                f"真实保存的summary：{record.get('summary')}"
            )
    return "\n\n".join(chunks) or "未找到已保存的人格总结文件。"


def real_sjt_example(runs: list[dict[str, Any]]) -> str:
    """Reconstruct one complete, real SJT call from persisted run data."""
    if not runs:
        return "未找到可重建的真实运行。"
    run = runs[0]
    technical = run["technical"]
    virtual = run["virtual"]
    summary_path_text = ((virtual.get("response_summary") or {}).get("persona_summary_path"))
    if not summary_path_text:
        return "未找到人格总结保存路径。"
    summary_path = Path(summary_path_text)
    if not summary_path.exists() or not run["items"]:
        return "未找到人格总结或题目文件。"
    summary_record = json.loads(summary_path.read_text(encoding="utf-8").splitlines()[0])
    respondent_id = summary_record.get("respondent_id")
    pool = read_json(ROOT / "sjt_system" / "data" / "virtual_respondents.json")
    respondent = next(
        (r for r in pool.get("respondents", []) if r.get("respondent_id") == respondent_id),
        None,
    )
    if not respondent:
        return "未找到对应虚拟被试的人格逐题作答。"
    scale = pool.get("response_scale") or {}
    personality_lines = []
    for item, value in zip(pool.get("items", []), respondent.get("response_values", [])):
        personality_lines.append(f"{item.get('statement')}：{scale.get(str(value), value)}")
    item = run["items"][0]
    sjt_path = Path(((virtual.get("response_summary") or {}).get("sjt_path", "")))
    response_record = None
    if sjt_path.exists():
        for line in sjt_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("respondent_id") == respondent_id and record.get("item_id") == item.get("item_id"):
                response_record = record
                break
    option_lines = [f"{o.get('option_id')}. {o.get('text')}" for o in item.get("response_options", [])]
    return "\n".join(
        [
            "【真实运行示例】",
            f"运行ID：{technical.get('run_id')}",
            f"被试ID：{respondent_id}",
            f"模型：{summary_record.get('model_id')}；提示词版本：{(response_record or {}).get('prompt_version', summary_record.get('prompt_version'))}",
            "",
            "人格提示（实际输入）：",
            "请想象你正在扮演一个特定的人。",
            "下面是这个人的人格逐题作答。每项作答使用五级：非常不同意、比较不同意、不确定、比较同意、非常同意。",
            "这些逐题作答是判断此人人格的主要且权威的依据。",
            "",
            "\n".join(personality_lines),
            "",
            "补充人格总结：",
            str(summary_record.get("summary")),
            "该总结只用于帮助整合上述逐题作答。若总结与逐题作答发生冲突，应以逐题作答为准；不要据此补充任何未提供的个人背景。",
            "",
            "SJT作答指令（实际输入）：",
            "现在请以这个人的真实行为倾向回答一道情境判断题。请选择此人最可能采取的行为，而不是理论上最好、最正确或社会赞许程度最高的行为。每次作答相互独立。只返回一个JSON对象，不要解释，格式为：{\"selected_option_id\":\"选项编号\"}。",
            "",
            f"情境：\n{item.get('scenario')}",
            f"作答要求：\n{item.get('response_instruction')}",
            "选项：\n" + "\n".join(option_lines),
            "",
            f"实际保存的模型输出：{{\"selected_option_id\":\"{(response_record or {}).get('selected_option_id', '未找到')}\"}}",
        ]
    )


PERSONA_PROMPT = """请想象你正在扮演一个特定的人。
下面是这个人的人格逐题作答。每项作答使用五级：非常不同意、比较不同意、不确定、比较同意、非常同意。
这些逐题作答是判断此人人格的主要且权威的依据。

{personality_items}

补充人格总结：
{personality_summary}
该总结只用于帮助整合上述逐题作答。若总结与逐题作答发生冲突，应以逐题作答为准；不要据此补充任何未提供的个人背景。"""


def build_html(runs: list[dict[str, Any]]) -> str:
    generated = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    comparison = []
    for run in runs:
        t = run["technical"]
        ts = t.get("test_statistics") or {}
        ev = ts.get("convergent_validity") or {}
        criterion = (ev.get("criterion") or {}).get("neo_ffi_dimension")
        rho = (ev.get("sjt_total_by_neo_dimension") or {}).get(criterion, {}).get("rho") if criterion else None
        comparison.append(
            "<tr>"
            f"<td>{esc(t.get('run_id'))}</td><td>{esc((t.get('construct_profile_ref') or {}).get('domain_name'))}</td>"
            f"<td>{esc(', '.join((t.get('construct_profile_ref') or {}).get('selected_facet_ids') or []))}</td>"
            f"<td>{esc(len(run['items']))}</td><td>{esc((run['virtual'] or {}).get('respondent_count'))}</td>"
            f"<td>{fmt(ts.get('cronbach_alpha'))}</td><td>Neo-FFI {esc(criterion)}: {fmt(rho)}</td>"
            "</tr>"
        )
    run_sections = "\n".join(run_html(run, i + 1) for i, run in enumerate(runs))
    agent_flow = """
<h2>二、五 Agent 流程规划</h2>
<p class="lead">这里按职责划分五个 Agent。它们是功能角色，不要求一定使用五个不同模型；程序负责状态、引用、版本和统计，Agent负责对应阶段的语言任务。</p>
<h3>共享状态与调度关系</h3>
<div class="architecture-diagram" role="img" aria-label="用户、共享状态、五个 Agent 与确定性工作流控制器的关系">
<svg viewBox="0 0 1180 570" xmlns="http://www.w3.org/2000/svg">
  <defs><marker id="architecture-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--accent)"/></marker><marker id="architecture-arrow-muted" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="var(--muted)"/></marker></defs>
  <g class="architecture-node user"><rect x="475" y="18" width="230" height="55"/><text x="590" y="51" text-anchor="middle">用户输入</text></g>
  <path class="architecture-arrow" d="M590 73 V104" marker-end="url(#architecture-arrow)"/>
  <g class="architecture-node shared"><rect x="70" y="105" width="1040" height="130"/><text x="590" y="134" text-anchor="middle" class="architecture-title">共享状态</text><text x="590" y="164" text-anchor="middle">测验规格 · 构念模型 · Behavior Evidence · Expansion</text><text x="590" y="189" text-anchor="middle">双向细目表 · 候选题库 · 验证报告 · 流程状态</text><text x="590" y="216" text-anchor="middle" class="architecture-note">逻辑上集中保存，实际按 Agent 契约限制读写范围</text></g>
  <text class="architecture-label" x="590" y="268" text-anchor="middle">各 Agent 读取共享状态中的必要数据，并写回自己的阶段产物</text>
  <path class="architecture-arrow-muted" d="M145 300 V235" marker-start="url(#architecture-arrow-muted)" marker-end="url(#architecture-arrow-muted)"/><path class="architecture-arrow-muted" d="M355 300 V235" marker-start="url(#architecture-arrow-muted)" marker-end="url(#architecture-arrow-muted)"/><path class="architecture-arrow-muted" d="M565 300 V235" marker-start="url(#architecture-arrow-muted)" marker-end="url(#architecture-arrow-muted)"/><path class="architecture-arrow-muted" d="M775 300 V235" marker-start="url(#architecture-arrow-muted)" marker-end="url(#architecture-arrow-muted)"/><path class="architecture-arrow-muted" d="M985 300 V235" marker-start="url(#architecture-arrow-muted)" marker-end="url(#architecture-arrow-muted)"/>
  <g class="architecture-node agent planning"><rect x="45" y="300" width="200" height="88"/><text x="145" y="327" text-anchor="middle" class="architecture-title">规划 Agent</text><text x="145" y="352" text-anchor="middle">规格 · 构念 · 蓝图</text><text x="145" y="374" text-anchor="middle" class="architecture-note">读写规划产物</text></g>
  <g class="architecture-node agent writing"><rect x="255" y="300" width="200" height="88"/><text x="355" y="327" text-anchor="middle" class="architecture-title">出题 Agent</text><text x="355" y="352" text-anchor="middle">生成 / 原子修复题目</text><text x="355" y="374" text-anchor="middle" class="architecture-note">只写当前题目版本</text></g>
  <g class="architecture-node agent review"><rect x="465" y="300" width="200" height="88"/><text x="565" y="327" text-anchor="middle" class="architecture-title">审查 Agent</text><text x="565" y="352" text-anchor="middle">审查 / 诊断局部问题</text><text x="565" y="374" text-anchor="middle" class="architecture-note">写 Review / RepairAdvice</text></g>
  <g class="architecture-node agent simulation"><rect x="675" y="300" width="200" height="88"/><text x="775" y="327" text-anchor="middle" class="architecture-title">施测 Agent</text><text x="775" y="352" text-anchor="middle">虚拟被试作答</text><text x="775" y="374" text-anchor="middle" class="architecture-note">只写作答矩阵</text></g>
  <g class="architecture-node agent validation"><rect x="885" y="300" width="200" height="88"/><text x="985" y="327" text-anchor="middle" class="architecture-title">验证 Agent</text><text x="985" y="352" text-anchor="middle">指标 / 状态 / 报告</text><text x="985" y="374" text-anchor="middle" class="architecture-note">写验证结果</text></g>
  <path class="architecture-arrow" d="M985 388 V446" marker-end="url(#architecture-arrow)"/><text class="architecture-label" x="1010" y="420" text-anchor="middle">验证结果</text>
  <g class="architecture-node controller"><rect x="300" y="446" width="580" height="72"/><text x="590" y="475" text-anchor="middle" class="architecture-title">确定性工作流控制器</text><text x="590" y="500" text-anchor="middle">按状态调度 Agent：下一题 · 原子修复 · 重测 · 冻结组卷</text></g>
  <path class="architecture-arrow" d="M300 482 H180 V235 H70" marker-end="url(#architecture-arrow)"/><text class="architecture-label" x="180" y="432" text-anchor="middle">回写状态 / 调度下一步</text>
</svg></div>
<div class="agent-flow">
  <div class="agent-node planning"><div class="agent-no">01</div><h3>规划 Agent</h3><p>明确需求、目标人群和目标 facet；读取构念档案与 Behavior Evidence；生成 Behavior Expansion、蓝图和题量分配。</p><small>输出：RequirementSpec、ConstructProfile、Behavior Evidence、Expansion、Blueprint</small></div>
  <div class="flow-arrow">→</div>
  <div class="agent-node writing"><div class="agent-no">02</div><h3>出题 Agent</h3><p>根据蓝图引用解析 Mechanism、Situation 和 Skeleton，把单题设计实现为情境、问题和四个行为选项。</p><small>输出：ItemDesign、题目版本、behavioral_level、程序派生 scoring_key</small></div>
  <div class="flow-arrow">→</div>
  <div class="agent-node review"><div class="agent-no">03</div><h3>审查 Agent</h3><p>进行首轮构念、内容、社会赞许性、答案泄露和选项结构审查；对异常题生成基于正常模型与观察证据的诊断意见。</p><small>输出：Review、AtomicRepairAdvice 或 defer</small></div>
  <div class="flow-arrow">→</div>
  <div class="agent-node simulation"><div class="agent-no">04</div><h3>施测 Agent</h3><p>让固定虚拟被试独立回答 SJT；复用未变题作答和人格总结，修改题只重新施测变更题。</p><small>输出：persona_summaries、SJT responses、Neo-FFI responses</small></div>
  <div class="flow-arrow">→</div>
  <div class="agent-node validation"><div class="agent-no">05</div><h3>验证 Agent</h3><p>汇总确定性题项指标和测验指标，判断是否需要诊断；确认题库覆盖、版本新鲜度和最终状态后组卷报告。</p><small>输出：PsychometricReport、accepted / accepted_with_warning、Final Bank</small></div>
</div>
"""
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>最近三次 PSJT 运行汇总</title>
<style>
:root {{ --ink:#1f2937; --muted:#64748b; --line:#dbe3ee; --panel:#fff; --bg:#f5f7fb; --accent:#2563eb; --soft:#eef4ff; --foreground:#1f2937; --border:#dbe3ee; --viz-series-1:#2563eb; --viz-series-2:#d97706; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; line-height:1.65; }}
main {{ max-width:1500px; margin:0 auto; padding:28px 34px 60px; }} h1 {{ margin:0 0 4px; font-size:30px; }} h2 {{ margin-top:38px; border-bottom:2px solid var(--accent); padding-bottom:8px; }} h3 {{ margin-top:26px; }} h4 {{ margin:8px 0; }} h5 {{ margin:18px 0 10px; }} h6 {{ margin:0 0 8px; font-size:15px; }}
.lead,.muted,.run-meta {{ color:var(--muted); }} .note {{ background:#fffbea; border-left:4px solid #eab308; padding:12px 16px; margin:18px 0; }}
.table-wrap {{ overflow-x:auto; }} table {{ border-collapse:collapse; width:100%; background:var(--panel); font-size:14px; }} th,td {{ border:1px solid var(--line); padding:8px 10px; vertical-align:top; text-align:left; }} th {{ background:#edf2f7; white-space:nowrap; }}
.metric-grid,.construct-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin:16px 0; }} .metric-grid>div,.construct-grid>div {{ background:var(--soft); border:1px solid #d9e6ff; border-radius:8px; padding:10px 12px; }}
.agent-flow {{ display:flex; align-items:stretch; gap:8px; margin:18px 0; }} .agent-node {{ flex:1; min-width:170px; border:1px solid var(--line); border-top:5px solid var(--accent); border-radius:10px; background:var(--panel); padding:12px 14px; box-shadow:0 2px 7px rgba(15,23,42,.05); }} .agent-node.writing {{ border-top-color:#7c3aed; }} .agent-node.review {{ border-top-color:#d97706; }} .agent-node.simulation {{ border-top-color:#059669; }} .agent-node.validation {{ border-top-color:#dc2626; }} .agent-no {{ color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.08em; }} .agent-node h3 {{ margin:3px 0 8px; }} .agent-node p {{ margin:0 0 8px; font-size:14px; }} .agent-node small {{ display:block; color:var(--muted); font-size:12px; line-height:1.45; }} .flow-arrow {{ align-self:center; color:var(--accent); font-size:26px; font-weight:700; }} .architecture-diagram {{ width:100%; overflow-x:auto; margin:14px 0 8px; background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:8px; }} .architecture-diagram svg {{ display:block; width:100%; min-width:760px; height:auto; }} .architecture-node rect {{ fill:var(--panel); stroke:var(--border); stroke-width:1.2; rx:9; }} .architecture-node.shared rect {{ fill:var(--soft); stroke:#b8cff9; }} .architecture-node.controller rect {{ fill:#fffbea; stroke:#f3d477; }} .architecture-node.user rect {{ fill:#eefcf5; stroke:#9bd8b5; }} .architecture-node.agent.planning rect {{ stroke:#2563eb; }} .architecture-node.agent.writing rect {{ stroke:#7c3aed; }} .architecture-node.agent.review rect {{ stroke:#d97706; }} .architecture-node.agent.simulation rect {{ stroke:#059669; }} .architecture-node.agent.validation rect {{ stroke:#dc2626; }} .architecture-node text {{ fill:var(--foreground); font-size:13px; }} .architecture-node .architecture-title {{ font-weight:700; }} .architecture-node .architecture-note,.architecture-label {{ fill:var(--muted); font-size:12px; }} .architecture-arrow {{ fill:none; stroke:var(--accent); stroke-width:1.8; }} .architecture-arrow-muted {{ fill:none; stroke:var(--muted); stroke-opacity:.7; stroke-width:1.3; }}
.comparison-chart-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; margin:16px 0; }} .comparison-chart {{ width:100%; height:auto; display:block; }} .chart-title,.chart-axis-title,.chart-tick-label,.chart-value-label,.heat-label {{ fill:var(--foreground); }} .chart-title {{ font-size:15px; font-weight:600; }} .chart-axis-title {{ font-size:12px; }} .chart-tick-label,.chart-value-label,.heat-label {{ font-size:11px; }} .chart-gridline {{ stroke:var(--border); stroke-opacity:.55; stroke-width:1; }} .chart-tick {{ stroke:var(--foreground); stroke-opacity:.65; }} .chart-legend {{ display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:var(--muted); margin:4px 0 22px; }} .chart-legend i {{ display:inline-block; width:12px; height:12px; margin-right:5px; vertical-align:-1px; border-radius:2px; }}
.facet-card,.evidence-card,.item-card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:12px 0; }} .evidence-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:12px; }} .evidence-card {{ margin:0; }}
dl.definition {{ display:grid; grid-template-columns:120px 1fr; margin:10px 0; }} dl.definition dt {{ font-weight:700; color:#475569; }} dl.definition dd {{ margin:0 0 4px; }}
details.item-card {{ padding:0; overflow:hidden; }} details.item-card summary {{ cursor:pointer; padding:13px 16px; font-weight:700; background:#fafcff; }} .item-body {{ padding:14px 16px; border-top:1px solid var(--line); }}
code,pre {{ font-family:ui-monospace,SFMono-Regular,Consolas,monospace; }} code {{ background:#eef2f7; padding:2px 5px; border-radius:4px; font-size:12px; }} pre.prompt {{ white-space:pre-wrap; background:#111827; color:#e5e7eb; padding:18px; border-radius:8px; overflow:auto; font-size:13px; line-height:1.6; }}
.raw-json {{ white-space:pre-wrap; background:#f8fafc; border:1px solid var(--line); padding:12px; border-radius:6px; overflow:auto; font-size:12px; line-height:1.45; }}
.source {{ font-size:13px; color:var(--muted); }} @media(max-width:700px) {{ main {{ padding:18px 12px 40px; }} h1 {{ font-size:24px; }} dl.definition {{ display:block; }} dl.definition dt {{ margin-top:8px; }} .agent-flow {{ flex-direction:column; }} .flow-arrow {{ transform:rotate(90deg); align-self:center; }} .comparison-chart-grid {{ grid-template-columns:1fr; gap:14px; }} }}
</style></head><body><main>
<h1>最近三次 PSJT 运行汇总</h1>
<p class="lead">生成时间：{esc(generated)}。本页按交付目录最近三次已完成运行读取真实 JSON，不补造缺失数据；题目详情可展开查看。</p>
<div class="note"><b>解释边界：</b>虚拟被试结果属于开发期模拟证据，不能替代真实被试效度。第三次运行是神经质 domain 级旧版本结果，题数和统计字段与后两次 facet 级运行不同。</div>
<h2>一、三次运行对比</h2>
<div class="table-wrap"><table><thead><tr><th>运行ID</th><th>域</th><th>目标分面</th><th>题数</th><th>虚拟样本</th><th>α</th><th>目标效标相关</th></tr></thead><tbody>{''.join(comparison)}</tbody></table></div>
{agent_flow}
{comparison_charts_html(runs[0])}
<h2>四、逐次运行详情</h2>
{run_sections}
<h2>五、当前虚拟被试提示词</h2>
<p class="source">来源：<code>sjt_system/evaluation/simulation.py</code>；人格总结逻辑约在 129–156 行，summary_plus_items 人格提示约在 159–190 行，SJT单题作答提示约在 193–226 行。以下是当前模板；运行时会把逐题人格作答、人格总结、情境和选项填入占位符。</p>
<h3>1. 人格总结生成提示词（中文翻译）</h3><pre class="prompt">{esc(SUMMARY_PROMPT_ZH)}</pre>
<h3>2. 真实运行中保存的人格总结（保留2条）</h3><pre class="prompt">{esc(real_summary_examples(runs))}</pre>
<h3>3. summary_plus_items 人格提示词模板</h3><pre class="prompt">{esc(PERSONA_PROMPT)}</pre>
<h3>4. 一个完整的真实 SJT 作答示例</h3><pre class="prompt">{esc(real_sjt_example(runs))}</pre>
</main></body></html>"""


def main() -> None:
    runs = [run_payload(path) for path in RUN_DIRS if (path / "technical_report.json").exists()]
    if len(runs) != 3:
        raise SystemExit(f"expected 3 runs, found {len(runs)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(build_html(runs), encoding="utf-8")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
