"""Generate a self-contained HTML report of meaningful validation results.

Only decision-relevant results are included:
- 30-class blind classification (5-class results are superseded; kept in JSON)
- Misclassification details for the two production banks (actionable)
- Per-item trait-response rho (anomalies highlighted)
"""
from __future__ import annotations

import html as html_mod
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blind_classification.catalog import build_catalog
from blind_classification.classify import (
    build_classification_prompt,
    facet_ids_for,
    load_gold_items,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TMP_DIR = Path(__file__).resolve().parents[1] / "tmp"
MIXED_EVALUATION_PATH = (
    RESULTS_DIR / "mixed_mussel_compliance_gregariousness_c30_evaluation.json"
)

SHORT_ZH = {
    "agreeableness_compliance": "A4 顺从",
    "agreeableness_straightforwardness": "A2 坦诚",
    "agreeableness_altruism": "A3 利他",
    "conscientiousness_self_discipline": "C5 自律",
    "extraversion_gregariousness": "E2 乐群",
    "neuroticism_self_consciousness": "N4 自我意识",
    "openness_ideas": "O5 观念",
}


def facet_zh(facet_id: str) -> str:
    if facet_id in SHORT_ZH:
        return SHORT_ZH[facet_id]
    domain = facet_id.split("_")[0]
    return f"{domain}·{facet_id.split('_')[-1]}"


def load_results(catalog_size: int) -> list[dict]:
    out = []
    for path in sorted(RESULTS_DIR.glob("*_classifications.json")):
        stem = path.stem[: -len("_classifications")]
        if "smoke" in stem:
            continue
        if catalog_size == 30 and not stem.endswith("_c30"):
            continue
        if catalog_size == 5 and stem.endswith("_c30"):
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        results = d["results"]
        valid = [r for r in results if r["predicted_facet"] is not None]
        hits = [r for r in valid if r["predicted_facet"] == r["true_facet"]]
        n_valid = len(valid)
        domain_valid = domain_hits = 0
        for r in valid:
            if "_" in str(r["true_facet"]) and "_" in str(r["predicted_facet"]):
                domain_valid += 1
                if r["true_facet"].split("_")[0] == r["predicted_facet"].split("_")[0]:
                    domain_hits += 1
        confusion: dict[tuple[str, str], int] = {}
        for r in valid:
            pair = (r["true_facet"], r["predicted_facet"])
            confusion[pair] = confusion.get(pair, 0) + 1
        out.append(
            {
                "label": d["manifest"]["label"],
                "catalog_size": catalog_size,
                "n_valid": n_valid,
                "n_total": len(results),
                "accuracy": len(hits) / n_valid if n_valid else 0.0,
                "domain_accuracy": domain_hits / domain_valid if domain_valid else None,
                "confusion": confusion,
                "errors": [
                    r for r in valid if r["predicted_facet"] != r["true_facet"]
                ],
            }
        )
    return out


def confusion_heatmap(entry: dict) -> str:
    facet_ids = sorted({f for pair in entry["confusion"] for f in pair})
    total = sum(entry["confusion"].values())
    rows = ['<table class="hm">']
    rows.append(
        "<tr><th>真值 \\ 预测</th>"
        + "".join(f'<th title="{fid}">{facet_zh(fid)}</th>' for fid in facet_ids)
        + "</tr>"
    )
    for tf in facet_ids:
        cells = [f"<td class='hm-label'>{facet_zh(tf)}</td>"]
        for pf in facet_ids:
            count = entry["confusion"].get((tf, pf), 0)
            if count == 0:
                cells.append('<td class="hm-cell" style="background:#f5f5f5">·</td>')
            else:
                intensity = min(1.0, count / max(1, total) * 8)
                color = (
                    f"rgba(34,139,34,{0.15 + intensity * 0.85})"
                    if tf == pf
                    else f"rgba(220,80,60,{0.12 + intensity * 0.88})"
                )
                cells.append(
                    f'<td class="hm-cell" style="background:{color}" '
                    f'title="{tf} → {pf}: {count}">{count}</td>'
                )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    rows.append("</table>")
    return "".join(rows)


def accuracy_table(entries: list[dict]) -> str:
    rows = ['<table class="styled">']
    rows.append(
        "<tr><th>分类对象</th><th>目录</th><th>有效题数</th><th>top-1 准确率</th>"
        "<th>随机基线</th><th>域级准确率</th></tr>"
    )
    for e in entries:
        if "gold" in e["label"]:
            label = "Mussel 金标准（校准）"
        elif "our_compliance" in e["label"]:
            label = "compliance 正式题库（16 题）"
        else:
            label = "gregariousness 冻结题库（16 题）"
        chance = "3.3%" if e["catalog_size"] == 30 else "20%"
        dom = (
            f"{e['domain_accuracy']:.1%}"
            if e["domain_accuracy"] is not None
            else "—"
        )
        rows.append(
            f"<tr><td>{label}</td><td>{e['catalog_size']} 类</td>"
            f"<td>{e['n_valid']}/{e['n_total']}</td>"
            f"<td><b>{e['accuracy']:.1%}</b></td><td>{chance}</td>"
            f"<td>{dom}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


def error_section(entry: dict, *, gold: bool) -> str:
    if gold:
        # Calibration context only: pattern summary, details stay in JSON.
        from collections import Counter

        pairs = Counter(
            (r["true_facet"], r["predicted_facet"]) for r in entry["errors"]
        )
        lines = [
            f"<li>{facet_zh(t)} → {facet_zh(p)}：{c} 题</li>"
            for (t, p), c in sorted(pairs.items(), key=lambda kv: -kv[1])
        ]
        return (
            "<p class='muted'>金标准 110 题错分 "
            f"{len(entry['errors'])} 处，用于校准分类器判别力"
            "（top-1 85%）。错分模式：</p><ul class='errors'>"
            + "".join(lines)
            + "</ul>"
        )
    if not entry["errors"]:
        return "<p class='muted'>无错分类。</p>"
    items = []
    for r in entry["errors"]:
        items.append(
            f"<li><b>{r['item_id']}</b>：真值 "
            f"<code>{facet_zh(r['true_facet'])}</code> → 预测 "
            f"<code>{facet_zh(r['predicted_facet'])}</code>"
            f"<div class='reason'>{r['reason']}</div></li>"
        )
    return "<ul class='errors'>" + "".join(items) + "</ul>"


def trait_section() -> str:
    parts = []
    parts.append(
        "<p>逐题 ρ(选项得分, 目标 Neo 维度)，n=100。负相关题目以 ⚠ 标记。</p>"
    )
    for path, title in [
        (TMP_DIR / "greg_trait_consistency.json", "gregariousness 运行（人格池覆盖 E2）"),
        (TMP_DIR / "compliance_trait_consistency.json", "compliance 运行（人格池无 A4 条目）"),
    ]:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data["items"]
        rhos = [i["rho_target"] for i in items]
        avg = sum(rhos) / len(rhos)
        pos = sum(1 for r in rhos if r > 0)
        rows = ['<table class="styled">']
        rows.append("<tr><th>题目</th><th>ρ(得分, 目标维度)</th></tr>")
        for i in items:
            r = i["rho_target"]
            mark = " ⚠" if r < 0 else ""
            rows.append(
                f"<tr><td>{i['item_id']}</td><td><b>{r:+.3f}</b>{mark}</td></tr>"
            )
        rows.append("</table>")
        parts.append(
            f"<h3>{title} — 平均 ρ = {avg:.2f}，{pos}/16 为正</h3>" + "".join(rows)
        )
    return "".join(parts)


MIXED_SOURCE_NAMES = {
    "mussel": "Mussel 金标准",
    "compliance": "compliance 正式题库",
    "gregariousness": "gregariousness 冻结题库",
}


def mixed_accuracy_chart(evaluation: dict) -> str:
    """Static grouped-bar chart for source-separated mixed results."""

    source_order = evaluation.get("source_order", [])
    width, height = 820, 350
    left, top, right, bottom = 78, 35, 24, 78
    plot_w = width - left - right
    plot_h = height - top - bottom
    baseline = top + plot_h
    colors = {"top1_accuracy": "#2e7d32", "domain_accuracy": "#ef6c00"}
    labels = {"top1_accuracy": "top-1 facet", "domain_accuracy": "域级"}
    parts = [
        f'<svg class="mixed-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Mussel、compliance、gregariousness 混合分类准确率比较">',
        '<title>混合题库分来源分类准确率</title>',
    ]
    for tick in (0, 0.25, 0.5, 0.75, 1.0):
        y = baseline - tick * plot_h
        parts.append(
            f'<line class="mixed-grid" x1="{left}" y1="{y:.1f}" '
            f'x2="{width - right}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="mixed-label" x="{left - 10}" y="{y + 4:.1f}" '
            f'text-anchor="end">{tick:.0%}</text>'
        )
    parts.append(
        f'<line class="mixed-axis" x1="{left}" y1="{top}" '
        f'x2="{left}" y2="{baseline}"/>'
    )
    parts.append(
        f'<line class="mixed-axis" x1="{left}" y1="{baseline}" '
        f'x2="{width - right}" y2="{baseline}"/>'
    )
    group_w = plot_w / max(1, len(source_order))
    bar_w = min(42, group_w / 4)
    for index, source in enumerate(source_order):
        row = evaluation["by_source"][source]
        center = left + group_w * (index + 0.5)
        for offset, metric in enumerate(("top1_accuracy", "domain_accuracy")):
            value = row.get(metric)
            if value is None:
                continue
            x = center + (offset - 0.5) * (bar_w + 10) - bar_w / 2
            y = baseline - value * plot_h
            h = value * plot_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{h:.1f}" fill="{colors[metric]}"><title>'
                f'{MIXED_SOURCE_NAMES.get(source, source)} {labels[metric]}：'
                f'{value:.1%}</title></rect>'
            )
            parts.append(
                f'<text class="mixed-value" x="{x + bar_w / 2:.1f}" '
                f'y="{max(top + 12, y - 6):.1f}" text-anchor="middle">'
                f'{value:.1%}</text>'
            )
        parts.append(
            f'<text class="mixed-label" x="{center:.1f}" y="{baseline + 28}" '
            f'text-anchor="middle">{html_mod.escape(MIXED_SOURCE_NAMES.get(source, source))}</text>'
        )
    parts.append(
        f'<text class="mixed-axis-title" x="18" y="{top + plot_h / 2:.1f}" '
        f'transform="rotate(-90 18 {top + plot_h / 2:.1f})" '
        f'text-anchor="middle">准确率</text>'
    )
    parts.append(
        f'<text class="mixed-axis-title" x="{left + plot_w / 2:.1f}" '
        f'y="{height - 18}" text-anchor="middle">题目来源（分类器不可见）</text>'
    )
    legend_x = width - 210
    for offset, metric in enumerate(("top1_accuracy", "domain_accuracy")):
        x = legend_x + offset * 105
        parts.append(
            f'<rect x="{x}" y="8" width="12" height="12" '
            f'fill="{colors[metric]}"/><text class="mixed-label" '
            f'x="{x + 17}" y="19">{labels[metric]}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def mixed_section() -> str:
    """Render the mixed-source analysis if its local result exists."""

    if not MIXED_EVALUATION_PATH.exists():
        return "<p class='muted'>尚未生成混合题库分析。</p>"
    evaluation = json.loads(MIXED_EVALUATION_PATH.read_text(encoding="utf-8"))
    rows = ['<table class="styled">']
    rows.append(
        "<tr><th>来源（仅用于本地分组）</th><th>有效题数</th>"
        "<th>top-1 facet 准确率</th><th>域级准确率</th>"
        "<th>同域内 facet 错分</th><th>跨域错分</th></tr>"
    )
    for source in evaluation.get("source_order", []):
        row = evaluation["by_source"][source]
        top1 = (
            f"{row['top1_correct']}/{row['n_valid']}（{row['top1_accuracy']:.1%}）"
            if row["top1_accuracy"] is not None
            else "证据不足"
        )
        domain = (
            f"{row['domain_correct']}/{row['domain_valid']}（{row['domain_accuracy']:.1%}）"
            if row["domain_accuracy"] is not None
            else "证据不足"
        )
        rows.append(
            f"<tr><td>{html_mod.escape(MIXED_SOURCE_NAMES.get(source, source))}</td>"
            f"<td>{row['n_valid']}/{row['n_total']}</td><td><b>{top1}</b></td>"
            f"<td><b>{domain}</b></td><td>{row['same_domain_facet_errors']}</td>"
            f"<td>{row['cross_domain_errors']}</td></tr>"
        )
    rows.append("</table>")

    error_blocks = []
    for source in evaluation.get("source_order", []):
        row = evaluation["by_source"][source]
        errors = row.get("errors", [])
        if not errors:
            error_blocks.append(
                f"<p><b>{html_mod.escape(MIXED_SOURCE_NAMES[source])}</b>：无错分。</p>"
            )
            continue
        error_rows = [
            "<table class='styled'><tr><th>题目</th><th>真实 facet</th>"
            "<th>预测 facet</th><th>类型</th></tr>"
        ]
        for error in errors:
            kind = "同域内 facet 错分" if error["same_domain"] else "跨域错分"
            error_rows.append(
                f"<tr><td>{html_mod.escape(error['item_id'])}</td>"
                f"<td>{html_mod.escape(facet_zh(error['true_facet']))}</td>"
                f"<td>{html_mod.escape(facet_zh(error['predicted_facet']))}</td>"
                f"<td>{kind}</td></tr>"
            )
        error_rows.append("</table>")
        error_blocks.append(
            f"<details><summary>{html_mod.escape(MIXED_SOURCE_NAMES[source])}："
            f"{len(errors)} 道错分</summary>{''.join(error_rows)}</details>"
        )

    overall = evaluation.get("overall") or {}
    overall_text = ""
    if overall.get("top1_accuracy") is not None:
        overall_text = (
            f"<p><b>混合总体：</b>有效 {overall['n_valid']}/{overall['n_total']}，"
            f"top-1 facet {overall['top1_correct']}/{overall['n_valid']} "
            f"（{overall['top1_accuracy']:.1%}），域级 "
            f"{overall['domain_correct']}/{overall['domain_valid']} "
            f"（{overall['domain_accuracy']:.1%}）。</p>"
        )

    return (
        "<h2>二、混合题库盲法分类（来源不进入提示词）</h2>"
        "<p>本节把 110 道 Mussel 金标准题、16 道 compliance 题和 16 道 "
        "gregariousness 题合并为 142 道题进行来源分组分析。分类器每次只看到"
        "30 个 facet 的目录、当前题情境和选项；来源标签只在本地结果中用于分组，"
        "不会进入模型输入。</p>"
        f"{mixed_accuracy_chart(evaluation)}{overall_text}{''.join(rows)}"
        "<p class='muted'>top-1 是具体 facet 命中；域级准确率只要求预测 facet 与真值"
        "属于同一个 Big Five 大域。同域内 facet 错分不会降低域级准确率。</p>"
        "<h3>混合结果中的错分（按真实来源区分）</h3>"
        + "".join(error_blocks)
    )


def prompt_section() -> str:
    """The exact 30-class prompt sent to the classifier, with one real example."""

    items = load_gold_items(30)
    prompt = build_classification_prompt(
        build_catalog(facet_ids_for(30)), items[0], facet_ids_for(30)
    )
    # Real model answer for the same item.
    example = ""
    for path in RESULTS_DIR.glob("*gold*c30_classifications.json"):
        d = json.loads(path.read_text(encoding="utf-8"))
        for r in d["results"]:
            if r["item_id"] == items[0]["item_id"]:
                example = json.dumps(
                    {
                        "facet_id": r["predicted_facet"],
                        "reason": r["reason"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                break
    escaped = html_mod.escape(prompt)
    return (
        "<p class='muted'>以下是一次真实分类调用的完整输入文本（每题独立调用，"
        "目录 + 规则 + 题目三部分；题目的真值标签与题号不发送）。</p>"
        f"<pre class='prompt'>{escaped}</pre>"
        + (f"<h4>模型返回示例（真值：{items[0]['facet']}）</h4>"
           f"<pre class='prompt'>{html_mod.escape(example)}</pre>"
           if example else "")
    )


def main() -> None:
    c30 = load_results(30)
    c5 = load_results(5)
    gold = next(e for e in c30 if "gold" in e["label"])
    ours = next(e for e in c30 if "our_compliance" in e["label"])
    bank = next(e for e in c30 if "bank_bank" in e["label"])
    gold5 = next(e for e in c5 if "gold" in e["label"])
    ours5 = next(e for e in c5 if "our_compliance" in e["label"])
    bank5 = next(e for e in c5 if "bank_bank" in e["label"])

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>SJT 验证结果（精简版）</title>
<style>
  body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 24px auto; max-width: 960px; padding: 0 16px;
         color: #222; background: #fafafa; }}
  h1 {{ border-bottom: 3px solid #2e7d32; padding-bottom: 8px; }}
  h2 {{ margin-top: 36px; color: #1b5e20; }}
  h3 {{ margin-top: 24px; }}
  table.styled {{ border-collapse: collapse; width: 100%;
                  background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table.styled th, table.styled td {{ border: 1px solid #ddd;
      padding: 6px 10px; text-align: center; font-size: 14px; }}
  table.styled th {{ background: #e8f5e9; }}
  table.hm {{ border-collapse: collapse; background: #fff;
              box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  table.hm th, table.hm td {{ border: 1px solid #eee; padding: 4px 6px;
      text-align: center; font-size: 12px; min-width: 58px; }}
  table.hm th {{ background: #263238; color: #fff; }}
  td.hm-label {{ background: #eceff1; font-weight: 600; text-align: right; }}
  .muted {{ color: #777; font-size: 13px; }}
  ul.errors {{ list-style: none; padding-left: 0; }}
  ul.errors li {{ background: #fff; border-left: 4px solid #c0392b;
      margin: 8px 0; padding: 8px 12px; box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  .reason {{ color: #555; font-size: 13px; margin-top: 4px; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }}
  pre.prompt {{ background: #263238; color: #e8f0e8; padding: 14px 16px;
      border-radius: 6px; overflow-x: auto; font-size: 12px;
      line-height: 1.5; white-space: pre-wrap; word-break: break-all; }}
  .key {{ background: #fff; border: 1px solid #ddd; padding: 14px;
          box-shadow: 0 1px 3px rgba(0,0,0,.12); }}
  .key li {{ margin: 6px 0; }}
  .mixed-chart {{ width: 100%; height: auto; display: block; margin: 14px 0 22px; }}
  .mixed-grid {{ stroke: #d9e2dc; stroke-width: 1; }}
  .mixed-axis {{ stroke: #555; stroke-width: 1.2; }}
  .mixed-label, .mixed-value, .mixed-axis-title {{ fill: #222; font-size: 12px; }}
  .mixed-value {{ font-weight: 600; }}
</style>
</head>
<body>
<h1>SJT 验证结果（精简版）</h1>

<div class="key">
<b>有决策价值的结果：</b>
<ul>
<li><b>语义：题目是准的。</b>两个题库在 30 类盲法分类下域级准确率
94%–100%，文本构念正确。</li>
<li><b>3 道题有 A4/A2 边界问题</b>（compliance 题库 Q004、Q009、Q012）：
行为对照实为"坦诚 vs 掩饰"（A2），建议按 A4 冲突-让步对照改写。</li>
<li><b>1 道题有 E2/A3 边界问题</b>（gregariousness 题库 row-12）：
"邀请落单同学"的对照偏利他。</li>
<li><b>虚拟作答一致性取决于人格池 facet 覆盖</b>：gregariousness 运行
平均 ρ=0.70，compliance 运行仅 0.24 —— 问题在被试端，不在题目端。</li>
<li><b>3 道题虚拟作答方向颠倒</b>：gregariousness row-11（ρ=-0.80，
高分选项锚定疑似反了）、compliance row-05 / row-09（ρ≈-0.01~-0.07）。</li>
</ul>
</div>

<h2>一、盲法分类准确率（30 类为主，5 类对照）</h2>
{accuracy_table([gold, ours, bank, gold5, ours5, bank5])}
<p class="muted">5 类目录只检验跨域混淆，无法发现 A4/A2、E2/A3 这类域内边界问题；
保留作为对照。</p>

{mixed_section()}

<h2>三、30 类混淆热图（绿=命中，红=错分）</h2>
<h3>Mussel 金标准（校准）</h3>
{confusion_heatmap(gold)}
{error_section(gold, gold=True)}
<h3>compliance 正式题库</h3>
{confusion_heatmap(ours)}
{error_section(ours, gold=False)}
<h3>gregariousness 冻结题库</h3>
{confusion_heatmap(bank)}
{error_section(bank, gold=False)}

<h2>四、5 类混淆热图（对照）</h2>
<h3>Mussel 金标准 · 5 类</h3>
{confusion_heatmap(gold5)}
<h3>compliance 正式题库 · 5 类</h3>
{confusion_heatmap(ours5)}
<h3>gregariousness 冻结题库 · 5 类</h3>
{confusion_heatmap(bank5)}

<h2>五、虚拟作答人格一致性（题级）</h2>
{trait_section()}

<h2>六、分类提示词（实际发送文本）</h2>
{prompt_section()}

<p class="muted" style="margin-top:40px">
数据源：blind_classification/results/*_c30_classifications.json（30 类）、
*_classifications.json（5 类）、
mixed_mussel_compliance_gregariousness_c30_evaluation.json（混合分组分析）、
tmp/*_trait_consistency.json。分类器为 LLM（deepseek-v4-pro-guan），
仅为开发期语义证据。</p>
</body>
</html>
"""
    out = RESULTS_DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    print(f"HTML 报告已生成: {out}")


if __name__ == "__main__":
    main()
