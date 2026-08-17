"""信效度迭代轨迹报告生成器（离线、确定性、无外部依赖）。

扫描 outputs/virtual_responses/*/bank-v*/psychometrics/ 的既有产物，
把每个版本的 信度(Cronbach α)、聚合效度(SJT×目标Neo维度 Spearman ρ)、
区分效度(目标ρ − 最大非目标ρ) 画成随迭代版本变化的曲线。

用法：
    python -X utf8 tools/iteration_trajectory.py
输出：
    outputs/iteration_trajectory.html
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "outputs" / "iteration_trajectory.html"
VR = ROOT / "outputs" / "virtual_responses"


def version_number(name: str) -> int:
    return int(name.split("-v")[1].split("-")[0])


def load_run_rows(run_dir: Path) -> list[dict]:
    rows = []
    for ver_dir in sorted(
        glob.glob(str(run_dir / "bank-v*")), key=lambda p: version_number(os.path.basename(p))
    ):
        me_path = Path(ver_dir) / "psychometrics" / "measurement_evaluation.json"
        if not me_path.exists():
            continue
        me = json.loads(me_path.read_text(encoding="utf-8"))
        ss_path = Path(ver_dir) / "psychometrics" / "scale_statistics.json"
        n = None
        if ss_path.exists():
            n = json.loads(ss_path.read_text(encoding="utf-8")).get("sample_size")
        rec = me.get("reliability") or {}
        conv = (me.get("validity") or {}).get("convergent") or {}
        disc = (me.get("validity") or {}).get("discriminant") or {}
        counts = me.get("item_recommendation_counts") or {}
        rows.append({
            "version": version_number(os.path.basename(ver_dir)),
            "n": n,
            "alpha": rec.get("cronbach_alpha"),
            "conv_rho": conv.get("rho"),
            "disc_rho": disc.get("largest_non_target_rho"),
            "margin": disc.get("target_margin"),
            "retain": counts.get("retain"),
            "revise": counts.get("revise"),
            "remove": counts.get("remove"),
        })
    return rows


def repair_events(run_id: str) -> int | None:
    cp = ROOT / "outputs" / "run_checkpoints" / f"{run_id}.json"
    if not cp.exists():
        return None
    state = json.loads(cp.read_text(encoding="utf-8")).get("state") or {}
    history = state.get("psychometric_repair_history") or []
    return len(history)


def esc(text) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def num(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # NaN 检查


# ---------------- SVG 折线图 ----------------

def line_chart(
    series: list[dict], key: str, title: str, y_label: str,
    lo: float, hi: float, thresholds: list[tuple[float, str]] = (),
    color: str = "#2563eb", width: int = 560, height: int = 240,
) -> str:
    points = []
    for i, row in enumerate(series):
        y = num(row.get(key))
        if y is None:
            continue
        points.append((i, y))
    x_values = list(range(len(series)))
    pad = 0.10 * (hi - lo)
    y_lo, y_hi = lo - pad, hi + pad
    margin_l, margin_r, margin_t, margin_b = 56, 20, 28, 34
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    def px(i):
        return margin_l + plot_w * (i / max(1, len(series) - 1))

    def py(y):
        return margin_t + plot_h * (1 - (y - y_lo) / (y_hi - y_lo))

    parts = [f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
             f'role="img" aria-label="{esc(title)}">']
    # 网格与 y 轴刻度
    ticks = 5
    for t in range(ticks + 1):
        yv = y_lo + (y_hi - y_lo) * t / ticks
        yy = py(yv)
        parts.append(f'<line x1="{margin_l}" y1="{yy:.1f}" x2="{width - margin_r}" '
                     f'y2="{yy:.1f}" stroke="#e2e8f0" stroke-width="1"/>')
        parts.append(f'<text x="{margin_l - 6}" y="{yy + 4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#64748b">{yv:.2f}</text>')
    # x 轴标签 = 版本号
    for i in x_values:
        parts.append(f'<text x="{px(i):.1f}" y="{height - 12}" text-anchor="middle" '
                     f'font-size="10" fill="#475569">v{series[i]["version"]}</text>')
    # 阈值线
    for tv, label in thresholds:
        if y_lo <= tv <= y_hi:
            yy = py(tv)
            parts.append(f'<line x1="{margin_l}" y1="{yy:.1f}" x2="{width - margin_r}" '
                         f'y2="{yy:.1f}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3"/>')
            parts.append(f'<text x="{width - margin_r - 4}" y="{yy - 4:.1f}" text-anchor="end" '
                         f'font-size="9" fill="#94a3b8">{esc(label)}</text>')
    # 折线
    if points:
        coords = " ".join(f"{px(i):.1f},{py(y):.1f}" for i, y in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        for i, y in points:
            parts.append(f'<circle cx="{px(i):.1f}" cy="{py(y):.1f}" r="3.6" fill="{color}"/>')
    # 标题
    parts.append(f'<text x="{margin_l}" y="{margin_t - 10}" font-size="12.5" font-weight="700" '
                 f'fill="#22303f">{esc(title)}</text>')
    parts.append(f'<text x="{margin_l}" y="{height - 16}" font-size="10" fill="#64748b">'
                 f'x = 题库版本（迭代）　y = {esc(y_label)}　点旁数字 = 样本量 n</text>')
    # 样本量标注
    for i, row in enumerate(series):
        y = num(row.get(key))
        if y is None or row.get("n") is None:
            continue
        parts.append(f'<text x="{px(i):.1f}" y="{py(y) + 16:.1f}" text-anchor="middle" '
                     f'font-size="8.5" fill="#94a3b8">n={row["n"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def run_card(run_id: str, rows: list[dict], events: int | None) -> str:
    alphas = [num(r.get("alpha")) for r in rows if num(r.get("alpha")) is not None]
    alphas = [a for a in alphas if -1 <= a <= 1]  # 过滤退化 α
    convs = [num(r.get("conv_rho")) for r in rows if num(r.get("conv_rho")) is not None]
    margins = [num(r.get("margin")) for r in rows if num(r.get("margin")) is not None]
    if not alphas and not convs:
        return ""
    all_vals = alphas + convs + margins
    lo, hi = min(all_vals), max(all_vals)
    lo = min(lo, 0.0)
    hi = max(hi, 1.0)
    lo = -0.05 if lo < 0 else lo

    events_note = ""
    if events is not None:
        events_note = f' · 心理测量修题事件 {events} 次'

    table_rows = "".join(
        f'<tr><td class="num">v{r["version"]}</td><td class="num">{r["n"] if r["n"] else "—"}</td>'
        f'<td class="num">{r["alpha"]:.3f}</td><td class="num">{r["conv_rho"]:.3f}</td>'
        f'<td class="num">{r["margin"]:.3f}</td>'
        f'<td class="num">{r["retain"]}/{r["revise"]}/{r["remove"]}</td></tr>'
        for r in rows
    )
    return f"""
<div class="card">
  <h3>run {esc(run_id[:8])} · {len(rows)} 个版本{events_note}</h3>
  <div class="charts">
    {line_chart(rows, "alpha", "信度：Cronbach α", "α", lo, hi,
                [(0.70, "可接受"), (0.80, "强")], color="#2563eb")}
    {line_chart(rows, "conv_rho", "聚合效度：SJT总分 × 目标Neo维度 ρ", "ρ", lo, hi,
                [(0.30, "0.30"), (0.40, "0.40")], color="#059669")}
    {line_chart(rows, "margin", "区分效度：目标ρ − 最大非目标ρ", "margin", lo, hi,
                [(0.0, "0")], color="#b45309")}
  </div>
  <table>
    <tr><th>版本</th><th class="num">n</th><th class="num">α</th><th class="num">聚合ρ</th>
        <th class="num">区分margin</th><th class="num">建议(保留/修/删)</th></tr>
    {table_rows}
  </table>
</div>"""


def main() -> None:
    runs = []
    for run_dir in sorted(glob.glob(str(VR / "*"))):
        run_id = os.path.basename(run_dir)
        rows = load_run_rows(Path(run_dir))
        if len(rows) >= 2:
            runs.append((run_id, rows, repair_events(run_id)))

    # 汇总：首尾版本变化
    deltas = {"alpha": [], "conv": [], "margin": []}
    for _, rows, _ in runs:
        first, last = rows[0], rows[-1]
        if num(first.get("alpha")) is not None and num(last.get("alpha")) is not None:
            deltas["alpha"].append(num(last["alpha"]) - num(first["alpha"]))
        if num(first.get("conv_rho")) is not None and num(last.get("conv_rho")) is not None:
            deltas["conv"].append(num(last["conv_rho"]) - num(first["conv_rho"]))
        if num(first.get("margin")) is not None and num(last.get("margin")) is not None:
            deltas["margin"].append(num(last["margin"]) - num(first["margin"]))

    def stat(values):
        if not values:
            return "—"
        mean = sum(values) / len(values)
        pos = sum(1 for v in values if v > 0.001)
        return (f"均值 {mean:+.3f} · 上升 {pos}/{len(values)} 个run")

    cards = "".join(run_card(run_id, rows, events) for run_id, rows, events in runs)

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8">
<title>信效度迭代轨迹报告</title>
<style>
body{{font-family:"Segoe UI","Microsoft YaHei",sans-serif;background:#f6f8fb;color:#22303f;
  line-height:1.7;margin:0}}
.wrap{{max-width:1160px;margin:0 auto;padding:26px 18px 80px}}
h1{{font-size:24px;margin:0 0 6px}}
.sub{{color:#64748b;font-size:13.5px;margin-bottom:20px}}
h2{{font-size:18px;border-left:4px solid #2563eb;padding-left:10px;margin:34px 0 12px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;
  margin:14px 0;box-shadow:0 1px 3px rgba(15,40,80,.05)}}
.card h3{{margin:0 0 12px;font-size:15.5px}}
.charts{{display:flex;gap:14px;flex-wrap:wrap}}
.charts svg{{flex:1;min-width:420px;max-width:560px;border:1px solid #e2e8f0;border-radius:10px;
  background:#fff}}
table{{border-collapse:collapse;margin-top:12px;font-size:12.5px}}
th,td{{border:1px solid #e2e8f0;padding:5px 10px;text-align:left}}
th{{background:#f1f5f9;color:#475569}}
td.num{{text-align:center}}
.summary td{{padding:8px 12px}}
.summary th{{width:180px}}
.caveat{{background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:12px 16px;
  font-size:13.5px;margin:10px 0}}
.caveat b{{color:#b45309}}
footer{{margin-top:50px;color:#94a3b8;font-size:12px;border-top:1px solid #e2e8f0;padding-top:12px}}
</style></head>
<body><div class="wrap">
<h1>信效度随迭代的轨迹</h1>
<div class="sub">数据源：outputs/virtual_responses/*/bank-v*/psychometrics/（既有存档，全部为虚拟被试开发期证据）·
生成器：tools/iteration_trajectory.py</div>

<h2>一、跨 run 汇总：从首版本到末版本的变化</h2>
<div class="card">
<table class="summary">
<tr><th>指标</th><th>首→末变化</th><th>含义</th></tr>
<tr><td>Cronbach α（信度）</td><td class="num">{stat(deltas["alpha"])}</td><td>修题迭代整体上是否提升内部一致性</td></tr>
<tr><td>聚合效度 ρ（SJT×目标Neo维度）</td><td class="num">{stat(deltas["conv"])}</td><td>题目整体是否更贴近目标构念</td></tr>
<tr><td>区分效度 margin（目标ρ−最大非目标ρ）</td><td class="num">{stat(deltas["margin"])}</td><td>是否更"只测目标、不测别的"</td></tr>
</table>
<p style="font-size:12.5px;color:#64748b">注意：不同 run 的样本量不同（30/50/100）、目标构念不同（compliance / gregariousness / self-discipline / openness_to_ideas 等），汇总只是趋势参考，不作统计推断。</p>
</div>

<h2>二、每个 run 的迭代轨迹（{len(runs)} 个有 ≥2 个版本的 run）</h2>
{cards}

<h2>三、结论与使用边界</h2>
<div class="caveat"><b>读图须知：</b>
① x 轴是<b>题库版本号</b>，不是严格的"修题轮次"——版本也会因重计分（评分键变更）而递增；与心理测量修题事件数对照使用。<br>
② 点旁标注样本量 n：同一 run 内 n 变化（如 30→100）时，前后不可直接比。<br>
③ 虚拟被试证据只能用于<b>开发期决策</b>，不是正式效度证据；聚合效度还会被"同一人格提示生成 SJT 与 Neo-FFI"的共同方法方差抬高。<br>
④ 部分早期版本 α 为负或 ρ 缺失，是 1–2 题的退化分量表所致，图已过滤。</div>
<footer>生成：tools/iteration_trajectory.py · 无 LLM 参与 · 可直接重新运行以包含最新 run</footer>
</div></body></html>"""

    OUT.write_text(html_doc, encoding="utf-8")
    print(f"[trajectory] {len(runs)} 个多版本 run 已写入 {OUT}")
    print(f"[trajectory] alpha: {stat(deltas['alpha'])}")
    print(f"[trajectory] conv:  {stat(deltas['conv'])}")
    print(f"[trajectory] margin: {stat(deltas['margin'])}")


if __name__ == "__main__":
    main()
