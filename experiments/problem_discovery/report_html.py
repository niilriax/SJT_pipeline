"""生成自包含 HTML 报告：算法公式 + 16 题真实数据诊断结果。

运行：
    python -X utf8 experiments/problem_discovery/report_html.py
输出：
    experiments/problem_discovery/out/problem_report.html
用浏览器直接打开即可，无需服务器。
"""

from __future__ import annotations

import csv
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.problem_discovery.demo import (  # noqa: E402
    PSYCH,
    RUN_ID,
    compute_option_order,
    compute_option_pbs,
    extract_findings,
    load_blind_map,
    load_neo_a_means,
    load_option_scores,
    load_rest_by_respondent,
    load_quality,
    load_selection_vectors,
)
from experiments.problem_discovery.diagnoser import decide  # noqa: E402
from experiments.problem_discovery.taxonomy import (  # noqa: E402
    FINDINGS,
    PROBLEMS,
)

BANK = "bank-v2-75babf8179a9"
OUT = Path(__file__).resolve().parent / "out"

LEVEL_ZH = {"low": "低", "medium_low": "中低", "medium_high": "中高", "high": "高"}

PROBLEM_COLORS = {
    "P_OPTION_ANCHOR_MISLABEL": "#b9770e",
    "P_OPTION_DEAD": "#e08a3c",
    "P_OPTION_MISLEAD": "#d9534f",
    "P_CONSTRUCT_MISALIGN": "#b062c8",
    "P_SCORING_REVERSED": "#c0392b",
    "P_RANGE_RESTRICTED": "#5b9bd5",
    "P_RELIABILITY_HARM": "#7f8c8d",
    "P_NONE": "#27ae60",
}

# 人工核对结论（与题目原文、选项作答数字逐条对照后写出）
MANUAL_REVIEW = {
    "01": ("属实", "4分档C「调整自己睡眠习惯」太圣、2分档D「找管理员」太小题大做，双双0人选；77%挤在A，题目只剩两档运转。"),
    "02": ("属实", "D全死、最高档B仅5人。原系统判retain，算法不同意——4分档半死、2分档全死，证据支持修选项。"),
    "03": ("属实·更严重", "C「事后在朋友圈吐槽」被标成中高顺从给3分——吐槽是记恨，属于行为水平标注错误；中间两档全废。"),
    "04": ("部分属实", "D仅1人选无统计意义；但「课后私下提醒」确实更接近坦率而非顺从，盲法指控有文本依据；难度0.233偏低源于1分档吸走67%。"),
    "05": ("属实", "66%选最高档导致难度0.873天花板；1分档D仅2人但A均分3.792全场最高——标注同样可疑。"),
    "06": ("轻症", "方向整体正确（B最低3.021、C最高3.824），仅2分/3分档挤在一起；算法未定罪，分寸恰当。"),
    "07": ("属实", "最高档D仅3人选，且A均分3.556低于2分档的3.979——4分档既不常被选、选的人宜人性还偏低。"),
    "08": ("属实", "B「立刻赞同日料好呀」谄媚到0人选、D「坚持说服」顽固到0人选，只剩中间两档，79%挤在C。"),
    "09": ("属实·全卷最重", "「低顺从1分」的A有75人选、宜人性均分3.727全场第二高，而2分档D的选择者只有3.368——得分与构念均分次序颠倒；「邀请对方一起讨论」是中高顺从，行为水平标反；盲法认成straightforwardness。三路证据一致。"),
    "10": ("健康属实", "方向清晰（1分档A均分3.104明显最低），各档都有人选。"),
    "11": ("健康属实", "四档全活，A均分3.217→3.483→3.774→3.759基本单调，全卷最好的题之一。"),
    "12": ("属实·更细致", "D「点头换话题」定为4分，但选它的3人A均分3.389全场最低——锚定与文本语义不符；盲法认成straightforwardness；2分档也死。"),
    "13": ("基本健康", "方向正确（C最高3.935）；D仅3人属小样本波动。"),
    "14": ("属实(锚定偏低)", "C「和对方商量时间地点」被标2分，却92%选择率、A均分3.721——被试把它当正常顺从；4分档「戴耳机妥协」0人选；负CITC由此而来。"),
    "15": ("属实", "C(1分)与D(2分)的A均分只差0.001，两档拉不开导致负CITC；4分档「借一整天」0人选。"),
    "16": ("算法正确拒绝定罪", "1→2→3→4分A均分3.528→3.636→3.817→3.875完全单调，方向全对；负CITC是3人/2人小样本噪声。现有系统会误判revise，算法避免了一次误修。"),
}


def esc(text: object) -> str:
    return html.escape(str(text or ""))


def load_items() -> dict[str, dict]:
    data = json.loads(
        (ROOT / "outputs" / "run_checkpoints" / f"{RUN_ID}.json").read_text(encoding="utf-8")
    )
    bank = data["state"]["frozen_item_bank"]
    return {item["item_id"]: item for item in bank}


def posterior_bar(problem: str, value: float, width_pct: float) -> str:
    color = PROBLEM_COLORS[problem]
    label = PROBLEMS[problem][0]
    return (
        f'<div class="pbar-row">'
        f'<span class="pbar-label">{esc(problem)}</span>'
        f'<div class="pbar-track"><div class="pbar-fill" '
        f'style="width:{width_pct:.1f}%;background:{color}"></div></div>'
        f'<span class="pbar-value">{value:.2f}</span>'
        f'<span class="pbar-name">{esc(label)}</span></div>'
    )


def decision_badge(decision: str, action: str) -> str:
    if decision == "healthy":
        return '<span class="badge badge-green">健康 · RETAIN 保留</span>'
    if decision == "act":
        action_zh = {"REVISE_OPTIONS": "修选项", "REVISE_SCENARIO": "修情境",
                     "REMOVE": "删除", "RETAIN": "保留"}.get(action, action)
        return f'<span class="badge badge-red">明确问题 · {esc(action_zh)}</span>'
    return '<span class="badge badge-amber">证据不足 · 升级人工/LLM</span>'


def finding_badge(code: str, detail: str) -> str:
    return (
        f'<div class="finding"><code>{esc(code)}</code>'
        f'<span class="finding-label">{esc(FINDINGS[code][1])}</span>'
        f'<span class="finding-detail">{esc(detail)}</span></div>'
    )


def build_item_card(
    item_id: str,
    item: dict,
    quality: dict,
    fired: dict[str, str],
    result: dict,
    options: list[dict],
    row_number: str,
) -> str:
    stats = quality[item_id]
    review_verdict, review_text = MANUAL_REVIEW[row_number]
    verdict_class = {"属实": "v-true", "属实·更严重": "v-true", "属实(锚定偏低)": "v-true",
                     "属实·全卷最重": "v-true", "属实·更细致": "v-true",
                     "部分属实": "v-partial", "轻症": "v-partial",
                     "基本健康": "v-ok", "健康属实": "v-ok",
                     "算法正确拒绝定罪": "v-ok"}[review_verdict]

    indicator_rows = []
    citc = stats["citc_r"]
    diff = stats["difficulty"]
    eff = stats["effective_option_count"]
    minr = stats["minimum_option_rate"]

    def ind(name: str, value: str, ok: bool, note: str = "") -> str:
        cls = "ind-ok" if ok else "ind-warn"
        return (f'<div class="ind"><span class="ind-name">{esc(name)}</span>'
                f'<span class="ind-value {cls}">{esc(value)}</span>'
                f'<span class="ind-note">{esc(note)}</span></div>')

    indicator_rows.append(ind("CITC", f"{citc:.3f}", citc >= 0.20,
                              "分面内校正题总相关"))
    indicator_rows.append(ind("难度", f"{diff:.3f}", 0.20 <= diff <= 0.80,
                              "标准化均分 (0.20–0.80)"))
    indicator_rows.append(ind("有效选项", str(eff), eff >= 3,
                              "选择率≥5%的选项数"))
    indicator_rows.append(ind("最低选择率", f"{minr:.0%}", minr >= 0.05,
                              "存在零选择选项即报警"))

    option_rows = []
    for opt in options:
        score = opt["score"]
        level = LEVEL_ZH.get(opt["level"], opt["level"])
        n = opt["n"]
        rate = opt["rate"]
        amean = opt["amean"]
        dead = rate < 0.05
        cls = "opt-dead" if dead else ""
        option_rows.append(
            f'<tr class="{cls}"><td>{esc(opt["oid"])}</td>'
            f'<td class="num">{score:.0f}</td>'
            f'<td class="lvl">{esc(level)}</td>'
            f'<td class="num">{n}</td>'
            f'<td class="num">{rate:.0%}</td>'
            f'<td class="num">{amean}</td>'
            f'<td class="opt-text">{esc(opt["text"])}</td></tr>'
        )

    findings_html = "".join(finding_badge(c, d) for c, d in sorted(fired.items()))
    ranked = result["ranked"][:3]
    max_value = ranked[0][1] if ranked else 0
    bars = "".join(
        posterior_bar(p, v, 100.0 * v / max(1e-9, max(1.0, max_value)))
        for p, v in ranked
    )

    return f"""
<div class="card">
  <div class="card-head">
    <span class="row-num">row-{esc(row_number)}</span>
    <span class="scenario">{esc(item.get("scenario"))}</span>
    {decision_badge(result["decision"], result["action"])}
  </div>
  <div class="indicators">{"".join(indicator_rows)}</div>
  <div class="findings">{findings_html or '<div class="finding-none">无 finding 触发</div>'}</div>
  <div class="posterior">{bars}</div>
  <table class="options">
    <tr><th>选项</th><th>分</th><th>行为水平</th><th>选择人数</th><th>选择率</th>
        <th>选择者<br>宜人性均分</th><th>原文</th></tr>
    {"".join(option_rows)}
  </table>
  <div class="review"><span class="verdict {verdict_class}">人工核对：{esc(review_verdict)}</span>
    <span class="review-text">{esc(review_text)}</span></div>
</div>"""


def main() -> None:
    quality = load_quality()
    items = load_items()
    blind_map = load_blind_map(set(quality))
    neo_means = load_neo_a_means()
    option_scores = load_option_scores()
    selectors = load_selection_vectors(set(quality), neo_means)
    rest_by_respondent = load_rest_by_respondent()

    # 每个选项的选择率（从 option_statistics.csv 读取权威值）
    option_rates: dict[str, dict[str, float]] = defaultdict(dict)
    with open(PSYCH / "option_statistics.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            option_rates[row["item_id"]][row["option_id"]] = float(row["rate"])

    cards = []
    counts = {"healthy": 0, "act": 0, "ambiguous": 0}
    for item_id in sorted(quality):
        row_number = item_id.split("row-")[1].split("-")[0]
        item = items[item_id]
        option_order = compute_option_order(item_id, option_scores, selectors)
        fired, observed = extract_findings(item_id, quality[item_id], blind_map, {item_id: option_order})
        result = decide(fired, observed)
        rounds = 1
        while result["action"].startswith("measure:") and rounds < 3:
            family = result["action"].split(":", 1)[1]
            if family == "OPTION_PBS":
                for code, detail in compute_option_pbs(
                    item_id, option_scores, selectors, rest_by_respondent
                ):
                    fired[code] = detail
                observed.add("OPTION_PBS")
            elif family == "ALPHA_DROP":
                # 该指标当前流水线未导出；保持未观察并停止追问。
                result["action"] = "INVESTIGATE"
                result["suggestion"] = "alpha_if_deleted 未导出，需补指标"
                break
            result = decide(fired, observed)
            rounds += 1
        counts[result["decision"]] += 1

        options = []
        for opt in item["response_options"]:
            oid = opt["option_id"]
            n = 0
            amean_sum = 0.0
            for value in selectors.get(item_id, {}).get(oid, {}).values():
                n += 1
                amean_sum += value
            options.append({
                "oid": oid,
                "score": float(item["scoring_key"].get(oid, 0)),
                "level": opt.get("behavioral_level", ""),
                "n": n,
                "rate": option_rates.get(item_id, {}).get(oid, 0.0),
                "amean": f"{amean_sum / n:.3f}" if n else "—",
                "text": opt.get("text", ""),
            })
        cards.append(build_item_card(item_id, item, quality, fired, result, options, row_number))

    summary = (
        f'<div class="stat-card stat-green"><div class="stat-num">{counts["healthy"]}</div>'
        f'<div class="stat-label">健康 · 保留</div></div>'
        f'<div class="stat-card stat-red"><div class="stat-num">{counts["act"]}</div>'
        f'<div class="stat-label">明确问题 · 直接动作</div></div>'
        f'<div class="stat-card stat-amber"><div class="stat-num">{counts["ambiguous"]}</div>'
        f'<div class="stat-label">证据不足 · 升级人工/LLM</div></div>'
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>SJT 题目问题发现报告 · {BANK}</title>
<style>
body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 0; background: #f4f6f9; color: #26303c; }}
.wrap {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px 80px; }}
h1 {{ font-size: 26px; margin: 0 0 4px; }}
.sub {{ color: #6b7686; font-size: 13px; margin-bottom: 24px; }}
h2 {{ font-size: 19px; border-left: 4px solid #3b6fd4; padding-left: 10px; margin: 36px 0 14px; }}
.stats {{ display: flex; gap: 14px; margin-bottom: 8px; }}
.stat-card {{ flex: 1; border-radius: 10px; padding: 14px 18px; color: #fff; }}
.stat-green {{ background: #27ae60; }} .stat-red {{ background: #d9534f; }} .stat-amber {{ background: #e08a3c; }}
.stat-num {{ font-size: 30px; font-weight: 700; }} .stat-label {{ font-size: 13px; opacity: .92; }}
.formula-box {{ background: #fff; border: 1px solid #e2e7ee; border-radius: 10px; padding: 18px 22px; margin: 12px 0; }}
.formula-box h3 {{ margin: 0 0 8px; font-size: 15px; color: #3b6fd4; }}
.formula {{ font-family: "Cambria Math", "Times New Roman", serif; font-size: 16px; line-height: 1.7; overflow-x: auto; }}
.formula i {{ font-style: italic; }}
.note {{ color: #6b7686; font-size: 13px; margin: 6px 0 0; }}
.card {{ background: #fff; border: 1px solid #e2e7ee; border-radius: 10px; padding: 16px 18px; margin: 16px 0; box-shadow: 0 1px 3px rgba(20,40,80,.05); }}
.card-head {{ display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; }}
.row-num {{ font-weight: 700; color: #3b6fd4; white-space: nowrap; }}
.scenario {{ font-size: 15px; font-weight: 600; flex: 1; min-width: 240px; }}
.badge {{ font-size: 12px; border-radius: 20px; padding: 3px 12px; color: #fff; white-space: nowrap; }}
.badge-green {{ background: #27ae60; }} .badge-red {{ background: #d9534f; }} .badge-amber {{ background: #e08a3c; }}
.indicators {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 6px; }}
.ind {{ background: #f4f6f9; border-radius: 8px; padding: 5px 10px; font-size: 12.5px; }}
.ind-name {{ color: #6b7686; margin-right: 6px; }}
.ind-value {{ font-weight: 700; }} .ind-ok {{ color: #27ae60; }} .ind-warn {{ color: #d9534f; }}
.ind-note {{ color: #9aa3b0; margin-left: 6px; }}
.findings {{ margin: 8px 0 10px; }}
.finding {{ display: inline-flex; align-items: center; gap: 8px; background: #fdf1f0; border: 1px solid #f3c6c3; color: #b03a2e; border-radius: 8px; padding: 4px 10px; margin: 2px 6px 2px 0; font-size: 12.5px; }}
.finding code {{ font-weight: 700; }}
.finding-label {{ color: #7a4a44; }}
.finding-detail {{ color: #9aa3b0; }}
.finding-none {{ color: #9aa3b0; font-size: 13px; }}
.posterior {{ margin: 6px 0 12px; }}
.pbar-row {{ display: flex; align-items: center; gap: 10px; margin: 4px 0; font-size: 12.5px; }}
.pbar-label {{ width: 190px; font-family: Consolas, monospace; font-size: 11.5px; color: #555; }}
.pbar-track {{ flex: 1; background: #eef1f5; border-radius: 5px; height: 14px; overflow: hidden; }}
.pbar-fill {{ height: 100%; border-radius: 5px; }}
.pbar-value {{ width: 38px; text-align: right; font-weight: 700; }}
.pbar-name {{ color: #6b7686; width: 210px; }}
table.options {{ width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
table.options th, table.options td {{ border: 1px solid #e2e7ee; padding: 6px 8px; text-align: left; vertical-align: top; }}
table.options th {{ background: #f4f6f9; color: #6b7686; font-weight: 600; }}
td.num {{ text-align: center; white-space: nowrap; }} td.lvl {{ text-align: center; white-space: nowrap; }}
tr.opt-dead td {{ background: #fdf6f0; }}
tr.opt-dead td.num {{ color: #c0562f; font-weight: 700; }}
.opt-text {{ color: #3a4554; }}
.review {{ margin-top: 10px; background: #f7f9fc; border-radius: 8px; padding: 8px 12px; font-size: 13px; }}
.verdict {{ font-weight: 700; margin-right: 8px; }}
.v-true {{ color: #b03a2e; }} .v-partial {{ color: #b9770e; }} .v-ok {{ color: #1e8449; }}
.review-text {{ color: #4a5564; }}
.highlight {{ background: #fff8e6; border: 1px solid #f0d98c; border-radius: 10px; padding: 12px 16px; margin: 10px 0; font-size: 13.5px; }}
</style>
</head>
<body><div class="wrap">
<h1>SJT 题目问题发现报告</h1>
<div class="sub">题库 {esc(BANK)} · 运行 {esc(RUN_ID[:8])} · 16 题 · 100 名虚拟被试 ·
方法：序贯贝叶斯问题发现（确定性 finding + 家族分组朴素贝叶斯 + 期望损失决策）</div>

<div class="stats">{summary}</div>

<div class="highlight">
<b>两个翻出原文才能看见、纯统计看不见的发现：</b><br>
① row-09：被标为「低顺从 1 分」的选项 A 实际是「邀请对方一起讨论」（中高顺从），75 人选、宜人性均分 3.727 全场第二高，而 2 分档 D 的选择者只有 3.368，得分与构念均分次序颠倒 —— 行为水平标注错误，全卷最重；<br>
② row-03：被标为「中高顺从 3 分」的选项 C 是「事后在朋友圈吐槽」——吐槽是记恨，方向完全标反。
</div>

<h2>算法公式</h2>
<div class="formula-box">
<h3>① finding 提取（确定性阈值）</h3>
<div class="formula"><i>f</i> = [ <i>x<sub>k</sub></i> ⊙ <i>τ<sub>k</sub></i> ]，⊙ ∈ {{&lt;, ≤, &gt;, ≥}}<br>
例：F_CITC_NEG = [ citc &lt; 0 ]，F_DIFF_HIGH = [ difficulty &gt; 0.80 ]，F_OPTION_FEW = [ 有效选项 &lt; 3 ]</div>
</div>
<div class="formula-box">
<h3>② 家族结果似然（族内互斥，未触发也是证据）</h3>
<div class="formula">L(F<sub>m</sub> = v | p) = <i>s</i>(f, p) 若 v 是触发成员 f；否则 max(0.05, 1 − Σ<sub>f∈F<sub>m</sub></sub> <i>s</i>(f, p))<br>
其中 <i>s</i>(f, p) = P(f 触发 | 问题 p 存在)，灵敏度表（专家先验，闭环 Laplace 更新）</div>
</div>
<div class="formula-box">
<h3>③ 后验（家族分组朴素贝叶斯；缺失 ≠ 未触发）</h3>
<div class="formula">P(p | obs) ∝ <i>π</i>(p) · ∏<sub>m ∈ observed</sub> L(F<sub>m</sub> = v<sub>m</sub> | p)，归一化 Z = Σ<sub>p′</sub> <i>π</i>(p′) ∏<sub>m</sub> L(F<sub>m</sub> = v<sub>m</sub> | p′)</div>
</div>
<div class="formula-box">
<h3>④ 熵与期望信息增益（选下一个该测的指标）</h3>
<div class="formula">H(P) = −Σ<sub>p</sub> P(p) log₂ P(p)<br>
IG(F<sub>m</sub>) = H(P) − Σ<sub>v</sub> P(F<sub>m</sub>=v) · H(P | F<sub>m</sub>=v)，P(F<sub>m</sub>=v) = Σ<sub>p</sub> P(p) · L(F<sub>m</sub>=v | p)</div>
</div>
<div class="formula-box">
<h3>⑤ 三带决策（θ_act=0.70，δ=0.20，ε=0.02）</h3>
<div class="formula">p* = P_NONE 且 P(p*) ≥ θ<sub>act</sub> → healthy，动作 RETAIN<br>
P(p*) ≥ θ<sub>act</sub> 且 P(p*) − P(p²) ≥ δ → act，a* = argmin<sub>a</sub> Σ<sub>p</sub> P(p) · L(a, p)（期望损失最小）<br>
否则 → ambiguous：m* = argmax<sub>m∉observed</sub> IG(F<sub>m</sub>)/c<sub>m</sub>；IG(F<sub>m*</sub>) &lt; ε 则 INVESTIGATE 升级，否则先测 F<sub>m*</sub> 回到 ②</div>
</div>
<div class="formula-box">
<h3>⑥ 闭环校准（Laplace 更新，修题→重施测→确认/推翻）</h3>
<div class="formula"><i>s</i>(f, p) ← (α + n(f,p)) / (α + β + n(p))，<i>π</i>(p) ← (α′ + n(p)) / (α′ + β′ + N)</div>
<p class="note">n(f,p)：问题 p 被确认时 finding f 同时出现的次数；n(p)：问题 p 被确认总次数；N：闭环题目总数。规则精度 / 问题解决率可监控。</p>
</div>

<h2>逐题诊断（指标 → finding → 问题后验 → 动作 → 人工核对）</h2>
{''.join(cards)}

<div class="sub">生成方式：experiments/problem_discovery/report_html.py · 数据来源：outputs/virtual_responses/{esc(RUN_ID[:8])}···/{esc(BANK)}/psychometrics/ 与盲法分类结果 · 后验为专家先验下的开发期估计，未接入主工作流</div>
</div></body></html>"""

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "problem_report.html"
    out_path.write_text(html_doc, encoding="utf-8")
    print(f"[report] 已生成 {out_path}")
    print(f"[report] healthy={counts['healthy']} act={counts['act']} "
          f"ambiguous={counts['ambiguous']}")


if __name__ == "__main__":
    main()
