# -*- coding: utf-8 -*-
# @Time :2025/12/16 16:18:16
# @Author : Scientist
# @File :neo_dataAna.py
# @Software :PyCharm

"""
NEO 自陈题数据分析脚本

功能：
1. 读取 NEO 题目计分规则（docs/Neo-FFI.json），处理反向计分；
2. 读取 GPT 作答记录（src/package/evaluators/neo_reports.json），
   计算每位被试在五大人格维度上的平均作答得分（已处理反向题）；
3. 从提示词（src/package/evaluators/filled_prompts_neo.json）中解析
   每位被试的五大人格 T 分数；
4. 合并成「一人一行」的数据表：
   - 被试ID
   - 提示词 T 分数：N_T, E_T, O_T, A_T, C_T
   - 作答平均分：N_mean, E_mean, O_mean, A_mean, C_mean
5. 计算上述 10 个变量的相关矩阵并导出。
"""

import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# 从公共工具模块导入常量和函数
from .common_utils import (
    NEO_DOMAIN_ORDER,
    load_neo_reverse_flags,
    map_domain_name_to_letter,
    parse_single_report,
    get_project_root,
)


# 使用公共工具函数获取项目根目录
ROOT_DIR = get_project_root()
DOCS_NEO_CONFIG = ROOT_DIR / "docs" / "Neo-FFI.json"
NEO_REPORTS_PATH = ROOT_DIR / "src" / "package" / "evaluators" / "neo_reports.json"
FILLED_PROMPTS_PATH = ROOT_DIR / "src" / "package" / "evaluators" / "filled_prompts_neo.json"


def load_neo_scoring() -> Tuple[Dict[int, str], Dict[int, bool]]:
    """
    从 docs/Neo-FFI.json 读取题目所属维度以及是否反向计分。

    返回：
    - item_domain: {题目编号(int) -> 域名("Neuroticism", ...)}
    - reverse_flags: {题目编号(int) -> 是否为反向题(True/False)}
    """
    with open(DOCS_NEO_CONFIG, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    item_domain: Dict[int, str] = {}
    # 使用公共工具函数加载反向计分标志
    reverse_flags = load_neo_reverse_flags(DOCS_NEO_CONFIG)

    for facet_code, facet_info in cfg.items():
        domain = facet_info["domain"]
        for item_id_str, item_info in facet_info["items"].items():
            item_id = int(item_id_str)
            item_domain[item_id] = domain

    return item_domain, reverse_flags


def load_neo_reports() -> pd.DataFrame:
    """
    读取 neo_reports.json，将每位被试的 Q1~Q30 作答展开成一行。

    兼容两种格式：
    1) report 是 list[{"题目ID": "Q1", "被试选择": "4"}, ...]
    2) report 是 {"Q1": "4", "Q2": "3", ...}
    """
    with open(NEO_REPORTS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows: List[Dict[str, int]] = []

    for rec in raw:
        sid = str(rec.get("被试ID"))
        report_str = rec.get("report")
        if report_str is None:
            continue

        try:
            parsed = json.loads(report_str)
        except json.JSONDecodeError:
            # 如果有奇怪格式，直接跳过该被试
            continue

        answers: Dict[str, int] = {}
        if isinstance(parsed, list):
            for item in parsed:
                qid = item.get("题目ID")
                choice = item.get("被试选择")
                if qid and choice is not None:
                    try:
                        answers[qid] = int(choice)
                    except (TypeError, ValueError):
                        continue
        elif isinstance(parsed, dict):
            for qid, choice in parsed.items():
                if qid and choice is not None:
                    try:
                        answers[qid] = int(choice)
                    except (TypeError, ValueError):
                        continue
        else:
            continue

        if not answers:
            continue

        row: Dict[str, int] = {"被试ID": sid}
        # 统一生成 Q1~Q30，缺失则设为 NaN（稍后由 pandas 处理）
        for i in range(1, 31):
            key = f"Q{i}"
            row[key] = answers.get(key, None)
        rows.append(row)

    df = pd.DataFrame(rows)
    # 尝试把所有 Q* 列转为 numeric
    for i in range(1, 31):
        col = f"Q{i}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def score_domains(df_q: pd.DataFrame, item_domain: Dict[int, str], reverse_flags: Dict[int, bool]) -> pd.DataFrame:
    """
    根据 item_domain 和 reverse_flags 计算五大人格维度的平均作答分。

    假设量表为 1~5 分，反向题记分方式为 new = 6 - old。
    """
    df = df_q.copy()

    # 先做反向计分
    for item_id, is_rev in reverse_flags.items():
        if not is_rev:
            continue
        col = f"Q{item_id}"
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 6 - x if pd.notna(x) else x)

    # 统计每个 domain 对应的题目编号
    domain_items: Dict[str, List[int]] = {}
    for item_id, domain in item_domain.items():
        domain_items.setdefault(domain, []).append(item_id)

    domain_score_cols: Dict[str, str] = {}
    for domain, items in domain_items.items():
        cols = [f"Q{i}" for i in items if f"Q{i}" in df.columns]
        if not cols:
            continue
        mean_col = f"{domain}_mean"
        df[mean_col] = df[cols].mean(axis=1, skipna=True)
        domain_score_cols[domain] = mean_col

    # 只保留被试ID和五个 domain 的均分列
    keep_cols = ["被试ID"] + list(domain_score_cols.values())
    return df[keep_cols].copy()


def parse_t_scores_from_prompt(prompt: str) -> List[float]:
    """
    从 filled_prompts_neo.json 中的一条 prompt 文本里解析五个 T 分数。

    文本示例片段：
    "1.神经质 :中等水平\\T53.46：..."
    使用正则提取连续出现的 "Txx.xx" 中的数值，按顺序返回。
    """
    # 匹配 T 后面的浮点数
    nums = re.findall(r"T(\d+\.\d+)", prompt)
    # 只取前 5 个（神经质、外向性、开放性、宜人性、尽责性）
    nums = nums[:5]
    return [float(x) for x in nums] if len(nums) == 5 else [float("nan")] * 5


def load_prompt_t_scores() -> pd.DataFrame:
    """
    读取 filled_prompts_neo.json，解析每位被试的五个 T 分数。

    返回列：
    - 被试ID
    - N_T, E_T, O_T, A_T, C_T
    """
    with open(FILLED_PROMPTS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for rec in raw:
        sid = str(rec.get("被试ID"))
        prompt = rec.get("prompt", "")
        t_scores = parse_t_scores_from_prompt(prompt)
        row = {
            "被试ID": sid,
            "N_T": t_scores[0],
            "E_T": t_scores[1],
            "O_T": t_scores[2],
            "A_T": t_scores[3],
            "C_T": t_scores[4],
        }
        rows.append(row)

    return pd.DataFrame(rows)


# 使用公共工具函数，无需重复定义


def reshape_domain_means(df_domain: pd.DataFrame, item_domain: Dict[int, str]) -> pd.DataFrame:
    """
    将以英文 domain 命名的 *_mean 列重命名为 N_mean/E_mean/... 形式。
    """
    # 从 item_domain 中推断有哪些 domain
    domains = {d for d in item_domain.values()}

    rename_map: Dict[str, str] = {}
    for domain in domains:
        mean_col = f"{domain}_mean"
        if mean_col in df_domain.columns:
            letter = map_domain_name_to_letter(domain)
            rename_map[mean_col] = f"{letter}_mean"

    return df_domain.rename(columns=rename_map)


def main() -> None:
    # 1. 读入计分规则
    item_domain, reverse_flags = load_neo_scoring()

    # 2. 展开每位被试的 Q1~Q30 作答，并计算五大人格维度均分（已处理反向计分）
    df_q = load_neo_reports()
    df_domain = score_domains(df_q, item_domain, reverse_flags)
    df_domain = reshape_domain_means(df_domain, item_domain)

    # 3. 解析提示词中的五大人格 T 分数
    df_t = load_prompt_t_scores()

    # 4. 合并成一人一行的表
    df_merged = pd.merge(df_t, df_domain, on="被试ID", how="inner")

    # 5. 计算 10 个变量的相关矩阵
    cols_for_corr = ["N_T", "E_T", "O_T", "A_T", "C_T",
                     "N_mean", "E_mean", "O_mean", "A_mean", "C_mean"]
    cols_for_corr = [c for c in cols_for_corr if c in df_merged.columns]
    data_for_corr = df_merged[cols_for_corr]
    corr_matrix = data_for_corr.corr()

    # 6. 计算相关的显著性水平（p 值），并在相关矩阵中直接标星
    n_vars = len(cols_for_corr)
    corr_with_sig = pd.DataFrame("", index=cols_for_corr, columns=cols_for_corr)

    def star_for_p(p: float) -> str:
        if not np.isfinite(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return ""

    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            x = data_for_corr.iloc[:, i]
            y = data_for_corr.iloc[:, j]
            mask = x.notna() & y.notna()
            if mask.sum() >= 3:
                r, p = stats.pearsonr(x[mask], y[mask])
            else:
                # 样本太少无法计算
                r, p = np.nan, np.nan
            corr_matrix.iat[i, j] = corr_matrix.iat[j, i] = r
            corr_with_sig.iat[i, j] = corr_with_sig.iat[j, i] = (
                "" if not np.isfinite(r) else f"{r:.3f}{star_for_p(p)}"
            )

    # 对角线填 1.000
    for k in range(n_vars):
        corr_matrix.iat[k, k] = 1.0
        corr_with_sig.iat[k, k] = "1.000"

    # 7. 导出结果
    out_dir = os.path.join(ROOT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    merged_path = os.path.join(out_dir, "neo_trait_scores.xlsx")
    corr_path = os.path.join(out_dir, "neo_trait_correlations.xlsx")

    with pd.ExcelWriter(merged_path, engine="openpyxl") as writer:
        df_merged.to_excel(writer, index=False, sheet_name="trait_scores")

    with pd.ExcelWriter(corr_path, engine="openpyxl") as writer:
        corr_with_sig.to_excel(writer, sheet_name="correlations_with_sig")

    # 控制台简单打印一下形状和最重要的相关系数，方便快速检查
    print(f"合并后的数据维度: {df_merged.shape}")
    print("相关矩阵：")
    print(corr_matrix)
    print("相关矩阵（含显著性标记）：")
    print(corr_with_sig)


if __name__ == "__main__":
    main()

