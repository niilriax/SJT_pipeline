# -*- coding: utf-8 -*-
# @Time :2025/12/16 16:50:58
# @Author : Scientist
# @File :sjt_dataAna.py
# @Software :PyCharm

"""
SJT 数据分析脚本：
1) 读取 SJT_all_traits.json 获取每题选项的 trait_level；
2) 将 SJT_[N/E/O/A/C].xlsx 转换为 0/1 计分（high=1, 其他=0）；
3) 将自陈表（自陈_*.xlsx）中的反向题做反向计分（按 1-5 量表，反向：6-x）；
4) 计算 SJT 的克隆巴赫 α（逐维度+整体）；
5) 计算 SJT（0/1 均分）与 NEO 自陈均分的收敛效度（Pearson r, p）。

输出：
- results/sjt_scored_{code}.xlsx     0/1 计分后的 SJT 数据（按特质）
- results/self_report_recoded_{code}.xlsx 反向计分后的自陈数据（若存在）
- results/sjt_analysis.xlsx          可靠性与收敛效度汇总
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# 从公共工具模块导入常量和函数
from .common_utils import (
    TRAIT_NAME_TO_CODE,
    TRAIT_CODE_TO_NAME,
    TRAIT_ORDER,
    cronbach_alpha,
    load_neo_reverse_flags,
    get_project_root,
)


# ----------------------------- 基础路径与映射 -----------------------------

ROOT_DIR = get_project_root()  # 使用公共工具函数
SJT_OUTPUT_DIR = ROOT_DIR / "src" / "package" / "utils" / "sjt_outputs"
SJT_TRAITS_JSON = SJT_OUTPUT_DIR / "SJT_all_traits.json"
DOCS_NEO_JSON = ROOT_DIR / "docs" / "Neo-FFI.json"  # 用于识别自陈的反向题
RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# 从 TRAIT_ORDER 提取特质代码列表，避免硬编码
TRAIT_CODES = [code for code, _ in TRAIT_ORDER]


def _parse_item_id(col: str) -> Tuple[str, int]:
    """
    从列名中解析出 trait_code 和 item_id。
    兼容格式：Q{code}_{id}, Q{code}{id}, Q{code}{sub}_{id}
    """
    m = re.search(r"Q([NEOAC])[A-Za-z]*_?(\d+)", col)
    if not m:
        return "", -1
    return m.group(1), int(m.group(2))


def _detect_reverse(col: str) -> bool:
    """
    检测自陈表列名是否为反向题（列名规则）：
    规则：列名包含 '_R' / '_r' / '-R' / '(反向)' / '(负向)' / 末尾 '-'
    """
    return bool(re.search(r"(_r|_R|\(反向\)|\(负向\)|-$)", col))


# ----------------------------- 元数据加载 -----------------------------

def load_sjt_metadata() -> Dict[str, Dict[int, Dict[str, str]]]:
    """
    返回结构：meta[trait_code][item_id][option_letter] = trait_level
    trait_level 取值 high/medium/low/medium_high/medium_low 等
    """
    data = json.loads(SJT_TRAITS_JSON.read_text(encoding="utf-8"))
    traits = data.get("traits", {})
    meta: Dict[str, Dict[int, Dict[str, str]]] = {}
    for trait_name, block in traits.items():
        code = TRAIT_NAME_TO_CODE.get(trait_name)
        if not code:
            continue
        for item in block.get("items", []):
            item_id = item.get("item_id")
            options = item.get("options", {})
            opt_levels = {}
            for opt_key, opt_val in options.items():
                level = opt_val.get("trait_level")
                opt_levels[opt_key.upper()] = level
            meta.setdefault(code, {})[item_id] = opt_levels
    return meta


# ----------------------------- SJT 0/1 计分 -----------------------------

def score_sjt_file(xlsx_path: Path, meta: Dict[str, Dict[int, Dict[str, str]]]) -> pd.DataFrame:
    """
    将单个 SJT_[code].xlsx 转换为 0/1 计分（high=1，其余=0）。
    """
    df = pd.read_excel(xlsx_path)
    df_scored = df.copy()
    for col in df.columns:
        if col.lower() in ["id", "被试id", "subject_id"]:
            continue
        trait_code, item_id = _parse_item_id(col)
        if trait_code == "" or item_id == -1:
            continue
        opt_levels = meta.get(trait_code, {}).get(item_id, {})
        def _score(val):
            if pd.isna(val):
                return np.nan
            opt = str(val).strip().upper()
            level = opt_levels.get(opt)
            if level is None:
                return np.nan
            return 1 if level == "high" else 0
        df_scored[col] = df[col].apply(_score)
    return df_scored


# ----------------------------- 自陈表反向计分 -----------------------------

# 使用公共工具函数，无需重复定义


def recode_self_report(xlsx_path: Path, scale_max: int = 5) -> pd.DataFrame:
    """
    对自陈表中反向题做反向计分：x -> scale_max+1 - x。
    反向识别优先级：
    1) 列名形如 Q{数字}，且在 docs/Neo-FFI.json 中标记 scoring == "-"；
    2) 列名包含标记（_R / _r / (反向) / (负向) / 末尾 -）。
    """
    df = pd.read_excel(xlsx_path)
    df_recoded = df.copy()
    reverse_flags = load_neo_reverse_flags(DOCS_NEO_JSON)
    for col in df.columns:
        if col.lower() in ["id", "被试id", "subject_id"]:
            continue
        # 规则 1：Q{num} 且在 reverse_flags 中标记
        m = re.match(r"Q(\d+)", str(col))
        is_rev = False
        if m:
            try:
                item_id = int(m.group(1))
                is_rev = reverse_flags.get(item_id, False)
            except Exception:
                is_rev = False
        # 规则 2：列名自带反向标记
        if not is_rev:
            is_rev = _detect_reverse(col)

        if is_rev:
            df_recoded[col] = df[col].apply(
                lambda x: scale_max + 1 - x if pd.notna(x) else x
            )
    return df_recoded


# ----------------------------- 收敛效度 -----------------------------

def compute_convergent_validity(df_sjt_mean: pd.DataFrame, neo_scores_path: Path) -> pd.DataFrame:
    """
    df_sjt_mean: 列 ['被试ID', 'N_sjt', ...] （或 id）
    neo_scores_path: results/neo_trait_scores.xlsx
    返回：DataFrame，行=特质，列=[r, p, n]
    """
    if not neo_scores_path.exists():
        return pd.DataFrame()
    df_neo = pd.read_excel(neo_scores_path, sheet_name=0)
    # 统一 id 名称
    id_cols = [c for c in df_sjt_mean.columns if c.lower() in ["被试id", "id", "subject_id"]]
    sjt_id_col = id_cols[0] if id_cols else df_sjt_mean.columns[0]
    df_sjt_mean = df_sjt_mean.rename(columns={sjt_id_col: "被试ID"})
    # NEO 列名：N_mean, E_mean, ...
    neo_id_col = [c for c in df_neo.columns if c.lower() in ["被试id", "id", "subject_id"]]
    neo_id_col = neo_id_col[0] if neo_id_col else "被试ID"
    df_neo = df_neo.rename(columns={neo_id_col: "被试ID"})

    merged = pd.merge(df_sjt_mean, df_neo, on="被试ID", how="inner")
    results = []
    for code in TRAIT_CODES:
        sjt_col = f"{code}_sjt"
        neo_col = f"{code}_mean"
        if sjt_col not in merged or neo_col not in merged:
            continue
        x = pd.to_numeric(merged[sjt_col], errors="coerce")
        y = pd.to_numeric(merged[neo_col], errors="coerce")
        mask = x.notna() & y.notna()
        n = mask.sum()
        if n < 3:
            r, p = np.nan, np.nan
        else:
            r, p = stats.pearsonr(x[mask], y[mask])
        results.append({"trait": code, "r": r, "p": p, "n": n})
    return pd.DataFrame(results)


# ----------------------------- 主流程 -----------------------------

def main() -> None:
    meta = load_sjt_metadata()
    RESULTS_DIR.mkdir(exist_ok=True)

    # 1) SJT 0/1 计分
    sjt_mean_rows = []
    reliability_rows = []
    for code in TRAIT_CODES:
        xlsx_path = SJT_OUTPUT_DIR / f"SJT_{code}.xlsx"
        if not xlsx_path.exists():
            continue
        df_scored = score_sjt_file(xlsx_path, meta)
        # 均分
        score_cols = [c for c in df_scored.columns if c.lower() not in ["id", "被试id", "subject_id"]]
        df_scored[f"{code}_sjt"] = df_scored[score_cols].mean(axis=1, skipna=True)

        # 可靠性
        alpha = cronbach_alpha(df_scored[score_cols]) if score_cols else np.nan
        reliability_rows.append({"trait": code, "cronbach_alpha": alpha, "k": len(score_cols)})

        # 保存计分后的表
        out_path = RESULTS_DIR / f"sjt_scored_{code}.xlsx"
        df_scored.to_excel(out_path, index=False)

        # 收敛效度用的均分
        id_col = [c for c in df_scored.columns if c.lower() in ["id", "被试id", "subject_id"]]
        id_col = id_col[0] if id_col else df_scored.columns[0]
        sjt_mean_rows.append(df_scored[[id_col, f"{code}_sjt"]])

    # 汇总 SJT 均分
    df_sjt_mean = None
    if sjt_mean_rows:
        df_sjt_mean = sjt_mean_rows[0]
        for extra in sjt_mean_rows[1:]:
            df_sjt_mean = pd.merge(df_sjt_mean, extra, on=df_sjt_mean.columns[0], how="outer")

    # 2) 自陈表反向计分
    self_report_files = [p for p in SJT_OUTPUT_DIR.glob("自陈_*.xlsx")]
    for sr_path in self_report_files:
        df_sr = recode_self_report(sr_path, scale_max=5)
        out_path = RESULTS_DIR / f"self_report_recoded_{sr_path.stem.replace('自陈_', '')}.xlsx"
        df_sr.to_excel(out_path, index=False)

    # 3) 整体可靠性（合并所有 SJT 题目）
    if sjt_mean_rows:
        # 读取已保存的 scored 文件并拼接
        all_cols = []
        for code in TRAIT_CODES:
            path = RESULTS_DIR / f"sjt_scored_{code}.xlsx"
            if not path.exists():
                continue
            df = pd.read_excel(path)
            cols = [c for c in df.columns if c.lower() not in ["id", "被试id", "subject_id", f"{code}_sjt"]]
            all_cols.append(df[cols])
        if all_cols:
            df_all = pd.concat(all_cols, axis=1)
            reliability_rows.append({"trait": "ALL", "cronbach_alpha": cronbach_alpha(df_all), "k": df_all.shape[1]})

    reliability_df = pd.DataFrame(reliability_rows)

    # 4) 收敛效度（SJT 均分 vs NEO 均分）
    convergent_df = pd.DataFrame()
    if df_sjt_mean is not None:
        neo_scores_path = RESULTS_DIR / "neo_trait_scores.xlsx"
        convergent_df = compute_convergent_validity(df_sjt_mean, neo_scores_path)

    # 5) 汇总输出
    summary_path = RESULTS_DIR / "sjt_analysis.xlsx"
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        reliability_df.to_excel(writer, index=False, sheet_name="reliability")
        if not convergent_df.empty:
            convergent_df.to_excel(writer, index=False, sheet_name="convergent_validity")
        if df_sjt_mean is not None:
            df_sjt_mean.to_excel(writer, index=False, sheet_name="sjt_trait_mean")

    print(f"✅ SJT 计分、可靠性与收敛效度已输出到：{summary_path}")


if __name__ == "__main__":
    main()
