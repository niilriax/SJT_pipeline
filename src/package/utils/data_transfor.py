# -*- codeing =utf-8 -*-
# @Time :2025/12/16 15:00:37
# @Author : Scientist
# @File :data_ana.py
# @Software :PyCharm
import pandas as pd
import numpy as np
import glob
import os
import re
from scipy.stats import pearsonr
import json
from pathlib import Path
from typing import Dict, List

# 从公共工具模块导入常量和函数
from .common_utils import (
    TRAIT_CODE_MAP,
    NEO_DOMAIN_ORDER,
    cronbach_alpha,
    parse_single_report,
)

def extract_trait(filename):
    match = re.search(r"[_\-]([A-Z])", filename)
    return match.group(1) if match else None

def export_sjt_reports_to_excels(
    reports_path: str = "src/package/evaluators/sjt_reports.json",
    output_dir: str = "src/package/utils/sjt_outputs",
) -> None:
    """
    将 sjt_reports.json（列表，每条包含被试ID和 report 字符串）按特质拆分为多个 SJT_*.xlsx。
    输出文件命名：SJT_<TraitCode>.xlsx（如 SJT_N.xlsx），保存在 output_dir。
    """
    # 统一把相对路径转换为项目根目录下的绝对路径
    from .common_utils import get_project_root
    project_root = get_project_root()
    reports_path = Path(reports_path)
    if not reports_path.is_absolute():
        reports_path = project_root / reports_path
    if not reports_path.exists():
        raise FileNotFoundError(f"找不到 sjt_reports.json：{reports_path}")

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_data = json.loads(reports_path.read_text(encoding="utf-8"))
    # 按特质累积：code -> list of records
    trait_rows: Dict[str, List[Dict[str, str]]] = {code: [] for code in TRAIT_CODE_MAP.keys()}

    for entry in reports_data:
        subject_id = str(entry.get("被试ID", ""))
        report_text = entry.get("report", "")
        answers = parse_single_report(report_text)
        if not answers:
            continue

        # 按题目 ID 分类
        trait_to_answers: Dict[str, Dict[str, str]] = {code: {} for code in TRAIT_CODE_MAP.keys()}
        for qid, ans in answers.items():
            # 期望格式 QN4_1 / QE2_1 / QO4_1 / QA4_1 / QC5_1
            m = re.match(r"Q([NEOAC])\d?_?\d*", qid)
            if not m:
                continue
            code = m.group(1)
            trait_to_answers.setdefault(code, {})[qid] = ans

        # 将该被试的作答添加到对应特质表
        for code, qa in trait_to_answers.items():
            if not qa:
                continue
            row = {"id": subject_id}
            # 按题目 ID 排序填充
            for q in sorted(qa.keys(), key=lambda x: (len(x), x)):
                row[q] = qa[q]
            trait_rows[code].append(row)

    # 输出到 Excel
    for code, rows in trait_rows.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        # 确保 id 列在最前
        cols = ["id"] + [c for c in df.columns if c != "id"]
        df = df[cols]
        out_path = output_dir / f"SJT_{code}.xlsx"
        df.to_excel(out_path, index=False)
        print(f"✅ 已导出 {TRAIT_CODE_MAP[code]} ({code})：{out_path}")


def _map_qid_to_neo_domain(qid: str) -> str:
    """根据题号映射到 N/E/O/A/C（每 6 题一组，共 30 题）。"""
    m = re.match(r"Q(\d+)", qid)
    if not m:
        return ""
    num = int(m.group(1))
    if num < 1 or num > 30:
        return ""
    idx = (num - 1) // 6  # 0-4
    return NEO_DOMAIN_ORDER[idx][0]


def export_neo_reports_to_excels(
    reports_path: str = "src/package/evaluators/neo_reports.json",
    output_dir: str = "src/package/utils/sjt_outputs",
) -> None:
    """
    将 neo_reports.json 按特质拆分为 自陈_*.xlsx。
    假定 30 题，每 6 题依次对应 N/E/O/A/C。
    """
    from .common_utils import get_project_root
    project_root = get_project_root()
    reports_path = Path(reports_path)
    if not reports_path.is_absolute():
        reports_path = project_root / reports_path
    if not reports_path.exists():
        raise FileNotFoundError(f"找不到 neo_reports.json：{reports_path}")

    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reports_data = json.loads(reports_path.read_text(encoding="utf-8"))
    trait_rows: Dict[str, List[Dict[str, str]]] = {code: [] for code, _ in NEO_DOMAIN_ORDER}

    for entry in reports_data:
        subject_id = str(entry.get("被试ID", ""))
        report_text = entry.get("report", "")
        answers = parse_single_report(report_text)
        if not answers:
            continue

        trait_to_answers: Dict[str, Dict[str, str]] = {code: {} for code, _ in NEO_DOMAIN_ORDER}
        for qid, ans in answers.items():
            domain = _map_qid_to_neo_domain(qid)
            if not domain:
                continue
            trait_to_answers[domain][qid] = ans

        for code, qa in trait_to_answers.items():
            if not qa:
                continue
            row = {"id": subject_id}
            for q in sorted(qa.keys(), key=lambda x: int(re.sub(r"\\D", "", x)) if re.search(r"\\d+", x) else x):
                row[q] = qa[q]
            trait_rows[code].append(row)

    for code, rows in trait_rows.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        cols = ["id"] + [c for c in df.columns if c != "id"]
        df = df[cols]
        out_path = output_dir / f"自陈_{code}.xlsx"
        trait_name = dict(NEO_DOMAIN_ORDER).get(code, code)
        df.to_excel(out_path, index=False)
        print(f"✅ 已导出自陈 {trait_name} ({code})：{out_path}")


# 如果需要单独执行导出，可在命令行运行：
# python src/package/utils/data_ana.py
if __name__ == "__main__":
    export_sjt_reports_to_excels()
    export_neo_reports_to_excels()
