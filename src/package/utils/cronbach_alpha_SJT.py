import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# 兼容脚本直接运行与包内导入
if __package__ is None or __package__ == "":
    # 直接运行时，将项目根目录/src 加入 sys.path
    _project_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(_project_root / "src"))
    from package.utils.common_utils import (
        TRAIT_ORDER,
        cronbach_alpha,
        get_project_root,
    )
    load_sjt_scoring_table = importlib.import_module(
        "package.utils.2PL_IRT_for_SJT"
    ).load_sjt_scoring_table
else:
    from .common_utils import TRAIT_ORDER, cronbach_alpha, get_project_root
    load_sjt_scoring_table = importlib.import_module(
        ".2PL_IRT_for_SJT", package=__package__
    ).load_sjt_scoring_table


def load_sjt_responses(responses_path: Optional[Path] = None) -> List[dict]:
    """
    读取 SJT 被试作答的原始 JSON 列表。
    - 默认路径为 src/package/evaluators/sjt_responses.json
    - 如果同级目录存在 sjt_responses.json，则优先使用
    """
    if responses_path is None:
        project_root = get_project_root()
        responses_path = project_root / "src" / "package" / "evaluators" / "sjt_responses.json"
        alt_path = Path("sjt_responses.json")
        if not responses_path.exists() and alt_path.exists():
            responses_path = alt_path
    responses_path = Path(responses_path)
    if not responses_path.exists():
        raise FileNotFoundError(f"找不到 SJT 作答文件: {responses_path}")
    return json.loads(responses_path.read_text(encoding="utf-8"))


def _load_sjt_scoring(scoring_table: Optional[Dict[str, Dict[str, int]]] = None) -> Dict[str, Dict[str, int]]:
    """加载或直接使用传入的 SJT 计分表。"""
    if scoring_table is not None:
        return scoring_table
    return load_sjt_scoring_table()


def sjt_responses_to_scored_matrix(
    responses_path: Optional[Path] = None,
    scoring_table: Optional[Dict[str, Dict[str, int]]] = None,
) -> pd.DataFrame:
    """
    导入 SJT 被试作答并完成计分，返回 DataFrame。

    计分规则：
    - 使用 SJT 计分表，选项得分由表中给出（高水平=1，低水平=0）
    - 结果以题目编号（如 QN_1）为列，按五大维度顺序排序
    """
    responses = load_sjt_responses(responses_path)
    scoring_table = _load_sjt_scoring(scoring_table)

    # 识别所有题目并排序（按维度顺序 + 题号）
    def _qid_key(qid: str):
        try:
            trait_code = qid[1]  # QN_1 -> N
            item_id = int(qid.split("_", 1)[1])
        except Exception:
            trait_code, item_id = "", 0
        trait_rank = {code: idx for idx, (code, _) in enumerate(TRAIT_ORDER)}
        return (trait_rank.get(trait_code, 999), item_id, qid)

    all_qids = set(scoring_table.keys())
    for subj in responses:
        if subj.get("test_type") != "SJT":
            continue
        for ans in subj.get("response") or []:
            qid = ans.get("题目ID")
            if qid:
                all_qids.add(str(qid))
    all_qids = sorted(all_qids, key=_qid_key)
    if not all_qids:
        return pd.DataFrame()

    rows = []
    for subj in responses:
        if subj.get("test_type") != "SJT":
            continue
        row_scores = {qid: np.nan for qid in all_qids}
        for ans in subj.get("response") or []:
            qid = ans.get("题目ID")
            if qid not in row_scores:
                continue
            choice = ans.get("被试选择")
            if not choice:
                continue
            if qid in scoring_table:
                score = scoring_table[qid].get(choice)
                row_scores[qid] = 0 if score is None else score
        rows.append(row_scores)

    df = pd.DataFrame(rows, columns=all_qids)
    return df.astype(float)


def cronbach_alpha_by_trait(df: pd.DataFrame) -> pd.DataFrame:
    """
    按五大维度计算 Cronbach's alpha。
    返回列：trait_code, trait_name, item_count, alpha
    """
    results = []
    for code, name in TRAIT_ORDER:
        cols = [c for c in df.columns if c.startswith(f"Q{code}_")]
        if not cols:
            continue
        sub_df = df[cols]
        alpha = cronbach_alpha(sub_df)
        results.append(
            {
                "trait_code": code,
                "trait_name": name,
                "item_count": len(cols),
                "alpha": alpha,
            }
        )
    return pd.DataFrame(results, columns=["trait_code", "trait_name", "item_count", "alpha"])


if __name__ == "__main__":
    # 作为脚本运行时，加载并计分 SJT 作答，输出数据概览
    df = sjt_responses_to_scored_matrix()
    print("SJT 作答计分完成")
    print(f"被试数量: {df.shape[0]}, 题目数量: {df.shape[1]}")
    # 显示每个维度前 3 个条目，便于快速核查
    sample_cols = []
    for code, _ in TRAIT_ORDER:
        sample_cols.extend([c for c in df.columns if c.startswith(f"Q{code}_")][:3])
    print("\n示例列（每个维度最多 3 个题目）:")
    if sample_cols:
        print(df[sample_cols].head())
    else:
        print("无示例列可展示")

    # 计算并输出各维度的 Cronbach's alpha
    alpha_df = cronbach_alpha_by_trait(df)
    print("\n各维度 Cronbach's alpha：")
    print(alpha_df if not alpha_df.empty else "无可计算的维度")
