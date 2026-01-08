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
    from package.utils import TRAIT_ORDER, cronbach_alpha, get_project_root
else:
    from . import TRAIT_ORDER, cronbach_alpha, get_project_root


def load_neo_responses(responses_path: Optional[Path] = None) -> List[dict]:
    """
    读取 NEO 被试作答的原始 JSON 列表。
    - 默认路径为 src/package/evaluators/neo_responses.json
    - 如果同级目录存在 neo_responses.json，则优先使用
    """
    if responses_path is None:
        project_root = get_project_root()
        responses_path = project_root / "src" / "package" / "evaluators" / "neo_responses.json"
        alt_path = Path("neo_responses.json")
        if not responses_path.exists() and alt_path.exists():
            responses_path = alt_path
    responses_path = Path(responses_path)
    if not responses_path.exists():
        raise FileNotFoundError(f"找不到 NEO 作答文件: {responses_path}")
    return json.loads(responses_path.read_text(encoding="utf-8"))


def _load_neo_reverse_map(neo_cfg_path: Optional[Path] = None) -> Dict[str, bool]:
    """
    将 docs/Neo-FFI.json 中的反向计分标志转换为 {qid: is_reverse} 映射。
    qid 形如 QN_1、QE_3，与作答文件保持一致。
    """
    if neo_cfg_path is None:
        project_root = get_project_root()
        neo_cfg_path = project_root / "docs" / "Neo-FFI.json"
    neo_cfg_path = Path(neo_cfg_path)
    if not neo_cfg_path.exists():
        return {}
    reverse_map: Dict[str, bool] = {}
    try:
        cfg = json.loads(neo_cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return reverse_map
    for domain_code, domain_info in cfg.items():
        items = domain_info.get("items", {})
        for item_code, item_info in items.items():
            try:
                item_id = item_code.split("_", 1)[1]
            except Exception:
                continue
            qid = f"Q{domain_code}_{item_id}"
            reverse_map[qid] = (item_info.get("scoring", "+") == "-")
    return reverse_map


def neo_responses_to_scored_matrix(
    responses_path: Optional[Path] = None,
    neo_cfg_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    导入 NEO 被试作答并完成计分（含反向计分），返回 DataFrame。

    计分规则：
    - 原始作答为 1~5 分的李克特量表
    - 反向题得分 = 6 - 原始分
    - 结果以题目编号（如 QN_1）为列，按五大维度顺序排序
    """
    responses = load_neo_responses(responses_path)
    reverse_map = _load_neo_reverse_map(neo_cfg_path)

    # 识别所有题目并排序（按维度顺序 + 题号）
    def _qid_key(qid: str):
        try:
            trait_code = qid[1]  # QN_1 -> N
            item_id = int(qid.split("_", 1)[1])
        except Exception:
            trait_code, item_id = "", 0
        trait_rank = {code: idx for idx, (code, _) in enumerate(TRAIT_ORDER)}
        return (trait_rank.get(trait_code, 999), item_id, qid)

    all_qids = set(reverse_map.keys())
    for subj in responses:
        if subj.get("test_type") != "NEO":
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
        if subj.get("test_type") != "NEO":
            continue
        row_scores = {qid: np.nan for qid in all_qids}
        for ans in subj.get("response") or []:
            qid = ans.get("题目ID")
            raw = ans.get("被试选择")
            if qid not in row_scores:
                continue
            try:
                score = float(raw)
            except Exception:
                continue
            if reverse_map.get(qid, False):
                score = 6 - score  # 5 级量表反向计分
            row_scores[qid] = score
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
    # 作为脚本运行时，加载并计分 NEO 作答，输出数据概览
    df = neo_responses_to_scored_matrix()
    print("NEO 作答计分完成")
    print(f"被试数量: {df.shape[0]}, 题目数量: {df.shape[1]}")
    # 显示每个维度前 3 个条目，便于快速核查
    sample_cols = []
    for code, _ in TRAIT_ORDER:
        sample_cols.extend([c for c in df.columns if c.startswith(f"Q{code}_")][:3])
    print("\n示例列（每个维度最多 3 个题目）:")
    print(df[sample_cols].head())

    # 计算并输出各维度的 Cronbach's alpha
    alpha_df = cronbach_alpha_by_trait(df)
    print("\n各维度 Cronbach's alpha：")
    print(alpha_df)
