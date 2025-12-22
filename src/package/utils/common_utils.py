# -*- coding: utf-8 -*-
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

# ==================== 常量定义 ====================
TRAIT_ORDER = [("N4", "神经质"), ("E2", "外向性"), ("O4", "开放性"), ("A4", "宜人性"), ("C5", "尽责性")]

TRAIT_CODE_MAP = {
    "N": "神经质",
    "E": "外向性",
    "O": "开放性",
    "A": "宜人性",
    "C": "尽责性",
}

TRAIT_NAME_TO_CODE = {
    "神经质": "N",
    "外向性": "E",
    "开放性": "O",
    "宜人性": "A",
    "尽责性": "C",
}

TRAIT_CODE_TO_NAME = {v: k for k, v in TRAIT_NAME_TO_CODE.items()}

NEO_DOMAIN_ORDER = [("N", "神经质"), ("E", "外向性"), ("O", "开放性"), ("A", "宜人性"), ("C", "尽责性")]

# ==================== JSON 解析工具 ====================

def extract_json_from_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    
    # 尝试提取 ```json ... ```
    if "```json" in text:
        json_start = text.find("```json") + 7
        json_end = text.find("```", json_start)
        if json_end != -1:
            return text[json_start:json_end].strip()
        else:
            return text[json_start:].strip()
    # 尝试提取 ``` ... ```
    elif "```" in text:
        json_start = text.find("```") + 3
        json_end = text.find("```", json_start)
        if json_end != -1:
            return text[json_start:json_end].strip()
        else:
            return text[json_start:].strip()
    
    return text


def parse_json_response(response_text: str, default_value: Any = None) -> Any:
    try:
        json_text = extract_json_from_text(response_text)
        if not json_text:
            return default_value
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return default_value


# ==================== LLM 调用工具 ====================

def LLM_call(
    prompt: str,
    model: Any,
) -> List[Dict[str, Any]]:
    try:
        response = model.invoke([{"role": "user", "content": prompt}])
        response_text = response.content if hasattr(response, 'content') else str(response)
        items = parse_json_response(response_text, default_value=[])
        if isinstance(items, dict):
            items = [items]
        elif not isinstance(items, list):
            items = []
        return items
    except Exception as e:
        print(f"  生成题目失败: {str(e)}")
        return []


# ==================== 报告解析工具 ====================

def parse_single_report(report_text: str) -> Dict[str, str]:
    """
    解析单个被试的报告文本，提取题目ID和答案的映射。
    
    支持多种格式：
    1. 直接 JSON 对象：{"Q1": "4", "Q2": "3", ...}
    2. JSON 数组：[{"题目ID": "Q1", "被试选择": "4"}, ...]
    3. 多个 JSON 对象串接
    4. 正则提取格式
    
    Args:
        report_text: 报告文本
        
    Returns:
        {题目ID: 答案} 的字典
    """
    report_text = report_text.strip()
    if not report_text:
        return {}

    # 尝试直接解析为 JSON
    try:
        obj = json.loads(report_text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, list):
            result = {}
            for item in obj:
                if not isinstance(item, dict):
                    continue
                qid = item.get("题目ID") or item.get("question_id") or item.get("id")
                ans = item.get("被试选择") or item.get("answer")
                if qid and isinstance(ans, str):
                    result[qid] = ans
            return result
    except json.JSONDecodeError:
        pass

    # 若是多个 JSON 对象串接，尝试逐段解析
    results: Dict[str, str] = {}
    parts = re.findall(r"\{[^{}]+\}", report_text)
    for part in parts:
        try:
            obj = json.loads(part)
            if isinstance(obj, dict):
                qid = obj.get("题目ID") or obj.get("question_id") or obj.get("id")
                ans = obj.get("被试选择") or obj.get("answer")
                if qid and isinstance(ans, str):
                    results[qid] = ans
        except json.JSONDecodeError:
            continue
    if results:
        return results

    # 最后使用正则提取 `"Q...": "A"` 或 `"题目ID": "Q..." "被试选择": "A"`
    pattern_kv = re.compile(r'"(Q[NEOAC]\d?_\d+)":\s*"([A-D])"')
    for qid, ans in pattern_kv.findall(report_text):
        results[qid] = ans

    pattern_cn = re.compile(r'"题目ID"\s*:\s*"([^"]+)"[^"]*"被试选择"\s*:\s*"([^"]+)"')
    for qid, ans in pattern_cn.findall(report_text):
        results[qid] = ans

    return results


# ==================== 统计分析工具 ====================

def cronbach_alpha(df: pd.DataFrame) -> float:
    """
    计算克隆巴赫α系数。
    
    Args:
        df: 数据框，每列为一个题目，每行为一个被试
        
    Returns:
        克隆巴赫α系数，如果无法计算则返回 np.nan
    """
    df = df.dropna(axis=0)
    if df.shape[1] <= 1:
        return np.nan
    k = df.shape[1]
    item_var = df.var(axis=0, ddof=1)
    total_var = df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - (item_var.sum() / total_var))


# ==================== NEO 相关工具 ====================

def load_neo_reverse_flags(neo_json_path: Optional[Path] = None) -> Dict[int, bool]:
    """从 Neo-FFI.json 读取题目的反向计分标志"""
    reverse_flags: Dict[int, bool] = {}
    
    if neo_json_path is None:
        # 默认路径：项目根目录/docs/Neo-FFI.json
        project_root = get_project_root()
        neo_json_path = project_root / "docs" / "Neo-FFI.json"
    
    neo_json_path = Path(neo_json_path)
    if not neo_json_path.exists():
        return reverse_flags
    
    try:
        cfg = json.loads(neo_json_path.read_text(encoding="utf-8"))
    except Exception:
        return reverse_flags
    
    for facet_info in cfg.values():
        for item_id_str, item_info in facet_info.get("items", {}).items():
            # 处理格式：N_1, E_2 等，提取数字部分
            try:
                # 如果格式是 "N_1"，提取 "1"
                if "_" in item_id_str:
                    item_id = int(item_id_str.split("_")[1])
                else:
                    item_id = int(item_id_str)
            except Exception:
                continue
            scoring_flag = item_info.get("scoring", "+")
            reverse_flags[item_id] = (scoring_flag == "-")
    
    return reverse_flags


def map_domain_name_to_letter(domain: str) -> str:
    mapping = {
        "Neuroticism": "N",
        "Extraversion": "E",
        "Openness": "O",
        "Agreeableness": "A",
        "Conscientiousness": "C",
    }
    return mapping.get(domain, domain)


# ==================== 评估统计工具 ====================

def calculate_evaluation_stats(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_cvis = []
    passed_items = []
    for eval_result in evaluation_results:
        i_cvi = eval_result.get("I-CVI", 0.0)
        if i_cvi > 0:
            all_cvis.append(i_cvi)
        if eval_result.get("通过", False):
            passed_items.append(eval_result)
    avg_cvi = round(sum(all_cvis) / len(all_cvis), 3) if all_cvis else 0.0
    overall_passed = avg_cvi >= 0.8
    return {
        "平均CVI": avg_cvi,
        "整体通过": overall_passed,
        "通过条目数": len(passed_items),
        "总条目数": len(evaluation_results),
        "all_cvis": all_cvis,
        "passed_items": passed_items
    }


# ==================== 路径工具 ====================

def get_project_root() -> Path:
    """
    获取项目根目录路径。
    
    Returns:
        项目根目录的 Path 对象
    """
    # 从 utils 目录向上三级：utils -> package -> src -> demo（项目根目录）
    # parents[0] = utils/, parents[1] = package/, parents[2] = src/, parents[3] = demo/
    return Path(__file__).resolve().parents[3]

