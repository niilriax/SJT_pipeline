# -*- coding: utf-8 -*-
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
# ==================== 常量定义 ====================
TRAIT_ORDER = [("N", "神经质"), ("E", "外向性"), ("O", "开放性"), ("A", "宜人性"), ("C", "尽责性")]

# ==================== JSON 解析工具 ====================
def extract_json_from_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    if "```json" in text:
        json_start = text.find("```json") + 7
        json_end = text.find("```", json_start)
        if json_end != -1:
            return text[json_start:json_end].strip()
        else:
            return text[json_start:].strip()
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
    max_retries: int = 30,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
) -> List[Dict[str, Any]]:
    def is_network_error(error: Exception) -> bool:
        error_str = str(error).lower()
        network_keywords = [
            'timeout', 'connection', 'network', 'socket', 
            'unreachable', 'refused', 'reset', 'broken pipe',
            'rate limit', '429', '503', '502', '500'
        ]
        return any(keyword in error_str for keyword in network_keywords)
    for attempt in range(max_retries):
        try:
            response = model.invoke([{"role": "user", "content": prompt}])
            response_text = response.content if hasattr(response, 'content') else str(response)
            # 检查响应是否为空
            if not response_text or not response_text.strip():
                if attempt < max_retries - 1:
                    # 指数退避：等待时间 = initial_delay * (2 ^ attempt)，但不超过max_delay
                    delay = min(initial_delay * (2 ** attempt), max_delay)
                    print(f"  ⚠️ 第 {attempt + 1} 次调用返回空响应，等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"  ⚠️ 警告: LLM调用返回空结果（已重试 {max_retries} 次）")
                    return []
            items = parse_json_response(response_text, default_value=[])
            if isinstance(items, dict):
                items = [items]
            elif not isinstance(items, list):
                items = []
            if not items:
                if attempt < max_retries - 1:
                    # JSON解析失败，使用较短的延迟（可能是格式问题，不需要等太久）
                    delay = min(initial_delay * (1.5 ** attempt), 10.0)
                    print(f"  ⚠️ 第 {attempt + 1} 次调用JSON解析失败，等待 {delay:.1f} 秒后重试...")
                    time.sleep(delay)
                    continue
                else:
                    print(f"  ⚠️ 警告: LLM调用返回空结果（已重试 {max_retries} 次）")
                    return []
            # 成功返回
            if attempt > 0:
                print(f"  ✓ 第 {attempt + 1} 次尝试成功")
            return items            
        except Exception as e:
            error_msg = str(e)
            is_network = is_network_error(e)
            if attempt < max_retries - 1:
                if is_network:
                    delay = min(initial_delay * (2 ** attempt), max_delay)
                    print(f"  ⚠️ 第 {attempt + 1} 次调用网络错误: {error_msg[:100]}，等待 {delay:.1f} 秒后重试...")
                else:
                    delay = min(initial_delay * (1.5 ** attempt), 10.0)
                    print(f"  ⚠️ 第 {attempt + 1} 次调用失败: {error_msg[:100]}，等待 {delay:.1f} 秒后重试...")
                time.sleep(delay)
                continue
            else:
                error_type = "网络错误" if is_network else "调用错误"
                print(f"  ❌ LLM调用失败（已重试 {max_retries} 次，{error_type}）: {error_msg}")
                return []
    return []

def LLM_call_concurrent(
    prompts: List[Tuple[str, Any]],
    max_workers: int = 50,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    if not prompts:
        return []
    results: List[Dict[str, Any]] = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(LLM_call, prompt, model, max_retries): idx
            for idx, (prompt, model) in enumerate(prompts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result() or []
                results[idx] = {
                    "index": idx,
                    "result": result,
                    "success": True
                }
            except Exception as e:
                results[idx] = {
                    "index": idx,
                    "result": [],
                    "success": False,
                    "error": str(e)
                }
    return results

'''
def LLM_call_concurrent_experts(
    prompt: str,
    experts: List[Any],
    max_retries: int = 3,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not experts:
        return [], [{"error": "No experts provided"}]
    all_evaluation_results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(experts)) as executor:
        future_to_expert = {
            executor.submit(LLM_call, prompt, expert_model): i
            for i, expert_model in enumerate(experts, start=1)
        }
        for fut in as_completed(future_to_expert):
            expert_index = future_to_expert[fut]
            try:
                eval_result = fut.result() or []
            except Exception as e:
                errors.append({"expert_index": expert_index, "error": repr(e)})
                continue

            if not eval_result:
                retry_ok = False
                for _ in range(max_retries):
                    eval_result = LLM_call(prompt, experts[expert_index - 1]) or []
                    if eval_result:
                        retry_ok = True
                        break
                if not retry_ok:
                    errors.append({"expert_index": expert_index, "error": "Empty return after retries"})
                    continue

            all_evaluation_results.append({
                "expert_index": expert_index,
                "results": eval_result,
            })

    return all_evaluation_results, errors
'''
# ==================== 报告解析工具 ====================
def parse_single_report(report_text: str) -> Dict[str, str]:
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
            try:
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

# ==================== 路径工具 ====================
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]

