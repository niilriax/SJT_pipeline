# -*- coding: utf-8 -*-
import csv
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from package.utils.common_utils import get_project_root
from langchain_openai import ChatOpenAI


def load_prompt_template() -> str:
    prompt_file = Path(__file__).parent / "prompts" / "content_validity_prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
    return prompt_file.read_text(encoding="utf-8")

def _format_single_item_block(trait_name: str, item: Dict[str, Any]) -> str:
    item_id = item.get("item_id")
    situation = item.get("situation", "")
    question = item.get("question", "")
    options = item.get("options", {})
    options_text = ""
    for key, value in options.items():
        if isinstance(value, dict):
            content = value.get("content", "")
            trait_level = value.get("trait_level", "")
            options_text += f"{key}. {content} (特质水平: {trait_level})\n"
        else:
            options_text += f"{key}. {value}\n"
    item_content = (
        f"题目ID: {item_id} 特质类型：{trait_name} 情境：{situation} "
        f"提问：{question} 选项：{options_text}"
    )
    return item_content

def format_all_items_prompt(data: Dict[str, Any]) -> str:
    all_items_content = []
    for trait_name, items in data.get("特质", {}).items():
        for item in items:
            item_content = _format_single_item_block(trait_name, item)
            all_items_content.append(item_content)
    items_text = "\n\n".join(all_items_content)
    template = load_prompt_template()
    prompt = template.replace("【题目内容】", items_text)
    return prompt

def get_content_validity_experts() -> List[ChatOpenAI]:
    experts: List[ChatOpenAI] = [
        ChatOpenAI(model="gpt-5.1", temperature=1, max_tokens=6000),
        ChatOpenAI(model="gpt-5.1", temperature=0.8, max_tokens=6000),
        ChatOpenAI(model="gpt-5.1", temperature=0.6, max_tokens=6000),
        ChatOpenAI(model="gpt-5.1", temperature=0.4, max_tokens=6000),
        ChatOpenAI(model="gpt-5.1", temperature=0.2, max_tokens=6000),
        ChatOpenAI(model="gpt-5.1", temperature=0, max_tokens=6000),
    ]
    return experts

def _extract_score_from_text(score_text: Any) -> Optional[int]:
    """从文本中提取分数（1-4）"""
    if score_text is None:
        return None
    if isinstance(score_text, (int, float)):
        score = int(score_text)
        if 1 <= score <= 4:
            return score
        return None
    if isinstance(score_text, str):
        # 尝试提取数字
        match = re.search(r'[1-4]', score_text)
        if match:
            return int(match.group())
    return None


from typing import List, Dict, Any, Optional

def calculate_cvi_from_evaluation_results(
    evaluation_results: List[Dict[str, Any]], 
    generated_items: Optional[List[Dict[str, Any]]] = None,
    expert_count: int = 6 
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    item_map: Dict[str, Dict[str, Any]] = {}
    if generated_items:
        for item in generated_items:
            item_id = str(item.get("item_id", ""))
            if item_id:
                item_map[item_id] = item
    item_scores: Dict[str, Dict[int, Optional[float]]] = {}
    item_evaluations: Dict[str, Dict[int, Dict[str, Any]]] = {} 
    for expert_result in evaluation_results:
        try:
            raw_idx = expert_result.get("expert_index", 0)
            expert_index = int(raw_idx)
        except (ValueError, TypeError):
            continue
        results = expert_result.get("results", [])
        for item in results:
            item_id = str(item.get("题目ID") or item.get("item_id") or "")
            if not item_id:
                continue
            if item_id not in item_scores:
                item_scores[item_id] = {}
                item_evaluations[item_id] = {}
            
            eval_data = item.get("内容效度评估", {})
            content_validity = eval_data.get("内容效度") if isinstance(eval_data, dict) else None
            try:
                score = _extract_score_from_text(content_validity)
            except Exception:
                score = None
            item_scores[item_id][expert_index] = score
            item_evaluations[item_id][expert_index] = item  
    cvi_data = []
    low_cvi_items = []
    passed_items = []
    cvi_threshold = 0.833
    for item_id, expert_map in item_scores.items():
        ordered_scores = []
        for i in range(1, expert_count + 1):
            ordered_scores.append(expert_map.get(i, None))
        valid_scores = [s for s in ordered_scores if s is not None]
        if valid_scores:
            high_scores = [s for s in valid_scores if s >= 3]
            i_cvi = len(high_scores) / len(valid_scores)
        else:
            i_cvi = 0.0
        row = {
            "题目ID": item_id,
            "I-CVI": round(i_cvi, 3)
        }
        for i in range(expert_count):
            row[f"专家{i+1}"] = ordered_scores[i]
        cvi_data.append(row)
        if i_cvi < cvi_threshold:
            original_item = item_map.get(item_id, {})
            expert_evaluations = []
            for expert_idx in range(1, expert_count + 1):
                evaluation_item = item_evaluations.get(item_id, {}).get(expert_idx)
                if evaluation_item:
                    expert_evaluations.append({
                        "expert_index": expert_idx,
                        "evaluation": evaluation_item
                    })
            low_cvi_item = {
                "item_id": item_id,
                "original_item": original_item,  
                "i_cvi": round(i_cvi, 3),
                "expert_evaluations": expert_evaluations
            }
            low_cvi_items.append(low_cvi_item)
        else:
            original_item = item_map.get(item_id, {})
            if original_item:
                passed_items.append(original_item)
    return cvi_data, low_cvi_items, passed_items


def calculate_cvi_from_evaluation_results_single_expert(
    evaluation_results: List[Dict[str, Any]], 
    generated_items: Optional[List[Dict[str, Any]]] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    item_map: Dict[str, Dict[str, Any]] = {}
    if generated_items:
        for item in generated_items:
            item_id = str(item.get("item_id", ""))
            if item_id:
                item_map[item_id] = item
    item_scores: Dict[str, Optional[int]] = {}
    item_evaluations: Dict[str, Dict[str, Any]] = {}
    if evaluation_results:
        expert_result = evaluation_results[0]
        results = expert_result.get("results", [])
        for item in results:
            item_id = str(item.get("题目ID") or item.get("item_id") or "")
            if not item_id:
                continue
            eval_data = item.get("内容效度评估", {})
            content_validity = eval_data.get("内容效度") if isinstance(eval_data, dict) else None
            try:
                score = _extract_score_from_text(content_validity)
            except Exception:
                score = None
            item_scores[item_id] = score
            item_evaluations[item_id] = item
    cvi_data = []
    low_cvi_items = []
    passed_items = []
    for item_id, score in item_scores.items():
        row = {
            "题目ID": item_id,
            "专家分数": score
        }
        cvi_data.append(row)
        original_item = item_map.get(item_id, {})
        evaluation_item = item_evaluations.get(item_id, {})
        if score is None:
            low_cvi_item = {
                "item_id": item_id,
                "original_item": original_item,
                "expert_evaluations": [{
                    "expert_index": 1,
                    "evaluation": evaluation_item
                }] if evaluation_item else []
            }
            low_cvi_items.append(low_cvi_item)
        elif score in [1, 2]:
            low_cvi_item = {
                "item_id": item_id,
                "original_item": original_item,
                "expert_evaluations": [{
                    "expert_index": 1,
                    "evaluation": evaluation_item
                }] if evaluation_item else []
            }
            low_cvi_items.append(low_cvi_item)
        elif score in [3, 4]:
            if original_item:
                passed_items.append(original_item)
    return cvi_data, low_cvi_items, passed_items


def convert_evaluation_results_to_csv(evaluation_results: List[Dict[str, Any]], output_path: Optional[Path] = None) -> Path:
    rows = []
    for expert_result in evaluation_results:
        expert_index = expert_result.get("expert_index", "")
        for item in expert_result.get("results", []):
            row = {
                "专家索引": expert_index,
                "题目ID": item.get("题目ID", ""),
                "待评估特质": item.get("待评估特质", ""),
            }
            eval_data = item.get("内容效度评估", {})
            dim_scores = eval_data.get("维度评分", {})
            for dim, score_text in dim_scores.items():
                row[dim] = score_text
            row["内容效度"] = eval_data.get("内容效度", "")
            rows.append(row)
    if not rows:
        raise ValueError("评估结果为空")
    if output_path is None:
        project_root = get_project_root()
        trait_name = rows[0].get("待评估特质", "")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = project_root / "results" / f"evaluation_table_{trait_name}_{timestamp}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return output_path