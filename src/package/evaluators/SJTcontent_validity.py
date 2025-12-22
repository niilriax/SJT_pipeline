# -*- coding: utf-8 -*-
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
    """内部工具：把单道题格式化成文本块。"""
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

def format_single_item_prompt(trait_name: str, item: Dict[str, Any]) -> str:
    item_text = _format_single_item_block(trait_name, item)
    template = load_prompt_template()
    prompt = template.replace("【题目内容】", item_text)
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


# ==================== CVI 计算与保存 ====================

def _to_float_score(value: Any) -> Optional[float]:
    """尽量从内容效度评分中提取数字（支持字符串/数值）。"""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            return float(m.group())
    return None


def compute_cvi_from_expert_results(
    expert_results: List[Dict[str, Any]],
    pass_threshold: float = 3.0,
) -> Dict[str, Any]:
    item_scores: Dict[str, List[float]] = {}
    for expert in expert_results or []:
        results = expert.get("results") or []
        for item in results:
            item_id = item.get("题目ID") or item.get("item_id")
            if not item_id:
                continue
            content_eval = item.get("内容效度评估") or {}
            raw_score = content_eval.get("内容效度")
            score = _to_float_score(raw_score)
            if score is None:
                continue
            item_scores.setdefault(str(item_id), []).append(score)
    item_cvi_list: List[Dict[str, Any]] = []
    for item_id, scores in item_scores.items():
        n = len(scores)
        agree = sum(1 for s in scores if s >= pass_threshold)
        i_cvi = round(agree / n, 3) if n else 0.0
        item_cvi_list.append({
            "item_id": item_id,
            "scores": scores,
            "N": n,
            "I-CVI": i_cvi,
        })

    s_cvi_ave = (
        round(sum(x["I-CVI"] for x in item_cvi_list) / len(item_cvi_list), 3)
        if item_cvi_list else 0.0
    )
    return {
        "items": item_cvi_list,
        "S-CVI-Ave": s_cvi_ave,
        "num_items": len(item_cvi_list),
        "num_experts": max((len(v) for v in item_scores.values()), default=0),
        "pass_threshold": pass_threshold,
    }

def save_cvi_results(
    cvi_results: Dict[str, Any],
    filename: str = "cvi_results.json",
) -> Path:
    project_root = get_project_root()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / filename
    payload = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "CVI": cvi_results,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return out_path

def compute_and_save_cvi(
    expert_results: List[Dict[str, Any]],
    filename: str = "cvi_results.json",
    pass_threshold: float = 3.0,
) -> Path:
    """便捷函数：直接计算并保存 CVI，返回文件路径。"""
    cvi = compute_cvi_from_expert_results(expert_results, pass_threshold=pass_threshold)
    return save_cvi_results(cvi, filename=filename)


def load_all_expert_results(
    filename: str = "evaluation_results.json",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    从项目根目录下 results/ 中读取所有特质的专家评估结果。
    预期结构与 workflow.py 写出的 evaluation_results.json 一致。
    """
    project_root = get_project_root()
    results_dir = project_root / "results"
    path = results_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到评估结果文件：{path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    # 顶层结构：{"生成时间": "...", "评价结果": {...}}
    return data.get("评价结果", {})


def main():
    """
    简单测试入口：
    1）从 results/evaluation_results.json 读取 6 个专家的内容效度评估结果
    2）对每个特质计算 CVI
    3）把所有特质的 CVI 汇总保存到 results/cvi_results_all_traits.json
    """
    all_expert_results = load_all_expert_results()
    all_cvi: Dict[str, Any] = {}
    for trait_name, expert_results in all_expert_results.items():
        cvi = compute_cvi_from_expert_results(expert_results)
        all_cvi[trait_name] = cvi
        print(f"{trait_name}：S-CVI-Ave = {cvi.get('S-CVI-Ave')}")
    out_path = save_cvi_results(
        all_cvi,
        filename="cvi_results_all_traits.json",
    )
    print(f"\n所有特质的 CVI 已保存到：{out_path}")
if __name__ == "__main__":
    main()