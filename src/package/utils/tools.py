# -*- coding: utf-8 -*-
# @Time :2025/12/13 20:31:25
# @Author : Scientist
# @File :tools.py
# @Software :PyCharm

"""
从 results_batch.json 中提取各特质的 edited_results 里的"修订后题目"，
按题号（1-30）排序，分别保存为每个特质一份 SJT 题库 JSON。
"""

import json
from pathlib import Path
from typing import Any, Dict, List
from .common_utils import get_project_root

def extract_sjt_from_generated_items(
    results_path: str = "results_batch.json",
    output_dir: str = "sjt_outputs",
) -> None:
    # 处理 results_path：如果是相对路径，转换为项目根目录的绝对路径
    results_path = Path(results_path)
    if not results_path.is_absolute():
        project_root = get_project_root()
        results_path = project_root / results_path

    if not results_path.exists():
        raise FileNotFoundError(f"results_path 不存在：{results_path}")

    # 处理 output_dir：如果是相对路径，转换为相对于当前文件所在目录的路径
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        # 默认输出到 src/package/utils/sjt_outputs/
        utils_dir = Path(__file__).parent
        output_dir = utils_dir / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取 JSON 文件
    data: Dict[str, Any] = json.loads(results_path.read_text(encoding="utf-8"))
    
    # 统一存到一个文件里，内部按特质分组
    all_traits: Dict[str, Dict[str, Any]] = {}
    for trait_name, trait_block in data.items():
        if not isinstance(trait_block, dict):
            continue
        # 优先使用generated_items，如果没有则尝试edited_results（兼容旧数据）
        items_list: List[Dict[str, Any]] = trait_block.get("generated_items") or []
        if not items_list:
            # 兼容旧格式：尝试从edited_results中提取
            edited_list: List[Dict[str, Any]] = trait_block.get("edited_results") or []
            if edited_list:
                items_list = []
                for er in edited_list:
                    q_info = er.get("修订后题目") or {}
                    if q_info:
                        items_list.append(q_info)
        
        if not items_list:
            continue
        
        sjt_items: List[Dict[str, Any]] = []
        for idx, item in enumerate(items_list, 1):
            # 从generated_items中提取
            item_id = item.get("item_id") or idx
            situation = item.get("situation") or item.get("情境题干", "")
            question = item.get("question") or item.get("提问项", "")
            options = item.get("options") or {}

            normalized_options: Dict[str, Any] = {}
            for opt_key in ["A", "B", "C", "D"]:
                if opt_key in options:
                    normalized_options[opt_key] = options[opt_key]

            sjt_item: Dict[str, Any] = {
                "item_id": item_id,
                "trait": trait_name,
                "situation": situation,
                "question": question,
                "options": normalized_options,
            }

            sjt_items.append(sjt_item)

        sjt_items.sort(key=lambda x: (x.get("item_id") is None, x.get("item_id")))

        all_traits[trait_name] = {
            "trait": trait_name,
            "items": sjt_items,
        }

    merged_obj: Dict[str, Any] = {
        "traits": all_traits
    }
    out_path = output_dir / "SJT_all_traits.json"
    out_path.write_text(
        json.dumps(merged_obj, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"已导出整合 SJT 题目到：{out_path}")


# 为了向后兼容，保留旧函数名
def extract_sjt_from_edited_results(
    results_path: str = "results_batch.json",
    output_dir: str = "sjt_outputs",
) -> None:
    """向后兼容的别名函数"""
    return extract_sjt_from_generated_items(results_path, output_dir)


if __name__ == "__main__":
    extract_sjt_from_generated_items()


