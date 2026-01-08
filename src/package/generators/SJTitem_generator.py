from pathlib import Path
from typing import List, Dict, Any, Optional

def load_prompt_template():
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "prompts" / "item_generation_prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template

def load_replacement_prompt_template():
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "prompts" / "item_replacement_prompt.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    return prompt_template

def format_prompt(target_trait: str):
    prompt = load_prompt_template()
    prompt = prompt.replace("【目标特质】", target_trait)
    return prompt

def build_prompt_with_suggestions(
    modification_suggestions: Optional[List[Dict[str, Any]]] = None
) -> str:
    prompt = load_replacement_prompt_template()
    suggestions_text = ""
    if modification_suggestions:
        for idx, suggestion in enumerate(modification_suggestions, 1):
            item_id = suggestion.get("题目ID", f"题目{idx}")
            content_validity = suggestion.get("内容效度评估", {})
            if content_validity:
                dimension_scores = content_validity.get("维度评分", {})
                if dimension_scores:
                    suggestions_text += f"\n题目 {item_id} 的评估建议：\n"
                    for dimension, feedback in dimension_scores.items():
                        suggestions_text += f"  - {dimension}: {feedback}\n"
    if suggestions_text:
        prompt = prompt.replace("【低内容效度题目及专家意见】", suggestions_text)
    else:
        prompt = prompt.replace("【低内容效度题目及专家意见】", "（暂无评估建议）")
    return prompt