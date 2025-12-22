from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from ..utils import TRAIT_ORDER, get_project_root

_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = get_project_root() 


def load_neo_items(neo_path: Optional[Path] = None) -> Dict[str, dict]:
    """
    读取 NEO-FFI 题目文件（docs/Neo-FFI.json）。
    返回结构与原 JSON 一致：{ domain_code: { domain, items{ id: {item, scoring} } } }
    支持格式：Neo-FFI.json (domain_code: N, E, O, A, C)
    """
    neo_path = neo_path or (_PROJECT_ROOT / "docs" / "Neo-FFI.json")
    neo_path = Path(neo_path)
    if not neo_path.exists():
        raise FileNotFoundError(f"找不到 NEO-FFI 题目文件：{neo_path}")
    return json.loads(neo_path.read_text(encoding="utf-8"))


def build_neo_items_block(neo_items: Dict[str, dict]) -> str:

    flat_items = []
    for facet_code, facet in neo_items.items():
        items = facet.get("items", {})
        for item_id, item_info in items.items():
            try:
                order = int(item_id)
            except ValueError:
                # 非数字 ID 就放在最后，保持原字符串顺序
                order = float("inf")
            flat_items.append(
                (
                    order,
                    item_id,
                    str(item_info.get("item", "")).strip(),
                )
            )

    # 按题号排序
    flat_items.sort(key=lambda x: x[0])

    lines = []
    for _, item_id, text in flat_items:
        if not text:
            continue
        lines.append(f"Q{item_id}. {text}")

    return "\n".join(lines)


def load_sjt_items(sjt_json_path: Optional[Path] = None) -> Dict[str, Any]:
    if sjt_json_path is None:
        sjt_json_path = _BASE_DIR.parent / "utils" / "sjt_outputs" / "SJT_all_traits.json"
    sjt_json_path = Path(sjt_json_path)
    if not sjt_json_path.exists():
        raise FileNotFoundError(f"找不到 SJT 题目文件：{sjt_json_path}")
    return json.loads(sjt_json_path.read_text(encoding="utf-8"))

def build_sjt_items_block(sjt_data: Dict[str, Any]) -> str:
    traits_data = sjt_data.get("traits", {})
    lines = []
    
    trait_name_to_code = {name: code for code, name in TRAIT_ORDER}
    trait_names = [name for _, name in TRAIT_ORDER]
    for trait_name in trait_names:
        if trait_name not in traits_data:
            continue
        trait_block = traits_data[trait_name]
        items = trait_block.get("items", [])
        items_sorted = sorted(items, key=lambda x: x.get("item_id", 0))
        
        for item in items_sorted:
            item_id = item.get("item_id", 0)
            situation = item.get("situation", "")
            question = item.get("question", "")
            options = item.get("options", {})
            
            if not situation or not question:
                continue

            trait_code = trait_name_to_code.get(trait_name, trait_name[:1])
            q_text = f"Q{trait_code}_{item_id}. {situation} {question}\n"

            for opt_key in ["A", "B", "C", "D"]:
                if opt_key in options:
                    opt_content = options[opt_key].get("content", "")
                    if opt_content:
                        q_text += f"  {opt_key}. {opt_content}\n"
            
            lines.append(q_text.strip())
    
    return "\n\n".join(lines)


def fill_template_with_sjt_items(template: str, sjt_json_path: Optional[Path] = None) -> str:
    if sjt_json_path is None or not Path(sjt_json_path).exists():
        return template
    
    sjt_data = load_sjt_items(sjt_json_path)
    items_block = build_sjt_items_block(sjt_data)
    
    # 支持两种占位符格式
    if "【SJT_ITEMS】" in template:
        return template.replace("【SJT_ITEMS】", items_block)
    elif "【情境判断题目】" in template:
        return template.replace("【情境判断题目】", items_block)
    
    return template


def fill_template_with_neo_items(template: str, neo_path: Optional[Path] = None) -> str:
    placeholder = "【NEO_FFI_R】"
    if placeholder not in template:
        return template

    neo_items = load_neo_items(neo_path)
    items_block = build_neo_items_block(neo_items)
    return template.replace(placeholder, items_block)


def load_template(template_path: Optional[Path] = None) -> str:
    template_path = template_path or (_BASE_DIR / "prompts" / "scorer_prompt.txt")
    return Path(template_path).read_text(encoding="utf-8")


def load_subjects(csv_path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = csv_path or (_BASE_DIR.parent / "generators" / "virtual_subjects.csv")
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到被试 CSV 文件：{csv_path}")
    return pd.read_csv(csv_path)


def fill_template_with_scores_only(template: str, scores: Dict[str, float]) -> str:
    filled = template
    for idx, (code, trait_name) in enumerate(TRAIT_ORDER, start=1):
        if code not in scores:
            raise ValueError(f"缺少特质 {code} 的分数，当前可用键：{list(scores.keys())}")

        value = float(scores[code])

        # 仅替换分数相关占位符
        filled = filled.replace(f"【T分数{idx}】", f"T{value:.2f}", 1)
        filled = filled.replace("【T分数】", f"T{value:.2f}", 1)

    return filled


def generate_filled_prompts_with_scores_only(
    csv_path: Optional[Path] = None,
    template_path: Optional[Path] = None,
    sjt_json_path: Optional[Path] = None,
    neo_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> List[Dict[str, str]]:
    # 根据是否有 SJT 题目来选择模板
    if template_path is None:
        if sjt_json_path is not None and Path(sjt_json_path).exists():
            # 使用 SJT 模板
            template_path = _BASE_DIR / "prompts" / "scorer_prompt.txt"
        else:
            # 使用 NEO 模板
            template_path = _BASE_DIR / "prompts" / "scorer_forNEO_prompt.txt"

    template = load_template(template_path)
    template = fill_template_with_neo_items(template, neo_path)
    template = fill_template_with_sjt_items(template, sjt_json_path)
    subjects = load_subjects(csv_path)

    if "被试ID" not in subjects.columns:
        subjects = subjects.copy()
        subjects["被试ID"] = subjects.index.astype(str)

    results: List[Dict[str, str]] = []
    for _, row in subjects.iterrows():
        subject_id = str(row["被试ID"])
        scores = {code: float(row[code]) for code, _ in TRAIT_ORDER}
        prompt_text = fill_template_with_scores_only(template, scores)
        results.append({"被试ID": subject_id, "prompt": prompt_text})

    if output_path:
        Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    return results