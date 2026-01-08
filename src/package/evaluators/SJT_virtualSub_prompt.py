from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import pandas as pd
from package.utils import TRAIT_ORDER, get_project_root

_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = get_project_root() 


def load_json_file(file_path: Optional[Path], default_path: Path, error_message: str) -> Dict[str, Any]:
    file_path = file_path or default_path
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"{error_message}：{file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))

def build_neo_items_block(neo_items: Dict[str, dict]) -> str:
    """将 NEO 题目数据格式化为文本"""
    lines = []
    for facet_code, facet in neo_items.items():
        items = facet.get("items", {})
        for item_id, item_info in items.items():
            text = str(item_info.get("item", "")).strip()
            if text:
                lines.append((int(item_id) if item_id.isdigit() else float("inf"), f"Q{item_id}. {text}"))
    return "\n".join(text for _, text in sorted(lines, key=lambda x: x[0]))

def build_sjt_items_block(sjt_data: Dict[str, Any]) -> str:
    """将 SJT 题目数据格式化为文本"""
    lines = []
    trait_name_to_code = {name: code for code, name in TRAIT_ORDER}
    for trait_name in [name for _, name in TRAIT_ORDER]:
        items = sjt_data.get("traits", {}).get(trait_name, {}).get("items", [])
        for item in sorted(items, key=lambda x: x.get("item_id", 0)):
            situation = item.get("situation", "")
            question = item.get("question", "")
            if not (situation and question):
                continue
            trait_code = trait_name_to_code.get(trait_name, trait_name[0])
            item_id = item.get("item_id", 0)
            q_text = f"Q{trait_code}_{item_id}. {situation} {question}\n"
            options = item.get("options", {})
            q_text += "\n".join(f"  {k}. {options[k].get('content', '')}" 
                              for k in ["A", "B", "C", "D"] 
                              if k in options and options[k].get("content"))
            lines.append(q_text.strip())
    return "\n\n".join(lines)


def load_subjects(csv_path: Optional[Path] = None) -> pd.DataFrame:
    csv_path = csv_path or (_BASE_DIR.parent / "generators" / "virtual_subjects.csv")
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到被试 CSV 文件：{csv_path}")
    return pd.read_csv(csv_path)


def generate_filled_prompts_with_scores_only(test_type: str):
    if test_type == "NEO":
        prompt_file = _BASE_DIR / "prompts" / "scorer_forNEO_prompt.txt"
        neo_path = _PROJECT_ROOT / "docs" / "Neo-FFI.json"
        neo_items = load_json_file(
            neo_path,
            _PROJECT_ROOT / "docs" / "Neo-FFI.json",
            "找不到 NEO-FFI 题目文件"
        )
        neo_block = build_neo_items_block(neo_items)
    elif test_type == "SJT":
        prompt_file = _BASE_DIR / "prompts" / "scorer_prompt.txt"
        sjt_path = _BASE_DIR.parent / "utils" / "sjt_outputs" / "SJT_all_traits.json"
        sjt_data = load_json_file(
            sjt_path,
            _BASE_DIR.parent / "utils" / "sjt_outputs" / "SJT_all_traits.json",
            "找不到 SJT 题目文件"
        )
        sjt_block = build_sjt_items_block(sjt_data)
    else:
        raise ValueError(f"不支持的测试类型: {test_type}，必须是 'NEO' 或 'SJT'")
    if not prompt_file.exists():
        raise FileNotFoundError(f"提示词文件不存在: {prompt_file}")
    prompt_template = prompt_file.read_text(encoding="utf-8")
    subjects = load_subjects()
    # 创建特质名称列表（按顺序：神经质, 外向性, 开放性, 宜人性, 尽责性）
    trait_codes = [code for code, _ in TRAIT_ORDER]  # N, E, O, A, C
    filled_prompts = []
    # 遍历每个被试
    for idx, row in subjects.iterrows():
        subject_id = row.get("被试ID", idx + 1)
        filled_prompt = prompt_template
        # 替换每个特质的T分数（按顺序替换【T分数】）
        for trait_code in trait_codes:
            score = row.get(trait_code, 0)
            # 替换第一个出现的【T分数】
            filled_prompt = filled_prompt.replace("【T分数】", str(round(score, 2)), 1)
        # 根据test_type替换题目占位符
        if test_type == "NEO":
            filled_prompt = filled_prompt.replace("【NEO_FFI_R】", neo_block)
        elif test_type == "SJT":
            filled_prompt = filled_prompt.replace("【情境判断题目】", sjt_block)
        filled_prompts.append({
            "被试ID": str(subject_id),
            "prompt": filled_prompt
        })
    return filled_prompts


def main():
    """主函数：测试生成填充后的提示词"""
    print("=" * 60)
    print("生成虚拟被试提示词")
    print("=" * 60)
    
    # 测试NEO类型
    print("\n--- 生成NEO提示词 ---")
    try:
        neo_prompts = generate_filled_prompts_with_scores_only("NEO")
        print(f"✓ 成功生成 {len(neo_prompts)} 个NEO提示词")
        
        # 保存到文件
        neo_output_path = _BASE_DIR / "filled_prompts_neo.json"
        with open(neo_output_path, 'w', encoding='utf-8') as f:
            json.dump(neo_prompts, f, ensure_ascii=False, indent=2)
        print(f"✓ NEO提示词已保存至: {neo_output_path}")
        
        # 显示第一个提示词的预览
        if neo_prompts:
            print(f"\n第一个被试（ID: {neo_prompts[0]['被试ID']}）的提示词预览（前500字符）：")
            print("-" * 60)
            print(neo_prompts[0]['prompt'][:500] + "...")
    except Exception as e:
        print(f"✗ NEO提示词生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试SJT类型
    print("\n--- 生成SJT提示词 ---")
    try:
        sjt_prompts = generate_filled_prompts_with_scores_only("SJT")
        print(f"✓ 成功生成 {len(sjt_prompts)} 个SJT提示词")
        
        # 保存到文件
        sjt_output_path = _BASE_DIR / "filled_prompts_sjt.json"
        with open(sjt_output_path, 'w', encoding='utf-8') as f:
            json.dump(sjt_prompts, f, ensure_ascii=False, indent=2)
        print(f"✓ SJT提示词已保存至: {sjt_output_path}")
        
        # 显示第一个提示词的预览
        if sjt_prompts:
            print(f"\n第一个被试（ID: {sjt_prompts[0]['被试ID']}）的提示词预览（前500字符）：")
            print("-" * 60)
            print(sjt_prompts[0]['prompt'][:500] + "...")
    except Exception as e:
        print(f"✗ SJT提示词生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
    