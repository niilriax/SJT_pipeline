# -*- coding: utf-8 -*-
# @Time : 2026/1/10
# @Author : Scientist
# @File : generate_single_dimension_prompts.py
# @Software : PyCharm

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Optional, Dict

# 大五人格特质的中文名称映射
TRAIT_NAMES = {
    "N": "神经质",
    "E": "外向性",
    "O": "开放性",
    "A": "宜人性",
    "C": "尽责性"
}

def load_neo_ffi_items(neo_ffi_file: Optional[Path] = None) -> Dict:
    """
    加载NEO-FFI题目数据
    
    Args:
        neo_ffi_file: JSON文件路径，如果为None则使用默认路径
    
    Returns:
        包含所有特质的题目字典
    """
    if neo_ffi_file is None:
        # 默认使用项目根目录下的 docs/Neo-FFI.json
        import os
        current_dir = Path(__file__).parent.resolve()
        
        # 方法1: 从脚本位置向上查找（最可靠）
        # Ana/single_VS_5/ -> Ana/ -> demo/
        project_root = current_dir.parent.parent
        neo_ffi_file = project_root / "docs" / "Neo-FFI.json"
        
        # 方法2: 如果不存在，从当前工作目录查找
        if not neo_ffi_file.exists():
            cwd = Path(os.getcwd()).resolve()
            neo_ffi_file = cwd / "docs" / "Neo-FFI.json"
        
        # 方法3: 如果还是不存在，尝试从工作目录的父目录查找
        if not neo_ffi_file.exists():
            cwd = Path(os.getcwd()).resolve()
            neo_ffi_file = cwd.parent / "demo" / "docs" / "Neo-FFI.json"
    
    neo_ffi_file = Path(neo_ffi_file).resolve()
    
    if not neo_ffi_file.exists():
        import os
        current_dir = Path(__file__).parent.resolve()
        cwd = Path(os.getcwd()).resolve()
        error_msg = (
            f"NEO-FFI文件不存在: {neo_ffi_file}\n"
            f"已尝试的路径:\n"
            f"  1. {current_dir.parent.parent / 'docs' / 'Neo-FFI.json'}\n"
            f"  2. {cwd / 'docs' / 'Neo-FFI.json'}\n"
            f"  3. {cwd.parent / 'demo' / 'docs' / 'Neo-FFI.json'}\n"
            f"请检查文件路径或手动指定neo_ffi_file参数"
        )
        raise FileNotFoundError(error_msg)
    
    with open(neo_ffi_file, "r", encoding="utf-8") as f:
        return json.load(f)

def format_items_for_trait(items_data: Dict, trait: str) -> str:
    """
    格式化指定特质的题目列表
    
    Args:
        items_data: 从JSON加载的完整数据
        trait: 人格特质代码
    
    Returns:
        格式化后的题目文本
    """
    if trait not in items_data:
        raise ValueError(f"特质 {trait} 不存在于题目数据中")
    
    trait_data = items_data[trait]
    items = trait_data.get("items", {})
    
    # 按题目ID中的数字排序（如 E_1, E_2, E_10）
    def item_sort_key(item_pair):
        item_id = item_pair[0]
        parts = item_id.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            return (parts[0], int(parts[1]))
        return (item_id, 0)

    sorted_items = sorted(items.items(), key=item_sort_key)
    
    # 格式化题目
    formatted_lines = []
    for item_id, item_info in sorted_items:
        item_text = item_info.get("item", "")
        formatted_lines.append(f"{item_id}: {item_text}")
    
    return "\n".join(formatted_lines)

def load_template(template_file: Optional[Path] = None) -> str:
    """
    从文件加载提示词模板
    
    Args:
        template_file: 模板文件路径，如果为None则使用默认路径
    
    Returns:
        模板文本
    """
    if template_file is None:
        # 默认使用当前脚本所在目录下的 virtual_sub_forOneFactor 文件
        current_dir = Path(__file__).parent
        template_file = current_dir / "virtual_sub_forOneFactor"
    
    template_file = Path(template_file)
    
    if not template_file.exists():
        raise FileNotFoundError(f"模板文件不存在: {template_file}")
    
    with open(template_file, "r", encoding="utf-8") as f:
        return f.read()

def generate_prompt(
    t_score: float, 
    trait: str, 
    subject_id: int, 
    template: Optional[str] = None,
    items_data: Optional[Dict] = None
) -> str:
    """
    生成单个被试的提示词
    
    Args:
        t_score: T分数
        trait: 人格特质代码
        subject_id: 被试ID
        template: 模板文本，如果为None则从文件加载
        items_data: 题目数据，如果为None则从文件加载
    
    Returns:
        完整的提示词文本
    """
    # 如果没有提供模板，从文件加载
    if template is None:
        template = load_template()
    
    # 如果没有提供题目数据，从文件加载
    if items_data is None:
        items_data = load_neo_ffi_items()
    
    trait_name = TRAIT_NAMES.get(trait, trait)
    
    # 替换【人格得分】部分
    score_line = f"{trait_name}: {t_score:.1f}"
    
    # 格式化题目列表
    items_text = format_items_for_trait(items_data, trait)
    
    # 查找并替换【人格得分】和【NEO_FFI_R】部分
    lines = template.split('\n')
    result_lines = []
    score_section_found = False
    items_section_found = False
    skip_next = False
    
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
            
        if line.strip() == "【人格得分】":
            result_lines.append(line)
            result_lines.append(score_line)
            score_section_found = True
            # 如果下一行是空行，跳过它（因为我们已经添加了得分行）
            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                skip_next = True
        elif line.strip() == "【NEO_FFI_R】":
            result_lines.append(line)
            result_lines.append(items_text)
            items_section_found = True
        else:
            result_lines.append(line)
    
    # 如果没有找到【人格得分】标记，在开头添加
    if not score_section_found:
        result_lines.insert(1, "【人格得分】")
        result_lines.insert(2, score_line)
    
    # 如果没有找到【NEO_FFI_R】标记，在适当位置添加
    if not items_section_found:
        # 查找"现在开始扮演"的位置，在其后添加题目
        for i, line in enumerate(result_lines):
            if "现在开始扮演" in line or "开始回答以下" in line:
                result_lines.insert(i + 1, "【NEO_FFI_R】")
                result_lines.insert(i + 2, items_text)
                break
    
    return '\n'.join(result_lines)


def generate_prompts_from_csv(
    csv_file: Path,
    trait: str,
    output_dir: Optional[Path] = None,
    output_format: str = "txt",
    template_file: Optional[Path] = None,
    neo_ffi_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    从CSV文件读取被试数据，生成提示词
    
    Args:
        csv_file: 包含被试数据的CSV文件路径
        trait: 人格特质代码
        output_dir: 输出目录，如果为None则使用CSV文件所在目录
        output_format: 输出格式，"txt"或"csv"
        template_file: 模板文件路径，如果为None则使用默认路径
        neo_ffi_file: NEO-FFI JSON文件路径，如果为None则使用默认路径
    
    Returns:
        包含提示词的DataFrame
    """
    # 读取模板和题目数据
    template = load_template(template_file)
    items_data = load_neo_ffi_items(neo_ffi_file)
    
    # 读取数据
    df = pd.read_csv(csv_file, index_col=0)
    
    if trait not in df.columns:
        raise ValueError(f"CSV文件中不存在特质列: {trait}")
    
    # 确定输出目录
    if output_dir is None:
        output_dir = csv_file.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成提示词
    prompts = []
    for idx, row in df.iterrows():
        t_score = row[trait]
        prompt = generate_prompt(t_score, trait, idx, template, items_data)
        prompts.append({
            "被试ID": idx,
            f"{trait}_T分数": t_score,
            "提示词": prompt
        })
    
    # 创建DataFrame
    prompts_df = pd.DataFrame(prompts)
    
    # 保存结果
    if output_format == "txt":
        # 为每个被试保存单独的txt文件
        txt_output_dir = output_dir / f"prompts_{trait}"
        txt_output_dir.mkdir(parents=True, exist_ok=True)
        
        for _, row in prompts_df.iterrows():
            subject_id = row["被试ID"]
            prompt_text = row["提示词"]
            txt_file = txt_output_dir / f"subject_{subject_id:04d}_prompt.txt"
            with open(txt_file, "w", encoding="utf-8") as f:
                f.write(prompt_text)
        
        print(f"已保存 {len(prompts_df)} 个提示词文件到: {txt_output_dir}")
    
    # 保存CSV文件（包含所有提示词）
    csv_output_file = output_dir / f"prompts_{trait}.csv"
    prompts_df.to_csv(csv_output_file, index=False, encoding="utf-8-sig")
    print(f"已保存提示词汇总文件到: {csv_output_file}")
    
    return prompts_df


def batch_generate_prompts(
    data_dir: Path,
    traits: list = ["N", "E", "O", "A", "C"],
    output_format: str = "txt",
    template_file: Optional[Path] = None,
    neo_ffi_file: Optional[Path] = None
):
    """
    批量生成所有特质的提示词
    
    Args:
        data_dir: 包含虚拟被试CSV文件的目录
        traits: 要处理的特质列表
        output_format: 输出格式
        template_file: 模板文件路径，如果为None则使用默认路径
        neo_ffi_file: NEO-FFI JSON文件路径，如果为None则使用默认路径
    """
    data_dir = Path(data_dir)
    
    print(f"开始批量生成提示词...")
    print(f"数据目录: {data_dir}")
    print(f"特质列表: {traits}\n")
    
    for trait in traits:
        csv_file = data_dir / f"virtual_subjects_{trait}.csv"
        
        if not csv_file.exists():
            print(f"警告: 文件 {csv_file} 不存在，跳过")
            continue
        
        print(f"=== 处理特质 {trait} ===")
        try:
            prompts_df = generate_prompts_from_csv(
                csv_file=csv_file,
                trait=trait,
                output_dir=data_dir,
                output_format=output_format,
                template_file=template_file,
                neo_ffi_file=neo_ffi_file
            )
            print(f"成功生成 {len(prompts_df)} 个提示词\n")
        except Exception as e:
            print(f"处理特质 {trait} 时出错: {e}\n")
    
    print("批量生成完成！")


def main():
    """主函数"""
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent
    
    # 默认使用 single_VS_5 目录（当前目录就是 single_VS_5）
    data_dir = current_dir
    
    # 确定项目根目录和JSON文件路径
    project_root = current_dir.parent.parent  # Ana/single_VS_5/ -> Ana/ -> demo/
    neo_ffi_file = project_root / "docs" / "Neo-FFI.json"
    
    # 批量生成提示词
    batch_generate_prompts(
        data_dir=data_dir,
        traits=["N", "E", "O", "A", "C"],
        output_format="txt",  # 可选: "txt" 或 "csv"
        neo_ffi_file=neo_ffi_file  # 明确指定JSON文件路径
    )


if __name__ == "__main__":
    main()

