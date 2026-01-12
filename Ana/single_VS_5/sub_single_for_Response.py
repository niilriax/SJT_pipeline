# -*- coding: utf-8 -*-
# @Time :2025/12/13 20:30:00
# @Author : Scientist
# @File :sub_single_for_Response.py
# @Software :PyCharm

from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
import json
from pathlib import Path
import pandas as pd
import sys

# 兼容脚本直接运行与包内导入
if __package__ is None or __package__ == "":
    _project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(_project_root / "src"))
    from package.utils import LLM_call
else:
    from package.utils import LLM_call


def load_prompts_from_csv(csv_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    prompts = []
    for _, row in df.iterrows():
        subject_id = int(row['被试ID'])
        prompt_text = row['提示词']
        prompts.append({
            "被试ID": subject_id,
            "prompt": prompt_text
        })
    return prompts


def load_prompts_from_txt_dir(txt_dir: Path) -> List[Dict[str, Any]]:
    prompts = []
    txt_dir = Path(txt_dir)
    
    if not txt_dir.exists():
        raise FileNotFoundError(f"提示词目录不存在: {txt_dir}")
    
    # 获取所有提示词文件并按被试ID排序
    txt_files = sorted(txt_dir.glob("subject_*_prompt.txt"))
    
    for txt_file in txt_files:
        try:
            # 从文件名提取被试ID: subject_0201_prompt.txt -> 201
            filename = txt_file.stem  # subject_0201_prompt
            parts = filename.split("_")
            if len(parts) >= 2:
                subject_id = int(parts[1])  # 0201 -> 201
            else:
                print(f"警告: 无法从文件名提取被试ID: {txt_file.name}")
                continue
            
            # 读取提示词内容
            with open(txt_file, 'r', encoding='utf-8') as f:
                prompt_text = f.read().strip()
            
            prompts.append({
                "被试ID": subject_id,
                "prompt": prompt_text
            })
        except Exception as e:
            print(f"警告: 读取文件 {txt_file.name} 时出错: {e}")
            continue
    
    return prompts


def parse_response_to_dict(response: List[Dict[str, Any]], trait: str) -> Dict[str, str]:
    result = {}
    if not isinstance(response, list):
        return result
    
    for item in response:
        if not isinstance(item, dict):
            continue
        qid = item.get("题目ID") or item.get("question_id") or item.get("id")
        ans = item.get("被试选择") or item.get("answer") or item.get("choice")
        if qid and ans:
            qid_str = str(qid).strip()
            ans_str = str(ans).strip()

            if qid_str.startswith("Q") and "_" in qid_str:
                qid_str = qid_str[1:]
            elif qid_str.startswith("Q") and qid_str[1:].isdigit():
                qid_str = f"{trait}_{qid_str[1:]}"
            elif qid_str.isdigit():
                qid_str = f"{trait}_{qid_str}"     
            result[qid_str] = ans_str
    
    return result


def generate_responses_for_trait(
    trait: str,
    model: ChatOpenAI,
    prompts_csv_path: Optional[Path] = None,
    prompts_txt_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None
) -> Path:
    # 确定文件路径
    current_dir = Path(__file__).parent
    
    # 优先使用文本文件目录，如果没有则尝试CSV文件
    if prompts_txt_dir is None and prompts_csv_path is None:
        prompts_txt_dir = current_dir / f"prompts_{trait}"
    
    if output_dir is None:
        output_dir = current_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载提示词
    print(f"\n=== 处理特质 {trait} ===")
    if prompts_txt_dir and Path(prompts_txt_dir).exists():
        print(f"正在从文本文件目录加载提示词: {prompts_txt_dir}")
        prompts = load_prompts_from_txt_dir(prompts_txt_dir)
    elif prompts_csv_path and Path(prompts_csv_path).exists():
        print(f"正在从CSV文件加载提示词: {prompts_csv_path}")
        prompts = load_prompts_from_csv(prompts_csv_path)
    else:
        raise FileNotFoundError(
            f"提示词文件不存在。请提供 prompts_txt_dir 或 prompts_csv_path。\n"
            f"尝试的路径: txt_dir={prompts_txt_dir}, csv_path={prompts_csv_path}"
        )
    
    print(f"已加载 {len(prompts)} 个提示词")
    # 调用大模型生成回答（顺序调用）
    print(f"开始调用大模型生成回答（顺序处理，共 {len(prompts)} 个被试）...")
    all_qids = set()  # 收集所有题目ID
    responses_dict = {}  # {被试ID: {题目ID: 选择}}
    
    for idx, prompt_item in enumerate(prompts, 1):
        subject_id = prompt_item["被试ID"]
        prompt_text = prompt_item["prompt"]
        
        print(f"  正在处理被试 {subject_id} ({idx}/{len(prompts)})...")
        try:
            response_data = LLM_call(prompt_text, model)
            
            # 如果返回的是列表的列表，取第一个
            if isinstance(response_data, list) and len(response_data) > 0:
                if isinstance(response_data[0], list):
                    response_data = response_data[0]
            
            if response_data:
                parsed = parse_response_to_dict(response_data, trait)
                responses_dict[subject_id] = parsed
                all_qids.update(parsed.keys())
                print(f"    ✓ 被试 {subject_id} 处理成功")
            else:
                print(f"    ⚠️ 警告: 被试 {subject_id} 返回空结果")
                responses_dict[subject_id] = {}
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ 警告: 被试 {subject_id} 生成失败: {error_msg}")
            responses_dict[subject_id] = {}
    
    # 解析回答并构建DataFrame
    print("正在构建结果...")
    # 排序题目ID（按特质代码和题号）
    def qid_sort_key(qid: str):
        try:
            parts = qid.split("_")
            if len(parts) == 2:
                trait_code = parts[0]
                item_id = int(parts[1])
                trait_order = {"N": 0, "E": 1, "O": 2, "A": 3, "C": 4}
                return (trait_order.get(trait_code, 999), item_id)
        except:
            pass
        return (999, 0)
    all_qids = sorted(all_qids, key=qid_sort_key)
    # 构建DataFrame
    rows = []
    for subject_id in sorted(responses_dict.keys()):
        row = {"被试ID": subject_id}
        for qid in all_qids:
            row[qid] = responses_dict[subject_id].get(qid, "")
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df.set_index("被试ID")
    
    # 保存CSV
    output_path = output_dir / f"virtual_subjects_{trait}_responses.csv"
    df.to_csv(output_path, encoding='utf-8-sig')
    print(f"✓ 已保存 {len(df)} 个被试的回答到: {output_path}")
    print(f"  包含 {len(all_qids)} 道题目")
    
    return output_path


def generate_responses_for_all_traits(
    traits: List[str] = ["N", "E", "O", "A", "C"],
    model: Optional[ChatOpenAI] = None,
    data_dir: Optional[Path] = None,
    use_txt_dir: bool = True
) -> List[Path]:
    if model is None:
        from dotenv import load_dotenv
        load_dotenv()
        model = ChatOpenAI(model="gpt-4o", temperature=0.5, max_tokens=7000)
    
    if data_dir is None:
        data_dir = Path(__file__).parent
    data_dir = Path(data_dir)
    
    output_paths = []
    for trait in traits:
        try:
            if use_txt_dir:
                # 使用文本文件目录
                txt_dir = data_dir / f"prompts_{trait}"
                csv_path = generate_responses_for_trait(
                    trait=trait,
                    model=model,
                    prompts_txt_dir=txt_dir,
                    output_dir=data_dir
                )
            else:
                # 使用CSV文件
                csv_path = generate_responses_for_trait(
                    trait=trait,
                    model=model,
                    prompts_csv_path=data_dir / f"prompts_{trait}.csv",
                    output_dir=data_dir
                )
            output_paths.append(csv_path)
        except Exception as e:
            print(f"❌ 处理特质 {trait} 时出错: {e}")
    
    print(f"\n=== 完成 ===")
    print(f"共处理 {len(output_paths)} 个特质")
    return output_paths


def main():
    from dotenv import load_dotenv
    load_dotenv()
    model = ChatOpenAI(model="gpt-5.1", temperature=0.5, max_tokens=7000)
    traits = ["N", "E", "O", "A", "C"]
    generate_responses_for_all_traits(
        traits=traits,
        model=model,
        use_txt_dir=True
    )
if __name__ == "__main__":
    main()
