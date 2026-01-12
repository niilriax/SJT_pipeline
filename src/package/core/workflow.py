# -*- codeing =utf-8 -*-
# @Time :2025/12/13 20:30:00
# @Author : Scientist
# @File :workflow.py
# @Software :PyCharm

from typing import TypedDict, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
import json
from datetime import datetime
from package.generators import format_prompt, build_prompt_with_suggestions
from package.generators.SJTvirtual_subject import run_virtual_subject_simulation
from package.evaluators import format_all_items_prompt
from package.evaluators.SJTcontent_validity import convert_evaluation_results_to_csv, calculate_cvi_from_evaluation_results_single_expert
from package.evaluators.SJT_virtualSub_prompt import generate_filled_prompts_with_scores_only
from package.utils import (
    LLM_call, 
    LLM_call_concurrent, 
    get_project_root,
    sjt_responses_to_matrix, 
    TRAIT_ORDER
)
from package.utils import data_Ana as citc_analysis
from pathlib import Path
import pandas as pd
import numpy as np


class WorkflowState(TypedDict, total=False):
    # --- 1. 基础配置 ---
    trait_names: List[str]         # 所有要处理的特质列表
    model: ChatOpenAI              # 模型
    experts: List[ChatOpenAI]      # 专家列表
    # --- 2. 生题循环---
    final_storage: List[Dict]      # 用来存所有通过的题目 (最终结果)
    batch_count: int               # 当前是第几个特质
    target_batches: int            # 大五
    # --- 3. 内循环变量 ---
    generated_items: List[Dict]    # 当前正在处理的题目
    evaluation_results: List[Dict] # 当前的专家评分结果
    evaluation_errors: List[Dict]  # 评估过程中的错误信息
    passed_items: List[Dict]       # 当前批次中，CVI 达标的题目
    low_cvi_items: List[Dict]      # 当前批次中，CVI 不达标的题目
    iteration: int                 # 当前批次修了第几次
    max_iterations: int            # 单批次最大修订次数
    # --- 4. 虚拟被试 ---
    virtual_subject_prompts_neo: List[Dict[str, str]]  # NEO虚拟被试提示词
    virtual_subject_prompts_sjt: List[Dict[str, str]]  # SJT虚拟被试提示词
    virtual_subject_responses_neo: List[Dict[str, Any]]  # NEO虚拟被试回答
    virtual_subject_responses_sjt: List[Dict[str, Any]]  # SJT虚拟被试回答
    # --- 5. 分析 ---
    irt_bad_items: List[Dict[str, Any]]        # 问题题目列表（若使用）
    irt_revision_prompt: str                   # 当前修订提示词
    irt_repair_mode: bool           # 标记：当前是否处于修复模式 (True/False)
    irt_iteration: int              # 计数：当前修复是第几轮 (防止死循环)
    irt_max_iterations: int         # 配置：最大修复轮次 (建议设为 3)
    irt_prompt_queue: List[str]     # 多批修订提示队列
    irt_bad_items_queue: List[List[Dict[str, Any]]]  # 多批坏题队列（带trait，用于回填）
    irt_prompt_queue: List[str] # 存储多批修订提示，逐批消耗
    irt_repair_trait_name: str     # trait for current repair batch


def _normalize_item_id(raw_id: Any) -> str:
    """Normalize item id, stripping prefixes like QN_1 -> 1."""
    if raw_id is None:
        return ""
    raw_str = str(raw_id).strip()
    match = re.match(r"^Q[A-Za-z]+_(\d+)$", raw_str)
    return match.group(1) if match else raw_str


def _normalize_item(item: Any) -> Any:
    """Ensure item_id is consistent regardless of Item_ID/ItemId/QN_1 formats."""
    if not isinstance(item, dict):
        return item
    normalized = item.copy()
    raw_id = normalized.get("item_id") or normalized.get("Item_ID") or normalized.get("ItemId")
    if raw_id is not None:
        normalized["item_id"] = _normalize_item_id(raw_id)
    return normalized


def _sort_final_storage(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort items by trait order then numeric item_id to keep stable ordering."""
    trait_order = {name: idx for idx, (_, name) in enumerate(TRAIT_ORDER)}
    def _key(item: Dict[str, Any]):
        trait = item.get("trait", "")
        order = trait_order.get(trait, 999)
        try:
            iid = int(item.get("item_id"))
        except Exception:
            iid = 0
        return (order, iid)
    return sorted(items, key=_key)


def _normalize_evaluation_results(evaluation_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_results = []
    for er in evaluation_results or []:
        norm_res = [_normalize_item(item) for item in er.get("results", [])]
        normalized_results.append({**er, "results": norm_res})
    return normalized_results


def _parse_revision_prompt_traits(prompt: str) -> Dict[str, str]:
    """Parse Item_ID -> Trait from CITC revision prompt text."""
    if not prompt:
        return {}
    id_to_trait: Dict[str, str] = {}
    current_id = ""
    current_trait = ""
    for line in prompt.splitlines():
        line = line.strip()
        if line.startswith("Item_ID:"):
            current_id = _normalize_item_id(line.split(":", 1)[1].strip())
        elif line.startswith("Trait:"):
            current_trait = line.split(":", 1)[1].strip()
            if current_id and current_trait:
                id_to_trait[current_id] = current_trait
                current_id = ""
                current_trait = ""
    return id_to_trait


def generate_items_node(state: WorkflowState) -> WorkflowState:
    model = state.get("model")
    low_cvi_items = state.get("low_cvi_items", [])
    irt_repair_mode = state.get("irt_repair_mode", False)
    trait_names = state.get("trait_names", [])
    batch_count = state.get("batch_count", 0)
    if low_cvi_items:
        num_low_cvi = len(low_cvi_items)
        print(f"🔄  正在根据专家意见修订 {num_low_cvi} 道题目...")
        modification_suggestions = []
        for item_info in low_cvi_items:
            item_info = _normalize_item(item_info)
            expert_evals = item_info.get("expert_evaluations", [])
            original_item = item_info.get("original_item", {})
            if expert_evals:
                expert_eval = expert_evals[0].get("evaluation", {})
                modification_suggestions.append({
                    "题目ID": item_info.get("item_id", ""),
                    "原题": original_item,
                    "内容效度评估": expert_eval.get("内容效度评估", {})
                })
        prompt = build_prompt_with_suggestions(modification_suggestions)
    elif irt_repair_mode:
        irt_iteration = state.get("irt_iteration", 0)
        irt_max_iterations = state.get("irt_max_iterations", 1)
        print(f"🔧  进入修复模式：第 {irt_iteration}/{irt_max_iterations} 轮...")
        prompt = state.get("irt_revision_prompt", "")
        if prompt:
            prompt = (
                f"{prompt}\n\n"
                "请严格保留原题的Item_ID，不要新增或修改Item_ID，确保后续可以精确替换。"
            )
        if not prompt:
            print("⚠️ 警告: 修复模式已启用，但未找到修订提示词！")
            state["generated_items"] = []
            return state
    else:
        if 0 <= batch_count < len(trait_names):
            trait_name = trait_names[batch_count]
            print(f"✨ [新批次] 正在为特质 [{trait_name}] 生成题目...")
            prompt = format_prompt(trait_name)
        else:
            print("⚠️ 批次计数超出范围或已完成。")
            state["generated_items"] = []
            return state
    try:
        items = LLM_call(prompt, model)
        normalized_items = [_normalize_item(item) for item in items]
        if irt_repair_mode:
            id_to_trait = _parse_revision_prompt_traits(state.get("irt_revision_prompt", ""))
            for item in normalized_items:
                item_id = item.get("item_id")
                if item_id is not None and "trait" not in item:
                    trait = id_to_trait.get(_normalize_item_id(item_id), "")
                    if trait:
                        item["trait"] = trait
        state["generated_items"] = normalized_items
    except Exception as e:
        print(f"❌ 生成出错: {e}")
        state["generated_items"] = []
    return state

def evaluate_items_node(state: WorkflowState) -> WorkflowState:
    experts = state.get("experts") or []
    items = state.get("generated_items", [])
    trait_names = state.get("trait_names", [])
    batch_count = state.get("batch_count", 0)
    trait_name = trait_names[batch_count] if 0 <= batch_count < len(trait_names) else ""
    irt_repair_mode = state.get("irt_repair_mode", False)
    if irt_repair_mode:
        id_to_trait = _parse_revision_prompt_traits(state.get("irt_revision_prompt", ""))
        items_by_trait = {}
        for idx, item in enumerate(items):
            item_trait = item.get("trait")
            if not item_trait:
                item_id = item.get("item_id")
                if item_id is not None:
                    item_trait = id_to_trait.get(_normalize_item_id(item_id), "")
            if not item_trait:
                item_trait = state.get("irt_repair_trait_name", trait_name)
            items_by_trait.setdefault(item_trait, []).append(item)
        prompt = format_all_items_prompt({"特质": items_by_trait})
    else:
        prompt = format_all_items_prompt({"特质": {trait_name: items}})
    if experts:
        expert = experts[0]
        evaluation_result = LLM_call(prompt, expert)
        if not isinstance(evaluation_result, list):
            evaluation_result = []
        evaluation_results = [{
            "expert_index": 1,
            "results": evaluation_result
        }]
        state["evaluation_results"] = evaluation_results
        state["evaluation_errors"] = []
    else:
        state["evaluation_results"] = []
        state["evaluation_errors"] = [{"error": "No experts available"}]
    return state

def convert_to_CVI_node(state: WorkflowState) -> WorkflowState:
    evaluation_results = _normalize_evaluation_results(state.get("evaluation_results", []))
    state["evaluation_results"] = evaluation_results
    generated_items = [_normalize_item(item) for item in state.get("generated_items", [])]
    state["generated_items"] = generated_items
    trait_names = state.get("trait_names", [])
    batch_count = state.get("batch_count", 0)
    trait_name = trait_names[batch_count] if 0 <= batch_count < len(trait_names) else ""
    irt_repair_mode = state.get("irt_repair_mode", False)
    irt_bad_items = state.get("irt_bad_items", [])
    print(f"--- 正在计算特质 [{trait_name}] 的CVI ---")
    try:
        cvi_data, low_cvi_items, passed_items = calculate_cvi_from_evaluation_results_single_expert(
            evaluation_results,
            generated_items=generated_items
        )
        low_cvi_items = [_normalize_item(item) for item in low_cvi_items]
        passed_items = [_normalize_item(item) for item in passed_items]
        state["low_cvi_items"] = low_cvi_items
        if low_cvi_items:
            iteration = state.get("iteration", 0)
            state["iteration"] = iteration + 1
        final_storage = state.get("final_storage", [])
        if irt_repair_mode and irt_bad_items:
            # 修复模式：按 item_id 精确替换旧题目
            irt_bad_items_by_id = {
                _normalize_item_id(item.get("item_id", "")): item for item in irt_bad_items
                if item.get("item_id") is not None
            }
            original_bad_count = len(irt_bad_items_by_id)
            for new_item in passed_items:
                item_id = new_item.get("item_id")
                normalized_id = _normalize_item_id(item_id) if item_id is not None else ""
                old_item = irt_bad_items_by_id.pop(normalized_id, None)
                if old_item:
                    item_with_trait = new_item.copy()
                    item_with_trait["trait"] = old_item.get("trait", trait_name)
                    item_with_trait["item_id"] = old_item.get("item_id", new_item.get("item_id", ""))
                    final_storage.append(item_with_trait)
            remaining_bad_items = list(irt_bad_items_by_id.values())
            if remaining_bad_items:
                final_storage.extend(remaining_bad_items)
                print(f"⚠️ 警告: {len(remaining_bad_items)} 道旧题未被替换，已保留原题")
            state["irt_bad_items"] = remaining_bad_items
            replaced_count = original_bad_count - len(irt_bad_items_by_id)
            print(f"✅ 成功修复 {replaced_count} 道题目，已替换原题目")
        else:
            for item in passed_items:
                item_with_trait = item.copy()
                item_with_trait["trait"] = trait_name
                final_storage.append(item_with_trait)
        state["final_storage"] = _sort_final_storage(final_storage)
        state["passed_items"] = passed_items
        print(f"已识别 {len(low_cvi_items)} 道低CVI题目，{len(passed_items)} 道合格题目（已累积到总库存）")
        csv_path = convert_evaluation_results_to_csv(evaluation_results)
        print(f"CSV文件已保存至: {csv_path}")
    except Exception as e:
        print(f"计算CVI时出错: {e}")
        raise
    return state


def check_quality(state: WorkflowState) -> str:
    low_cvi_items = state.get("low_cvi_items", [])
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 3)
    if low_cvi_items:
        if iteration >= max_iterations:
            print(f"⚠️ 已达到最大CVI修订次数，强制归档")
            return "archive"
        print(f"🔄 CVI不合格，进入第 {state['iteration']} 次内容修订")
        return "revise"
    print("✅ CVI评估通过，准备归档")
    return "archive"

def check_quantity(state: WorkflowState) -> str:
    current_batch = state.get("batch_count", 0)
    target_batches = state.get("target_batches", 5)
    if current_batch < target_batches:
        return "next_batch"
    return "finish"


def virtual_subject_node(state: WorkflowState) -> WorkflowState:
    project_root = get_project_root()
    final_storage = state.get("final_storage", [])
    sjt_output_dir = project_root / "src" / "package" / "utils" / "sjt_outputs"
    sjt_output_dir.mkdir(parents=True, exist_ok=True)
    sjt_json_path = sjt_output_dir / "SJT_all_traits.json"
    traits_data: Dict[str, Dict[str, Any]] = {}
    trait_names = state.get("trait_names", [])
    for trait_name in trait_names:
        traits_data[trait_name] = {
            "trait": trait_name,
            "items": []
        }
    for item in final_storage:
        trait_name = item.get("trait", "")
        if trait_name and trait_name in traits_data:
            item_clean = {k: v for k, v in item.items() if k != "trait"}
            traits_data[trait_name]["items"].append(item_clean)
        else:
            print(f"⚠️ 警告: 题目 {item.get('item_id', 'unknown')} 缺少特质信息，已跳过")
    sjt_data = {"traits": traits_data}
    with open(sjt_json_path, 'w', encoding='utf-8') as f:
        json.dump(sjt_data, f, ensure_ascii=False, indent=2)
    print(f"SJT题目已保存至: {sjt_json_path}")
    print("\n--- 开始生成虚拟被试 ---")
    virtual_subjects_df = run_virtual_subject_simulation(
        n_subjects=300,
        driving_facet="N",
        mean=50.0,
        std=10.0,
        seed=1
    )
    csv_path = project_root / "src" / "package" / "generators" / "virtual_subjects.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    virtual_subjects_df.to_csv(csv_path, index=True, encoding='utf-8-sig')
    print(f"虚拟被试分数已保存至: {csv_path}")
    # 生成NEO提示词
    print("\n--- 生成NEO提示词 ---")
    neo_prompts_output_path = project_root / "src" / "package" / "evaluators" / "filled_prompts_neo.json"
    virtual_subject_prompts_neo = generate_filled_prompts_with_scores_only(test_type="NEO")
    print(f"已生成 {len(virtual_subject_prompts_neo)} 个虚拟被试的NEO提示词，保存至: {neo_prompts_output_path}")
    print("\n--- 生成SJT提示词 ---")
    sjt_prompts_output_path = project_root / "src" / "package" / "evaluators" / "filled_prompts_sjt.json"
    virtual_subject_prompts_sjt = generate_filled_prompts_with_scores_only(test_type="SJT")
    print(f"已生成 {len(virtual_subject_prompts_sjt)} 个虚拟被试的SJT提示词，保存至: {sjt_prompts_output_path}")
    state["virtual_subject_prompts_neo"] = virtual_subject_prompts_neo
    state["virtual_subject_prompts_sjt"] = virtual_subject_prompts_sjt
    return state


def virtual_subject_response_node(state: WorkflowState) -> WorkflowState:
    model = state.get("model")
    virtual_subject_prompts_neo = state.get("virtual_subject_prompts_neo", [])
    virtual_subject_prompts_sjt = state.get("virtual_subject_prompts_sjt", [])
    if not virtual_subject_prompts_neo and not virtual_subject_prompts_sjt:
        print("⚠️ 警告: 未找到虚拟被试提示词，跳过回答")
        return state
    project_root = get_project_root()
    max_workers = 50
    # 处理NEO回答
    neo_responses: List[Dict[str, Any]] = []
    if virtual_subject_prompts_neo:
        print(f"\n--- 开始生成NEO回答（共 {len(virtual_subject_prompts_neo)} 个被试，并发数: {max_workers}）---")
        neo_prompts = [(item["prompt"], model) for item in virtual_subject_prompts_neo]
        neo_results = LLM_call_concurrent(neo_prompts, max_workers=max_workers)
        for idx, (prompt_item, result) in enumerate(zip(virtual_subject_prompts_neo, neo_results)):
            subject_id = str(prompt_item["被试ID"])
            if result.get("success", False):
                response_text = result.get("result", [])
                if isinstance(response_text, list) and len(response_text) > 0:
                    response_text = response_text[0] if len(response_text) == 1 else response_text
                neo_responses.append({
                    "被试ID": subject_id,
                    "test_type": "NEO",
                    "response": response_text
                })
            else:
                error = result.get("error", "未知错误")
                neo_responses.append({
                    "被试ID": subject_id,
                    "test_type": "NEO",
                    "response": f"生成失败: {error}"
                })
        # 保存NEO回答
        neo_output_path = project_root / "src" / "package" / "evaluators" / "neo_responses.json"
        neo_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(neo_output_path, 'w', encoding='utf-8') as f:
            json.dump(neo_responses, f, ensure_ascii=False, indent=2)
        print(f"已生成 {len(neo_responses)} 条NEO回答，保存至: {neo_output_path}")
    # 处理SJT回答
    sjt_responses: List[Dict[str, Any]] = []
    if virtual_subject_prompts_sjt:
        print(f"\n--- 开始生成SJT回答（共 {len(virtual_subject_prompts_sjt)} 个被试，并发数: {max_workers}）---")
        sjt_prompts = [(item["prompt"], model) for item in virtual_subject_prompts_sjt]
        sjt_results = LLM_call_concurrent(sjt_prompts, max_workers=max_workers)
        for idx, (prompt_item, result) in enumerate(zip(virtual_subject_prompts_sjt, sjt_results)):
            subject_id = str(prompt_item["被试ID"])
            if result.get("success", False):
                response_text = result.get("result", [])
                if isinstance(response_text, list) and len(response_text) > 0:
                    response_text = response_text[0] if len(response_text) == 1 else response_text
                sjt_responses.append({
                    "被试ID": subject_id,
                    "test_type": "SJT",
                    "response": response_text
                })
            else:
                error = result.get("error", "未知错误")
                sjt_responses.append({
                    "被试ID": subject_id,
                    "test_type": "SJT",
                    "response": f"生成失败: {error}"
                })
        sjt_output_path = project_root / "src" / "package" / "evaluators" / "sjt_responses.json"
        sjt_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sjt_output_path, 'w', encoding='utf-8') as f:
            json.dump(sjt_responses, f, ensure_ascii=False, indent=2)
        print(f"已生成 {len(sjt_responses)} 条SJT回答，保存至: {sjt_output_path}")
    state["virtual_subject_responses_neo"] = neo_responses
    state["virtual_subject_responses_sjt"] = sjt_responses
    return state


def analysis_node(state: WorkflowState) -> WorkflowState:
    """执行CITC分析，生成修订提示（不删除题目、不跑2PL IRT）。"""
    print("\n--- 开始执行CITC分析 ---")
    try:
        data = citc_analysis.sjt_responses_to_matrix()
        citc_df = citc_analysis.citc_by_trait(data, items_per_trait=None, corrected=True)
        project_root = get_project_root()
        output_dir = project_root / "output" / "CITC_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "CITC_分析报告.csv"
        citc_df.to_csv(report_path, index=False, encoding='utf-8-sig')
        print(f"📄 CITC分析报告已保存: {report_path}")
        prompt_paths, prompt_item_ids = citc_analysis.generate_citc_prompts_to_files(
            citc_df, batch_size=5, output_dir=output_dir
        )
        if prompt_paths:
            prompt_texts = [Path(p).read_text(encoding="utf-8") for p in prompt_paths]
            first_prompt = prompt_texts.pop(0)
            state["irt_prompt_queue"] = prompt_texts
            state["irt_revision_prompt"] = first_prompt
            # 删除坏题并记录按批次的坏题队列
            bad_df = citc_df[(citc_df["citc"].isna()) | (citc_df["citc"] < 0.5)]
            bad_qids_all = bad_df["item"].astype(str).tolist()
            name_to_code = {name: code for code, name in TRAIT_ORDER}
            final_storage = state.get("final_storage", [])
            kept, removed = [], []
            matched_bad_ids = set()
            for item in final_storage:
                trait = item.get("trait", "")
                item_id_num = _normalize_item_id(item.get("item_id", ""))
                trait_code = name_to_code.get(trait, "")
                item_id_str = f"Q{trait_code}_{item_id_num}" if trait_code and item_id_num else str(item_id_num)
                if item_id_str in bad_qids_all:
                    matched_bad_ids.add(item_id_str)
                    removed.append(item)
                else:
                    kept.append(item)
            state["final_storage"] = _sort_final_storage(kept)
            removed_map: Dict[str, Dict[str, Any]] = {}
            for item in removed:
                trait = item.get("trait", "")
                item_id_num = _normalize_item_id(item.get("item_id", ""))
                trait_code = name_to_code.get(trait, "")
                item_id_str = f"Q{trait_code}_{item_id_num}" if trait_code and item_id_num else str(item_id_num)
                removed_map[item_id_str] = item
            bad_items_queue: List[List[Dict[str, Any]]] = []
            for batch_ids in prompt_item_ids:
                batch_items = [removed_map[qid] for qid in batch_ids if qid in removed_map]
                bad_items_queue.append(batch_items)
            current_bad = bad_items_queue.pop(0) if bad_items_queue else []
            state["irt_bad_items_queue"] = bad_items_queue
            state["irt_bad_items"] = current_bad
            state["irt_repair_trait_name"] = current_bad[0].get("trait", "") if current_bad else ""
            state["irt_repair_mode"] = True
            print(f"🧹 坏题匹配命中 {len(matched_bad_ids)} / {len(bad_qids_all)}")
            print(f"📄 已生成 {len(prompt_paths)} 个 CITC 修订提示，已从库存删除 {len(removed)} 道问题题目")
        else:
            print("✅ 所有题目的CITC均在 0.5 以上")
            state["irt_revision_prompt"] = ""
            state["irt_bad_items"] = []
            state["irt_repair_mode"] = False
            state["irt_iteration"] = 0

        params_path = output_dir / "CITC参数.csv"
        citc_df[['trait', 'item', 'citc', 'quality']].to_csv(
            params_path, index=False, encoding='utf-8-sig'
        )
        print("✅ CITC分析完成")
    except Exception as e:
        print(f"❌ CITC分析出错: {e}")
        state["irt_analysis_error"] = str(e)
    return state

def _update_sjt_all_traits_file(state: WorkflowState) -> None:
    """更新SJT_all_traits.json文件，使其与final_storage保持一致"""
    project_root = get_project_root()
    final_storage = state.get("final_storage", [])
    sjt_output_dir = project_root / "src" / "package" / "utils" / "sjt_outputs"
    sjt_output_dir.mkdir(parents=True, exist_ok=True)
    sjt_json_path = sjt_output_dir / "SJT_all_traits.json"
    traits_data: Dict[str, Dict[str, Any]] = {}
    trait_names = state.get("trait_names", [])
    for trait_name in trait_names:
        traits_data[trait_name] = {
            "trait": trait_name,
            "items": []
        }
    for item in final_storage:
        trait_name = item.get("trait", "")
        if trait_name and trait_name in traits_data:
            item_clean = {k: v for k, v in item.items() if k != "trait"}
            traits_data[trait_name]["items"].append(item_clean)
    sjt_data = {"traits": traits_data}
    with open(sjt_json_path, 'w', encoding='utf-8') as f:
        json.dump(sjt_data, f, ensure_ascii=False, indent=2)
    print(f"📄 已更新SJT题目文件: {sjt_json_path}（共 {len(final_storage)} 道题目）")


def check_irt_repair(state: WorkflowState) -> str:
    """检查是否需要修复"""
    irt_iteration = state.get("irt_iteration", 0)
    irt_max_iterations = state.get("irt_max_iterations", 3)
    irt_revision_prompt = state.get("irt_revision_prompt", "")
    prompt_queue = state.get("irt_prompt_queue", [])
    bad_queue = state.get("irt_bad_items_queue", [])

    if not irt_revision_prompt:
        if prompt_queue:
            next_prompt = prompt_queue.pop(0)
            state["irt_revision_prompt"] = next_prompt
            state["irt_prompt_queue"] = prompt_queue
            # 同步下一批坏题
            if bad_queue:
                next_bad = bad_queue.pop(0)
                state["irt_bad_items"] = next_bad
                state["irt_bad_items_queue"] = bad_queue
                state["irt_repair_trait_name"] = next_bad[0].get("trait", "") if next_bad else ""
        else:
            # 没有更多提示词，检查是否还有未恢复的坏题
            current_bad = state.get("irt_bad_items", [])
            final_storage = state.get("final_storage", [])
            if current_bad:
                # 只恢复那些还没有被成功修复的题目
                existing_item_ids = {
                    (item.get("item_id"), item.get("trait")) 
                    for item in final_storage
                }
                to_restore = [
                    item for item in current_bad
                    if (item.get("item_id"), item.get("trait")) not in existing_item_ids
                ]
                if to_restore:
                    final_storage.extend(to_restore)
                    state["final_storage"] = _sort_final_storage(final_storage)
                    print(f"⚠️ 修复失败，已恢复 {len(to_restore)} 道未修复的原题目到库存")
            # CITC修复完成后，重新保存SJT_all_traits.json以保持一致性
            _update_sjt_all_traits_file(state)
            return "finish"

    if irt_iteration >= irt_max_iterations:
        print(f"已达到最大修复次数 ({irt_max_iterations} 次)，停止修复")
        # 只恢复那些还没有被成功修复的题目
        current_bad = state.get("irt_bad_items", [])
        final_storage = state.get("final_storage", [])
        restored_count = 0
        
        # 检查并恢复当前批次中未修复的题目
        if current_bad:
            # 检查哪些题目还没有被替换（通过 item_id 和 trait 匹配）
            existing_item_ids = {
                (item.get("item_id"), item.get("trait")) 
                for item in final_storage
            }
            to_restore = [
                item for item in current_bad
                if (item.get("item_id"), item.get("trait")) not in existing_item_ids
            ]
            if to_restore:
                final_storage.extend(to_restore)
                restored_count += len(to_restore)
        if bad_queue:
            for bad_batch in bad_queue:
                final_storage.extend(bad_batch)
                restored_count += len(bad_batch)
            state["irt_bad_items_queue"] = []
        
        if restored_count > 0:
            state["final_storage"] = _sort_final_storage(final_storage)
            print(f"⚠️ 修复失败，已恢复 {restored_count} 道未修复的原题目到库存，确保题目数量")
        
        # CITC修复完成后，重新保存SJT_all_traits.json以保持一致性
        _update_sjt_all_traits_file(state)
        
        return "finish"
    state["irt_repair_mode"] = True
    state["irt_iteration"] = irt_iteration + 1
    state["irt_prompt_queue"] = prompt_queue
    state["irt_bad_items_queue"] = bad_queue
    print(f"🔄 开始第 {state['irt_iteration']}/{irt_max_iterations} 轮修复")
    return "repair"

def accumulator_node(state: WorkflowState) -> WorkflowState:
    current_passed = state.get("passed_items", [])
    final_storage = state.get("final_storage", [])
    current_batch = state.get("batch_count", 0)
    irt_repair_mode = state.get("irt_repair_mode", False)
    if irt_repair_mode:
        new_batch_count = current_batch
        print(f"🔧 [归档-修复] IRT修复题目已入库，批次保持: {new_batch_count}")
    else:
        new_batch_count = current_batch + 1
        print(f"\n [归档完成] 第 {new_batch_count} 个批次结束。")
        print(f"本批合格: {len(current_passed)} 题|总库存: {len(final_storage)} 题")

    return {
        "final_storage": final_storage,
        "batch_count": new_batch_count,
        "generated_items": [],
        "passed_items": [],
        "low_cvi_items": [],
        "evaluation_results": [],
        "iteration": 0,
        # 清空当前提示，下一轮由队列补充
        "irt_repair_mode": False,
        "irt_revision_prompt": "",
    }


def create_sjt_workflow(model: ChatOpenAI = None) -> StateGraph:
    workflow = StateGraph(WorkflowState)
    workflow.add_node("generate_items", generate_items_node)
    workflow.add_node("evaluate_items", evaluate_items_node)
    workflow.add_node("convert_to_CVI", convert_to_CVI_node) # 计算CVI节点
    workflow.add_node("accumulator", accumulator_node)       # 归档节点
    workflow.add_node("virtual_subject", virtual_subject_node) # 虚拟被试节点
    workflow.add_node("virtual_subject_response", virtual_subject_response_node) # 虚拟被试回答节点
    workflow.add_node("analysis", analysis_node) # 分析节点
    workflow.set_entry_point("generate_items")
    workflow.add_edge("generate_items", "evaluate_items")
    workflow.add_edge("evaluate_items", "convert_to_CVI")
    workflow.add_conditional_edges(
        "convert_to_CVI",
        check_quality,
        {
            "revise": "generate_items",
            "archive": "accumulator",
        }
    )
    workflow.add_conditional_edges(
        "accumulator",
        check_quantity,
        {
            "next_batch": "generate_items",
            "finish": "virtual_subject"
        }
    )
    workflow.add_edge("virtual_subject", "virtual_subject_response")
    workflow.add_edge("virtual_subject_response", "analysis")
    workflow.add_conditional_edges(
        "analysis",
        check_irt_repair,
        {
            "repair": "generate_items",
            "finish": END
        }
    )
    return workflow.compile()


def create_sjt_repair_workflow(model: ChatOpenAI = None) -> StateGraph:
    """Repair-only workflow that skips virtual subjects and CITC analysis."""
    workflow = StateGraph(WorkflowState)
    workflow.add_node("generate_items", generate_items_node)
    workflow.add_node("evaluate_items", evaluate_items_node)
    workflow.add_node("convert_to_CVI", convert_to_CVI_node)
    workflow.add_node("accumulator", accumulator_node)
    workflow.set_entry_point("generate_items")
    workflow.add_edge("generate_items", "evaluate_items")
    workflow.add_edge("evaluate_items", "convert_to_CVI")
    workflow.add_conditional_edges(
        "convert_to_CVI",
        check_quality,
        {
            "revise": "generate_items",
            "archive": "accumulator",
        }
    )
    workflow.add_conditional_edges(
        "accumulator",
        check_irt_repair,
        {
            "repair": "generate_items",
            "finish": END,
        }
    )
    return workflow.compile()

def run_workflow(
    trait_names: List[str],
    model: ChatOpenAI = None,
    experts: List[ChatOpenAI] = None,
    irt_max_iterations: int = 1,
) -> Dict[str, Any]:
    workflow = create_sjt_workflow(model)
    initial_state: WorkflowState = {
        "target_trait": trait_names[0] if trait_names else "",
        "trait_names": trait_names,
        "model": model,
        "experts": experts,
        "target_batches": 5,
        "batch_count": 0,
        "final_storage": [],
        "generated_items": [],
        "low_cvi_items": [],
        "iteration": 0,
        # 修复相关字段初始化
        "irt_repair_mode": False,
        "irt_iteration": 0,
        "irt_max_iterations": irt_max_iterations,
        "irt_bad_items": [],
        "irt_revision_prompt": "",
        "irt_prompt_queue": [],
        "irt_repair_trait_name": ""
    }
    result = workflow.invoke(
        initial_state,
        config={"recursion_limit": 150}
    )
    final_items = result.get("final_storage", [])
    return {"final_items": final_items}

def main():
    from dotenv import load_dotenv
    from package.utils import TRAIT_ORDER
    from package.evaluators.SJTcontent_validity import get_content_validity_experts
    load_dotenv()
    model = ChatOpenAI(model="gpt-5-mini", temperature=0.5, max_tokens=7000)
    trait_names = [name for _, name in TRAIT_ORDER]
    experts = get_content_validity_experts()
    result = run_workflow(
        trait_names=trait_names,
        model=model,
        experts=experts,
        irt_max_iterations=3,
    )
    
    # 保存结果到文件
    project_root = get_project_root()
    output_dir = project_root / "output" / "workflow_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"workflow_result_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"📄 工作流结果已保存至: {result_file}")
    print(f"\n工作流执行完成！")
if __name__ == "__main__":
    main()
