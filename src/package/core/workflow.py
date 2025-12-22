# -*- codeing =utf-8 -*-
# @Time :2025/12/13 20:30:00
# @Author : Scientist
# @File :workflow.py
# @Software :PyCharm

from typing import TypedDict, List, Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
import json
from datetime import datetime
from package.generators import format_prompt
from package.evaluators import format_all_items_prompt
from package.utils import LLM_call, get_project_root


class WorkflowState(TypedDict, total=False):
    target_trait: str  # 当前特质名称
    generated_items: List[Dict[str, Any]]  # 当前特质生成好的题目
    evaluation_results: List[Dict[str, Any]]  # 当前特质的评价结果
    model: ChatOpenAI  # 模型
    experts: List[ChatOpenAI]  # 多个专家模型
    iteration: int  # 迭代次数
    modification_suggestions: List[Dict[str, Any]]  # 修改建议
    max_iterations: int  # 最大迭代次数


def generate_items_node(state: WorkflowState) -> WorkflowState:
    model = state.get("model")
    trait_name = state.get("target_trait", "")
    prompt = format_prompt(trait_name)
    items = LLM_call(prompt, model)
    state["generated_items"] = items
    return state


def evaluate_items_node(state: WorkflowState) -> WorkflowState:
    experts = state.get("experts")
    items = state.get("generated_items", [])
    trait_name = state.get("target_trait", "")
    data = {"特质": {trait_name: items}}
    prompt = format_all_items_prompt(data)
    all_evaluation_results: List[Dict[str, Any]] = []
    for idx, expert_model in enumerate(experts, start=1):
        eval_result = LLM_call(prompt, expert_model)
        all_evaluation_results.append({
            "expert_index": idx,
            "results": eval_result,
        })
        state["evaluation_results"] = all_evaluation_results
    return state


def create_sjt_workflow(model: ChatOpenAI = None) -> StateGraph:
    workflow = StateGraph(WorkflowState)
    workflow.add_node("generate_items", generate_items_node)
    workflow.add_node("evaluate_items", evaluate_items_node)
    workflow.set_entry_point("generate_items")
    workflow.add_edge("generate_items", "evaluate_items")
    return workflow.compile()


def run_workflow(
    trait_names: List[str],
    model: ChatOpenAI = None,
    experts: List[ChatOpenAI] = None,
) -> Dict[str, Any]:
    workflow = create_sjt_workflow(model)
    all_generated_items: Dict[str, List[Dict[str, Any]]] = {}
    all_evaluation_results: Dict[str, List[Dict[str, Any]]] = {}
    for trait_name in trait_names:
        initial_state: WorkflowState = {
            "target_trait": trait_name,
            "model": model,
            "experts": experts,
        }
        result = workflow.invoke(initial_state)
        items = result.get("generated_items", [])
        evaluation_results = result.get("evaluation_results", [])
        all_generated_items[trait_name] = items
        all_evaluation_results[trait_name] = evaluation_results
    project_root = get_project_root()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    # 1）保存生成的题目
    generated_items_path = results_dir / "generated_items.json"
    save_items = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "特质": all_generated_items,
    }
    generated_items_path.write_text(
        json.dumps(save_items, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    # 2）保存评估结果
    evaluation_results_path = results_dir / "evaluation_results.json"
    save_evals = {
        "生成时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "评价结果": all_evaluation_results,
    }
    evaluation_results_path.write_text(
        json.dumps(save_evals, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    result_dict: Dict[str, Any] = {}
    for trait_name in trait_names:
        result_dict[trait_name] = {
            "items": all_generated_items.get(trait_name, []),
            "evaluation_results": all_evaluation_results.get(trait_name, []),
        }
    return result_dict
def main():
    from dotenv import load_dotenv
    from package.utils import TRAIT_ORDER
    from package.evaluators.SJTcontent_validity import get_content_validity_experts
    load_dotenv()
    model = ChatOpenAI(model="gpt-5.1", temperature=0.5, max_tokens=7000)
    trait_names = [name for _, name in TRAIT_ORDER]
    experts = get_content_validity_experts()
    run_workflow(trait_names=trait_names, model=model, experts=experts)
    print("\n工作流执行完成！")

if __name__ == "__main__":
    main()