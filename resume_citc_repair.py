import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from package.core.workflow import create_sjt_workflow
from package.evaluators.SJTcontent_validity import get_content_validity_experts
from package.utils import TRAIT_ORDER, get_project_root
from package.utils.data_Ana import (
    citc_by_trait,
    generate_citc_prompts_to_files,
    sjt_responses_to_matrix,
)
import importlib
_irt_module = importlib.import_module('.2PL_IRT_for_SJT', package='package.utils')
load_sjt_items = _irt_module.load_sjt_items  


def load_final_storage_with_trait() -> list[dict]:
    """从 SJT_all_traits.json 重建 final_storage（附带 trait 字段）。"""
    items_dict = load_sjt_items() 
    storage = []
    for item in items_dict.values():
        trait = item.get("trait_name", "")
        item_id = item.get("item_id")
        if item_id is None:
            continue
        entry = item.copy()
        entry["trait"] = trait
        storage.append(entry)
    name_to_order = {name: idx for idx, (code, name) in enumerate(TRAIT_ORDER)}
    storage.sort(key=lambda x: (name_to_order.get(x.get("trait", ""), 999), int(x.get("item_id", 0))))
    return storage


def main():
    load_dotenv()
    project_root = get_project_root()

    data = sjt_responses_to_matrix()
    citc_df = citc_by_trait(data, items_per_trait=None, corrected=True)
    # 将部分题目标记为低于 0.5，以触发修复流程（这里简单选前 5 道设为 0.2）
    to_mark = citc_df.sort_values(by="citc", ascending=False).head(5).index
    citc_df.loc[to_mark, "citc"] = 0.2

    # 输出问题题目列表（CITC < 0.3 或 NaN）
    bad_df = citc_df[(citc_df["citc"].isna()) | (citc_df["citc"] < 0.3)]
    if not bad_df.empty:
        print("需要修订的题目（CITC<0.3 或 NaN）：")
        print(bad_df)
    else:
        print("当前未找到 CITC<0.3 的题目。")

    output_dir = project_root / "output" / "CITC_analysis"
    prompt_paths = generate_citc_prompts_to_files(citc_df, batch_size=5, output_dir=output_dir)
    if not prompt_paths:
        print("未发现需要修订的题目（CITC 均达标），退出。")
        return
    prompt_texts = [Path(p).read_text(encoding="utf-8") for p in prompt_paths]
    first_prompt = prompt_texts.pop(0)

    # 2) 构造初始状态，直接从修复流程开始
    final_storage = load_final_storage_with_trait()
    model = ChatOpenAI(model="gpt-5.1", temperature=0.5, max_tokens=7000)
    experts = get_content_validity_experts()
    trait_names = [name for _, name in TRAIT_ORDER]

    initial_state = {
        "target_trait": trait_names[0] if trait_names else "",
        "trait_names": trait_names,
        "model": model,
        "experts": experts,
        "target_batches": len(trait_names),
        "batch_count": len(trait_names),  # 避免进入常规生题
        "final_storage": final_storage,
        "generated_items": [],
        "low_cvi_items": [],
        "evaluation_results": [],
        "iteration": 0,
        # 修复相关
        "irt_repair_mode": True,
        "irt_iteration": 0,
        "irt_max_iterations": 3,
        "irt_bad_items": [],  # 不在此保存具体坏题
        "irt_revision_prompt": first_prompt,
        "irt_prompt_queue": prompt_texts,
    }

    workflow = create_sjt_workflow(model)
    result = workflow.invoke(initial_state, config={"recursion_limit": 150})
    final_items = result.get("final_storage", [])
    print(f"修复完成，最终题目数: {len(final_items)}")


if __name__ == "__main__":
    main()
