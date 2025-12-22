# -*- codeing =utf-8 -*-
# @Time :2025/12/14
# @Author : Scientist
# @File :app_batch_example.py
# @Software :PyCharm
from dotenv import load_dotenv
import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from package.core import run_workflow, postprocess_sjt_results
from package.core.sjt_postprocess import run_neo_virtual_subject_pipeline, run_sjt_virtual_subject_pipeline
from package.generators import run_virtual_subject_simulation
from package.evaluators.SJT_virtualSub_prompt import generate_filled_prompts_with_scores_only
from package.utils import get_project_root, TRAIT_ORDER

load_dotenv()
# 创建模型实例
model = ChatOpenAI(model="gpt-5.1", temperature=0.5, top_p=1, max_tokens=7000)
print("\n" + "=" * 60)
print("批量生成题目")
print("=" * 60)
trait_names = [name for _, name in TRAIT_ORDER]
result_custom = run_workflow(
    trait_names=trait_names,
    model=model,
)
with open("results_batch.json", "w", encoding="utf-8") as f:
    json.dump(result_custom, f, ensure_ascii=False, indent=2)
postprocess_sjt_results(
    results_path="results_batch.json",
    sjt_output_dir="sjt_outputs",
)
print("\n" + "=" * 60)
print("生成完成！统计信息：")
print("=" * 60)
for trait_name, data in result_custom.items():
    if trait_name == "_summary":
        continue
    print(f"\n{trait_name}:")
    print(f"  题目数: {data.get('total_items', 0)} 道")
if "_summary" in result_custom:
    summary = result_custom["_summary"]
    print(f"\n总体统计：")
    print(f"  总题目数: {summary.get('总题目数', 0)}")
    print(f"  平均CVI: {summary.get('平均CVI', 0.0)}")
    print(f"  整体通过: {'是' if summary.get('整体通过', False) else '否'}")
    print(f"  迭代次数: {summary.get('迭代次数', 0)}")
print("\n结果已保存到 results_batch.json")
print("SJT 题库已导出到文件夹 sjt_outputs/SJT_all_traits.json")

print("\n" + "=" * 60)
print("生成虚拟被试...")
print("=" * 60)
project_root = get_project_root()
virtual_subjects_path = project_root / "src" / "package" / "generators" / "virtual_subjects.csv"

if not virtual_subjects_path.exists():
    print("正在生成虚拟被试 CSV 文件...")
    virtual_subjects_df = run_virtual_subject_simulation(
        n_subjects=100,  # 可根据需要调整数量
    )
    virtual_subjects_path.parent.mkdir(parents=True, exist_ok=True)
    virtual_subjects_df.to_csv(virtual_subjects_path, encoding="utf-8-sig", index=True)
    print(f"虚拟被试已生成: {virtual_subjects_path}")
else:
    print(f"使用已有虚拟被试文件: {virtual_subjects_path}")

# 生成提示词
print("\n" + "=" * 60)
print("生成提示词...")
print("=" * 60)

# 设置文件路径（使用项目根目录）
evaluators_dir = project_root / "src" / "package" / "evaluators"
utils_dir = project_root / "src" / "package" / "utils"
docs_dir = project_root / "docs"

neo_json_path = docs_dir / "Neo-FFI.json"
sjt_json_path = utils_dir / "sjt_outputs" / "SJT_all_traits.json"

# 生成 NEO 提示词
print("生成 NEO 提示词...")
neo_prompts_path = evaluators_dir / "filled_prompts_neo.json"
try:
    neo_prompts = generate_filled_prompts_with_scores_only(
        csv_path=virtual_subjects_path,
        sjt_json_path=None,  # NEO 流程不使用 SJT
        neo_path=neo_json_path if neo_json_path.exists() else None,
        output_path=neo_prompts_path,
    )
    print(f"✓ 已生成 {len(neo_prompts)} 条 NEO 提示词，保存至: {neo_prompts_path}")
except Exception as e:
    print(f"✗ NEO 提示词生成失败: {e}")

# 生成 SJT 提示词
print("\n生成 SJT 提示词...")
sjt_prompts_path = evaluators_dir / "filled_prompts_sjt.json"
try:
    if sjt_json_path.exists():
        sjt_prompts = generate_filled_prompts_with_scores_only(
            csv_path=virtual_subjects_path,
            sjt_json_path=sjt_json_path,
            neo_path=neo_json_path if neo_json_path.exists() else None,
            output_path=sjt_prompts_path,
        )
        print(f"✓ 已生成 {len(sjt_prompts)} 条 SJT 提示词，保存至: {sjt_prompts_path}")
    else:
        print(f"✗ 未找到 SJT 题目文件: {sjt_json_path}")
except Exception as e:
    print(f"✗ SJT 提示词生成失败: {e}")

# 调用大模型生成报告
print("\n" + "=" * 60)
print("调用大模型生成报告...")
print("=" * 60)

# 创建用于生成报告的模型实例（可能需要不同的参数）
report_model = ChatOpenAI(model="gpt-5.1", temperature=0, max_tokens=7000)

# 生成 NEO 报告
print("\n生成 NEO 报告...")
try:
    neo_reports = run_neo_virtual_subject_pipeline(
        model=report_model,
        prompts_input_path=neo_prompts_path,
        max_workers=50,
    )
    print(f"✓ 已生成 {len(neo_reports)} 条 NEO 报告")
except FileNotFoundError as e:
    print(f"✗ 错误: {e}")
    print("提示: 请先运行生成提示词步骤")
    neo_reports = None
except Exception as e:
    print(f"✗ NEO 报告生成失败: {e}")
    neo_reports = None

# 生成 SJT 报告
print("\n生成 SJT 报告...")
if sjt_json_path.exists():
    try:
        sjt_reports = run_sjt_virtual_subject_pipeline(
            model=report_model,
            prompts_input_path=sjt_prompts_path,
            max_workers=5,
        )
        print(f"✓ 已生成 {len(sjt_reports)} 条 SJT 报告")
    except FileNotFoundError as e:
        print(f"✗ 错误: {e}")
        print("提示: 请先运行生成提示词步骤")
        sjt_reports = None
    except Exception as e:
        print(f"✗ SJT 报告生成失败: {e}")
        sjt_reports = None
else:
    print(f"✗ 未找到 SJT 题目文件: {sjt_json_path}")
    sjt_reports = None

print("\n" + "=" * 60)
print("全部流程完成！")
print("=" * 60)

