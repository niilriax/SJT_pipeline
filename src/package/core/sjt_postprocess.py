# -*- coding: utf-8 -*-
# @Time :2025/12/16
# @Author : Scientist
# @File :sjt_postprocess.py
# @Software :PyCharm
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from package.utils import extract_sjt_from_edited_results

def postprocess_sjt_results(
    results_path: str = "results_batch.json",
    sjt_output_dir: str = "sjt_outputs",
) -> None:
    extract_sjt_from_edited_results(
        results_path=results_path,
        output_dir=sjt_output_dir,
    )


def _generate_single_report(
    model: ChatOpenAI, subject_id: str, prompt_text: str
) -> Dict[str, Any]:
    try:
        resp = model.invoke([{"role": "user", "content": prompt_text}])
        report_text = resp.content if hasattr(resp, "content") else str(resp)
        return {"被试ID": subject_id, "report": report_text}
    except Exception as e:
        return {"被试ID": subject_id, "report": f"生成失败: {str(e)}"}

def _run_virtual_subject_pipeline_common(
    model: ChatOpenAI,
    prompts_input_path: Path,
    reports_output_path: Path,
    pipeline_name: str,
    max_workers: int = 5,
) -> List[Dict[str, Any]]:

    if not prompts_input_path.exists():
        raise FileNotFoundError(
            f"找不到提示词文件：{prompts_input_path}\n"
            f"请先运行 SJT_virtualSub_prompt.py 生成提示词文件。"
        )
    print(f"读取提示词文件: {prompts_input_path}")
    prompts = json.loads(prompts_input_path.read_text(encoding="utf-8"))
    print(f"已加载 {len(prompts)} 条 {pipeline_name} 提示词")
    print(f"\n开始生成 {pipeline_name} 作答（共 {len(prompts)} 个被试，并发数: {max_workers}）...")
    reports: List[Dict[str, Any]] = [None] * len(prompts)  
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                _generate_single_report,
                model,
                str(item["被试ID"]),
                item["prompt"]
            ): idx
            for idx, item in enumerate(prompts)
        }
        completed = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                reports[idx] = future.result()
                completed += 1
                if completed % 10 == 0 or completed == len(prompts):
                    print(f"  已处理 {completed}/{len(prompts)} 个被试...")
            except Exception as e:
                subject_id = str(prompts[idx]["被试ID"])
                reports[idx] = {"被试ID": subject_id, "report": f"生成失败: {str(e)}"}
                completed += 1
                print(f"  警告：被试 {subject_id} 处理失败: {e}")

    reports_output_path.write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"已生成 {len(reports)} 条 {pipeline_name} 报告，保存至: {reports_output_path}")

    return reports


def run_neo_virtual_subject_pipeline(
    model: ChatOpenAI,
    prompts_input_path: Optional[Path] = None,
    reports_output_path: Optional[Path] = None,
    max_workers: int = 5,
) -> List[Dict[str, Any]]:
    """运行 NEO 虚拟被试管道"""
    base_dir = Path(__file__).resolve().parent
    
    if prompts_input_path is None:
        prompts_input_path = base_dir.parent / "evaluators" / "filled_prompts_neo.json"
    
    if reports_output_path is None:
        reports_output_path = base_dir.parent / "evaluators" / "neo_reports.json"
    
    return _run_virtual_subject_pipeline_common(
        model=model,
        prompts_input_path=prompts_input_path,
        reports_output_path=reports_output_path,
        pipeline_name="NEO",
        max_workers=max_workers,
    )


def run_sjt_virtual_subject_pipeline(
    model: ChatOpenAI,
    prompts_input_path: Optional[Path] = None,
    reports_output_path: Optional[Path] = None,
    max_workers: int = 5,
) -> List[Dict[str, Any]]:
    """运行 SJT 虚拟被试管道"""
    base_dir = Path(__file__).resolve().parent
    
    if prompts_input_path is None:
        prompts_input_path = base_dir.parent / "evaluators" / "filled_prompts_sjt.json"
    
    if reports_output_path is None:
        reports_output_path = base_dir.parent / "evaluators" / "sjt_reports.json"
    
    return _run_virtual_subject_pipeline_common(
        model=model,
        prompts_input_path=prompts_input_path,
        reports_output_path=reports_output_path,
        pipeline_name="SJT",
        max_workers=max_workers,
    )


def _run_pipeline_with_error_handling(
    pipeline_func,
    pipeline_name: str,
    model: ChatOpenAI,
    max_workers: int = 5
) -> Optional[List[Dict[str, Any]]]:
    print("=" * 60)
    print(f"{pipeline_name} 虚拟被试评分流程")
    print("=" * 60)
    
    try:
        reports = pipeline_func(model=model, max_workers=max_workers)
        print("\n" + "=" * 60)
        print(f"{pipeline_name} 流程完成！")
        print("=" * 60)
        print(f"成功生成 {len(reports)} 条 {pipeline_name} 解释报告")
        return reports
    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print("\n提示: 请先运行 SJT_virtualSub_prompt.py 生成提示词文件")
        return None
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # 创建模型实例
    model = ChatOpenAI(model="gpt-5.1", temperature=0, max_tokens=7000)
    
    # 运行 NEO 流程
    neo_reports = _run_pipeline_with_error_handling(
        run_neo_virtual_subject_pipeline,
        "NEO-PI-R",
        model,
        max_workers=5
    )
    
    # 运行 SJT 流程
    print("\n\n")
    sjt_reports = _run_pipeline_with_error_handling(
        run_sjt_virtual_subject_pipeline,
        "SJT",
        model,
        max_workers=5
    )
    
    # 总结
    print("\n\n" + "=" * 60)
    print("全部流程完成！")
    print("=" * 60)
    if neo_reports:
        print(f"✓ NEO 报告: {len(neo_reports)} 条")
    else:
        print("✗ NEO 报告: 生成失败")
    if sjt_reports:
        print(f"✓ SJT 报告: {len(sjt_reports)} 条")
    else:
        print("✗ SJT 报告: 生成失败")
    print("=" * 60)
