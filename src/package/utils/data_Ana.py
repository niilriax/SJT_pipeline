import json
import numpy as np
import pandas as pd
from pathlib import Path
from psychometric.item_analysis import ItemAnalysis

from package.utils import TRAIT_ORDER, get_project_root


def load_sjt_scoring_table() -> dict:
    project_root = get_project_root()
    sjt_path = project_root / "src" / "package" / "utils" / "sjt_outputs" / "SJT_all_traits.json"
    if not sjt_path.exists():
        alt_path = Path("SJT_all_traits.json")
        if alt_path.exists():
            sjt_path = alt_path
        else:
            raise FileNotFoundError(f"找不到 SJT_all_traits.json: {sjt_path}")
    sjt_data = json.loads(sjt_path.read_text(encoding="utf-8"))
    name_to_code = {name: code for code, name in TRAIT_ORDER}
    scoring_table: dict = {}
    traits = sjt_data.get("traits", {})
    for trait_name, trait_block in traits.items():
        items = trait_block.get("items", [])
        trait_code = name_to_code.get(trait_name)
        if not trait_code:
            continue
        for item in items:
            item_id = item.get("item_id")
            if item_id is None:
                continue
            qid = f"Q{trait_code}_{item_id}"
            option_scores = {}
            options = item.get("options", {})
            for opt_key, opt_info in options.items():
                level = (opt_info.get("trait_level") or "").lower()
                option_scores[opt_key] = 1 if level == "high" else 0
            scoring_table[qid] = option_scores
    return scoring_table


def sjt_responses_to_matrix() -> pd.DataFrame:
    project_root = get_project_root()
    resp_path = project_root / "src" / "package" / "evaluators" / "sjt_responses.json"
    if not resp_path.exists():
        alt_path = Path("sjt_responses.json")
        if alt_path.exists():
            resp_path = alt_path
        else:
            raise FileNotFoundError(f"找不到 sjt_responses.json: {resp_path}")
    responses = json.loads(resp_path.read_text(encoding="utf-8"))
    scoring_table = load_sjt_scoring_table()
    code_order = {code: idx for idx, (code, _) in enumerate(TRAIT_ORDER)}

    def _qid_sort_key(qid: str):
        try:
            trait_code = qid[1]
            item_part = qid.split("_", 1)[1]
            item_id = int(item_part)
        except Exception:
            trait_code = qid[1] if len(qid) > 1 else ""
            item_id = 0
        return (code_order.get(trait_code, 999), item_id, qid)

    all_qids = sorted(scoring_table.keys(), key=_qid_sort_key)
    rows = []
    for subj in responses:
        if subj.get("test_type") != "SJT":
            continue
        ans_list = subj.get("response") or []
        row_scores = {qid: np.nan for qid in all_qids}
        for ans in ans_list:
            qid = ans.get("题目ID")
            choice = ans.get("被试选择")
            if not qid or not choice:
                continue
            if qid in scoring_table:
                score = scoring_table[qid].get(choice)
                row_scores[qid] = 0 if score is None else score
        rows.append(row_scores)

    df = pd.DataFrame(rows, columns=all_qids)
    df = df.astype(float)
    return df


def _load_citc_template() -> str:
    project_root = get_project_root()
    template_path = project_root / "src" / "package" / "utils" / "sjt_outputs" / "CITC_prompt.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"找不到 CITC_prompt.txt: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _load_sjt_items() -> dict:
    project_root = get_project_root()
    sjt_path = project_root / "src" / "package" / "utils" / "sjt_outputs" / "SJT_all_traits.json"
    if not sjt_path.exists():
        raise FileNotFoundError(f"找不到 SJT_all_traits.json: {sjt_path}")
    data = json.loads(sjt_path.read_text(encoding="utf-8"))
    name_to_code = {name: code for code, name in TRAIT_ORDER}
    items = {}
    for trait_name, block in data.get("traits", {}).items():
        code = name_to_code.get(trait_name)
        for item in block.get("items", []):
            item_id = item.get("item_id")
            if item_id is None:
                continue
            qid = f"Q{code}_{item_id}" if code else str(item_id)
            items[qid] = {**item, "trait_name": trait_name}
    return items


def _format_citc_block(qid: str, item: dict, citc: float) -> str:
    trait = item.get("trait_name", "")
    situation = item.get("situation", "")
    question = item.get("question", "")
    options = item.get("options", {})
    lines = [
        f"Item_ID: {qid}",
        f"Trait: {trait}",
        "原题目:",
        f"  - 情境描述: {situation}",
        f"  - 提问: {question}",
        "  - 选项:"
    ]
    for opt_key in sorted(options.keys()):
        opt = options[opt_key]
        lines.append(f"    {opt_key}. {opt.get('content','')} (特质水平: {opt.get('trait_level','')})")
    diag = "CITC 无法计算（零方差/数据缺失）" if pd.isna(citc) else f"CITC={citc:.3f} < 0.3"
    lines.append(f"【数据诊断意见】\n{diag}")
    return "\n".join(lines)


def generate_citc_revision_prompt(citc_df: pd.DataFrame) -> str:
    if citc_df is None or citc_df.empty:
        return ""
    items_map = _load_sjt_items()
    template = _load_citc_template()
    bad_df = citc_df[(citc_df["citc"].isna()) | (citc_df["citc"] < 0.3)]
    blocks = []
    for _, row in bad_df.iterrows():
        qid = str(row.get("item") or "")
        if qid not in items_map:
            continue
        blocks.append(_format_citc_block(qid, items_map[qid], row.get("citc", np.nan)))
    if not blocks:
        return ""
    body = "\n\n".join(blocks)
    return template.replace("【待优化题目及数据诊断意见】", body)


def generate_citc_prompts_to_files(
    citc_df: pd.DataFrame,
    batch_size: int = 5,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    按批次生成多个 CITC 修订提示词文件。
    - batch_size: 每个提示包含的坏题数量
    - output_dir: 保存目录，默认 output/CITC_prompts
    """
    if citc_df is None or citc_df.empty:
        return []
    project_root = get_project_root()
    out_dir = output_dir or (project_root / "output" / "CITC_prompts")
    out_dir.mkdir(parents=True, exist_ok=True)
    bad_df = citc_df[(citc_df["citc"].isna()) | (citc_df["citc"] < 0.3)]
    items_map = _load_sjt_items()
    template = _load_citc_template()
    paths: list[Path] = []
    if bad_df.empty:
        return paths
    total = len(bad_df)
    for idx in range(0, total, batch_size):
        chunk = bad_df.iloc[idx : idx + batch_size]
        blocks = []
        for _, row in chunk.iterrows():
            qid = str(row.get("item") or "")
            if qid not in items_map:
                continue
            blocks.append(_format_citc_block(qid, items_map[qid], row.get("citc", np.nan)))
        if not blocks:
            continue
        body = "\n\n".join(blocks)
        prompt = template.replace("【待优化题目及数据诊断意见】", body)
        path = out_dir / f"CITC_prompt_batch_{idx//batch_size + 1}.txt"
        path.write_text(prompt, encoding="utf-8")
        paths.append(path)
    return paths


def citc_by_trait(df: pd.DataFrame, items_per_trait: int = None, corrected: bool = True) -> pd.DataFrame:
    records = []
    for code, trait_name in TRAIT_ORDER:
        prefix = f"Q{code}_"
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            continue
        if items_per_trait and len(cols) > items_per_trait:
            cols = cols[:items_per_trait]
        sub_df = df[cols]
        ia = ItemAnalysis(sub_df)
        res = ia.citc(corrected=corrected)
        res["trait"] = trait_name
        records.append(res)
    if not records:
        return pd.DataFrame(columns=["trait", "item", "citc", "quality"])
    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(by="citc", ascending=True)
    return out[["trait", "item", "citc", "quality"]]


if __name__ == "__main__":
    data = sjt_responses_to_matrix()
    citc_df = citc_by_trait(data, items_per_trait=None, corrected=True)
    print(citc_df)
    paths = generate_citc_prompts_to_files(citc_df, batch_size=5)
    if paths:
        print("\n生成的 CITC 提示文件:")
        for p in paths:
            print(p)
