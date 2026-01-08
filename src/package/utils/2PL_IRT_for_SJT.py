import numpy as np
from pathlib import Path
import json
import pandas as pd
from girth import twopl_mml
from package.utils import TRAIT_ORDER, get_project_root

# ================= 1. 数据加载部分 =================
def load_sjt_scoring_table() -> dict:
    project_root = get_project_root()
    sjt_path = project_root / "src" / "package" / "utils" / "sjt_outputs" / "SJT_all_traits.json"
    if not sjt_path.exists():
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
        raw_sid = subj.get("被试ID")
        try:
            sid = str(int(float(raw_sid)))
        except Exception:
            sid = str(raw_sid)
        ans_list = subj.get("response") or []
        row_scores = {qid: None for qid in all_qids}
        for ans in ans_list:
            qid = ans.get("题目ID")
            choice = ans.get("被试选择")
            if not qid or not choice:
                continue
            if qid in scoring_table:
                score = scoring_table[qid].get(choice)
                if score is None:
                    score = 0
                row_scores[qid] = score
        rows.append(row_scores)

    df = pd.DataFrame(rows, columns=all_qids)
    df = df.astype(float)
    # 处理缺失值：将所有 NaN 填充为 0（未回答的题目计为0分）
    df = df.fillna(0)
    return df


def run_irt_analysis(raw_data):
    if isinstance(raw_data, pd.DataFrame):
        data_array = raw_data.values
        item_ids = raw_data.columns.tolist()
    else:
        data_array = np.array(raw_data)
        item_ids = [f"Item_{i}" for i in range(data_array.shape[1])]
    irt_data = data_array.T.astype(int)
    if irt_data.ndim != 2:
        raise ValueError(f"数据维度错误！期望 2维矩阵，实际为 {irt_data.ndim}维。请检查数据是否为空或格式不对。")
    print(f"✅ 数据检查通过")
    print(f"   - 题目数量 (Rows): {irt_data.shape[0]}")
    print(f"   - 被试数量 (Cols): {irt_data.shape[1]}")
    print("🚀 正在运行 2PL IRT 模型 (可能需要几秒钟)...")
    try:
        results = twopl_mml(irt_data)
    except Exception as e:
        print("\n❌ IRT 模型计算失败！")
        print("常见原因：")
        print("1. 某道题所有人全对(全1)或全错(全0) -> 导致方差为0")
        print("2. 数据包含空值(NaN)")
        raise e
    discrimination = results['Discrimination']  # a 参数
    difficulty = results['Difficulty']  # b 参数
    df_result = pd.DataFrame({
        'Item_ID': item_ids,
        'Discrimination_a': discrimination,
        'Difficulty_b': difficulty
    })
    return df_result


def load_sjt_items() -> dict:
    project_root = get_project_root()
    sjt_path = project_root / "src" / "package" / "utils" / "sjt_outputs" / "SJT_all_traits.json"
    if not sjt_path.exists():
        raise FileNotFoundError(f"找不到 SJT_all_traits.json: {sjt_path}")
    
    sjt_data = json.loads(sjt_path.read_text(encoding="utf-8"))
    name_to_code = {name: code for code, name in TRAIT_ORDER}
    items_dict = {}
    
    for trait_name, trait_block in sjt_data.get("traits", {}).items():
        trait_code = name_to_code.get(trait_name)
        if not trait_code:
            continue
        for item in trait_block.get("items", []):
            item_id = item.get("item_id")
            if item_id is None:
                continue
            qid = f"Q{trait_code}_{item_id}"
            items_dict[qid] = {**item, "trait_name": trait_name}
    return items_dict


def _load_irt_template() -> str:
    project_root = get_project_root()
    template_path = project_root / "src" / "package" / "utils" / "sjt_outputs" / "IRT_prompt.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"找不到IRT_prompt.txt模板文件: {template_path}")
    return template_path.read_text(encoding="utf-8")


def _format_item_content(item_id: str, item: dict) -> str:
    trait_name = item.get("trait_name", "未知特质")
    situation = item.get("situation", "")
    question = item.get("question", "")
    options = item.get("options", {})
    lines = [
        f"Item_ID: {item_id}",
        f"Trait: {trait_name}",
        "原题目:",
        f"  - 情境描述: {situation}",
        f"  - 提问: {question}",
        "  - 选项:"
    ]
    for opt_key in sorted(options.keys()):
        opt_info = options[opt_key]
        content = opt_info.get("content", "")
        trait_level = opt_info.get("trait_level", "")
        lines.append(f"    {opt_key}. {content} (特质水平: {trait_level})")
    return "\n".join(lines)


def _generate_diagnosis(discrimination_a: float, difficulty_b: float) -> str:
    diagnosis_parts = []
    if discrimination_a < 0.5:
        diagnosis_parts.append(
            f"【严重问题：区分度极低 (a={discrimination_a:.3f})】该题目无法有效区分被试水平，可能属于噪音数据。"
            f"高特质和低特质的被试在选什么选项上没有明显差异。建议彻底重写情境，确保高分选项和低分选项在行为逻辑上有本质区别，选项界限过于模糊。"
        )
    elif discrimination_a < 0.8:
        diagnosis_parts.append(
            f"【问题：区分度一般 (a={discrimination_a:.3f})】题目区分能力有待提升。"
            f"建议微调选项措辞，增强高分选项的特质指向性，使选项之间的差异更加明显。"
        )
    if difficulty_b < -3.0:
        diagnosis_parts.append(
            f"【严重问题：题目太容易 (b={difficulty_b:.3f})】几乎所有人都选了高分项（得1分），说明低分干扰项毫无吸引力。"
            f"建议增加干扰项(低分选项)的合理性和吸引力，不要让低分项看起来太愚蠢或不合理。"
        )
    elif difficulty_b > 3.0:
        diagnosis_parts.append(
            f"【严重问题：题目太难 (b={difficulty_b:.3f})】几乎没人选高分项（得0分），说明高分选项太极端或太怪异。"
            f"建议降低正确选项(高分选项)的门槛，使其反应更符合常理，不要过于极端或理想化。"
        )
    return "\n".join(diagnosis_parts) if diagnosis_parts else "【参数表现一般，建议优化】"


def generate_item_revision_prompt(bad_items: pd.DataFrame) -> str:
    if bad_items.empty:
        return ""
    template = _load_irt_template()
    items_dict = load_sjt_items()
    all_items_text = []
    for _, row in bad_items.iterrows():
        item_id = row['Item_ID']
        if item_id not in items_dict:
            continue
        item_text = _format_item_content(item_id, items_dict[item_id])
        diagnosis_text = _generate_diagnosis(row['Discrimination_a'], row['Difficulty_b'])
        all_items_text.append(f"{item_text}\n【数据诊断意见】\n{diagnosis_text}")
    if not all_items_text:
        return ""
    return template.replace("【待优化题目及数据诊断意见】", "\n\n".join(all_items_text))
