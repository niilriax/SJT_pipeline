import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from package.utils import TRAIT_ORDER, get_project_root
from package.utils.reliability_validity import neo_responses_to_scored_matrix
from package.utils.reliability_validity_SJT import sjt_responses_to_scored_matrix


def _sum_by_trait(df: pd.DataFrame) -> pd.DataFrame:
    """按 TRAIT_ORDER 汇总各特质得分，返回列为 trait_code，索引为被试顺序。"""
    scores: Dict[str, pd.Series] = {}
    for code, _ in TRAIT_ORDER:
        cols = [c for c in df.columns if c.startswith(f"Q{code}_")]
        if not cols:
            continue
        scores[code] = df[cols].sum(axis=1)
    return pd.DataFrame(scores)


def compute_convergent_validity() -> pd.DataFrame:
    """计算 SJT 与 NEO 的汇聚效度（按特质的皮尔逊相关）。"""
    sjt_df = sjt_responses_to_scored_matrix()
    neo_df = neo_responses_to_scored_matrix()
    if sjt_df.empty or neo_df.empty:
        raise ValueError("SJT 或 NEO 计分矩阵为空，无法计算汇聚效度。")

    sjt_trait = _sum_by_trait(sjt_df)
    neo_trait = _sum_by_trait(neo_df)

    rows = []
    for code, name in TRAIT_ORDER:
        if code not in sjt_trait.columns or code not in neo_trait.columns:
            continue
        corr = np.nan
        try:
            corr = sjt_trait[code].corr(neo_trait[code])
        except Exception:
            pass
        rows.append({"trait_code": code, "trait_name": name, "correlation": corr})
    return pd.DataFrame(rows)


def save_convergent_validity(df: pd.DataFrame, output_path: Path | None = None) -> Path:
    if output_path is None:
        project_root = get_project_root()
        output_dir = project_root / "output" / "convergent_validity"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "convergent_validity.csv"
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def compute_correlation_matrix() -> pd.DataFrame:
    """计算 SJT/NEO 全部特质总分的相关矩阵（含跨量表）。"""
    sjt_df = sjt_responses_to_scored_matrix()
    neo_df = neo_responses_to_scored_matrix()
    if sjt_df.empty or neo_df.empty:
        raise ValueError("SJT 或 NEO 计分矩阵为空，无法计算相关矩阵。")
    sjt_trait = _sum_by_trait(sjt_df).add_prefix("SJT_")
    neo_trait = _sum_by_trait(neo_df).add_prefix("NEO_")
    merged = pd.concat([sjt_trait, neo_trait], axis=1)
    return merged.corr(method="pearson")


def save_corr_heatmap(corr_df: pd.DataFrame, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = get_project_root() / "output" / "convergent_validity"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_path = output_dir / "convergent_validity_heatmap.png"
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_df, annot=False, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("SJT/NEO 汇聚效度相关矩阵")
    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()
    return img_path


def main():
    # 对应特质的相关
    result_df = compute_convergent_validity()
    path = save_convergent_validity(result_df)
    print("汇聚效度（同名特质）：")
    print(result_df)
    print(f"结果已保存至: {path}")

    # 全相关矩阵 + 热力图
    corr_df = compute_correlation_matrix()
    project_root = get_project_root()
    output_dir = project_root / "output" / "convergent_validity"
    corr_path = output_dir / "convergent_validity_matrix.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(corr_path, encoding="utf-8-sig")
    heatmap_path = save_corr_heatmap(corr_df, output_dir=output_dir)
    print(f"\n完整相关矩阵已保存: {corr_path}")
    print(f"热力图已保存: {heatmap_path}")


if __name__ == "__main__":
    main()
