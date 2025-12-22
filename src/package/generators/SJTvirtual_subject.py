# -*- coding: utf-8 -*-
# @Time :2025/12/13 20:18:12
# @Author : Scientist
# @File :SJTvirtual_subject.py
# @Software :PyCharm

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from ..utils.common_utils import get_project_root


def load_corr_matrix_from_excel() -> Tuple[np.ndarray, List[str]]:
    project_root = get_project_root()
    file_path = project_root / "docs" / "neo_facet_cor.xlsx"
    df = pd.read_excel(file_path, header=0, index_col=0)
    facet_names = [str(x).strip() for x in df.columns.tolist()]
    df = df.loc[facet_names, facet_names]
    corr_matrix = df.astype(float).to_numpy()
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError(f"相关矩阵必须为方阵，当前形状={corr_matrix.shape}")
    return corr_matrix, facet_names


def create_virtual_subjects(
    n_subjects: int,
    corr_matrix: np.ndarray,
    facet_names: List[str],
    driving_facet: str,
    mean: float = 50.0,
    std: float = 10.0,
    noise_std: Optional[float] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    if driving_facet not in facet_names:
        raise ValueError(f"{driving_facet} 不在 facet 列表中：{facet_names}")
    n_dims = len(facet_names)
    if corr_matrix.shape != (n_dims, n_dims):
        raise ValueError(f"corr_matrix 形状应为 ({n_dims},{n_dims})，实际为 {corr_matrix.shape}")
    rng = np.random.default_rng(seed)
    k = facet_names.index(driving_facet)
    driving_scores = rng.normal(mean, std, n_subjects)
    scores = np.zeros((n_subjects, n_dims), dtype=float)
    scores[:, k] = driving_scores
    for j in range(n_dims):
        if j == k:
            continue
        r = float(corr_matrix[j, k])
        r = max(-1.0, min(1.0, r))
        y_mean = mean + r * (driving_scores - mean)
        if noise_std is None:
            eps_sd = std * np.sqrt(max(0.0, 1.0 - r * r))
        else:
            eps_sd = float(noise_std)
        scores[:, j] = y_mean + rng.normal(0.0, eps_sd, n_subjects)

    return pd.DataFrame(
        scores,
        columns=facet_names,
        index=pd.RangeIndex(1, n_subjects + 1, name="被试ID")
    )


def run_virtual_subject_simulation(
    n_subjects: int = 500,
    driving_facet: str = "N4",
    mean: float = 50.0,
    std: float = 10.0,
    noise_std: Optional[float] = None,
    seed: Optional[int] = 1
) -> pd.DataFrame:
    corr_matrix, facet_names = load_corr_matrix_from_excel()
    return create_virtual_subjects(
        n_subjects=n_subjects,
        corr_matrix=corr_matrix,
        facet_names=facet_names,
        driving_facet=driving_facet,
        mean=mean,
        std=std,
        noise_std=noise_std,
        seed=seed
    )
