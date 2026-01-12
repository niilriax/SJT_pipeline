from pathlib import Path

import matplotlib
import numpy as np
from psychometric import Validity
import semopy
import pandas as pd
import seaborn as sns
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from package.utils.cronbach_alpha_NEO import neo_responses_to_scored_matrix


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)
np.set_printoptions(threshold=np.inf)


BASE_DIR = Path(__file__).parent
TRAITS = ["N", "E", "O", "A", "C"]


def load_trait_csv(trait: str) -> pd.DataFrame:
    csv_path = BASE_DIR / f"virtual_subjects_{trait}_responses.csv"
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "被试ID" in df.columns:
        df = df.set_index("被试ID")
    return df


def combine_traits() -> pd.DataFrame:
    parts = [load_trait_csv(t) for t in TRAITS]
    combined = pd.concat(parts, axis=1)
    combined.index.name = "被试ID"
    return combined


def build_neo_responses_json(df: pd.DataFrame) -> list[dict]:
    responses = []
    for subject_id, row in df.iterrows():
        items = []
        for col, val in row.items():
            if pd.isna(val):
                continue
            items.append({"题目ID": f"Q{col}", "被试选择": str(val)})
        responses.append({"被试ID": str(subject_id), "test_type": "NEO", "response": items})
    return responses


def compute_item_correlations(df: pd.DataFrame) -> pd.DataFrame:
    df_num = df.apply(pd.to_numeric, errors="coerce")
    return df_num.corr()


combined_df = combine_traits()
neo_json_path = BASE_DIR / "neo_responses_from_csv.json"
neo_json_path.write_text(
    pd.Series(build_neo_responses_json(combined_df)).to_json(orient="values", force_ascii=False),
    encoding="utf-8",
)

data = neo_responses_to_scored_matrix(responses_path=neo_json_path)

val = Validity(data)
fig, axes = plt.subplots(1, 2, figsize=(4, 2))
for ax, img, title, (xlab, ylab) in [
    (axes[0], data, "Input DataFrame", ("Items", "Subjects")),
    (axes[1], data.corr(), "Item Correlation", ("Items", "Items")),
]:
    ax.imshow(img, cmap="viridis", aspect="auto")
    ax.set(title=title, xlabel=xlab, ylabel=ylab)
fig.tight_layout()
output_path = "cfa_test_output.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Plot saved to: {output_path}")
plt.close()

struct = {code: [f"Q{code}_{i}" for i in range(1, 13)] for code in TRAITS}
cfa = val.cfa(struct)
print("=" * 50)
print("fit_indices:")
print("=" * 50)
print(cfa["fit_indices"])
print(cfa["model"])






model = cfa["model"]
try:
    inspect = model.inspect(std_est=True)
except Exception:
    model.fit(data)
    inspect = model.inspect(std_est=True)
factors = ['N', 'E', 'O', 'A', 'C']
correlations = inspect[
    (inspect['op'] == '~~') &
    (inspect['lval'].isin(factors)) &
    (inspect['rval'].isin(factors)) &
    (inspect['lval'] != inspect['rval'])
]

print("--- 因子间相关系数矩阵 (Factor Correlations) ---")

print(correlations[['lval', 'rval', 'Est. Std']])

matrix = pd.DataFrame(index=factors, columns=factors)
for _, row in correlations.iterrows():
    matrix.loc[row['lval'], row['rval']] = row['Est. Std']
    matrix.loc[row['rval'], row['lval']] = row['Est. Std']
for f in factors:
    matrix.loc[f, f] = 1.0
matrix = matrix.astype(float)
plt.figure(figsize=(8, 6))
sns.heatmap(matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Latent Factor Correlations (CFA)')
plt.show()
