import matplotlib
import numpy as np
from psychometric import Validity
import semopy
import pandas as pd
import seaborn as sns
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from package.utils.cronbach_alpha_NEO import neo_responses_to_scored_matrix

# 设置 pandas 显示选项，确保完整输出不省略
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
np.set_printoptions(threshold=np.inf)

data = neo_responses_to_scored_matrix()
data # n_subjects x n_items DataFrame
val = Validity(data)
fig, axes = plt.subplots(1, 2, figsize=(4, 2))
for ax, img, title, (xlab, ylab) in [
    (axes[0], data, "Input DataFrame", ("Items", "Subjects")),
    (axes[1], data.corr(), ">Item Correlation", ("Items", "Items")),
]:
    ax.imshow(img, cmap="viridis", aspect="auto")
    ax.set(title=title, xlabel=xlab, ylabel=ylab)
fig.tight_layout()
# 保存图片而不是显示（避免显示错误）
output_path = "cfa_test_output.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"图片已保存到: {output_path}")
plt.close()  # 关闭图形释放内存
# 根据 NEO-FFI 数据格式，每个特质有12道题，格式为 Q{code}_{item_id}
# 特质顺序：N(神经质), E(外向性), O(开放性), A(宜人性), C(尽责性)
struct = {
    'N': [f"QN_{i}" for i in range(1, 13)],  # QN_1 到 QN_12
    'E': [f"QE_{i}" for i in range(1, 13)],  # QE_1 到 QE_12
    'O': [f"QO_{i}" for i in range(1, 13)],  # QO_1 到 QO_12
    'A': [f"QA_{i}" for i in range(1, 13)],  # QA_1 到 QA_12
    'C': [f"QC_{i}" for i in range(1, 13)]   # QC_1 到 QC_12
}
cfa = val.cfa(struct)
cfa.keys()
print("=" * 50)
print("fit_indices 完整输出:")
print("=" * 50)
print(cfa['fit_indices'])





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
