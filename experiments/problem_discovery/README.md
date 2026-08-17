# 问题发现算法实验（Problem Discovery）

把"根据明确的指标发现明确的问题"做成一个**确定性 + 概率**的算法骨架，
并在真实运行产物上演示。属于实验原型，未接入主工作流。

## 算法：序贯贝叶斯问题发现

```
后验(p) ∝ 先验(p) · ∏_家族 P(该家族观察结果 | 问题 p)
```

1. **finding 提取**（确定性阈值）：CITC、难度、选项选择率、盲法分类、
   选项得分与构念均分次序、选项点二列相关（可从真实作答重算）。
2. **家族分组朴素贝叶斯**：互斥 finding 按家族只贡献一个结果（触发成员
   或"未触发"）；"未触发"本身也是证据（盲法分类正确会强力压低构念错位）。
3. **三带决策**：healthy（P_NONE 领跑）/ act（最高后验 ≥ 0.70 且领先
   次名 ≥ 0.20，按期望损失最小化选动作）/ ambiguous（升级或追问）。
4. **下一步指标选择**：在未观察的候选指标里按 信息增益/成本 排序选最
   该补测的指标；增益不足则升级 INVESTIGATE（对应主流程的 LLM 诊断）。
5. **闭环**：修题 → 重施测 → 真实结果用于 Laplace 更新似然与先验，
   每条规则的 precision/resolution rate 可监控（见 docs 讨论）。

## 文件

- `taxonomy.py`：问题分类学、finding 家族、灵敏度/先验/损失/成本表。
  所有数字是专家先验，可被闭环数据更新。
- `diagnoser.py`：后验、熵、期望信息增益、期望损失、三带决策。
- `demo.py`：在 `outputs/virtual_responses/.../bank-v2-75babf8179a9` 的
  真实心理测量产物 + 盲法分类结果上运行，写出问题登记表。
- `test_diagnoser.py`：确定性单元测试。

## 运行

```powershell
python -X utf8 experiments\problem_discovery\demo.py
python -X utf8 -m pytest experiments\problem_discovery\test_diagnoser.py -q
```

## 已知边界

- 似然表是专家先验；`our_Q00X ↔ row-0X` 的盲法映射是批次假设。
- `alpha_if_deleted` 当前流水线未导出，是歧义题最常被建议补测的指标
  （这本身是一个对主流水线的真实改进项）。
- 阈值（0.70/0.20/0.02）与成本表可调；正式化时应按历史运行校准。
