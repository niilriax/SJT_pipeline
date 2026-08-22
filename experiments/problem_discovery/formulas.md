# 序贯贝叶斯问题发现：算法公式

## 记号

- 题目集合中的一道题：`item`
- 问题假说（含"健康"）：`p ∈ P`，共 7 个：
  `P_CONSTRUCT_MISALIGN`、`P_OPTION_DEAD`、`P_OPTION_MISLEAD`、
  `P_SCORING_REVERSED`、`P_RANGE_RESTRICTED`、`P_RELIABILITY_HARM`、`P_NONE`
- 指标（indicator）：`x₁…x_K`（CITC、难度、选项选择率、盲法分类、
  选项得分与构念均分次序、选项点二列相关、α-if-deleted……）
- finding：阈值化后的布尔事件 `f`
- finding 家族：`F₁…F_M`，族内互斥，每题每族至多一个成员触发
- 先验：`π(p)`
- 动作：`a ∈ {RETAIN, REVISE_OPTIONS, REVISE_SCENARIO, REMOVE}`
  （另有 `INVESTIGATE` 只用于升级，不参与期望损失最小化）

## 1. finding 提取（确定性）

对每个指标应用阈值谓词，产生 finding：

```
f = [ x_k ⊙ τ_k ]    ⊙ ∈ {<, ≤, >, ≥}
```

例：`F_CITC_NEG = [ citc < 0 ]`、`F_DIFF_HIGH = [ difficulty > 0.80 ]`、
`F_OPTION_FEW = [ effective_options < 3 ]`。选项点二列相关采用**符号感知**定义
（低分选项负相关、高分选项正相关是健康方向，不报警）：

```
F_OPTION_HIGH_NEG_PBS = [ score_o > 题均分 且 r_pb < −0.15 ]   # 高分不吸引高构念者
F_OPTION_LOW_POS_PBS  = [ score_o < 题均分 且 r_pb > +0.15 ]   # 低分选项吸引高构念者
```

## 2. 家族结果似然

每个家族对一道题只产生一个结果 `v_m`：触发的成员，或"未触发"（none）。
"未触发"本身也是证据：

```
              ⎧ s(f, p)                              v = f ∈ F_m 且 f 触发
L(F_m = v | p) = ⎨ max(0.05, 1 − Σ_{f ∈ F_m} s(f, p))  v = none
              ⎩
```

其中 `s(f, p) = P(f 触发 | 问题 p 存在)` 是灵敏度表（专家先验，
闭环后用 Laplace 更新，见第 6 节）。

## 3. 后验（家族分组朴素贝叶斯）

只对**已观察**（真的算过）的家族更新；缺失 ≠ 未触发：

```
                      π(p) · ∏_{m ∈ observed} L(F_m = v_m | p)
P(p | obs) = ─────────────────────────────────────────────────
              Σ_{p'} π(p') · ∏_{m ∈ observed} L(F_m = v_m | p')
```

## 4. 熵与期望信息增益

```
H(P) = − Σ_p P(p) · log₂ P(p)

IG(F_m) = H(P) − Σ_v P(F_m = v) · H(P | F_m = v)

P(F_m = v) = Σ_p P(p) · L(F_m = v | p)
```

`IG(F_m)` 衡量"再测家族 m 的指标，能消掉多少后验不确定性"。

## 5. 三带决策 + 下一步指标选择

记 `p*` 为后验最高的假说、`p²` 为次高：

```
if  p* = P_NONE 且 P(p*) ≥ θ_act:                      → healthy, a = RETAIN
elif P(p*) ≥ θ_act 且 P(p*) − P(p²) ≥ δ:               → act
        a* = argmin_a Σ_p P(p) · L(a, p)                （期望损失最小）
else:                                                  → ambiguous
        m* = argmax_{m ∉ observed} IG(F_m) / c_m        （信息增益/测量成本）
        if IG(F_m*) < ε:                                → INVESTIGATE（升级 LLM/人工）
        else:                                            → 先测 F_m*，回到第 2 节
```

默认参数：`θ_act = 0.70`，`δ = 0.20`，`ε = 0.02`。

`L(a, p)` 是损失矩阵（真实问题为 p 时采取动作 a 的开发周期损失）：
坏题保留损失最高（8–10），针对性修复损失低（2），INVESTIGATE 恒定成本 1。

## 6. 闭环校准（Laplace 更新）

每完成一轮"修题 → 重施测 → 确认/推翻"，用结果更新灵敏度与先验：

```
          α + n(f, p)
s(f, p) ← ────────────
          α + β + n(p)

π(p) ← (α' + n(p)) / (α' + β' + N)
```

`n(f, p)`：问题 p 被确认时 finding f 也出现的次数；
`n(p)`：问题 p 被确认的总次数；`N`：闭环处理的题目总数。
`α, β, α', β'` 为伪计数。这样先验表会随项目自己的运行数据进化，
每条规则的 precision / resolution rate 可监控。

## 7. 复杂度

- 每次决策：`O(|P| · M)`；
- 每个候选家族的 IG：`O(|P| · |F_m|)`；
- 全部确定性计算，无随机成分，可单元测试、可追溯。
