# 盲法构念分类研究（Blind Facet Classification）

用 LLM 作为盲法评判者，对 SJT 题目做构念分类，检验题目文本语义是否
对应其目标 facet。

## 方法

1. **金标准校准**：Mussel et al. (2018) 的 110 道题（5 facet × 22 题，
   `docs/mussel_zh.json`）作为已知标签校准集。分类器看不到任何标签，
   只看到情境 + 四个选项。
2. **待检验题目**：最近一次正式运行的 16 道题（目标 facet = compliance，
   来自被试卷 `respondent_form`，天然不含构念标签、计分键和蓝图信息）。
3. **盲法要求**：输入只含情境与选项文本；item_id、facet、计分键、
   construct_rationale、behavioral_level 一律不发送。
4. **分类器独立性**：默认使用与出题模型（deepseek-v4-flash）不同的
   推理模型 deepseek-v4-pro-guan（thinking enabled, reasoning_effort=high），
   温度不设置，避免"自己出题自己分类"的循环放大。
5. **单题独立调用**，固定随机打乱题序，避免批次 anchoring。
6. 指标：总体准确率 vs 随机基线（1/5 = 20%，二项检验）、各 facet
   精确率/召回率/F1、混淆矩阵、错分类明细。

## 目录结构

```
blind_classification/
  catalog.py      # 5 个 facet 定义目录（来自版本化构念注册表）
  classify.py     # 分类主脚本
  evaluate.py     # 指标计算与报告生成
  results/
    gold_mussel_classifications.json       # Mussel 金标准分类结果
    our_compliance_items_classifications.json  # 本次 16 题分类结果
    evaluation_report.md                   # 汇总报告
```

## 运行

```powershell
# 第一步：金标准校准（110 题）
.\env\Scripts\python.exe -X utf8 blind_classification\classify.py --skip-ours

# 第二步：对本次 16 题分类
.\env\Scripts\python.exe -X utf8 blind_classification\classify.py --skip-gold

# 生成评估报告
.\env\Scripts\python.exe -X utf8 blind_classification\evaluate.py
```

## 结果解释边界

- 分类器是 LLM 而非人类专家；结果只作为**开发期语义证据**。
- 只有金标准准确率显著高于随机且混淆矩阵合理时，才对本次题目的
  分类结果下结论。
- 高分类准确率证明"题目文本说的是这个构念"（内容效度），不证明
  "被试作答按这个构念运作"（需要心理测量证据，见虚拟被试分析）。
- 发表级构念效度证据仍需人类专家执行同样的盲法任务。
