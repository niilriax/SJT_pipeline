# 题数驱动的情境容量设计

## 目标

用户要求的最终题数只决定双向细目表的题目槽位数，以及上游需要准备的唯一情境引用数量；它不反向强迫 Behavior Evidence 或 Activation Mechanism 凑数。

## 数据流

```text
final_item_count = N
  -> Expansion 至少准备 N 个唯一 Situation 引用
  -> Blueprint 选择 N 个不同引用
  -> 每个 Blueprint 行固定生成 1 道题
```

在多 facet 请求中，程序将 N 个所需情境引用按 facet 均衡分配；单 facet MVP 中，该 facet 直接承担 N 个引用。

## 约束

- Behavior Evidence 数量完全由 curated evidence 决定。
- 每条 Behavior Evidence 可有 1–4 个有实质差异的 Activation Mechanism。
- 不得为了题量创建重复或伪机制。
- 每个机制至少包含 1 个 Situation，不设置固定的每机制情境上限。
- Expansion 总情境容量不足当前 facet 配额时，重试 Expansion，而不是由 Blueprint 重复同一引用。
- Blueprint 必须返回 N 个唯一引用；每行的生成数和保留数均为 1。

## 不在本次范围

- 不修改 Behavior Evidence、Skeleton、Item Writer 或审题逻辑。
- 不增加文本相似度或程序化语义多样性评分。
- 不恢复增量开发或备用题机制。

