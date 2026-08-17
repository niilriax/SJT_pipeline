# 当前核心 Prompt 契约

## Behavior Evidence

输入 facet 构念档案与 IPIP 条目，输出可观察的行为证据。不得生成情境、题干、选项、审查结论或工作流状态。

## Behavior Expansion

每个 facet 调用一次。输入该 facet 的全部 Behavior Evidence、构念档案与目标人群；每条行为输出 2–4 个 `activation_mechanism`，每个机制输出 3–6 个扁平情境条目。情境条目只含 `domain`、`actor_relation` 和 `event_class`。

## 双向细目表

输入全部 facet 与 Expansion，以及精确的生成和保留总数。Agent 只选择 `facet_id + behavior_id + mechanism_id + situation_id` 组合并分配题量。行 ID、槽位 ID、引用解析和总数确认由程序负责。

## Skeleton

输入当前细目表行解析出的 Behavior、Mechanism、Situation 与构念边界，只输出：

- `situation_type`
- `stakes_level`
- `social_context`
- `behavioral_tension`
- 四级 `option_structure`

## Item Writer

将 Skeleton 实现为完整题目。不得改变程序分配的题目身份、构念引用或评分层级。

Prompt 的可执行版本位于 `sjt_system/prompt/` 与 `sjt_system/authoring/situation_space.py`。
