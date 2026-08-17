# PSJT 构念档案

当前注册表提供 NEO-PI-R 的 5 个领域与 30 个 facet。构念档案只承担构念界定，不再保存预制情境。

每个 facet 的核心内容包括：

- `definition`
- `high_trait_behavior` / `low_trait_behavior`
- `common_confounds`
- `confounding_contexts` / `inappropriate_contexts`
- `forbidden_patterns`
- `option_design_rules`
- `hard_constraints`

情境内容由运行时的 Behavior Expansion 根据 Behavior Evidence、当前构念档案和 `target_population` 一次性生成。构念档案与情境空间因而没有重复字段。

实现来源：`sjt_system/authoring/legacy_construct_resources.py`，由 `sjt_system/authoring/construct_registry.py` 注册和加载。
