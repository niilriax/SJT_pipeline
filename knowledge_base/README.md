# 人格知识库

本目录保存人格题目来源与可复用的 Behavior Evidence。

## 目录

- `items/ipip_neo_items.json`：从 `docs/ipip.md` 解析的 IPIP 条目。
- `evidence_library/<facet_name>/stage2_evidence_library_curated.json`：正式运行读取的
  curated 行为证据库；运行时默认只使用 `stable` family。
- `outputs/behavior_evidence_candidates/`：离线抽取 CLI 的临时输出，不会自动进入正式运行。

每条行为证据只含行为维度、可观察行为、高低表达、边界条件和来源题号。运行时若找不到某个 facet 的资源，系统调用一次抽取 Agent，写入 `resources/` 后立即使用。

```powershell
.\env\Scripts\python.exe -m sjt_system.knowledge.behavior_evidence_cli parse-ipip
.\env\Scripts\python.exe -m sjt_system.knowledge.behavior_evidence_cli build --facet A5
.\env\Scripts\python.exe -m sjt_system.knowledge.behavior_evidence_cli show outputs\behavior_evidence_candidates\agreeableness_modesty.json
```

IPIP 条目仅作为开发期行为证据来源。正式发表或分发前，仍需项目负责人核对所用内容的许可和引用要求。
