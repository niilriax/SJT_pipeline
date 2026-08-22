"""问题分类学：finding 家族、问题假说、专家初始化的似然与损失。

所有数字是开发期的专家先验（可被闭环数据 Laplace 更新），不是从
数据估计出来的。闭环机制（修题 → 重施测 → 确认/推翻）是把它们
逐步变成经验似然的通道。

设计约定：
- finding 按"家族"分组，同一家族内互斥（如 CITC 家族只会触发
  NEG/LOW 其中一个）。贝叶斯更新时每个家族只贡献一个结果：
  触发的成员，或"未触发"（概率 = 1 − Σ该家族灵敏度）。
- sensitivity = P(该 finding 触发 | 该问题存在)。缺省 0.15。
- P_NONE 是"健康题"假说，所有 finding 对它的灵敏度都很低。
- LOSS[a][p] = 当真实问题是 p 时采取动作 a 的损失（开发周期单位）。
  INVESTIGATE 不参与期望损失最小化——它只用于"歧义且信息增益不足"
  时升级给 LLM 诊断。
"""

from __future__ import annotations

# 可被观察的 finding 家族。每族至多一个成员对同一题触发。
FAMILIES = [
    "CITC",            # 校正题总相关
    "DIFF",            # 标准化难度
    "OPTION",          # 选项选择率
    "BLIND",           # 盲法构念分类
    "CRITERION_ORDER", # 选项均分与评分键方向
    "OPTION_PBS",      # 选项点二列相关（对分面其余总分）
    "ALPHA_DROP",      # 删除题目后的 alpha 变化
]

FINDINGS: dict[str, tuple[str, str]] = {
    "F_CITC_NEG": ("CITC", "校正题总相关为负"),
    "F_CITC_LOW": ("CITC", "校正题总相关低于0.20"),
    "F_DIFF_HIGH": ("DIFF", "标准化难度高于0.80"),
    "F_DIFF_LOW": ("DIFF", "标准化难度低于0.20"),
    "F_OPTION_FEW": ("OPTION", "有效选项少于3个"),
    "F_OPTION_ZERO": ("OPTION", "存在零选择选项"),
    "F_BLIND_CROSS": ("BLIND", "盲法分类跨域错配"),
    "F_BLIND_SAME": ("BLIND", "盲法分类同域异facet错配"),
    "F_OPTION_ORDER_BROKEN": ("CRITERION_ORDER", "选项得分与选择者构念均分次序颠倒"),
    # 符号感知的点二列相关：低分选项负相关、高分选项正相关都是健康方向，
    # 只有方向与得分相反才是问题。
    "F_OPTION_HIGH_NEG_PBS": ("OPTION_PBS", "高分选项与分面其余总分负相关（高顺从不吸引高构念者）"),
    "F_OPTION_LOW_POS_PBS": ("OPTION_PBS", "低分选项与分面其余总分正相关（低分选项吸引高构念者）"),
    "F_ALPHA_DROP": ("ALPHA_DROP", "删除该题后α上升超过0.01"),
}

PROBLEMS: dict[str, tuple[str, str | None, str | None]] = {
    # code: (标签, locus, severity)
    "P_CONSTRUCT_MISALIGN": ("构念错位：题目语义偏离目标facet", "scenario", "high"),
    "P_OPTION_ANCHOR_MISLABEL": ("选项锚定标注错误：行为水平与选项文本不符", "response_options", "high"),
    "P_OPTION_DEAD": ("选项锚定失效：低行为水平选项无人选择", "response_options", "medium"),
    "P_OPTION_MISLEAD": ("干扰项误导：错误选项吸引高构念被试", "response_options", "high"),
    "P_SCORING_REVERSED": ("评分方向疑反：高构念者系统性选低分选项", "scoring_key", "critical"),
    "P_RANGE_RESTRICTED": ("难度偏移致分数范围受限", "scenario", "medium"),
    "P_RELIABILITY_HARM": ("信度损害：题目与卷内其余题目不一致", "item", "medium"),
    "P_NONE": ("未发现明确问题（健康）", None, None),
}

DEFAULT_SENSITIVITY = 0.15

SENSITIVITY: dict[str, dict[str, float]] = {
    "P_CONSTRUCT_MISALIGN": {
        "F_BLIND_CROSS": 0.75, "F_BLIND_SAME": 0.45, "F_CITC_LOW": 0.45,
        "F_OPTION_HIGH_NEG_PBS": 0.25, "F_OPTION_LOW_POS_PBS": 0.30,
        "F_OPTION_ORDER_BROKEN": 0.25,
        "F_CITC_NEG": 0.06, "F_OPTION_FEW": 0.15, "F_OPTION_ZERO": 0.10,
        "F_DIFF_HIGH": 0.08, "F_DIFF_LOW": 0.08, "F_ALPHA_DROP": 0.10,
    },
    "P_OPTION_ANCHOR_MISLABEL": {
        # 锚定标注错误最常见的后果：高低分档次序颠倒、极端难度、死选项。
        "F_OPTION_ORDER_BROKEN": 0.55, "F_DIFF_HIGH": 0.45, "F_DIFF_LOW": 0.45,
        "F_OPTION_ZERO": 0.40, "F_OPTION_FEW": 0.35,
        "F_BLIND_SAME": 0.30, "F_CITC_LOW": 0.30,
        "F_OPTION_LOW_POS_PBS": 0.25, "F_BLIND_CROSS": 0.20,
        "F_OPTION_HIGH_NEG_PBS": 0.20, "F_ALPHA_DROP": 0.15, "F_CITC_NEG": 0.10,
    },
    "P_OPTION_DEAD": {
        "F_OPTION_FEW": 0.85, "F_OPTION_ZERO": 0.55,
        "F_OPTION_ORDER_BROKEN": 0.30,
        "F_OPTION_HIGH_NEG_PBS": 0.25, "F_OPTION_LOW_POS_PBS": 0.30,
        "F_BLIND_CROSS": 0.12, "F_CITC_NEG": 0.08, "F_CITC_LOW": 0.30,
        "F_DIFF_HIGH": 0.10, "F_DIFF_LOW": 0.10, "F_ALPHA_DROP": 0.15,
        "F_BLIND_SAME": 0.20,
    },
    "P_OPTION_MISLEAD": {
        "F_CITC_NEG": 0.60, "F_OPTION_HIGH_NEG_PBS": 0.55,
        "F_OPTION_LOW_POS_PBS": 0.20,
        "F_OPTION_ORDER_BROKEN": 0.45, "F_OPTION_FEW": 0.30,
        "F_OPTION_ZERO": 0.20, "F_CITC_LOW": 0.35,
        "F_BLIND_CROSS": 0.15, "F_BLIND_SAME": 0.25,
        "F_DIFF_HIGH": 0.10, "F_DIFF_LOW": 0.10, "F_ALPHA_DROP": 0.20,
    },
    "P_SCORING_REVERSED": {
        "F_OPTION_ORDER_BROKEN": 0.55,
        "F_OPTION_HIGH_NEG_PBS": 0.35, "F_OPTION_LOW_POS_PBS": 0.35,
        "F_CITC_NEG": 0.35, "F_CITC_LOW": 0.25,
        "F_BLIND_CROSS": 0.20, "F_BLIND_SAME": 0.20,
        "F_OPTION_FEW": 0.20, "F_OPTION_ZERO": 0.15,
        "F_DIFF_HIGH": 0.15, "F_DIFF_LOW": 0.15, "F_ALPHA_DROP": 0.15,
    },
    "P_RANGE_RESTRICTED": {
        "F_DIFF_HIGH": 0.70, "F_DIFF_LOW": 0.70, "F_CITC_LOW": 0.25,
        "F_CITC_NEG": 0.15, "F_OPTION_ZERO": 0.15, "F_OPTION_FEW": 0.20,
        "F_OPTION_HIGH_NEG_PBS": 0.10, "F_OPTION_LOW_POS_PBS": 0.10,
        "F_OPTION_ORDER_BROKEN": 0.15,
        "F_BLIND_CROSS": 0.12, "F_BLIND_SAME": 0.15, "F_ALPHA_DROP": 0.10,
    },
    "P_RELIABILITY_HARM": {
        "F_ALPHA_DROP": 0.80, "F_CITC_LOW": 0.30, "F_CITC_NEG": 0.30,
        "F_OPTION_FEW": 0.25, "F_OPTION_ZERO": 0.20,
        "F_OPTION_HIGH_NEG_PBS": 0.15, "F_OPTION_LOW_POS_PBS": 0.15,
        "F_OPTION_ORDER_BROKEN": 0.20, "F_BLIND_CROSS": 0.15,
        "F_BLIND_SAME": 0.15, "F_DIFF_HIGH": 0.10, "F_DIFF_LOW": 0.10,
    },
    "P_NONE": {
        "F_CITC_NEG": 0.02, "F_CITC_LOW": 0.10, "F_DIFF_HIGH": 0.06,
        "F_DIFF_LOW": 0.06, "F_OPTION_FEW": 0.10, "F_OPTION_ZERO": 0.12,
        "F_BLIND_CROSS": 0.04, "F_BLIND_SAME": 0.10,
        "F_OPTION_ORDER_BROKEN": 0.06,
        "F_OPTION_HIGH_NEG_PBS": 0.06, "F_OPTION_LOW_POS_PBS": 0.06,
        "F_ALPHA_DROP": 0.10,
    },
}

PRIOR: dict[str, float] = {
    "P_CONSTRUCT_MISALIGN": 0.10,
    "P_OPTION_ANCHOR_MISLABEL": 0.15,
    "P_OPTION_DEAD": 0.25,
    "P_OPTION_MISLEAD": 0.08,
    "P_SCORING_REVERSED": 0.05,
    "P_RANGE_RESTRICTED": 0.15,
    "P_RELIABILITY_HARM": 0.10,
    "P_NONE": 0.12,
}

# 未被观察的候选指标家族（按获取成本排序）；decide() 会过滤掉已观察的。
CANDIDATE_FAMILIES = ["OPTION_PBS", "CRITERION_ORDER", "BLIND", "ALPHA_DROP"]

# 每族的边际测量成本（开发周期单位；LLM 批量成本高于本地重算）。
# 选下一个指标按 IG/COST 排序，与临床“按性价比选检查”同理。
FAMILY_COST = {
    "OPTION_PBS": 0.5,
    "CRITERION_ORDER": 0.5,
    "ALPHA_DROP": 1.0,
    "BLIND": 2.0,
}

# "明确行动"时参与期望损失最小化的动作。
ACTIONS = ["RETAIN", "REVISE_OPTIONS", "REVISE_SCENARIO", "REMOVE"]

LOSS: dict[str, dict[str, float]] = {
    "RETAIN": {
        "P_NONE": 0.0, "P_OPTION_DEAD": 6.0, "P_RANGE_RESTRICTED": 5.0,
        "P_RELIABILITY_HARM": 6.0, "P_CONSTRUCT_MISALIGN": 8.0,
        "P_OPTION_MISLEAD": 8.0, "P_OPTION_ANCHOR_MISLABEL": 8.0,
        "P_SCORING_REVERSED": 10.0,
    },
    "REVISE_OPTIONS": {
        "P_OPTION_DEAD": 2.0, "P_OPTION_MISLEAD": 2.0,
        "P_OPTION_ANCHOR_MISLABEL": 2.0, "P_RELIABILITY_HARM": 3.0,
        "P_CONSTRUCT_MISALIGN": 4.0, "P_RANGE_RESTRICTED": 4.0,
        "P_SCORING_REVERSED": 5.0, "P_NONE": 0.5,
    },
    "REVISE_SCENARIO": {
        "P_CONSTRUCT_MISALIGN": 2.0, "P_RANGE_RESTRICTED": 2.0,
        "P_OPTION_DEAD": 4.0, "P_OPTION_MISLEAD": 4.0,
        "P_OPTION_ANCHOR_MISLABEL": 4.0, "P_SCORING_REVERSED": 5.0,
        "P_RELIABILITY_HARM": 4.0, "P_NONE": 0.5,
    },
    "REMOVE": {
        "P_OPTION_MISLEAD": 3.0, "P_CONSTRUCT_MISALIGN": 3.0,
        "P_SCORING_REVERSED": 3.0, "P_OPTION_ANCHOR_MISLABEL": 3.5,
        "P_RELIABILITY_HARM": 3.5, "P_OPTION_DEAD": 4.0,
        "P_RANGE_RESTRICTED": 4.0, "P_NONE": 1.0,
    },
}

# 决策带：直接行动需要最高后验 ≥ THETA_ACT 且领先次名 ≥ MARGIN。
# 类比临床 rule-in 的"高门槛"：宁可歧义升级，不误修好题。
THETA_ACT = 0.70
MARGIN = 0.20
# 下一步指标选择的停止阈值：最大信息增益低于它就不值得再测，升级 LLM/人工。
EPS_IG = 0.02
