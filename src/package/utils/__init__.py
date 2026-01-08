# -*- coding: utf-8 -*-
# @Time :2025/12/13 20:20:07
# @Author : Scientist
# @File :__init__.py
# @Software :PyCharm

from .common_utils import (
    # 常量
    TRAIT_ORDER,
    # JSON 解析工具
    parse_json_response,
    extract_json_from_text,
    # LLM 调用工具
    LLM_call,
    LLM_call_concurrent,
    # 报告解析工具
    parse_single_report,
    # 路径工具
    get_project_root,
    # 其他工具
    cronbach_alpha,
    load_neo_reverse_flags,
    map_domain_name_to_letter,
)
from .reliability_validity import (
    load_neo_responses,
    neo_responses_to_scored_matrix,
    cronbach_alpha_by_trait,
)
# 为避免循环引用和非法模块名，推迟导入 SJT 相关函数
try:
    from .reliability_validity_SJT import (
        load_sjt_responses,
        sjt_responses_to_scored_matrix,
        cronbach_alpha_by_trait as cronbach_alpha_by_trait_sjt,
    )
except Exception:
    # 在部分环境中允许延迟导入；调用方可直接 from package.utils.reliability_validity_SJT import ...
    pass

import importlib
_irt_module = importlib.import_module('.2PL_IRT_for_SJT', package=__name__)
sjt_responses_to_matrix = _irt_module.sjt_responses_to_matrix
run_irt_analysis = _irt_module.run_irt_analysis
load_sjt_scoring_table = _irt_module.load_sjt_scoring_table
generate_item_revision_prompt = _irt_module.generate_item_revision_prompt

__all__ = [
    # 常量
    "TRAIT_ORDER",
    # JSON 解析工具
    "parse_json_response",
    "extract_json_from_text",
    # LLM 调用工具
    "LLM_call",
    "LLM_call_concurrent",
    # 报告解析工具
    "parse_single_report",
    # 路径工具
    "get_project_root",
    # 其他工具
    "cronbach_alpha",
    "load_neo_reverse_flags",
    "map_domain_name_to_letter",
    # NEO 作答导入与计分
    "load_neo_responses",
    "neo_responses_to_scored_matrix",
    "cronbach_alpha_by_trait",
    # SJT 作答导入与计分（如导入失败请直接从 reliability_validity_SJT 导入）
    "load_sjt_responses",
    "sjt_responses_to_scored_matrix",
    "cronbach_alpha_by_trait_sjt",
    # IRT分析工具
    "sjt_responses_to_matrix",
    "run_irt_analysis",
    "load_sjt_scoring_table",
    "generate_item_revision_prompt",
]


