"""
Configuration and constants for the SJT generation system.

This module holds:
1. System-wide settings (thresholds, limits)
2. NEO-PI-R based facet definitions (30 facets across 5 domains)
3. Review rubrics
4. Construct boundaries and inappropriate conditions

Sources for facet definitions:
- Costa, P. T., & McCrae, R. R. (1992/2010). NEO-PI-R Manual.
- Costa & McCrae (1995). Domains and facets: Hierarchical personality assessment.
- Tett & Burnett (2003). A personality trait-based interactionist model of job performance.
- Christian et al. (2010). A meta-analysis of SJT criterion-related validity.

Sources for construct boundaries and inappropriate contexts:
- LLM-elicited based on NEO-PI-R facet definitions
- Culturally adapted for Chinese university students (collectivism, face culture, academic pressure)
- Contemporary college life scenarios (dormitory relationships, bao yan competition, employment stress)
- Designed for ecological validity in Chinese higher education context
"""

from dataclasses import dataclass, field
from typing import Dict, List

ROLE_STAGES = ["student", "early_career", "mid_career", "team_lead"]
GENDERS = ["woman", "man", "nonbinary"]

# =============================================================================
# NEO-PI-R Facet Definitions
# =============================================================================


@dataclass
class FacetDefinition:

    domain: str
    facet_name: str
    definition: str
    high_trait_behavior: str
    low_trait_behavior: str
    common_confounds: List[str]
    forbidden_patterns: List[str]
    option_design_rules: List[str]
    scoring_logic: str
    confounding_contexts: List[str] = field(default_factory=list)
    inappropriate_contexts: List[str] = field(default_factory=list)


def get_neo_pi_r_facets() -> Dict[str, FacetDefinition]:

    facets = {}

    # ==========================================================================
    # NEUROTICISM FACETS (N1-N6)
    # ==========================================================================

    facets["neuroticism_anxiety"] = FacetDefinition(
        domain="neuroticism",
        facet_name="anxiety",
        definition="容易体验担忧、紧张和不安，并倾向于预期潜在危险或负面后果。",
        high_trait_behavior="反复扫描风险、寻求确认、担心事情出错，并回避不确定情境",
        low_trait_behavior="面对不确定性时较平静，不轻易预设问题，相信自己能处理意外",
        common_confounds=[
            "需要区分真实威胁水平和稳定焦虑倾向",
            "压力负荷高时可能与压力脆弱性重叠",
        ],
        forbidden_patterns=[
            "不要把焦虑描写成愚蠢或不理性",
            "避免惊恐发作等临床症状",
        ],
        option_design_rules=[
            "高焦虑选项应体现担忧但仍能行动",
            "低焦虑选项应平静但不轻视真实风险",
        ],
        scoring_logic="计分为 1 的选项表示较低焦虑：在不确定中仍能适应性行动",
        confounding_contexts=[
            "同时面临多个 deadline（也激活 stress_vulnerability）",
            "在很多人面前发言（也激活 self_consciousness）",
            "身体出现不明症状（可能激活 depression）",
        ],
        inappropriate_contexts=[
            "家人重病等待手术结果（过于极端，超出学生日常）",
            "面临退学警告（情境过于严重，不是正常压力）",
            "涉及心理疾病诊断场景（可能触发真实焦虑症患者）",
        ],
    )

    facets["neuroticism_angry_hostility"] = FacetDefinition(
        domain="neuroticism",
        facet_name="angry_hostility",
        definition="容易体验愤怒、烦躁和受冒犯感，并对阻碍或不公反应较强的倾向。",
        high_trait_behavior="明显表现出挫败、责怪他人、长时间介意，并在受阻时冲动反应",
        low_trait_behavior="受挫时保持克制，更倾向从情境因素解释问题，并较快放下",
        common_confounds=[
            "需要区分合理维护权益和愤怒反应",
            "可能与宜人性的顺从维度重叠",
        ],
        forbidden_patterns=[
            "不要把愤怒描写成必然错误",
            "避免攻击性、暴力或明显不道德反应",
        ],
        option_design_rules=[
            "高愤怒选项可体现强烈不满但不越界",
            "低愤怒选项应体现冷静沟通或重新解释",
        ],
        scoring_logic="计分为 1 的选项表示较低愤怒敌意：受挑衅时仍能保持克制",
        confounding_contexts=[
            "被不公平对待（也激活 agreeableness_compliance 的低分端）",
            "被公开批评（也激活 self_consciousness）",
            "计划被打乱（可能激活 conscientiousness_order）",
        ],
        inappropriate_contexts=[
            "涉及暴力冲突场景",
            "师生权力不对等的正面冲突（可能引导学生对抗权威）",
            "涉及地域歧视等敏感话题的引发愤怒场景",
        ],
    )

    facets["neuroticism_depression"] = FacetDefinition(
        domain="neuroticism",
        facet_name="depression",
        definition="容易体验悲伤、沮丧、孤独和消极自我评价的倾向。",
        high_trait_behavior="遇到失败后退缩、反复自责、预期负面结果，并感到低能量和低动力",
        low_trait_behavior="受挫后仍能维持希望和参与感，较快恢复自信并继续行动",
        common_confounds=[
            "需要区分正常哀伤和特质性低落",
            "可能与低外向性的积极情绪不足重叠",
        ],
        forbidden_patterns=[
            "不要描写自杀或自伤意念",
            "不要把低落选项写成道德软弱",
        ],
        option_design_rules=[
            "高抑郁选项可体现退缩但仍在日常功能范围内",
            "低抑郁选项应体现恢复力但不否认情绪",
        ],
        scoring_logic="计分为 1 的选项表示较低抑郁倾向：受挫后仍能保持投入和希望",
        confounding_contexts=[
            "社交失败后被孤立（也激活 self_consciousness）",
            "长期压力下的情绪低落（也激活 stress_vulnerability）",
            "失去兴趣精力不足（可能激活 extraversion 低分端）",
        ],
        inappropriate_contexts=[
            "自杀/自残相关场景",
            "重度抑郁症状描写（可能触发真实抑郁症患者）",
            "亲人去世的哀伤场景（这是正常哀伤不是特质抑郁）",
        ],
    )

    facets["neuroticism_self_consciousness"] = FacetDefinition(
        domain="neuroticism",
        facet_name="self_consciousness",
        definition="在社交或被评价情境中容易感到尴尬、羞耻或过度在意他人眼光的倾向。",
        high_trait_behavior="监控自我呈现、避免成为焦点、害怕出丑，并在发言前犹豫",
        low_trait_behavior="被关注时较自在，能自然表达，也能接受轻微尴尬",
        common_confounds=[
            "需要区分自我意识和内向偏好",
            "可能与社交焦虑重叠",
        ],
        forbidden_patterns=[
            "不要简单写成害羞",
            "不要把低自我意识写成自大或冒失",
        ],
        option_design_rules=[
            "高自我意识选项应体现顾虑但仍可参与",
            "低自我意识选项应体现自在投入而非炫耀",
        ],
        scoring_logic="计分为 1 的选项表示较低自我意识：被评价时仍能自然参与",
        confounding_contexts=[
            "和权威人士交谈（也激活 anxiety）",
            "不擅长的社交场合（可能激活 introversion）",
            "被评价表现（也激活 anxiety）",
        ],
        inappropriate_contexts=[
            "社交焦虑障碍级别的场景（如完全无法出门）",
            "涉及身体缺陷被嘲笑的场景",
            "霸凌/羞辱场景",
        ],
    )

    facets["neuroticism_impulsiveness"] = FacetDefinition(
        domain="neuroticism",
        facet_name="impulsiveness",
        definition="面对诱惑、冲动或即时满足时，较难延迟反应和考虑后果的倾向。",
        high_trait_behavior="容易顺从欲望、先行动后思考，难以延迟满足，常做临时决定",
        low_trait_behavior="能抵抗诱惑，先考虑后果再行动，并在冲动下保持自控",
        common_confounds=[
            "需要区分冲动性和健康的自发性",
            "可能与低尽责性的计划不足重叠",
        ],
        forbidden_patterns=[
            "不要涉及成瘾、药物或违法行为",
            "不要把冲动选项写成明显荒唐",
        ],
        option_design_rules=[
            "高冲动选项应承认欲望并较快行动",
            "低冲动选项应体现延迟、折中或完全抵抗",
        ],
        scoring_logic="计分为 1 的选项表示较低冲动性：面对诱惑仍能保持自控",
        confounding_contexts=[
            "追求新鲜刺激（也激活 extraversion_excitement_seeking）",
            "拖延不完成任务（也激活 conscientiousness_self_discipline 低分端）",
            "冒险行为（可能激活 conscientiousness_deliberation 低分端）",
        ],
        inappropriate_contexts=[
            "涉及药物/酒精/赌博的冲动场景",
            "违法行为（如偷窃、作弊）",
            "危险性行为场景",
        ],
    )

    facets["neuroticism_vulnerability"] = FacetDefinition(
        domain="neuroticism",
        facet_name="vulnerability",
        definition="在压力和多重要求下容易感到难以应对、混乱或依赖他人的倾向。",
        high_trait_behavior="压力下容易失序、过度求助、感到被压垮，并做出较差判断",
        low_trait_behavior="压力下仍能组织任务、处理多重要求，并保持基本应对信心",
        common_confounds=[
            "需要区分真实能力不足和压力脆弱性",
            "不确定情境下可能与焦虑重叠",
        ],
        forbidden_patterns=[
            "不要描写完全崩溃",
            "避免惊恐发作或临床危机描述",
        ],
        option_design_rules=[
            "高脆弱性选项可体现不知所措但仍尝试处理",
            "低脆弱性选项应体现冷静排序和适度求助",
        ],
        scoring_logic="计分为 1 的选项表示较低压力脆弱性：压力下仍保持功能",
        confounding_contexts=[
            "担心未来结果（也激活 anxiety）",
            "任务太多做不完（可能激活 conscientiousness_order 低分端）",
            "需要帮助时求助（也激活 anxiety）",
        ],
        inappropriate_contexts=[
            "灾难性事件（地震、火灾等）",
            "完全超出学生能力的危机场景",
            "需要专业心理干预的压力场景",
        ],
    )

    # ==========================================================================
    # EXTRAVERSION FACETS (E1-E6)
    # ==========================================================================

    facets["extraversion_warmth"] = FacetDefinition(
        domain="extraversion",
        facet_name="warmth",
        definition="愿意建立亲近关系并表达友好、关心和情感连接的倾向。",
        high_trait_behavior="表达关心和亲近，维护关系，优先考虑人与人之间的连接",
        low_trait_behavior="保持情感距离，较少表达亲近，关系更正式或工具化",
        common_confounds=[
            "需要区分温暖和同情助人",
            "文化规范会影响亲近表达方式",
        ],
        forbidden_patterns=[
            "避免浪漫或暧昧化表达",
            "不要把低温暖写成冷酷",
        ],
        option_design_rules=[
            "高温暖选项应自然表达关心",
            "低温暖选项应保持礼貌距离而非敌意",
        ],
        scoring_logic="计分为 1 的选项表示较高温暖性：适度表达关心和连接",
        confounding_contexts=[
            "帮助他人（也激活 agreeableness_altruism）",
            "理解他人感受（也激活 agreeableness_tender_mindedness）",
            "表达情感（可能激活 openness_feelings）",
        ],
        inappropriate_contexts=[
            "过度侵入他人边界的热情（可能让人不适）",
            "浪漫/恋爱场景（超出友谊范围）",
            "物质付出过多的场景（可能涉及经济条件差异）",
        ],
    )

    facets["extraversion_gregariousness"] = FacetDefinition(
        domain="extraversion",
        facet_name="gregariousness",
        definition="喜欢与他人相处、享受群体场合并主动参与集体互动的倾向。",
        high_trait_behavior="主动寻找群体互动，喜欢热闹场合，并从集体中获得能量",
        low_trait_behavior="更偏好独处或小范围互动，长时间群体活动后容易疲惫",
        common_confounds=[
            "需要区分低群居性和社交焦虑",
            "活动兴趣会影响参与意愿",
        ],
        forbidden_patterns=[
            "不要把低群居性写成反社会",
            "避免强迫社交式情境",
        ],
        option_design_rules=[
            "高群居性选项应主动加入并享受互动",
            "低群居性选项应体现偏好安静但不排斥他人",
        ],
        scoring_logic="计分为 1 的选项表示较高群居性：主动寻求并享受群体陪伴",
        confounding_contexts=[
            "喜欢热闹（也激活 extraversion_excitement_seeking）",
            "需要合作的任务（可能激活 teamwork 相关）",
            "社交是为了建立关系网（可能是工具性而非真正喜欢）",
        ],
        inappropriate_contexts=[
            "派对/饮酒场景（可能引导不良行为）",
            "逃课出去玩的场景",
            "小团体/排他性社交场景",
        ],
    )

    facets["extraversion_assertiveness"] = FacetDefinition(
        domain="extraversion",
        facet_name="assertiveness",
        definition="倾向于清楚表达观点、影响他人并在需要时承担主导角色。",
        high_trait_behavior="自信表达意见、推动决策、愿意协调或领导他人",
        low_trait_behavior="更倾向倾听和等待他人带头，在表达立场时较谨慎",
        common_confounds=[
            "需要区分果断和攻击性",
            "权力距离会影响表达方式",
        ],
        forbidden_patterns=[
            "不要把果断写成支配控制",
            "不要把安静写成软弱",
        ],
        option_design_rules=[
            "高果断性选项应直接但尊重",
            "低果断性选项应谨慎参与而非消极逃避",
        ],
        scoring_logic="计分为 1 的选项表示较高果断性：适度表达并推动协作",
        confounding_contexts=[
            "表达不同意见（也激活 agreeableness_straightforwardness）",
            "领导团队（可能激活 conscientiousness_competence 自信）",
            "争取个人利益（可能激活 self_interest 而非特质）",
        ],
        inappropriate_contexts=[
            "对抗权威的挑衅场景",
            "侵犯他人权益的'assertiveness'",
            "在不适当场合强行发言的场景",
        ],
    )

    facets["extraversion_activity"] = FacetDefinition(
        domain="extraversion",
        facet_name="activity",
        definition="日常活动中的节奏、精力水平和保持忙碌的倾向。",
        high_trait_behavior="行动节奏快，喜欢保持忙碌，处理任务时有较强能量感",
        low_trait_behavior="偏好较慢节奏和较少安排，需要更多恢复时间",
        common_confounds=[
            "需要区分稳定活动水平和短期疲劳",
            "健康和睡眠会影响活动表现",
        ],
        forbidden_patterns=[
            "不要把低活动性写成懒惰",
            "不要把高活动性写成过劳",
        ],
        option_design_rules=[
            "高活动性选项应保持高效活跃但不过度透支",
            "低活动性选项应体现稳妥节奏",
        ],
        scoring_logic="计分为 1 的选项表示较高活动性：保持有活力的行动节奏",
        confounding_contexts=[
            "工作效率高（也激活 conscientiousness_achievement_striving）",
            "不喜欢拖延（也激活 conscientiousness_self_discipline）",
            "精力旺盛（可能激活 excitement_seeking）",
        ],
        inappropriate_contexts=[
            "工作狂场景（可能引导不健康生活方式）",
            "忽视休息的场景",
            "影响他人的过度活跃",
        ],
    )

    facets["extraversion_excitement_seeking"] = FacetDefinition(
        domain="extraversion",
        facet_name="excitement_seeking",
        definition="喜欢新鲜、刺激和变化体验，并愿意寻找较高唤起活动的倾向。",
        high_trait_behavior="主动尝试新鲜刺激的活动，偏好变化和较强感官体验",
        low_trait_behavior="更喜欢熟悉、平稳和可预测的活动",
        common_confounds=[
            "需要区分寻求刺激和不负责任冒险",
            "可能与开放性尝新重叠",
        ],
        forbidden_patterns=[
            "不要涉及违法、危险或伤害性活动",
            "不要把低刺激寻求写成无趣",
        ],
        option_design_rules=[
            "高刺激寻求选项应在合理边界内尝新",
            "低刺激寻求选项应选择稳定安全的体验",
        ],
        scoring_logic="计分为 1 的选项表示较高刺激寻求：在适当范围内追求新鲜刺激",
        confounding_contexts=[
            "冲动行为（也激活 neuroticism_impulsiveness）",
            "尝试新事物（也激活 openness_actions）",
            "追求变化（可能激活 openness_values）",
        ],
        inappropriate_contexts=[
            "危险冒险行为（无保护攀岩等）",
            "违法边缘的刺激追求",
            "可能伤害自己或他人的行为",
        ],
    )

    facets["extraversion_positive_emotions"] = FacetDefinition(
        domain="extraversion",
        facet_name="positive_emotions",
        definition="容易体验和表达愉快、兴奋、热情等积极情绪的倾向。",
        high_trait_behavior="经常感到开心和兴奋，容易表达赞赏和热情",
        low_trait_behavior="积极情绪表达较少或较平稳，不容易外显兴奋",
        common_confounds=[
            "需要区分积极情绪和社交礼貌",
            "低表达不等于抑郁",
        ],
        forbidden_patterns=[
            "不要把低积极情绪写成悲观或冷漠",
            "不要把高积极情绪写成夸张失控",
        ],
        option_design_rules=[
            "高积极情绪选项应自然表达喜悦",
            "低积极情绪选项应平稳回应但不否定价值",
        ],
        scoring_logic="计分为 1 的选项表示较高积极情绪：自然体验并表达愉快和热情",
        confounding_contexts=[
            "感到快乐（也激活 openness_feelings）",
            "表达情感（可能激活 warmth）",
            "积极态度（可能激活 optimism 而非特质）",
        ],
        inappropriate_contexts=[
            "过度兴奋/躁狂场景",
            "不合时宜的高兴（如他人悲伤时）",
            "表面化的强颜欢笑",
        ],
    )

    # ==========================================================================
    # OPENNESS FACETS (O1-O6)
    # ==========================================================================

    facets["openness_fantasy"] = FacetDefinition(
        domain="openness",
        facet_name="fantasy",
        definition="倾向于运用想象、构思可能情境，并让想象丰富日常体验。",
        high_trait_behavior="经常进行想象和联想，用虚构或可能性扩展体验",
        low_trait_behavior="更关注现实和具体事实，较少沉浸于想象",
        common_confounds=[
            "需要区分想象力和逃避现实",
            "可能与创造性重叠",
        ],
        forbidden_patterns=[
            "不要写成幼稚或不成熟",
            "不要写成脱离现实无法行动",
        ],
        option_design_rules=[
            "高幻想选项应建设性运用想象",
            "低幻想选项应体现现实取向而非贫乏",
        ],
        scoring_logic="计分为 1 的选项表示较高幻想性：用想象丰富经验",
        confounding_contexts=[
            "创造力（也激活 openness_aesthetics）",
            "规划未来（可能激活 conscientiousness_deliberation）",
            "逃避现实（可能激活 neuroticism_depression）",
        ],
        inappropriate_contexts=[
            "脱离现实的幻想",
            "病理性白日梦场景",
            "影响正常功能的想象",
        ],
    )

    facets["openness_aesthetics"] = FacetDefinition(
        domain="openness",
        facet_name="aesthetics",
        definition="对艺术、美感、音乐和环境审美元素敏感并愿意欣赏的倾向。",
        high_trait_behavior="容易注意并欣赏美感，能被艺术、音乐或自然景象打动",
        low_trait_behavior="更重视功能和实用，对审美元素关注较少",
        common_confounds=[
            "需要区分艺术训练和自然欣赏",
            "审美偏好受文化经验影响",
        ],
        forbidden_patterns=[
            "不要要求专业艺术知识",
            "不要把低审美敏感性写成没文化",
        ],
        option_design_rules=[
            "高审美选项应自然注意并欣赏美",
            "低审美选项应体现实用优先",
        ],
        scoring_logic="计分为 1 的选项表示较高审美性：自然欣赏美和艺术",
        confounding_contexts=[
            "喜欢设计（也激活 openness_fantasy）",
            "追求美感（可能激活 materialism）",
            "艺术爱好（可能涉及 socioeconomic status）",
        ],
        inappropriate_contexts=[
            "精英主义的艺术品味",
            "需要特定知识才能欣赏的场景",
            "昂贵消费场景（可能涉及经济条件差异）",
        ],
    )

    facets["openness_feelings"] = FacetDefinition(
        domain="openness",
        facet_name="feelings",
        definition="愿意觉察、接纳并探索自己和他人情绪体验的倾向。",
        high_trait_behavior="关注内在感受，愿意思考和表达情绪的细微差异",
        low_trait_behavior="较少关注情绪体验，更重视事实、任务或外在结果",
        common_confounds=[
            "需要区分情绪开放和情绪波动",
            "文化规范会影响情绪表达",
        ],
        forbidden_patterns=[
            "不要把低情感开放写成冷漠",
            "不要把高情感开放写成情绪失控",
        ],
        option_design_rules=[
            "高情感开放选项应探索和承认感受",
            "低情感开放选项应偏事实处理但不否定情绪",
        ],
        scoring_logic="计分为 1 的选项表示较高情感开放：重视并觉察情绪体验",
        confounding_contexts=[
            "情绪化（也激活 neuroticism）",
            "表达情感（可能激活 extraversion_warmth）",
            "共情能力（也激活 agreeableness_tender_mindedness）",
        ],
        inappropriate_contexts=[
            "情绪失控场景",
            "过度情绪化的表现",
            "情绪依赖场景",
        ],
    )

    facets["openness_actions"] = FacetDefinition(
        domain="openness",
        facet_name="actions",
        definition="愿意尝试新活动、新方法和不同日常安排的倾向。",
        high_trait_behavior="主动尝试新方式，愿意改变惯常做法并探索未知体验",
        low_trait_behavior="偏好熟悉流程和稳定习惯，对新做法较谨慎",
        common_confounds=[
            "需要区分尝新和寻求刺激",
            "资源和时间限制会影响尝试意愿",
        ],
        forbidden_patterns=[
            "不要把低尝新写成僵化",
            "避免危险或不道德尝试",
        ],
        option_design_rules=[
            "高行动开放选项应愿意试新方式",
            "低行动开放选项应偏好熟悉可靠方案",
        ],
        scoring_logic="计分为 1 的选项表示较高行动开放：愿意尝试新活动和新方式",
        confounding_contexts=[
            "追求刺激（也激活 extraversion_excitement_seeking）",
            "变化（可能激活 neuroticism_vulnerability 低分端）",
            "多样性（可能激活 openness_ideas）",
        ],
        inappropriate_contexts=[
            "危险的新尝试",
            "不负责任的改变",
            "影响他人的随意变动",
        ],
    )

    facets["openness_ideas"] = FacetDefinition(
        domain="openness",
        facet_name="ideas",
        definition="喜欢抽象思考、复杂概念和智性探索的倾向。",
        high_trait_behavior="享受理论讨论，主动探索新观点和复杂问题",
        low_trait_behavior="更偏好具体、实用和熟悉的内容",
        common_confounds=[
            "需要区分智力能力和观点开放兴趣",
            "专业背景会影响参与程度",
        ],
        forbidden_patterns=[
            "不要要求特定专业知识",
            "不要把低想法开放写成低智力",
        ],
        option_design_rules=[
            "高想法开放选项应投入抽象或概念内容",
            "低想法开放选项应强调实用和具体",
        ],
        scoring_logic="计分为 1 的选项表示较高想法开放：喜欢智性探索和抽象思考",
        confounding_contexts=[
            "学习成绩好（也激活 competence）",
            "知识储备（可能激活 education level）",
            "智力（可能混淆 intelligence）",
        ],
        inappropriate_contexts=[
            "知识分子优越感场景",
            "需要特定知识背景的场景",
            "学术精英主义",
        ],
    )

    facets["openness_values"] = FacetDefinition(
        domain="openness",
        facet_name="values",
        definition="愿意反思既有价值观、规则和社会惯例，并考虑替代观点的倾向。",
        high_trait_behavior="愿意重新审视假设，接纳多元观点并思考规则背后的理由",
        low_trait_behavior="更重视传统、稳定规范和既有价值框架",
        common_confounds=[
            "需要区分价值开放和道德立场",
            "政治敏感话题会干扰测量",
        ],
        forbidden_patterns=[
            "避免政治争议或敏感议题",
            "不要把低价值开放写成愚昧保守",
        ],
        option_design_rules=[
            "高价值开放选项应愿意审视假设",
            "低价值开放选项应重视稳定规范但不贬低他人",
        ],
        scoring_logic="计分为 1 的选项表示较高价值开放：愿意重新审视价值和规则",
        confounding_contexts=[
            "政治立场（也激活 political liberalism）",
            "道德判断（可能激活 moral reasoning）",
            "反传统（可能激活 rebellion 而非开放）",
        ],
        inappropriate_contexts=[
            "敏感政治话题",
            "争议性道德议题",
            "挑战底线的价值观",
        ],
    )

    # ==========================================================================
    # AGREEABLENESS FACETS (A1-A6)
    # ==========================================================================

    facets["agreeableness_trust"] = FacetDefinition(
        domain="agreeableness",
        facet_name="trust",
        definition="倾向于相信他人诚实、可信并怀有善意。",
        high_trait_behavior="愿意给他人善意解释，在合作中预期对方可靠",
        low_trait_behavior="更谨慎怀疑，会先核实动机和可信度",
        common_confounds=[
            "过往受骗经历会影响反应",
            "对方实际可信度需要控制",
        ],
        forbidden_patterns=[
            "不要把信任写成天真",
            "不要把怀疑写成绝对高明",
        ],
        option_design_rules=[
            "高信任选项应相信但保留基本判断",
            "低信任选项应谨慎但不偏执",
        ],
        scoring_logic="计分为 1 的选项表示较高信任：倾向善意解释并愿意合作",
        confounding_contexts=[
            "容易相信（也激活 naivety）",
            "不给怀疑（可能激活 lack of critical thinking）",
            "信任权威（可能激活 authority orientation）",
        ],
        inappropriate_contexts=[
            "盲目信任导致被骗场景",
            "涉及重大利益的信任决策",
            "明显应该警惕的场景",
        ],
    )

    facets["agreeableness_straightforwardness"] = FacetDefinition(
        domain="agreeableness",
        facet_name="straightforwardness",
        definition="在人际互动中诚实、直接、真诚表达而较少操控或掩饰的倾向。",
        high_trait_behavior="表达真实想法，沟通坦诚，较少策略性隐瞒",
        low_trait_behavior="更讲究策略和包装，可能根据利益调整表达",
        common_confounds=[
            "需要区分坦诚和不顾场合的冒犯",
            "文化中委婉表达会影响行为",
        ],
        forbidden_patterns=[
            "不要把低坦诚写成邪恶欺骗",
            "不要把高坦诚写成鲁莽伤人",
        ],
        option_design_rules=[
            "高坦诚选项应诚实但顾及方式",
            "低坦诚选项应体现策略性表达而非恶意欺骗",
        ],
        scoring_logic="计分为 1 的选项表示较高坦诚性：真诚、直接且有分寸",
        confounding_contexts=[
            "直接（也激活 extraversion_assertiveness）",
            "诚实（可能激活 honesty 道德）",
            "不圆滑（可能激活 social skills 低）",
        ],
        inappropriate_contexts=[
            "伤害他人感情的'直率'",
            "不分场合的直言不讳",
            "以直率为借口的粗鲁",
        ],
    )

    facets["agreeableness_altruism"] = FacetDefinition(
        domain="agreeableness",
        facet_name="altruism",
        definition="愿意主动关心他人需要并提供实际帮助的倾向。",
        high_trait_behavior="看到他人需要时主动帮助，愿意投入时间或资源",
        low_trait_behavior="更优先考虑自身事务，只有在必要或方便时帮助",
        common_confounds=[
            "需要区分助人意愿和助人能力",
            "资源不足会限制帮助行为",
        ],
        forbidden_patterns=[
            "不要把低利他写成自私冷血",
            "避免过度牺牲自我的英雄式帮助",
        ],
        option_design_rules=[
            "高利他选项应主动且适度帮助",
            "低利他选项应保持边界但不恶意拒绝",
        ],
        scoring_logic="计分为 1 的选项表示较高利他性：主动关心并愿意帮助他人",
        confounding_contexts=[
            "帮助行为（也激活 social desirability）",
            "热心（可能激活 extraversion_warmth）",
            "利他（可能激活 moral identity）",
        ],
        inappropriate_contexts=[
            "自我牺牲式的帮助",
            "超越能力的助人行为",
            "可能被视为讨好型人格的场景",
        ],
    )

    facets["agreeableness_compliance"] = FacetDefinition(
        domain="agreeableness",
        facet_name="compliance",
        definition="面对冲突时愿意克制对抗、让步或寻找和解方式的倾向。",
        high_trait_behavior="避免正面冲突，愿意退让、协商和维护关系",
        low_trait_behavior="更坚持自身立场，面对冲突时较直接或强硬",
        common_confounds=[
            "需要区分顺从和焦虑性回避",
            "也可能与果断性低分重叠",
        ],
        forbidden_patterns=[
            "不要把低顺从写成攻击性",
            "不要把高顺从写成没有原则",
        ],
        option_design_rules=[
            "高顺从选项应体面让步或缓和冲突",
            "低顺从选项应坚持立场但不伤害他人",
        ],
        scoring_logic="计分为 1 的选项表示较高顺从性：愿意为减少冲突而适度退让",
        confounding_contexts=[
            "避免冲突（也激活 neuroticism_anxiety）",
            "顺从（可能激活 authority orientation）",
            "passive（可能激活 low assertiveness）",
        ],
        inappropriate_contexts=[
            "被欺负不还手场景",
            "放弃正当权益",
            "助长他人不当行为的忍让",
        ],
    )

    facets["agreeableness_modesty"] = FacetDefinition(
        domain="agreeableness",
        facet_name="modesty",
        definition="对自己的成绩和优点保持谦逊、不夸耀的倾向。",
        high_trait_behavior="淡化自我表现，承认贡献但不强调优越感",
        low_trait_behavior="更愿意展示成就，强调自己的能力和贡献",
        common_confounds=[
            "需要区分谦逊和低自尊",
            "自我展示也可能是情境要求",
        ],
        forbidden_patterns=[
            "不要把低谦逊写成傲慢",
            "不要把高谦逊写成自我贬低",
        ],
        option_design_rules=[
            "高谦逊选项应自然承认成绩但不过度突出自己",
            "低谦逊选项应自信展示但不贬低他人",
        ],
        scoring_logic="计分为 1 的选项表示较高谦逊性：对成就保持低调和分寸",
        confounding_contexts=[
            "谦虚（也激活 cultural norm）",
            "缺乏自信（可能激活 low competence）",
            "自我贬低（可能激活 false modesty）",
        ],
        inappropriate_contexts=[
            "过度自我贬低",
            "虚假谦虚（心里不这么想）",
            "影响正常自我展示的场景",
        ],
    )

    facets["agreeableness_tender_mindedness"] = FacetDefinition(
        domain="agreeableness",
        facet_name="tender_mindedness",
        definition="容易被他人处境打动，并以同情和人道关怀看待问题的倾向。",
        high_trait_behavior="关注他人痛苦和处境，愿意从关怀角度判断问题",
        low_trait_behavior="更重视原则、效率或事实判断，较少被情绪打动",
        common_confounds=[
            "可能与利他和同情心重叠",
            "需要区分同情和情绪脆弱",
        ],
        forbidden_patterns=[
            "不要把低同情心写成残酷",
            "不要把高同情心写成没有判断",
        ],
        option_design_rules=[
            "高慈悲心选项应体现同情但保留判断",
            "低慈悲心选项应重视原则但不冷酷",
        ],
        scoring_logic="计分为 1 的选项表示较高慈悲心：对他人处境保持同情和关怀",
        confounding_contexts=[
            "同情（也激活 neuroticism_depression）",
            "感性（可能激活 openness_feelings）",
            "情绪化（可能激活 emotional volatility）",
        ],
        inappropriate_contexts=[
            "过度感伤",
            "因同情而放弃原则",
            "情绪卷入过深的场景",
        ],
    )

    # ==========================================================================
    # CONSCIENTIOUSNESS FACETS (C1-C6)
    # ==========================================================================

    facets["conscientiousness_competence"] = FacetDefinition(
        domain="conscientiousness",
        facet_name="competence",
        definition="相信自己有能力处理任务、解决问题并有效应对要求的倾向。",
        high_trait_behavior="对完成任务有信心，能评估资源并采取有效行动",
        low_trait_behavior="更容易怀疑自己的能力，需要更多确认或指导",
        common_confounds=[
            "需要区分真实能力和自我效能感",
            "经验不足会影响信心",
        ],
        forbidden_patterns=[
            "不要把低胜任感写成无能",
            "不要把高胜任感写成自负",
        ],
        option_design_rules=[
            "高胜任感选项应自信且现实",
            "低胜任感选项应谨慎但不完全放弃",
        ],
        scoring_logic="计分为 1 的选项表示较高胜任感：相信自己能有效处理任务",
        confounding_contexts=[
            "自信（也激活 extraversion_assertiveness）",
            "能力（可能激活 actual ability）",
            "过度自信（可能激活 Dunning-Kruger）",
        ],
        inappropriate_contexts=[
            "傲慢自大场景",
            "不切实际的自信",
            "轻视他人能力的场景",
        ],
    )

    facets["conscientiousness_order"] = FacetDefinition(
        domain="conscientiousness",
        facet_name="order",
        definition="偏好有序、整洁和结构清晰，并主动维护秩序的倾向。",
        high_trait_behavior="保持环境、资料和流程整齐有序，重视分类和结构",
        low_trait_behavior="能接受一定混乱，组织方式更随性或灵活",
        common_confounds=[
            "需要区分秩序偏好和强迫症状",
            "时间压力会影响整理程度",
        ],
        forbidden_patterns=[
            "不要把低秩序写成邋遢无能",
            "避免强迫症刻板描述",
        ],
        option_design_rules=[
            "高秩序选项应适度维护组织系统",
            "低秩序选项应体现灵活但仍能运转",
        ],
        scoring_logic="计分为 1 的选项表示较高秩序性：维持组织和整洁",
        confounding_contexts=[
            "整洁（也激活 cleanliness）",
            "组织（可能激活 time management）",
            "强迫（可能激活 OCD tendency）",
        ],
        inappropriate_contexts=[
            "强迫症级别的整洁",
            "对他人整洁度要求过高",
            "影响正常功能的整理行为",
        ],
    )

    facets["conscientiousness_dutifulness"] = FacetDefinition(
        domain="conscientiousness",
        facet_name="dutifulness",
        definition="重视责任、规则和义务，并倾向于按承诺行事的程度。",
        high_trait_behavior="认真履行承诺，遵守规则，对责任有较强内在要求",
        low_trait_behavior="对规则和义务更灵活，可能根据情境调整承诺",
        common_confounds=[
            "需要区分责任感和盲从",
            "规则合理性会影响行为",
        ],
        forbidden_patterns=[
            "不要把低责任写成不道德",
            "不要把高责任写成僵化服从",
        ],
        option_design_rules=[
            "高责任选项应可靠履行义务",
            "低责任选项应体现灵活调整但不恶意失信",
        ],
        scoring_logic="计分为 1 的选项表示较高责任义务感：理解并履行责任",
        confounding_contexts=[
            "遵守规则（也激活 authority orientation）",
            "可靠（可能激活 dependability）",
            "盲从（可能激活 lack of critical thinking）",
        ],
        inappropriate_contexts=[
            "盲目服从不当指令",
            "规则本身有问题时的遵守",
            "影响健康的过度尽责",
        ],
    )

    facets["conscientiousness_achievement_striving"] = FacetDefinition(
        domain="conscientiousness",
        facet_name="achievement_striving",
        definition="追求高标准、努力达成目标并希望表现出色的倾向。",
        high_trait_behavior="设定较高目标，持续投入，努力取得优秀结果",
        low_trait_behavior="目标较温和，更重视够用、平衡或过程体验",
        common_confounds=[
            "需要区分成就追求和外部压力",
            "能力差异会影响结果",
        ],
        forbidden_patterns=[
            "不要把低成就追求写成懒惰",
            "避免美化过度竞争或牺牲健康",
        ],
        option_design_rules=[
            "高成就追求选项应努力提升表现但不过度极端",
            "低成就追求选项应重视适度目标",
        ],
        scoring_logic="计分为 1 的选项表示较高成就追求：有达成和超越目标的动力",
        confounding_contexts=[
            "努力（也激活 self_discipline）",
            "成就动机（可能激活 extrinsic motivation）",
            "竞争（可能激活 competitiveness）",
        ],
        inappropriate_contexts=[
            "工作狂/过劳场景",
            "不择手段追求成就",
            "忽视健康和关系的奋斗",
        ],
    )

    facets["conscientiousness_self_discipline"] = FacetDefinition(
        domain="conscientiousness",
        facet_name="self_discipline",
        definition="即使任务枯燥或困难，也能启动、坚持并完成工作的倾向。",
        high_trait_behavior="能克服拖延，坚持推进困难或无聊任务",
        low_trait_behavior="容易受分心或厌倦影响，启动和坚持任务较困难",
        common_confounds=[
            "需要区分自律和任务兴趣",
            "压力或疲劳会影响坚持",
        ],
        forbidden_patterns=[
            "不要把低自律写成没有纪律或品行差",
            "避免把高自律写成过度压榨自己",
        ],
        option_design_rules=[
            "高自律选项应适当坚持并完成任务",
            "低自律选项应体现拖延或分心但不荒唐",
        ],
        scoring_logic="计分为 1 的选项表示较高自律性：面对困难仍能坚持推进",
        confounding_contexts=[
            "坚持（也激活 grit）",
            "自律（可能激活 self_control）",
            "忍耐（可能激活 suppression）",
        ],
        inappropriate_contexts=[
            "自我惩罚式的自律",
            "忽视身心信号的坚持",
            "影响健康的过度自律",
        ],
    )

    facets["conscientiousness_deliberation"] = FacetDefinition(
        domain="conscientiousness",
        facet_name="deliberation",
        definition="行动前倾向于谨慎思考、权衡后果并避免草率决定的程度。",
        high_trait_behavior="做决定前分析风险和后果，愿意花时间思考",
        low_trait_behavior="更快做决定，较少反复权衡，行动更直接",
        common_confounds=[
            "需要区分审慎和焦虑性犹豫",
            "过度分析可能影响行动",
        ],
        forbidden_patterns=[
            "不要把低审慎写成鲁莽",
            "不要把高审慎写成无法行动",
        ],
        option_design_rules=[
            "高审慎选项应考虑后果后行动",
            "低审慎选项应较快行动但不冒失",
        ],
        scoring_logic="计分为 1 的选项表示较高审慎性：行动前认真权衡后果",
        confounding_contexts=[
            "谨慎（也激活 anxiety）",
            "犹豫（可能激活 indecisiveness）",
            "分析（可能激活 analysis paralysis）",
        ],
        inappropriate_contexts=[
            "过度分析导致无法行动",
            "错失时机的犹豫",
            "不必要的过度思考",
        ],
    )

    _apply_neo_pi_r_chinese_overrides(facets)
    return facets


def _apply_neo_pi_r_chinese_overrides(facets: Dict[str, FacetDefinition]) -> None:
    """Localize NEO-PI-R construct materials used in prompts to Simplified Chinese."""
    overrides = {
        "neuroticism_anxiety": {
            "definition": "容易体验担忧、紧张和不安，并倾向于预期潜在危险或负面后果。",
            "high_trait_behavior": "反复扫描风险、寻求确认、担心事情出错，并回避不确定情境",
            "low_trait_behavior": "面对不确定性时较平静，不轻易预设问题，相信自己能处理意外",
            "common_confounds": ["需要区分真实威胁水平和稳定焦虑倾向", "压力负荷高时可能与压力脆弱性重叠"],
            "forbidden_patterns": ["不要把焦虑描写成愚蠢或不理性", "避免惊恐发作等临床症状"],
            "option_design_rules": ["高焦虑选项应体现担忧但仍能行动", "低焦虑选项应平静但不轻视真实风险"],
            "scoring_logic": "计分为 1 的选项表示较低焦虑：在不确定中仍能适应性行动",
        },
        "neuroticism_angry_hostility": {
            "definition": "容易体验愤怒、烦躁和受冒犯感，并对阻碍或不公反应较强的倾向。",
            "high_trait_behavior": "明显表现出挫败、责怪他人、长时间介意，并在受阻时冲动反应",
            "low_trait_behavior": "受挫时保持克制，更倾向从情境因素解释问题，并较快放下",
            "common_confounds": ["需要区分合理维护权益和愤怒反应", "可能与宜人性的顺从维度重叠"],
            "forbidden_patterns": ["不要把愤怒描写成必然错误", "避免攻击性、暴力或明显不道德反应"],
            "option_design_rules": ["高愤怒选项可体现强烈不满但不越界", "低愤怒选项应体现冷静沟通或重新解释"],
            "scoring_logic": "计分为 1 的选项表示较低愤怒敌意：受挑衅时仍能保持克制",
        },
        "neuroticism_depression": {
            "definition": "容易体验悲伤、沮丧、孤独和消极自我评价的倾向。",
            "high_trait_behavior": "遇到失败后退缩、反复自责、预期负面结果，并感到低能量和低动力",
            "low_trait_behavior": "受挫后仍能维持希望和参与感，较快恢复自信并继续行动",
            "common_confounds": ["需要区分正常哀伤和特质性低落", "可能与低外向性的积极情绪不足重叠"],
            "forbidden_patterns": ["不要描写自杀或自伤意念", "不要把低落选项写成道德软弱"],
            "option_design_rules": ["高抑郁选项可体现退缩但仍在日常功能范围内", "低抑郁选项应体现恢复力但不否认情绪"],
            "scoring_logic": "计分为 1 的选项表示较低抑郁倾向：受挫后仍能保持投入和希望",
        },
        "neuroticism_self_consciousness": {
            "definition": "在社交或被评价情境中容易感到尴尬、羞耻或过度在意他人眼光的倾向。",
            "high_trait_behavior": "监控自我呈现、避免成为焦点、害怕出丑，并在发言前犹豫",
            "low_trait_behavior": "被关注时较自在，能自然表达，也能接受轻微尴尬",
            "common_confounds": ["需要区分自我意识和内向偏好", "可能与社交焦虑重叠"],
            "forbidden_patterns": ["不要简单写成害羞", "不要把低自我意识写成自大或冒失"],
            "option_design_rules": ["高自我意识选项应体现顾虑但仍可参与", "低自我意识选项应体现自在投入而非炫耀"],
            "scoring_logic": "计分为 1 的选项表示较低自我意识：被评价时仍能自然参与",
        },
        "neuroticism_impulsiveness": {
            "definition": "面对诱惑、冲动或即时满足时，较难延迟反应和考虑后果的倾向。",
            "high_trait_behavior": "容易顺从欲望、先行动后思考，难以延迟满足，常做临时决定",
            "low_trait_behavior": "能抵抗诱惑，先考虑后果再行动，并在冲动下保持自控",
            "common_confounds": ["需要区分冲动性和健康的自发性", "可能与低尽责性的计划不足重叠"],
            "forbidden_patterns": ["不要涉及成瘾、药物或违法行为", "不要把冲动选项写成明显荒唐"],
            "option_design_rules": ["高冲动选项应承认欲望并较快行动", "低冲动选项应体现延迟、折中或完全抵抗"],
            "scoring_logic": "计分为 1 的选项表示较低冲动性：面对诱惑仍能保持自控",
        },
        "neuroticism_vulnerability": {
            "definition": "在压力和多重要求下容易感到难以应对、混乱或依赖他人的倾向。",
            "high_trait_behavior": "压力下容易失序、过度求助、感到被压垮，并做出较差判断",
            "low_trait_behavior": "压力下仍能组织任务、处理多重要求，并保持基本应对信心",
            "common_confounds": ["需要区分真实能力不足和压力脆弱性", "不确定情境下可能与焦虑重叠"],
            "forbidden_patterns": ["不要描写完全崩溃", "避免惊恐发作或临床危机描述"],
            "option_design_rules": ["高脆弱性选项可体现不知所措但仍尝试处理", "低脆弱性选项应体现冷静排序和适度求助"],
            "scoring_logic": "计分为 1 的选项表示较低压力脆弱性：压力下仍保持功能",
        },
        "extraversion_warmth": {
            "definition": "愿意建立亲近关系并表达友好、关心和情感连接的倾向。",
            "high_trait_behavior": "表达关心和亲近，维护关系，优先考虑人与人之间的连接",
            "low_trait_behavior": "保持情感距离，较少表达亲近，关系更正式或工具化",
            "common_confounds": ["需要区分温暖和同情助人", "文化规范会影响亲近表达方式"],
            "forbidden_patterns": ["避免浪漫或暧昧化表达", "不要把低温暖写成冷酷"],
            "option_design_rules": ["高温暖选项应自然表达关心", "低温暖选项应保持礼貌距离而非敌意"],
            "scoring_logic": "计分为 1 的选项表示较高温暖性：适度表达关心和连接",
        },
        "extraversion_gregariousness": {
            "definition": "喜欢群体场合、享受与多人在一起并主动参与集体活动的倾向。",
            "high_trait_behavior": "主动寻找群体互动，喜欢热闹场合，并从集体中获得能量",
            "low_trait_behavior": "更偏好独处或小范围互动，长时间群体活动后容易疲惫",
            "common_confounds": ["需要区分低群居性和社交焦虑", "活动兴趣会影响参与意愿"],
            "forbidden_patterns": ["不要把低群居性写成反社会", "避免强迫社交式情境"],
            "option_design_rules": ["高群居性选项应主动加入并享受互动", "低群居性选项应体现偏好安静但不排斥他人"],
            "scoring_logic": "计分为 1 的选项表示较高群居性：主动寻求并享受群体陪伴",
        },
        "extraversion_assertiveness": {
            "definition": "倾向于清楚表达观点、影响他人并在需要时承担主导角色。",
            "high_trait_behavior": "自信表达意见、推动决策、愿意协调或领导他人",
            "low_trait_behavior": "更倾向倾听和等待他人带头，在表达立场时较谨慎",
            "common_confounds": ["需要区分果断和攻击性", "权力距离会影响表达方式"],
            "forbidden_patterns": ["不要把果断写成支配控制", "不要把安静写成软弱"],
            "option_design_rules": ["高果断性选项应直接但尊重", "低果断性选项应谨慎参与而非消极逃避"],
            "scoring_logic": "计分为 1 的选项表示较高果断性：适度表达并推动协作",
        },
        "extraversion_activity": {
            "definition": "日常生活中节奏较快、保持忙碌和行动活力的倾向。",
            "high_trait_behavior": "行动节奏快，喜欢保持忙碌，处理任务时有较强能量感",
            "low_trait_behavior": "偏好较慢节奏和较少安排，需要更多恢复时间",
            "common_confounds": ["需要区分稳定活动水平和短期疲劳", "健康和睡眠会影响活动表现"],
            "forbidden_patterns": ["不要把低活动性写成懒惰", "不要把高活动性写成过劳"],
            "option_design_rules": ["高活动性选项应保持高效活跃但不过度透支", "低活动性选项应体现稳妥节奏"],
            "scoring_logic": "计分为 1 的选项表示较高活动性：保持有活力的行动节奏",
        },
        "extraversion_excitement_seeking": {
            "definition": "喜欢新鲜、刺激和变化体验，并愿意寻找较高唤起活动的倾向。",
            "high_trait_behavior": "主动尝试新鲜刺激的活动，偏好变化和较强感官体验",
            "low_trait_behavior": "更喜欢熟悉、平稳和可预测的活动",
            "common_confounds": ["需要区分寻求刺激和不负责任冒险", "可能与开放性尝新重叠"],
            "forbidden_patterns": ["不要涉及违法、危险或伤害性活动", "不要把低刺激寻求写成无趣"],
            "option_design_rules": ["高刺激寻求选项应在合理边界内尝新", "低刺激寻求选项应选择稳定安全的体验"],
            "scoring_logic": "计分为 1 的选项表示较高刺激寻求：在适当范围内追求新鲜刺激",
        },
        "extraversion_positive_emotions": {
            "definition": "容易体验和表达愉快、兴奋、热情等积极情绪的倾向。",
            "high_trait_behavior": "经常感到开心和兴奋，容易表达赞赏和热情",
            "low_trait_behavior": "积极情绪表达较少或较平稳，不容易外显兴奋",
            "common_confounds": ["需要区分积极情绪和社交礼貌", "低表达不等于抑郁"],
            "forbidden_patterns": ["不要把低积极情绪写成悲观或冷漠", "不要把高积极情绪写成夸张失控"],
            "option_design_rules": ["高积极情绪选项应自然表达喜悦", "低积极情绪选项应平稳回应但不否定价值"],
            "scoring_logic": "计分为 1 的选项表示较高积极情绪：自然体验并表达愉快和热情",
        },
        "openness_fantasy": {
            "definition": "倾向于运用想象、构思可能情境，并让想象丰富日常体验。",
            "high_trait_behavior": "经常进行想象和联想，用虚构或可能性扩展体验",
            "low_trait_behavior": "更关注现实和具体事实，较少沉浸于想象",
            "common_confounds": ["需要区分想象力和逃避现实", "可能与创造性重叠"],
            "forbidden_patterns": ["不要写成幼稚或不成熟", "不要写成脱离现实无法行动"],
            "option_design_rules": ["高幻想选项应建设性运用想象", "低幻想选项应体现现实取向而非贫乏"],
            "scoring_logic": "计分为 1 的选项表示较高幻想性：用想象丰富经验",
        },
        "openness_aesthetics": {
            "definition": "对艺术、美感、音乐和环境审美元素敏感并愿意欣赏的倾向。",
            "high_trait_behavior": "容易注意并欣赏美感，能被艺术、音乐或自然景象打动",
            "low_trait_behavior": "更重视功能和实用，对审美元素关注较少",
            "common_confounds": ["需要区分艺术训练和自然欣赏", "审美偏好受文化经验影响"],
            "forbidden_patterns": ["不要要求专业艺术知识", "不要把低审美敏感性写成没文化"],
            "option_design_rules": ["高审美选项应自然注意并欣赏美", "低审美选项应体现实用优先"],
            "scoring_logic": "计分为 1 的选项表示较高审美性：自然欣赏美和艺术",
        },
        "openness_feelings": {
            "definition": "愿意觉察、接纳并探索自己和他人情绪体验的倾向。",
            "high_trait_behavior": "关注内在感受，愿意思考和表达情绪的细微差异",
            "low_trait_behavior": "较少关注情绪体验，更重视事实、任务或外在结果",
            "common_confounds": ["需要区分情绪开放和情绪波动", "文化规范会影响情绪表达"],
            "forbidden_patterns": ["不要把低情感开放写成冷漠", "不要把高情感开放写成情绪失控"],
            "option_design_rules": ["高情感开放选项应探索和承认感受", "低情感开放选项应偏事实处理但不否定情绪"],
            "scoring_logic": "计分为 1 的选项表示较高情感开放：重视并觉察情绪体验",
        },
        "openness_actions": {
            "definition": "愿意尝试新活动、新方法和不同日常安排的倾向。",
            "high_trait_behavior": "主动尝试新方式，愿意改变惯常做法并探索未知体验",
            "low_trait_behavior": "偏好熟悉流程和稳定习惯，对新做法较谨慎",
            "common_confounds": ["需要区分尝新和寻求刺激", "资源和时间限制会影响尝试意愿"],
            "forbidden_patterns": ["不要把低尝新写成僵化", "避免危险或不道德尝试"],
            "option_design_rules": ["高行动开放选项应愿意试新方式", "低行动开放选项应偏好熟悉可靠方案"],
            "scoring_logic": "计分为 1 的选项表示较高行动开放：愿意尝试新活动和新方式",
        },
        "openness_ideas": {
            "definition": "喜欢抽象思考、复杂概念和智性探索的倾向。",
            "high_trait_behavior": "享受理论讨论，主动探索新观点和复杂问题",
            "low_trait_behavior": "更偏好具体、实用和熟悉的内容",
            "common_confounds": ["需要区分智力能力和观点开放兴趣", "专业背景会影响参与程度"],
            "forbidden_patterns": ["不要要求特定专业知识", "不要把低想法开放写成低智力"],
            "option_design_rules": ["高想法开放选项应投入抽象或概念内容", "低想法开放选项应强调实用和具体"],
            "scoring_logic": "计分为 1 的选项表示较高想法开放：喜欢智性探索和抽象思考",
        },
        "openness_values": {
            "definition": "愿意反思既有价值观、规则和社会惯例，并考虑替代观点的倾向。",
            "high_trait_behavior": "愿意重新审视假设，接纳多元观点并思考规则背后的理由",
            "low_trait_behavior": "更重视传统、稳定规范和既有价值框架",
            "common_confounds": ["需要区分价值开放和道德立场", "政治敏感话题会干扰测量"],
            "forbidden_patterns": ["避免政治争议或敏感议题", "不要把低价值开放写成愚昧保守"],
            "option_design_rules": ["高价值开放选项应愿意审视假设", "低价值开放选项应重视稳定规范但不贬低他人"],
            "scoring_logic": "计分为 1 的选项表示较高价值开放：愿意重新审视价值和规则",
        },
        "agreeableness_trust": {
            "definition": "倾向于相信他人诚实、可信并怀有善意。",
            "high_trait_behavior": "愿意给他人善意解释，在合作中预期对方可靠",
            "low_trait_behavior": "更谨慎怀疑，会先核实动机和可信度",
            "common_confounds": ["过往受骗经历会影响反应", "对方实际可信度需要控制"],
            "forbidden_patterns": ["不要把信任写成天真", "不要把怀疑写成绝对高明"],
            "option_design_rules": ["高信任选项应相信但保留基本判断", "低信任选项应谨慎但不偏执"],
            "scoring_logic": "计分为 1 的选项表示较高信任：倾向善意解释并愿意合作",
        },
        "agreeableness_straightforwardness": {
            "definition": "在人际互动中诚实、直接、真诚表达而较少操控或掩饰的倾向。",
            "high_trait_behavior": "表达真实想法，沟通坦诚，较少策略性隐瞒",
            "low_trait_behavior": "更讲究策略和包装，可能根据利益调整表达",
            "common_confounds": ["需要区分坦诚和不顾场合的冒犯", "文化中委婉表达会影响行为"],
            "forbidden_patterns": ["不要把低坦诚写成邪恶欺骗", "不要把高坦诚写成鲁莽伤人"],
            "option_design_rules": ["高坦诚选项应诚实但顾及方式", "低坦诚选项应体现策略性表达而非恶意欺骗"],
            "scoring_logic": "计分为 1 的选项表示较高坦诚性：真诚、直接且有分寸",
        },
        "agreeableness_altruism": {
            "definition": "愿意主动关心他人需要并提供实际帮助的倾向。",
            "high_trait_behavior": "看到他人需要时主动帮助，愿意投入时间或资源",
            "low_trait_behavior": "更优先考虑自身事务，只有在必要或方便时帮助",
            "common_confounds": ["需要区分助人意愿和助人能力", "资源不足会限制帮助行为"],
            "forbidden_patterns": ["不要把低利他写成自私冷血", "避免过度牺牲自我的英雄式帮助"],
            "option_design_rules": ["高利他选项应主动且适度帮助", "低利他选项应保持边界但不恶意拒绝"],
            "scoring_logic": "计分为 1 的选项表示较高利他性：主动关心并愿意帮助他人",
        },
        "agreeableness_compliance": {
            "definition": "面对冲突时愿意克制对抗、让步或寻找和解方式的倾向。",
            "high_trait_behavior": "避免正面冲突，愿意退让、协商和维护关系",
            "low_trait_behavior": "更坚持自身立场，面对冲突时较直接或强硬",
            "common_confounds": ["需要区分顺从和焦虑性回避", "也可能与果断性低分重叠"],
            "forbidden_patterns": ["不要把低顺从写成攻击性", "不要把高顺从写成没有原则"],
            "option_design_rules": ["高顺从选项应体面让步或缓和冲突", "低顺从选项应坚持立场但不伤害他人"],
            "scoring_logic": "计分为 1 的选项表示较高顺从性：愿意为减少冲突而适度退让",
        },
        "agreeableness_modesty": {
            "definition": "对自己的成绩和优点保持谦逊、不夸耀的倾向。",
            "high_trait_behavior": "淡化自我表现，承认贡献但不强调优越感",
            "low_trait_behavior": "更愿意展示成就，强调自己的能力和贡献",
            "common_confounds": ["需要区分谦逊和低自尊", "自我展示也可能是情境要求"],
            "forbidden_patterns": ["不要把低谦逊写成傲慢", "不要把高谦逊写成自我贬低"],
            "option_design_rules": ["高谦逊选项应自然承认成绩但不过度突出自己", "低谦逊选项应自信展示但不贬低他人"],
            "scoring_logic": "计分为 1 的选项表示较高谦逊性：对成就保持低调和分寸",
        },
        "agreeableness_tender_mindedness": {
            "definition": "容易被他人处境打动，并以同情和人道关怀看待问题的倾向。",
            "high_trait_behavior": "关注他人痛苦和处境，愿意从关怀角度判断问题",
            "low_trait_behavior": "更重视原则、效率或事实判断，较少被情绪打动",
            "common_confounds": ["可能与利他和同情心重叠", "需要区分同情和情绪脆弱"],
            "forbidden_patterns": ["不要把低同情心写成残酷", "不要把高同情心写成没有判断"],
            "option_design_rules": ["高慈悲心选项应体现同情但保留判断", "低慈悲心选项应重视原则但不冷酷"],
            "scoring_logic": "计分为 1 的选项表示较高慈悲心：对他人处境保持同情和关怀",
        },
        "conscientiousness_competence": {
            "definition": "相信自己有能力处理任务、解决问题并有效应对要求的倾向。",
            "high_trait_behavior": "对完成任务有信心，能评估资源并采取有效行动",
            "low_trait_behavior": "更容易怀疑自己的能力，需要更多确认或指导",
            "common_confounds": ["需要区分真实能力和自我效能感", "经验不足会影响信心"],
            "forbidden_patterns": ["不要把低胜任感写成无能", "不要把高胜任感写成自负"],
            "option_design_rules": ["高胜任感选项应自信且现实", "低胜任感选项应谨慎但不完全放弃"],
            "scoring_logic": "计分为 1 的选项表示较高胜任感：相信自己能有效处理任务",
        },
        "conscientiousness_order": {
            "definition": "偏好有序、整洁和结构清晰，并主动维护秩序的倾向。",
            "high_trait_behavior": "保持环境、资料和流程整齐有序，重视分类和结构",
            "low_trait_behavior": "能接受一定混乱，组织方式更随性或灵活",
            "common_confounds": ["需要区分秩序偏好和强迫症状", "时间压力会影响整理程度"],
            "forbidden_patterns": ["不要把低秩序写成邋遢无能", "避免强迫症刻板描述"],
            "option_design_rules": ["高秩序选项应适度维护组织系统", "低秩序选项应体现灵活但仍能运转"],
            "scoring_logic": "计分为 1 的选项表示较高秩序性：维持组织和整洁",
        },
        "conscientiousness_dutifulness": {
            "definition": "重视责任、规则和义务，并倾向于按承诺行事的程度。",
            "high_trait_behavior": "认真履行承诺，遵守规则，对责任有较强内在要求",
            "low_trait_behavior": "对规则和义务更灵活，可能根据情境调整承诺",
            "common_confounds": ["需要区分责任感和盲从", "规则合理性会影响行为"],
            "forbidden_patterns": ["不要把低责任写成不道德", "不要把高责任写成僵化服从"],
            "option_design_rules": ["高责任选项应可靠履行义务", "低责任选项应体现灵活调整但不恶意失信"],
            "scoring_logic": "计分为 1 的选项表示较高责任义务感：理解并履行责任",
        },
        "conscientiousness_achievement_striving": {
            "definition": "追求高标准、努力达成目标并希望表现出色的倾向。",
            "high_trait_behavior": "设定较高目标，持续投入，努力取得优秀结果",
            "low_trait_behavior": "目标较温和，更重视够用、平衡或过程体验",
            "common_confounds": ["需要区分成就追求和外部压力", "能力差异会影响结果"],
            "forbidden_patterns": ["不要把低成就追求写成懒惰", "避免美化过度竞争或牺牲健康"],
            "option_design_rules": ["高成就追求选项应努力提升表现但不过度极端", "低成就追求选项应重视适度目标"],
            "scoring_logic": "计分为 1 的选项表示较高成就追求：有达成和超越目标的动力",
        },
        "conscientiousness_self_discipline": {
            "definition": "即使任务枯燥或困难，也能启动、坚持并完成工作的倾向。",
            "high_trait_behavior": "能克服拖延，坚持推进困难或无聊任务",
            "low_trait_behavior": "容易受分心或厌倦影响，启动和坚持任务较困难",
            "common_confounds": ["需要区分自律和任务兴趣", "压力或疲劳会影响坚持"],
            "forbidden_patterns": ["不要把低自律写成没有纪律或品行差", "避免把高自律写成过度压榨自己"],
            "option_design_rules": ["高自律选项应适当坚持并完成任务", "低自律选项应体现拖延或分心但不荒唐"],
            "scoring_logic": "计分为 1 的选项表示较高自律性：面对困难仍能坚持推进",
        },
        "conscientiousness_deliberation": {
            "definition": "行动前倾向于谨慎思考、权衡后果并避免草率决定的程度。",
            "high_trait_behavior": "做决定前分析风险和后果，愿意花时间思考",
            "low_trait_behavior": "更快做决定，较少反复权衡，行动更直接",
            "common_confounds": ["需要区分审慎和焦虑性犹豫", "过度分析可能影响行动"],
            "forbidden_patterns": ["不要把低审慎写成鲁莽", "不要把高审慎写成无法行动"],
            "option_design_rules": ["高审慎选项应考虑后果后行动", "低审慎选项应较快行动但不冒失"],
            "scoring_logic": "计分为 1 的选项表示较高审慎性：行动前认真权衡后果",
        },
    }

    for key, values in overrides.items():
        facet = facets.get(key)
        if facet is None:
            continue
        for field_name, value in values.items():
            setattr(facet, field_name, value)

    text_replacements = {
        "deadline": "截止任务",
        "PPT": "演示文稿",
        "notification": "消息提醒",
        "offer": "录用通知",
        "pros/cons": "利弊清单",
        "anxiety": "焦虑",
        "self_consciousness": "自我意识",
        "depression": "抑郁倾向",
        "stress_vulnerability": "压力脆弱性",
        "vulnerability": "压力脆弱性",
        "introversion": "内向倾向",
        "extraversion": "外向性",
        "agreeableness_compliance": "宜人性顺从",
        "conscientiousness_order": "尽责性秩序",
        "conscientiousness_self_discipline": "尽责性自律",
        "conscientiousness_deliberation": "尽责性审慎",
        "indecisiveness": "优柔寡断",
        "analysis paralysis": "过度分析导致停滞",
    }

    for facet in facets.values():
        for field_name in ("confounding_contexts", "inappropriate_contexts"):
            values = getattr(facet, field_name)
            cleaned_values = []
            for text in values:
                for source, target in text_replacements.items():
                    text = text.replace(source, target)
                cleaned_values.append(text)
            setattr(facet, field_name, cleaned_values)

    cleanup_replacements = {
        "多个 截止任务": "多个截止任务",
        "激活 ": "激活",
        "可能激活 ": "可能激活",
        "外向性_excitement_seeking": "外向性刺激寻求",
        "外向性_warmth": "外向性温暖",
        "外向性_assertiveness": "外向性果断",
        "neuroticism_impulsiveness": "神经质冲动性",
        "neuroticism_抑郁倾向": "神经质抑郁倾向",
        "neuroticism_压力脆弱性": "神经质压力脆弱性",
        "neuroticism_焦虑": "神经质焦虑",
        "neuroticism": "神经质",
        "openness_aesthetics": "开放性审美",
        "openness_fantasy": "开放性幻想",
        "openness_feelings": "开放性情感",
        "openness_actions": "开放性行动",
        "openness_ideas": "开放性想法",
        "openness_values": "开放性价值",
        "agreeableness_altruism": "宜人性利他",
        "agreeableness_tender_mindedness": "宜人性慈悲心",
        "agreeableness_straightforwardness": "宜人性坦诚",
        "conscientiousness_achievement_striving": "尽责性成就追求",
        "self_discipline": "自律",
        "competence": "胜任感",
        "teamwork": "团队合作",
        "self_interest": "个人利益",
        "materialism": "物质主义",
        "socioeconomic status": "社会经济地位",
        "education level": "教育水平",
        "intelligence": "智力",
        "gap year": "间隔年",
        "political liberalism": "政治自由主义",
        "moral reasoning": "道德推理",
        "rebellion": "叛逆倾向",
        "naivety": "天真轻信",
        "lack of critical thinking": "缺乏批判性思维",
        "authority orientation": "权威取向",
        "honesty": "诚实",
        "social skills": "社交技能",
        "social desirability": "社会赞许性",
        "moral identity": "道德认同",
        "passive": "被动",
        "low assertiveness": "低果断性",
        "cultural norm": "文化规范",
        "low competence": "低胜任感",
        "false modesty": "虚假谦逊",
        "emotional volatility": "情绪波动",
        "actual ability": "实际能力",
        "Dunning-Kruger": "能力错觉",
        "cleanliness": "洁净偏好",
        "time management": "时间管理",
        "OCD tendency": "强迫倾向",
        "dependability": "可靠性",
        "extrinsic motivation": "外部动机",
        "competitiveness": "竞争性",
        "grit": "坚毅",
        "self_control": "自我控制",
        "suppression": "压抑控制",
        "conscientiousness_胜任感": "尽责性胜任感",
        "'assertiveness'": "果断行为",
        "excitement_seeking": "刺激寻求",
        "warmth": "温暖性",
        "optimism": "乐观倾向",
        "low 胜任感": "低胜任感",
    }

    for facet in facets.values():
        for field_name in ("confounding_contexts", "inappropriate_contexts"):
            values = getattr(facet, field_name)
            cleaned_values = []
            for text in values:
                for source, target in cleanup_replacements.items():
                    text = text.replace(source, target)
                cleaned_values.append(text)
            setattr(facet, field_name, cleaned_values)


def get_facet(domain: str, facet_name: str) -> FacetDefinition:
    """Get a specific facet definition."""
    facets = get_neo_pi_r_facets()
    key = f"{domain}_{facet_name}"
    if key not in facets:
        raise ValueError(f"Unknown facet: {domain}.{facet_name}")
    return facets[key]


def get_facets_by_domain(domain: str) -> Dict[str, FacetDefinition]:
    """Get all facets for a domain."""
    all_facets = get_neo_pi_r_facets()
    return {
        k: v for k, v in all_facets.items()
        if v.domain == domain
    }


# =============================================================================
# Self-Report Items (for validation)
# =============================================================================

SELF_REPORT_ITEMS = {
    "neuroticism": [
        "I become tense quickly when several demands arrive at once.",
        "Small setbacks stay on my mind longer than I would like.",
    ],
    "extraversion": [
        "I usually bring energy into group situations.",
        "I am comfortable starting conversations with unfamiliar people.",
    ],
    "openness": [
        "I enjoy trying a new approach even when the old one still works.",
        "Ideas that challenge my routines usually interest me.",
    ],
    "agreeableness": [
        "I try to understand other people's reasons before I judge them.",
        "I usually look for a cooperative solution during friction.",
    ],
    "conscientiousness": [
        "I keep track of details before they create problems.",
        "I follow through on plans even when the work becomes repetitive.",
    ],
}



# =============================================================================
# Inventory access
# =============================================================================

def get_inventory(inventory_name: str = "neo_pi_r") -> Dict[str, FacetDefinition]:
    """Return the NEO-PI-R facet definitions."""
    if inventory_name.lower() != "neo_pi_r":
        raise ValueError(f"Unknown inventory: {inventory_name}. Use neo_pi_r.")
    return get_neo_pi_r_facets()
