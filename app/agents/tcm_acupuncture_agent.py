"""
================================================================================
中医针灸 32B 领域微调大模型专家节点 (TCM Acupuncture Specialist Sub-Agent)
================================================================================
架构设计说明（对应简历与面试手册核心亮点）：
为什么不把 32B 针灸微调大模型单纯封装成一个阻塞式 MCP Tool？
1. 【避免双重生成延迟 (Double Generation Penalty)】：若每次问诊都先经总控 DeepSeek 调工具、
   等 32B 模型生成完 600 字再返回给总控 DeepSeek 总结转述，首字延迟 (TTFT) 会翻倍，
   原本 vLLM + SSE 的流式低延迟优势完全丢失。
2. 【保留领域微调分布】：经 53.3 万条单轮 + 13.9 万条多轮数据微调出的辨证思维链与追问节奏
   (专家评测 0.9215 / 0.8110)，必须由该专家节点直接向用户输出才能保持原汁原味。

因此本模块提供【双模接口】：
- 模式 A (`stream_tcm_specialist_handoff`)：Sub-Agent Handoff 直连流式模式，接管多轮中医问诊；
- 模式 B (`consult_tcm_for_brain_diagnosis`)：中西医跨模态协同会诊模式，结合脑病 MCP 定量预测结果给出针灸方案。
"""
import re
from typing import AsyncGenerator, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from openai import AsyncOpenAI

from app.config import settings

# 连接 vLLM 部署的 32B 中医针灸微调大模型 (OpenAI 兼容接口)
tcm_vllm_client = AsyncOpenAI(
    api_key=settings.TCM_VLLM_API_KEY,
    base_url=settings.TCM_VLLM_BASE_URL,
)

# ============================================================================
# 1. 否定症状识别与结构化临床槽位更新 (对应论文：否定症状识别 + 结构化状态槽位)
# ============================================================================
NEGATIVE_PREFIXES = ("无", "没有", "未见", "不", "否认", "并无")
COMMON_TCM_SYMPTOMS = (
    "恶寒", "发热", "畏寒", "汗出", "盗汗", "自汗", "失眠", "多梦", "头晕",
    "耳鸣", "腰膝酸软", "五心烦热", "口干", "口苦", "舌红", "少苔", "苔黄腻",
    "记忆力减退", "健忘", "神疲乏力", "肢体麻木", "头痛",
)


def update_tcm_clinical_slots(user_text: str, existing_slots: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    从患者输入中提取【阳性症状】与【阴性/否定症状】，存入不随历史滑窗截断的结构化槽位。
    例如："我最近失眠多梦、头晕耳鸣，但没有发热恶寒"
    -> positive_symptoms: ["失眠", "多梦", "头晕", "耳鸣"]
    -> negative_symptoms: ["无发热", "无恶寒"]
    """
    pos_set = set(existing_slots.get("positive_symptoms", []))
    neg_set = set(existing_slots.get("negative_symptoms", []))

    clauses = re.split(r"[，,。；;！!\s]+", user_text)
    for clause in clauses:
        if not clause:
            continue
        has_neg = any(neg in clause for neg in NEGATIVE_PREFIXES)
        for sym in COMMON_TCM_SYMPTOMS:
            if sym in clause:
                if has_neg:
                    neg_set.add(f"无{sym}")
                    pos_set.discard(sym)
                else:
                    pos_set.add(sym)
                    neg_set.discard(f"无{sym}")

    return {
        "positive_symptoms": sorted(pos_set),
        "negative_symptoms": sorted(neg_set),
    }


# ============================================================================
# 2. 中医针灸知识库 RAG 检索与施针禁忌校验
# ============================================================================
TCM_ACUPUNCTURE_KB = [
    {
        "keywords": ["记忆力减退", "健忘", "阿尔茨海默", "AD", "认知", "髓海不足"],
        "syndrome": "髓海不足 / 脾肾两虚证",
        "acupoints": "主穴：百会、四神聪、神庭、本神、风池；配穴：太溪、足三里、神门、悬钟（绝骨）。",
        "method": "醒脑开窍、填精益髓。百会、四神聪平刺 0.5-0.8 寸，施以小幅度捻转补法；太溪、足三里直刺行补法。",
        "contraindication": "风池穴严禁向对侧眼眶方向深刺，以防伤及延髓；高龄体弱患者宜轻刺激，留针 20-30 分钟。",
        "source": "《针灸学（十四五规划教材）》· 痴呆/健忘篇 & 醒脑开窍针刺临床指南",
    },
    {
        "keywords": ["失眠", "多梦", "头晕", "耳鸣", "腰膝酸软", "五心烦热", "舌红", "少苔"],
        "syndrome": "肝肾阴虚 / 心肾不交证",
        "acupoints": "主穴：照海、申脉、神门、三阴交、安眠、百会；配穴：太溪、太冲、心俞、肾俞。",
        "method": "滋阴降火、交通心肾。神门、太溪、三阴交行捻转补法，照海补、申脉泻。",
        "contraindication": "孕妇禁针三阴交；背部心俞穴宜斜刺 0.5-0.8 寸，不可深刺以防气胸。",
        "source": "《中医内科学》与《针灸甲乙经》· 不寐辨证选穴",
    },
]


def retrieve_tcm_rag_evidence(query_text: str, clinical_slots: dict) -> str:
    """根据当前症状槽位与问题检索中医针灸配穴依据及禁忌规则"""
    matched_docs = []
    search_corpus = query_text + " " + " ".join(clinical_slots.get("positive_symptoms", []))

    for item in TCM_ACUPUNCTURE_KB:
        if any(kw in search_corpus for kw in item["keywords"]):
            matched_docs.append(
                f"【参考证型】{item['syndrome']}\n"
                f"【推荐配穴】{item['acupoints']}\n"
                f"【刺灸法】{item['method']}\n"
                f"【安全禁忌】{item['contraindication']}\n"
                f"【文献来源】{item['source']}"
            )

    if not matched_docs:
        return "【RAG 检索提示】当前四诊信息尚不完整，请先围绕寒热、汗出、睡眠、二便及舌脉进行辨证追问，暂勿盲目开具完整处方。"
    return "\n\n".join(matched_docs)


# ============================================================================
# 3. 模式 A：Sub-Agent Handoff 直连流式问诊 (绕过总控二次总结，直接流式输出)
# ============================================================================
TCM_SPECIALIST_SYSTEM_PROMPT = """您是 DAO 实验室基于 DeepSeek-R1 微调的【中医针灸临床专家大模型】。
请严格遵循中医“望、闻、问、切”四诊合参原则与八纲脏腑辨证逻辑进行多轮问诊或开具针灸方案：

【当前结构化患者槽位（严禁忽略否定症状）】
- 已确认阳性症状：{positive_symptoms}
- 已确认否定症状（排除项）：{negative_symptoms}

【RAG 检索到的经络穴位与禁忌证据】
{rag_evidence}

【回复规范】
1. 若患者症状信息不足以锁定证型，请先给出初步病机推测，并提出 2-3 个精准的辨证追问（如询问舌象、脉象、寒热喜恶）；
2. 若辨证信息已充足，请按【中医辨证分析】、【治则治法】、【针灸处方（主穴+配穴）】、【针刺操作与禁忌事项】结构化输出；
3. 结尾固定注明：“以上针灸方案仅供科研与临床参考，具体施针请由执业中医师操作。”
"""


async def stream_tcm_specialist_handoff(
    windowed_messages: List[BaseMessage],
    clinical_slots: dict,
    latest_user_query: str,
) -> AsyncGenerator[dict, None]:
    """
    当总控 DeepSeek 执行 Sub-Agent Handoff 将控制权移交给中医针灸专家节点时调用。
    直接对接 vLLM 进行流式生成，确保低首字延迟 (TTFT) 与微调模型的原生问诊风格。
    """
    rag_evidence = retrieve_tcm_rag_evidence(latest_user_query, clinical_slots)
    sys_prompt = TCM_SPECIALIST_SYSTEM_PROMPT.format(
        positive_symptoms=clinical_slots.get("positive_symptoms") or ["待问诊采集"],
        negative_symptoms=clinical_slots.get("negative_symptoms") or ["暂无"],
        rag_evidence=rag_evidence,
    )

    openai_msgs = [{"role": "system", "content": sys_prompt}]
    for m in windowed_messages:
        role = "user" if isinstance(m, HumanMessage) else "assistant"
        openai_msgs.append({"role": role, "content": str(m.content)})

    stream = await tcm_vllm_client.chat.completions.create(
        model=settings.TCM_MODEL_NAME,
        messages=openai_msgs,
        temperature=0.2,
        max_tokens=1200,
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        # 若 vLLM 返回 DeepSeek-R1 的 reasoning_content，则作为思考链事件推送
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            yield {"event": "reasoning", "data": {"source": "tcm_32b_subagent", "delta": reasoning}}
        if delta.content:
            yield {"event": "token", "data": {"source": "tcm_32b_subagent", "delta": delta.content}}


# ============================================================================
# 4. 模式 B：跨模态中西医协同会诊 (接收 api_new_mcp 脑病预测结果并给出针灸康养方案)
# ============================================================================
async def consult_tcm_for_brain_diagnosis(
    brain_prediction: dict,
    user_query: str,
    clinical_slots: dict,
) -> str:
    """
    当用户上传了脑部 MRI (.nii.gz / .npy) 并要求“结合脑病预测结果给出中医针灸方案”时，
    由总控编排器在调用完 `api_new_mcp` 后调用本专家会诊接口。
    """
    label_cn = brain_prediction.get("label_cn", "未知")
    confidence = brain_prediction.get("confidence", 0.0)
    probs = brain_prediction.get("probabilities", {})

    rag_evidence = retrieve_tcm_rag_evidence("记忆力减退 阿尔茨海默 认知 " + user_query, clinical_slots)

    consult_prompt = (
        f"【西医脑部 MRI 影像 AI 筛查定量结论】\n"
        f"- 预测类别：{label_cn}\n"
        f"- 模型置信度：{confidence:.2%}\n"
        f"- 概率分布：认知正常(CN)={probs.get('0', 0):.2%}, 阿尔茨海默病风险(AD)={probs.get('1', 0):.2%}\n\n"
        f"【患者补充主诉】{user_query}\n\n"
        f"请结合上述西医脑影像定量筛查结果与以下针灸文献证据，出具【中西医结合脑健康与针灸康养建议】"
        f"（包含影像结果解读、醒脑开窍/填精益髓针灸选穴方案、穴位按摩与日常照护建议）：\n{rag_evidence}"
    )

    resp = await tcm_vllm_client.chat.completions.create(
        model=settings.TCM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "您是精通脑病影像解读与中医醒脑开窍针灸学的主任医师。"},
            {"role": "user", "content": consult_prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
    )
    return resp.choices[0].message.content or ""
