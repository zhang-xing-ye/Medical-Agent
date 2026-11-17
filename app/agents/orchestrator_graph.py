"""
================================================================================
总控 DeepSeek + LangGraph 多智能体编排器 (核心文件：演示 Sub-Agent Handoff 与 MCP 调度)
================================================================================
本文件展示了如何将整個系统组织为一个真正的 Multi-Agent + MCP 架构：

1. 【入口粘性路由守卫 (`entry_sticky_router`)】：
   - 如果当前会话处于中医多轮问诊状态 (`sticky_agent == "tcm_specialist"`)，且用户本轮没有上传新的
     脑影像文件 (`.nii.gz`/`.npy`)，也没有要求做西医影像检测，则**直接进入中医针灸 32B Sub-Agent**，
     零中转延迟！
   - 一旦检测到用户上传了新的脑影像资产，立即触发**粘性中断 (Interrupt)**，将控制权交回总控 DeepSeek。

2. 【总控 DeepSeek 节点 (`supervisor_deepseek_node`)】：
   - 分析用户意图与当前挂载的 `asset_id` 凭据，通过 LangGraph 的 `Command(goto=...)` 原语做出三种路由决策：
     a) **Handoff 移交控制权 (`goto="tcm_specialist_node"`)**：纯中医针灸辨证问诊，设置 `sticky_agent="tcm_specialist"`；
     b) **调用脑病预测 MCP 服务 (`goto="brain_mcp_tool_node"`)**：用户上传了 `.nii.gz` 或 `.npy` 脑影像；
     c) **直接回答 (`goto="__end__"`)**：平台使用咨询或通用医学常识问答。

3. 【脑病预测 MCP 工具节点 (`brain_mcp_tool_node`) + 中西医协同会诊】：
   - 通过 MCP 协议调用 `api_new_mcp` 的 `predict_brain_ad_from_nifti` / `predict_brain_ad_from_npy`（仅传 `asset_id`）；
   - 若预测出 AD 风险或用户要求中医调理建议，自动串联调用中医针灸 32B 模型进行**中西医联合会诊**！
"""
import json
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.agents.tcm_acupuncture_agent import (
    consult_tcm_for_brain_diagnosis,
    update_tcm_clinical_slots,
)
from app.asset_hub.asset_service import get_asset_metadata
from app.cache.task_progress_bus import publish_task_progress
from app.config import settings
from app.mcp_client.brain_mcp_client import brain_mcp_client


# ============================================================================
# 1. 定义 LangGraph 全局状态 Schema (含 Handoff 粘性状态与结构化槽位)
# ============================================================================
class AgentPlatformState(TypedDict):
    # 对话消息历史
    messages: List[BaseMessage]
    # 当前最新用户输入
    user_query: str
    # 本轮新上传的多模态文件凭据列表 (例如 ["ast_901"])
    new_asset_ids: List[str]
    # 会话累计挂载的所有文件资产元数据 (Claim-Check 凭据池)
    active_assets: List[Dict[str, Any]]
    # 【核心】Sub-Agent Handoff 粘性标记：若为 "tcm_specialist"，后续多轮问诊直达针灸大模型
    sticky_agent: Optional[Literal["tcm_specialist"]]
    # 【核心】不随滑动窗口淘汰的结构化中医临床槽位 (阳性症状 + 否定症状)
    clinical_slots: Dict[str, List[str]]
    # 是否需要在脑影像 MCP 推理完成后追加中医针灸专家协同会诊
    require_tcm_joint_consult: bool
    # 脑病 MCP 工具返回的定量结果与前端画布 _artifact
    latest_brain_mcp_output: Optional[Dict[str, Any]]
    # 搜集本轮待推送给前端 SSE 的事件列表
    sse_events: List[Dict[str, Any]]


# ============================================================================
# 2. 动态 Token 预算滑动窗口 (对应简历：Token 预算滑窗 + 结构化槽位双轨制)
# ============================================================================
def estimate_tokens(text: str) -> int:
    """中英文混合轻量 Token 估算（约 1 个中文字 ≈ 0.6~1 Token，此处按保守上界估算）"""
    return max(1, int(len(text) * 0.85))


def apply_token_budget_sliding_window(
    messages: List[BaseMessage],
    token_budget: int = settings.MAX_HISTORY_TOKEN_BUDGET,
) -> List[BaseMessage]:
    """
    在扣除 System Prompt、结构化槽位与 RAG 预算后，从最新一轮向前贪心截取历史消息：
    $$ \\sum_{m_i \\in W} \\operatorname{tokens}(m_i) \\leq B_{\\text{history}} $$
    """
    selected: List[BaseMessage] = []
    used_tokens = 0
    for msg in reversed(messages):
        msg_tokens = estimate_tokens(str(msg.content))
        if used_tokens + msg_tokens > token_budget and selected:
            break
        selected.append(msg)
        used_tokens += msg_tokens
    return list(reversed(selected))


# ============================================================================
# 3. 节点一：上下文预处理与 Sticky Handoff 入口守卫节点
# ============================================================================
async def context_and_sticky_guard_node(
    state: AgentPlatformState,
) -> Command[Literal["tcm_specialist_node", "supervisor_deepseek_node"]]:
    """
    负责：
    1. 将新上传的 asset_id 解析为轻量元数据放入 active_assets；
    2. 提取并更新患者阳性/否定症状槽位；
    3. 判断是否满足 Sticky Handoff 直达中医针灸 Sub-Agent 的条件。
    """
    active_assets = list(state.get("active_assets") or [])
    existing_ids = {a["asset_id"] for a in active_assets}
    for aid in state.get("new_asset_ids") or []:
        if aid not in existing_ids:
            active_assets.append(await get_asset_metadata(aid))

    clinical_slots = update_tcm_clinical_slots(
        state["user_query"],
        state.get("clinical_slots") or {"positive_symptoms": [], "negative_symptoms": []},
    )
    windowed_msgs = apply_token_budget_sliding_window(state.get("messages") or [])

    # 检查是否存在强制打断中医粘性路由的信号（如新上传了脑影像，或明确提及脑部 MRI / 蛋白质预测）
    has_new_imaging = len(state.get("new_asset_ids") or []) > 0
    mentions_imaging = any(
        kw in state["user_query"].lower()
        for kw in (".nii", ".npy", "脑部", "核磁", "mri", "影像", "分割", "分类")
    )

    if state.get("sticky_agent") == "tcm_specialist" and not has_new_imaging and not mentions_imaging:
        # 【Sticky Routing 生效】：处于多轮中医问诊中，无需再经过总控 DeepSeek，直达针灸 32B 模型！
        return Command(
            update={
                "active_assets": active_assets,
                "clinical_slots": clinical_slots,
                "messages": windowed_msgs,
                "sse_events": [
                    {
                        "event": "router_trace",
                        "data": {
                            "mode": "sticky_handoff",
                            "target": "tcm_specialist_node",
                            "message": "保持中医针灸专家会话粘性，直接进入多轮辨证问诊",
                        },
                    }
                ],
            },
            goto="tcm_specialist_node",
        )

    # 否则进入总控 DeepSeek 进行全局意图路由与工具规划
    return Command(
        update={
            "active_assets": active_assets,
            "clinical_slots": clinical_slots,
            "messages": windowed_msgs,
            "sticky_agent": None,  # 清除旧粘性状态
        },
        goto="supervisor_deepseek_node",
    )


# ============================================================================
# 4. 节点二：总控 DeepSeek 编排与 Handoff 决策节点 (Supervisor)
# ============================================================================
SUPERVISOR_ROUTING_PROMPT = """您是 DAO 多模态医学智能体平台的【总控认知路由器 (DeepSeek Supervisor)】。
请根据用户的当前指令以及会话中已挂载的资产凭据 (active_assets)，输出严格的 JSON 路由决策：

【当前已挂载的文件资产凭据】
{active_assets_json}

【可选的目标动作 (action)】
1. "handoff_to_tcm": 用户在进行中医症状咨询、把脉舌象描述、针灸选穴配伍问答（未要求对脑部影像文件做推理）。系统将执行 Sub-Agent Handoff 将控制权移交给 32B 中医针灸大模型。
2. "call_brain_mcp": 用户上传了脑部影像（.nii / .nii.gz / .npy）或要求对已挂载的脑影像进行阿尔茨海默病 (AD) 风险预测。
   - 若用户同时希望结合中医针灸给出康养/配穴方案（或默认提供中西医结合建议），请将 "require_tcm_joint_consult" 设为 true。
3. "direct_reply": 用户仅询问平台功能介绍或无需调用专业模型的普通问题。

请仅输出如下格式的 JSON（不要包含 Markdown 代码块标记）：
{{
  "action": "handoff_to_tcm" | "call_brain_mcp" | "direct_reply",
  "target_asset_id": "若 action 为 call_brain_mcp 则填入对应的 asset_id，否则为 null",
  "require_tcm_joint_consult": true | false,
  "direct_answer": "若 action 为 direct_reply 则填入回复内容，否则为 null",
  "reason": "简要说明路由决策原因"
}}
"""


def _init_supervisor_llm() -> ChatDeepSeek:
    return ChatDeepSeek(
        api_key=settings.DEEPSEEK_API_KEY,
        model=settings.SUPERVISOR_MODEL,
        temperature=0.0,
        max_tokens=512,
    )


async def supervisor_deepseek_node(
    state: AgentPlatformState,
) -> Command[Literal["tcm_specialist_node", "brain_mcp_tool_node", "__end__"]]:
    """
    总控 DeepSeek 节点：决定是执行 Sub-Agent Handoff 移交给中医针灸大模型，
    还是调度 `api_new_mcp` 脑病预测 MCP 工具。
    """
    user_q = state["user_query"]
    assets = state.get("active_assets") or []
    events = list(state.get("sse_events") or [])

    llm = _init_supervisor_llm()
    prompt_str = SUPERVISOR_ROUTING_PROMPT.format(
        active_assets_json=json.dumps(assets, ensure_ascii=False)
    )
    resp = await llm.ainvoke(
        [
            SystemMessage(content=prompt_str),
            HumanMessage(content=user_q),
        ]
    )

    raw_text = str(resp.content).strip().strip("`").removeprefix("json").strip()
    try:
        decision = json.loads(raw_text)
    except json.JSONDecodeError:
        # 兜底启发式规则：有影像优先调脑病 MCP，否则移交中医专家
        decision = {
            "action": "call_brain_mcp" if assets else "handoff_to_tcm",
            "target_asset_id": assets[-1]["asset_id"] if assets else None,
            "require_tcm_joint_consult": True,
            "reason": "启发式回退路由",
        }

    action = decision.get("action")

    # ------------------------------------------------------------------------
    # 分支 1：执行 Sub-Agent Handoff -> 移交控制权给中医针灸 32B 专家节点
    # ------------------------------------------------------------------------
    if action == "handoff_to_tcm":
        events.append(
            {
                "event": "handoff",
                "data": {
                    "from": "supervisor_deepseek",
                    "to": "tcm_specialist_node",
                    "sticky": True,
                    "reason": decision.get("reason", "识别为中医针灸辨证问诊需求，移交控制权"),
                },
            }
        )
        return Command(
            update={
                "sticky_agent": "tcm_specialist",
                "sse_events": events,
            },
            goto="tcm_specialist_node",
        )

    # ------------------------------------------------------------------------
    # 分支 2：调度脑病预测 FastMCP 工具 (`api_new_mcp`)
    # ------------------------------------------------------------------------
    if action == "call_brain_mcp":
        events.append(
            {
                "event": "tool_start",
                "data": {
                    "mcp_server": "Brain-AD-Diagnosis-MCP",
                    "asset_id": decision.get("target_asset_id") or (assets[-1]["asset_id"] if assets else ""),
                    "reason": decision.get("reason", "调用脑部 MRI 阿尔茨海默病预测 MCP 工具"),
                },
            }
        )
        return Command(
            update={
                "require_tcm_joint_consult": bool(decision.get("require_tcm_joint_consult", True)),
                "sse_events": events,
            },
            goto="brain_mcp_tool_node",
        )

    # ------------------------------------------------------------------------
    # 分支 3：普通平台问答直接结束
    # ------------------------------------------------------------------------
    reply_text = decision.get("direct_answer") or "您好！我是 DAO 多模态医学智能体，支持脑部 MRI 影像筛查与中医针灸辨证问诊。"
    events.append({"event": "token", "data": {"source": "supervisor_deepseek", "delta": reply_text}})
    return Command(
        update={
            "messages": [*state["messages"], AIMessage(content=reply_text)],
            "sse_events": events,
        },
        goto=END,
    )


# ============================================================================
# 5. 节点三：脑病预测 MCP 工具执行与中西医协同会诊节点
# ============================================================================
async def brain_mcp_tool_node(
    state: AgentPlatformState,
) -> Command[Literal["__end__"]]:
    """
    1. 从 active_assets 中取出最新的脑影像凭据 `asset_id`；
    2. 根据文件类型选择调用 `api_new_mcp` 的 `predict_brain_ad_from_nifti` 或 `predict_brain_ad_from_npy`；
    3. 提取返回的 `_artifact` 推送给前端画布；
    4. 若 `require_tcm_joint_consult=True`，自动将脑病定量预测结果传给中医针灸 32B 模型生成中西医联合建议！
    """
    assets = state.get("active_assets") or []
    events = list(state.get("sse_events") or [])

    if not assets:
        err_msg = "请先上传脑部 MRI 影像文件（支持 `.nii`、`.nii.gz` 或 `.npy` 格式），我再为您调用脑病预测 MCP 模型。"
        events.append({"event": "token", "data": {"source": "supervisor_deepseek", "delta": err_msg}})
        return Command(
            update={"messages": [*state["messages"], AIMessage(content=err_msg)], "sse_events": events},
            goto=END,
        )

    target_asset = assets[-1]
    asset_id = target_asset["asset_id"]
    modality = target_asset.get("modality", "")

    tool_name = (
        "predict_brain_ad_from_nifti"
        if modality == "nifti_brain_mri"
        else "predict_brain_ad_from_npy"
    )
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    await publish_task_progress(task_id, "running", 15, f"开始调用 {tool_name} ({asset_id})")

    # 通过标准 MCP 协议调用 api_new_mcp 服务（仅传递轻量 asset_id 凭据！）
    mcp_res = await brain_mcp_client.call_brain_tool(
        tool_name=tool_name,
        arguments={"asset_id_or_path": asset_id},
    )

    if not mcp_res.get("success"):
        await publish_task_progress(task_id, "failed", 100, mcp_res.get("error", "未知错误"))
        err_text = f"脑病预测 MCP 工具执行失败：{mcp_res.get('error', '未知错误')}"
        events.append({"event": "token", "data": {"source": "brain_mcp", "delta": err_text}})
        return Command(
            update={"messages": [*state["messages"], AIMessage(content=err_text)], "sse_events": events},
            goto=END,
        )

    await publish_task_progress(task_id, "succeeded", 100, "脑病预测 MCP 推理完成", result_asset_id=asset_id)

    # 若 MCP 工具返回了 _artifact，立即追加画布渲染事件
    if "_artifact" in mcp_res:
        events.append({"event": "artifact", "data": mcp_res["_artifact"]})

    prediction = mcp_res["prediction"]

    # 协同调用中医针灸 32B 专家模型出具【中西医结合脑健康与针灸康养报告】
    if state.get("require_tcm_joint_consult", True):
        events.append(
            {
                "event": "consultation_start",
                "data": {
                    "specialist": "TCM-Acupuncture-32B",
                    "message": "正在结合脑部 MRI 定量预测结果进行中医针灸专家协同会诊...",
                },
            }
        )
        joint_advice = await consult_tcm_for_brain_diagnosis(
            brain_prediction=prediction,
            user_query=state["user_query"],
            clinical_slots=state.get("clinical_slots") or {},
        )
    else:
        joint_advice = (
            f"脑部影像 AI 筛查完成：预测类别为 **{prediction['label_cn']}** "
            f"(置信度 {prediction['confidence']:.2%})。具体诊疗请遵医嘱。"
        )

    events.append({"event": "token", "data": {"source": "joint_synthesis", "delta": joint_advice}})
    return Command(
        update={
            "latest_brain_mcp_output": mcp_res,
            "messages": [*state["messages"], AIMessage(content=joint_advice)],
            "sse_events": events,
        },
        goto=END,
    )


# ============================================================================
# 6. 节点四：中医针灸 32B Sub-Agent 专家接管节点 (Handoff Target)
# ============================================================================
async def tcm_specialist_node(
    state: AgentPlatformState,
) -> Command[Literal["__end__"]]:
    """
    当控制权通过 Handoff 移交至本节点时，标记当前回合由 `tcm_32b_subagent` 直接向前端 SSE 流式输出。
    （实际流式 Token 生成在网关 SSE 生成器中调用 `stream_tcm_specialist_handoff` 实时吐出，避免内存缓冲阻塞）
    """
    events = list(state.get("sse_events") or [])
    events.append(
        {
            "event": "subagent_stream_ready",
            "data": {
                "subagent": "tcm_acupuncture_32b",
                "sticky_agent": "tcm_specialist",
                "clinical_slots": state.get("clinical_slots"),
            },
        }
    )
    return Command(
        update={"sse_events": events},
        goto=END,
    )


# ============================================================================
# 7. 编译 LangGraph 状态图
# ============================================================================
def build_medical_agent_graph():
    builder = StateGraph(AgentPlatformState)

    builder.add_node("context_and_sticky_guard_node", context_and_sticky_guard_node)
    builder.add_node("supervisor_deepseek_node", supervisor_deepseek_node)
    builder.add_node("brain_mcp_tool_node", brain_mcp_tool_node)
    builder.add_node("tcm_specialist_node", tcm_specialist_node)

    builder.add_edge(START, "context_and_sticky_guard_node")
    return builder.compile()


medical_agent_graph = build_medical_agent_graph()
