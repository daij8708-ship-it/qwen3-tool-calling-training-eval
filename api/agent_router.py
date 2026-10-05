"""五个专业入口的路由提示词与结果校验；此模块不执行工具。"""

from __future__ import annotations

import re


ROUTES = {
    "consult_technical_expert": "处理电脑、软件、系统故障、操作指导和知识库技术问答。",
    "search_realtime_information": "处理今天或最近的天气、新闻、股价、赛事等需要联网核验的实时信息。",
    "query_service_station_and_navigate": "处理维修站或服务中心查询、地点导航和路线规划。",
    "manage_support_ticket": "处理创建或查询工单、补充工单信息、查询处理进度和转人工。",
    "handle_general_conversation": "处理问候、概念解释、学习答疑和文本整理等不需要专业工具的请求。",
}

TOOL_TO_ROUTE = {
    "consult_technical_expert": "technical",
    "search_realtime_information": "realtime",
    "query_service_station_and_navigate": "service",
    "manage_support_ticket": "ticket",
    "handle_general_conversation": "general",
}

MULTI_TASK = re.compile(
    r"然后|同时|接着|顺便|再帮我|再查|另外还|并且|并帮|并创建|没解决就|修不好就|"
    r"如果.{1,20}(解决不了|修不好|不行|没有用).{0,10}(找|联系|转|建)"
)


def build_route_prompt(user_request: str, now: str, prefix: str = "") -> str:
    lines = [
        "你是多智能体系统的第一层分诊器，只选择下一位处理者，不执行工具，不生成答案。",
        "本次可用工具都是分发入口，不是最终业务能力。当前请求只有一个任务时，必须从五个入口选择一个。",
        "即使缺少工单号、地点、待改写原文或权限信息，也要转给相应智能体；追问和权限校验由下游负责。",
        "技术知识库问题转技术专家；工单更新和转人工转工单专家；普通聊天、概念解释、文本处理转通用助手。",
        "只允许输出本次列出的工具名，不能调用其他工具。query 参数逐字使用用户请求。",
        "只输出一个严格 JSON 对象：{\"action\":\"call\",\"tool\":\"工具名\",\"arguments\":{\"query\":\"原始请求\"}}。",
        "本次可用工具:",
    ]
    lines.extend(f"{name}(query:str):{description}" for name, description in ROUTES.items())
    lines.extend((f"服务端当前时间: {now}", f"用户请求: {user_request}"))
    return "\n".join(lines)


def build_route_prompt_v1(user_request: str, now: str) -> str:
    """第二代分诊：单任务选五路，跨 Agent 多任务交由云端编排。"""
    lines = [
        "你是多智能体系统的第一层分诊器。你不回答问题，只输出一个 JSON 对象。",
        "先判断用户是否明确提出需要不同智能体处理的两个或更多任务。",
        "若是跨智能体多任务，输出 {\"action\":\"delegate\",\"reason\":\"multi_task\"}，交给云端编排全部任务。",
        "同一智能体领域内的多个步骤仍属于单路由，例如排查故障再给排查步骤、查两地天气、查维修店再导航过去。",
        "单路由时输出 {\"action\":\"call\",\"tool\":\"工具名\",\"arguments\":{\"query\":\"原始请求\"}}。",
        "缺少工单号、地点或原文也要选对应智能体；信息追问由下游负责。",
        "只能使用以下五个工具名，query 必须逐字保留用户请求；不得生成答案或其他文字。",
        "本次可用工具:",
    ]
    lines.extend(f"{name}(query:str):{description}" for name, description in ROUTES.items())
    lines.extend((f"服务端当前时间: {now}", f"用户请求: {user_request}"))
    return "\n".join(lines)


def validate_route_decision(user_request: str, decision: object) -> tuple[str | None, str]:
    """只采纳白名单中的单一路由，失败时交还原有云端调度。"""
    if MULTI_TASK.search(user_request):
        return None, "multi_task"
    if not isinstance(decision, dict) or decision.get("action") != "call":
        return None, "non_call"
    tool = decision.get("tool")
    if tool not in TOOL_TO_ROUTE:
        return None, "unknown_tool"
    arguments = decision.get("arguments")
    if not isinstance(arguments, dict) or set(arguments) != {"query"}:
        return None, "invalid_arguments"
    if not isinstance(arguments["query"], str) or not arguments["query"].strip():
        return None, "empty_query"
    return TOOL_TO_ROUTE[tool], "model"


def validate_route_decision_v1(decision: object) -> tuple[str | None, str]:
    """多任务由模型显式交还云端；单路由仍要通过严格白名单。"""
    if not isinstance(decision, dict):
        return None, "invalid_json"
    if decision.get("action") == "delegate":
        if decision.get("reason") == "multi_task":
            return None, "model_multi_task"
        return None, "invalid_delegate"
    if decision.get("action") != "call":
        return None, "non_call"
    tool = decision.get("tool")
    if tool not in TOOL_TO_ROUTE:
        return None, "unknown_tool"
    arguments = decision.get("arguments")
    if not isinstance(arguments, dict) or set(arguments) != {"query"}:
        return None, "invalid_arguments"
    if not isinstance(arguments["query"], str) or not arguments["query"].strip():
        return None, "empty_query"
    return TOOL_TO_ROUTE[tool], "model"
