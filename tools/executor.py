#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型决策到工具执行之间的服务端闸门。

输入三样东西:模型决策、服务端可信上下文(本次可用工具 + 服务端确认状态 + 当前时间)、模拟数据。
依次判定:决策格式 -> 工具是否可用 -> 参数 Schema -> 跨字段约束 -> 执行。
任何一步失败都不执行工具、不写数据;写操作还要额外过服务端确认这一关。

契约校验不在这里重写,直接复用第 1 步的 eval/validate_contracts.py:
Schema 子集校验与跨字段语义规则和手工案例用的是同一份实现,避免两套判据漂移。

    from executor import Trusted, execute_decision
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "eval", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))  # 复用第 1 步的校验实现,不重复造一份

import simulator  # noqa: E402  同目录下的本地模拟执行
import validate_contracts as contracts  # noqa: E402

RESULT_CONTRACT_FILE = ROOT / "tools" / "result_contract.json"
LOC = "decision"


class ServerConfigError(RuntimeError):
    """服务端自身配置或契约不合法,与模型输出无关。"""


def load_contracts():
    """加载六个工具契约与决策契约;契约不合格就不接受任何执行请求。"""
    schemas = {}
    for name in contracts.EXPECTED_TOOLS:
        errors, doc = contracts.check_one_tool(name)
        if errors:
            raise ServerConfigError(f"工具契约 {name} 未通过第 1 步校验: {errors}")
        schemas[name] = doc["input_schema"]
    decision_errors, decision_schema = contracts.check_decision_contract()
    if decision_errors:
        raise ServerConfigError(f"决策契约未通过第 1 步校验: {decision_errors}")
    return schemas, decision_schema


class Trusted:
    """服务端可信上下文。模型输出里没有任何字段能改写它。"""

    def __init__(self, now, user_confirmed_write, available_tools):
        parsed = contracts.parse_dt(now)
        if parsed is None:
            raise ServerConfigError(f"服务端当前时间 {now} 不符合 {contracts.TIME_FORMAT}")
        if not isinstance(user_confirmed_write, bool):
            raise ServerConfigError("user_confirmed_write 必须是服务端给出的布尔值")
        if not isinstance(available_tools, (list, tuple)) or not available_tools:
            raise ServerConfigError("available_tools 必须是非空列表")
        if len(set(available_tools)) != len(available_tools):
            raise ServerConfigError("available_tools 含重复项")
        self.now = parsed
        self.now_text = now
        self.user_confirmed_write = user_confirmed_write
        self.available_tools = list(available_tools)
        self.confirmation_source = "context.user_confirmed_write"


class Executor:
    def __init__(self, schemas=None, decision_schema=None, result_contract=None):
        if schemas is None or decision_schema is None:
            schemas, decision_schema = load_contracts()
        self.schemas = schemas
        self.decision_schema = decision_schema
        self.result_contract = result_contract or json.loads(RESULT_CONTRACT_FILE.read_text(encoding="utf-8"))
        unknown = sorted(set(self.result_contract["tools"]) - set(schemas))
        if unknown:
            raise ServerConfigError(f"执行结果契约里有未知工具: {unknown}")

    def run(self, decision, trusted, store=None):
        store = store if store is not None else simulator.Store.load()
        before = list(store.calendar_ids())

        def report(stage, gate_code, reasons, executed=False, action=None, tool=None, result=None):
            after = list(store.calendar_ids())
            return {
                "stage": stage,
                "allowed": stage == "executed",
                "executed": executed,
                "gate_code": gate_code,
                "reasons": list(reasons),
                "action": action,
                "tool": tool,
                "result": result,
                "assistant_text": None if not isinstance(decision, dict) else (decision.get("question") or decision.get("reason")),
                "confirmation": {
                    "source": trusted.confirmation_source,
                    "value": trusted.user_confirmed_write,
                    "model_output_consulted": False,
                    "user_text_consulted": False,
                },
                "calendar_event_ids_before": before,
                "calendar_event_ids_after": after,
                "calendar_changed": before != after,
            }

        if not isinstance(decision, dict):
            return report("decision_format", "decision_invalid", ["决策必须是 JSON 对象"])

        errors = []
        contracts.validate(decision, self.decision_schema, LOC, self.schemas, errors)
        action = decision.get("action")
        tool = decision.get("tool")
        if errors:
            return report(
                "decision_format",
                "decision_invalid",
                [f"{code} @ {location}: {message}" for location, code, message in errors],
                action=action,
                tool=tool,
            )
        if action != "call":
            return report("not_executed", "action_not_call", [f"动作 {action} 不触发工具执行"], action=action)

        if tool not in self.schemas:
            return report("tool_availability", "tool_unavailable", [f"{tool} 没有对应的工具契约"], action=action, tool=tool)
        if tool not in trusted.available_tools:
            return report(
                "tool_availability",
                "tool_unavailable",
                [f"本次服务端提供的工具是 {trusted.available_tools},{tool} 不在其中"],
                action=action,
                tool=tool,
            )

        arguments = decision.get("arguments")
        schema_errors = []
        contracts.validate(arguments, self.schemas[tool], f"{LOC}/arguments", self.schemas, schema_errors)
        if schema_errors:
            return report(
                "argument_schema",
                "arguments_invalid",
                [f"{code} @ {location}: {message}" for location, code, message in schema_errors],
                action=action,
                tool=tool,
            )
        semantic_errors = []
        contracts.tool_semantics(tool, arguments, f"{LOC}/arguments", trusted.now, semantic_errors)
        if semantic_errors:
            return report(
                "semantic_constraint",
                "arguments_invalid",
                [f"{code} @ {location}: {message}" for location, code, message in semantic_errors],
                action=action,
                tool=tool,
            )

        result = simulator.execute_tool(store, tool, arguments, now=trusted.now, confirmed=trusted.user_confirmed_write)
        report_out = report("executed", None, [], executed=True, action=action, tool=tool, result=result)
        if report_out["calendar_changed"] and not (
            tool == "create_event" and self.result_contract["tools"][tool]["writes"] and result["status"] == "ok"
        ):
            raise RuntimeError(f"写入边界被破坏:{tool} 返回 {result['status']} 却改变了日历")
        return report_out


_CACHED = {}


def execute_decision(decision, trusted, store=None):
    """模块级入口。契约只加载一次,数据默认每次新读,保证测试互不干扰。"""
    if "executor" not in _CACHED:
        _CACHED["executor"] = Executor()
    return _CACHED["executor"].run(decision, trusted, store=store)
