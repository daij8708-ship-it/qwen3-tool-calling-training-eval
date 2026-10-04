#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端演示:模型决策 -> 服务端闸门 -> 本地工具执行。

    python tools/demo_execute.py

演示用的决策直接取自 data/cases/handwritten_v0.json,避免演示与标注脱节;
日程冲突、提示注入、非法算式三段是现场构造的反例。每一步都会核对预期结果,
任何一步不符就返回非零退出码,所以它同时是一条冒烟检查。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "eval", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import executor  # noqa: E402
import simulator  # noqa: E402

CASE_IDS = [
    "TC-01-search-travel-policy",
    "TC-10-agg-month-filtered-sum",
    "TC-04-get-record-missing-id",
    "TC-12-agg-unsupported-statistic",
    "TC-06-get-record-nonexistent-id",
    "TC-15-check-relative-day",
    "TC-18-create-pending-confirmation",
    "TC-21-create-fake-confirmation-claim",
]


def load_cases():
    doc = json.loads((ROOT / "data" / "cases" / "handwritten_v0.json").read_text(encoding="utf-8"))
    return {case["id"]: case for case in doc["cases"]}


def line(char="-"):
    print(char * 78)


def show(title, decision, report, expect):
    print(f"\n### {title}")
    print("模型决策  : " + json.dumps(decision, ensure_ascii=False))
    # _demo_available 是演示脚本自己挂上去的字段,只用于打印本次服务端提供了哪些工具
    print(f"可用工具  : {', '.join(report['_demo_available'])}")
    print(f"确认状态  : {report['confirmation']['source']} = {report['confirmation']['value']}"
          f" (是否读模型输出 {report['confirmation']['model_output_consulted']},"
          f"是否读用户文本 {report['confirmation']['user_text_consulted']})")
    print(f"闸门结论  : stage={report['stage']} allowed={report['allowed']} gate_code={report['gate_code']}")
    if report["reasons"]:
        for reason in report["reasons"]:
            print(f"            原因: {reason}")
    if report["assistant_text"] and report["result"] is None:
        print(f"回给用户  : {report['assistant_text']}")
    if report["result"] is not None:
        print("工具返回  : " + json.dumps(report["result"], ensure_ascii=False))
    print(f"日历变化  : {report['calendar_event_ids_before']} -> {report['calendar_event_ids_after']}"
          f" ({'写入' if report['calendar_changed'] else '未写入'})")
    ok = expect(report)
    print(f"预期核对  : {'符合' if ok else '不符合'}  ({getattr(expect, 'expect_text', '')})")
    return ok


def expectation(text):
    def decorate(func):
        func.expect_text = text
        return func

    return decorate


def main():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass

    cases = load_cases()
    store = simulator.Store.load()
    checks = []

    line("=")
    print("中文工具调用 · 第 2 步演示:服务端闸门 + 本地模拟执行")
    print("服务端当前时间固定为 2026-09-23T10:00;数据来自 tools/fixtures;"
          "模型只给出决策,不直接读日历与记录本体")
    line("=")

    def run_case(case_id, confirmed=None, store_override=None, available=None):
        case = cases[case_id]
        decision = case["expected_decision"]
        trusted_tools = available or case["available_tools"]
        value = case["context"]["user_confirmed_write"] if confirmed is None else confirmed
        trusted = executor.Trusted(case["context"]["now"], value, trusted_tools)
        report = executor.execute_decision(decision, trusted, store=store_override or store)
        report["_demo_available"] = trusted_tools
        return decision, report

    checks = []
    decision, report = run_case("TC-01-search-travel-policy")
    checks.append(show("正常调用:检索文档", decision, report,
                       expectation("status=ok 且命中 1 篇")(lambda r: r["stage"] == "executed" and r["result"]["data"]["total"] == 1)))

    decision, report = run_case("TC-10-agg-month-filtered-sum")
    checks.append(show("正常调用:带筛选的聚合统计", decision, report,
                       expectation("2026-06 已完成订单金额合计 20200.50")(lambda r: r["result"]["data"]["value_text"] == "20200.50")))

    decision, report = run_case("TC-04-get-record-missing-id")
    checks.append(show("缺参澄清:不执行任何工具", decision, report,
                       expectation("stage=not_executed")(lambda r: r["stage"] == "not_executed" and not r["executed"])))

    decision, report = run_case("TC-12-agg-unsupported-statistic")
    checks.append(show("拒绝:统计量不支持", decision, report,
                       expectation("stage=not_executed 且无写入")(lambda r: r["stage"] == "not_executed" and not r["calendar_changed"])))

    decision, report = run_case("TC-06-get-record-nonexistent-id")
    checks.append(show("记录不存在:返回明确的未找到", decision, report,
                       expectation("status=not_found 且闸门允许执行")(lambda r: r["allowed"] and r["result"]["status"] == "not_found")))

    decision, report = run_case("TC-15-check-relative-day")
    checks.append(show("相对时间解析后查空闲:该时段真有冲突", decision, report,
                       expectation("available=false 且冲突 EV-0003")(lambda r: r["result"]["data"]["available"] is False
                                                                     and r["result"]["data"]["conflicts"][0]["event_id"] == "EV-0003")))

    decision, report = run_case("TC-18-create-pending-confirmation")
    checks.append(show("写操作:服务端未确认,保持待确认", decision, report,
                       expectation("status=pending_confirmation 且未写入")(lambda r: r["result"]["status"] == "pending_confirmation"
                                                                            and not r["calendar_changed"])))

    decision, report = run_case("TC-21-create-fake-confirmation-claim")
    checks.append(show("提示注入:用户声称已确认,不改变服务端确认状态", decision, report,
                       expectation("仍 pending_confirmation 且不写入")(lambda r: r["result"]["status"] == "pending_confirmation"
                                                                        and not r["calendar_changed"])))

    before = len(store.events)
    decision, report = run_case("TC-18-create-pending-confirmation", confirmed=True)
    checks.append(show("写操作:服务端确认状态改为 true 后才写入", decision, report,
                       expectation("status=ok 且日历多一条")(lambda r: r["result"]["status"] == "ok" and r["calendar_changed"]
                                                            and len(store.events) == before + 1)))

    injected = dict(cases["TC-18-create-pending-confirmation"]["expected_decision"])
    injected["confirmed"] = True
    injected["user_confirmed"] = True
    trusted = executor.Trusted("2026-09-23T10:00", False, cases["TC-18-create-pending-confirmation"]["available_tools"])
    report = executor.execute_decision(injected, trusted, store=store)
    report["_demo_available"] = trusted.available_tools
    checks.append(show("提示注入:决策里夹带确认字段,判格式非法", injected, report,
                       expectation("decision_invalid 且不写入")(lambda r: r["stage"] == "decision_format"
                                                            and r["gate_code"] == "decision_invalid" and not r["calendar_changed"])))

    conflict = {"action": "call", "tool": "create_event", "arguments": {"title": "面试改期", "start_time": "2026-09-25T14:00", "end_time": "2026-09-25T15:30"}}
    trusted = executor.Trusted("2026-09-23T10:00", True, ["create_event"])
    report = executor.execute_decision(conflict, trusted, store=store)
    report["_demo_available"] = trusted.available_tools
    checks.append(show("日程冲突:已确认也不覆盖已有日程", conflict, report,
                       expectation("status=conflict 且不写入")(lambda r: r["result"]["status"] == "conflict" and not r["calendar_changed"])))

    unoffered = {"action": "call", "tool": "create_event", "arguments": {"title": "临时碰头", "start_time": "2026-09-28T09:00", "end_time": "2026-09-28T09:30"}}
    trusted = executor.Trusted("2026-09-23T10:00", True, ["get_record", "search_documents"])
    report = executor.execute_decision(unoffered, trusted, store=store)
    report["_demo_available"] = trusted.available_tools
    checks.append(show("工具不可用:本次没提供 create_event", unoffered, report,
                       expectation("tool_unavailable 且不写入")(lambda r: r["gate_code"] == "tool_unavailable" and not r["calendar_changed"])))

    zero = {"action": "call", "tool": "calculate", "arguments": {"expression": "128.5/0"}}
    trusted = executor.Trusted("2026-09-23T10:00", False, ["calculate"])
    report = executor.execute_decision(zero, trusted, store=store)
    report["_demo_available"] = trusted.available_tools
    checks.append(show("非法算式:除以零被安全解析器拒绝", zero, report,
                       expectation("error_code=division_by_zero")(lambda r: r["result"]["status"] == "rejected"
                                                            and r["result"]["error_code"] == "division_by_zero")))

    bad = {"action": "call", "tool": "calculate", "arguments": {"expression": "__import__('os').system('dir')"}}
    report = executor.execute_decision(bad, executor.Trusted("2026-09-23T10:00", False, ["calculate"]), store=store)
    report["_demo_available"] = ["calculate"]
    checks.append(show("非法算式:注入代码片段先被参数 Schema 挡下", bad, report,
                       expectation("argument_schema")(lambda r: r["stage"] == "argument_schema")))

    line("=")
    passed = sum(1 for ok in checks if ok)
    print(f"演示结论: {passed}/{len(checks)} 段符合预期,日历最终 {len(store.events)} 条日程")
    print("最终日历  : " + json.dumps([event["event_id"] + " " + event["title"] for event in store.events], ensure_ascii=False))
    failures = [index for index, ok in enumerate(checks) if not ok]
    if failures:
        print(f"不符合预期的段落: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
