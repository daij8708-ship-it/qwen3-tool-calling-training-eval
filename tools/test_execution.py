#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地模拟执行与服务端闸门的测试。

    python tools/test_execution.py

覆盖六个工具的正常路径、无效参数、不可用工具、记录不存在、日程冲突,
以及最重要的写入边界:未确认不写入、确认后才写入、任何闸门失败都不写。
不调用大模型,不联网,数据固定可重复。
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "eval", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import executor  # noqa: E402
import simulator  # noqa: E402
import validate_contracts as contracts  # noqa: E402

NOW = "2026-09-23T10:00"
ALL_TOOLS = list(contracts.EXPECTED_TOOLS)
RESULT_CONTRACT = json.loads((ROOT / "tools" / "result_contract.json").read_text(encoding="utf-8"))


def run(decision, available_tools=None, confirmed=False, store=None, now=NOW):
    trusted = executor.Trusted(now, confirmed, available_tools or ALL_TOOLS)
    return executor.execute_decision(decision, trusted, store=store)


def call(tool, arguments):
    return {"action": "call", "tool": tool, "arguments": arguments}


class HappyPath(unittest.TestCase):
    def test_search_documents_finds_fixture(self):
        report = run(call("search_documents", {"query": "差旅报销制度"}))
        self.assertEqual(report["stage"], "executed")
        self.assertTrue(report["allowed"])
        data = report["result"]["data"]
        self.assertEqual(data["total"], 1)
        self.assertEqual(data["matches"][0]["id"], "DOC-001")

    def test_search_documents_multi_token_and_zero_hit(self):
        hit = run(call("search_documents", {"query": "合同 期限"}))
        self.assertEqual([item["id"] for item in hit["result"]["data"]["matches"]], ["DOC-003"])
        miss = run(call("search_documents", {"query": "火星殖民条例"}))
        self.assertEqual(miss["result"]["status"], "ok")
        self.assertEqual(miss["result"]["data"]["total"], 0)

    def test_get_record_returns_fields(self):
        report = run(call("get_record", {"record_type": "contract", "record_id": "HT045771"}))
        self.assertEqual(report["result"]["status"], "ok")
        self.assertEqual(report["result"]["data"]["fields"]["amount"], 420000.0)

    def test_calculate_matches_hand_check(self):
        report = run(call("calculate", {"expression": "3*128.5*0.88+23.9"}))
        self.assertEqual(report["result"]["data"]["value_text"], "363.14")
        percent = run(call("calculate", {"expression": "200*10%"}))
        self.assertEqual(percent["result"]["data"]["value_text"], "20.00")
        unit = run(call("calculate", {"expression": "2.5*60"}))
        self.assertEqual(unit["result"]["data"]["value_text"], "150.00")

    def test_aggregate_filtered_sum_and_count(self):
        total = run(call("aggregate_data", {
            "dataset": "orders", "metric": "amount", "operation": "sum",
            "filters": {"time_range": {"start": "2026-06-01T00:00", "end": "2026-06-30T23:59"}, "status": ["completed"]},
        }))
        self.assertEqual(total["result"]["data"]["value_text"], "20200.50")
        self.assertEqual(total["result"]["data"]["sample_count"], 2)
        count = run(call("aggregate_data", {"dataset": "invoices", "metric": "record", "operation": "count", "filters": {"status": ["paid"]}}))
        self.assertEqual(count["result"]["data"]["value"], 2)

    def test_check_availability_free_and_busy(self):
        free = run(call("check_availability", {"start_time": "2026-09-24T11:00", "end_time": "2026-09-24T12:00"}))
        self.assertTrue(free["result"]["data"]["available"])
        busy = run(call("check_availability", {"start_time": "2026-09-25T14:00", "end_time": "2026-09-25T16:00"}))
        self.assertFalse(busy["result"]["data"]["available"])
        self.assertEqual([item["event_id"] for item in busy["result"]["data"]["conflicts"]], ["EV-0003"])

    def test_back_to_back_slots_do_not_conflict(self):
        report = run(call("check_availability", {"start_time": "2026-09-25T15:00", "end_time": "2026-09-25T16:00"}))
        self.assertTrue(report["result"]["data"]["available"])

    def test_create_event_writes_only_when_confirmed(self):
        decision = call("create_event", {"title": "需求评审会", "start_time": "2026-09-24T09:30", "end_time": "2026-09-24T10:30"})
        pending = run(decision, confirmed=False)
        self.assertEqual(pending["result"]["status"], "pending_confirmation")
        self.assertFalse(pending["calendar_changed"])
        self.assertEqual(pending["calendar_event_ids_before"], pending["calendar_event_ids_after"])

        store = simulator.Store.load()
        confirmed = run(decision, confirmed=True, store=store)
        self.assertEqual(confirmed["result"]["status"], "ok")
        self.assertTrue(confirmed["calendar_changed"])
        self.assertEqual(confirmed["calendar_event_ids_after"][-1], "EV-0005")
        self.assertEqual(store.events[-1]["title"], "需求评审会")

    def test_confirm_flag_reported_as_server_source(self):
        pending = run(call("create_event", {"title": "架构评审", "start_time": "2026-09-26T10:00", "end_time": "2026-09-26T11:00"}), confirmed=False)
        self.assertEqual(pending["confirmation"]["source"], "context.user_confirmed_write")
        self.assertFalse(pending["confirmation"]["value"])
        self.assertFalse(pending["confirmation"]["model_output_consulted"])
        self.assertFalse(pending["confirmation"]["user_text_consulted"])

    def test_confirmation_source_is_pinned_in_the_executor_source(self):
        """变异测试:把闸门改成采纳 decision['confirmed'] 时,这条必须红。
        单看行为看不出来(多余字段先被格式层拦掉),所以直接钉住取值来源。"""
        source = (ROOT / "tools" / "executor.py").read_text(encoding="utf-8")
        self.assertIn("confirmed=trusted.user_confirmed_write", source)
        self.assertNotIn("decision.get(\"confirmed\"", source)
        self.assertNotIn("decision.get('confirmed'", source)

    def test_title_claiming_confirmation_does_not_write(self):
        """标题里写"已确认"仍只是文本,确认状态不看内容。"""
        decision = call("create_event", {"title": "已确认的架构评审", "start_time": "2026-09-26T09:00", "end_time": "2026-09-26T10:00"})
        report = run(decision, confirmed=False)
        self.assertEqual(report["result"]["status"], "pending_confirmation")
        self.assertFalse(report["calendar_changed"])

    def test_simulator_alone_refuses_write_without_confirmation(self):
        """绕开闸门直接调用模拟器,同样不会写入:写入判断不只在闸门一层。"""
        store = simulator.Store.load()
        before = len(store.events)
        result = simulator.execute_tool(
            store, "create_event",
            {"title": "绕过闸门的直调", "start_time": "2026-09-26T09:00", "end_time": "2026-09-26T10:00"},
            now=contracts.parse_dt(NOW), confirmed=False,
        )
        self.assertEqual(result["status"], "pending_confirmation")
        self.assertEqual(len(store.events), before)


class WriteBoundary(unittest.TestCase):
    def test_user_text_claiming_confirmation_does_not_write(self):
        """用户说"我确认过了"不改变服务端确认状态;闸门也从不读用户文本。"""
        decision = call("create_event", {"title": "临时对齐会", "start_time": "2026-09-26T14:00", "end_time": "2026-09-26T15:00"})
        report = run(decision, confirmed=False)
        self.assertEqual(report["result"]["status"], "pending_confirmation")
        self.assertFalse(report["calendar_changed"])

    def test_decision_carrying_confirmation_field_is_rejected(self):
        decision = call("create_event", {"title": "临时对齐会", "start_time": "2026-09-26T14:00", "end_time": "2026-09-26T15:00"})
        decision["confirmed"] = True
        decision["user_confirmed"] = True
        report = run(decision, confirmed=False)
        self.assertEqual(report["stage"], "decision_format")
        self.assertEqual(report["gate_code"], "decision_invalid")
        self.assertFalse(report["calendar_changed"])
        self.assertTrue(any("EXTRA_FIELD" in reason for reason in report["reasons"]))

    def test_conflicting_slot_not_written_even_when_confirmed(self):
        report = run(
            call("create_event", {"title": "候选人面试改期", "start_time": "2026-09-25T14:00", "end_time": "2026-09-25T15:30"}),
            confirmed=True,
        )
        self.assertEqual(report["result"]["status"], "conflict")
        self.assertFalse(report["calendar_changed"])
        self.assertEqual([item["event_id"] for item in report["result"]["data"]["conflicts"]], ["EV-0003"])

    def test_every_gate_failure_leaves_calendar_untouched(self):
        bad_decisions = [
            {"action": "call", "tool": "create_event", "arguments": {"title": "缺结束"}},
            {"action": "call", "tool": "create_event", "arguments": {"title": "错前缀", "start_time": "2026-09-26T09:00", "end_time": "2026-09-26T10:00", "note": "顺带"}},
            {"action": "call", "tool": "create_event", "arguments": {"title": "时间倒序", "start_time": "2026-09-26T16:00", "end_time": "2026-09-26T14:00"}},
            {"action": "call", "tool": "create_event", "arguments": {"title": "过去时间", "start_time": "2026-09-20T09:00", "end_time": "2026-09-20T10:00"}},
            {"action": "call", "tool": "get_record", "arguments": {"record_type": "order", "record_id": "FP123456"}},
            {"action": "explain", "tool": "create_event", "arguments": {"title": "动作不合法", "start_time": "2026-09-26T09:00", "end_time": "2026-09-26T10:00"}},
        ]
        store = simulator.Store.load()
        baseline = store.calendar_ids()
        for decision in bad_decisions:
            report = run(decision, confirmed=True, store=store)
            self.assertFalse(report["allowed"], f"不该通过闸门: {decision}")
            self.assertFalse(report["calendar_changed"], f"闸门失败却写入了日历: {decision}")
        self.assertEqual(store.calendar_ids(), baseline)

    def test_stage_order_reports_the_first_failing_check(self):
        self.assertEqual(run({"action": "call", "tool": "create_event"}, confirmed=True)["stage"], "decision_format")
        self.assertEqual(
            run(call("create_event", {"title": "未提供", "start_time": "2026-09-26T09:00", "end_time": "2026-09-26T10:00"}), available_tools=["get_record"], confirmed=True)["stage"],
            "tool_availability",
        )
        self.assertEqual(
            run(call("create_event", {"title": "缺结束时间", "start_time": "2026-09-26T09:00"}), confirmed=True)["stage"],
            "argument_schema",
        )
        self.assertEqual(
            run(call("create_event", {"title": "超过八小时", "start_time": "2026-09-26T09:00", "end_time": "2026-10-06T09:00"}), confirmed=True)["stage"],
            "semantic_constraint",
        )
        self.assertEqual(
            run(call("check_availability", {"start_time": "2026-09-26T16:00", "end_time": "2026-09-26T14:00"}))["stage"],
            "semantic_constraint",
        )


class InvalidInput(unittest.TestCase):
    def test_unavailable_tool_is_rejected(self):
        report = run(call("aggregate_data", {"dataset": "orders", "metric": "amount", "operation": "sum"}), available_tools=["get_record", "search_documents"])
        self.assertEqual(report["stage"], "tool_availability")
        self.assertEqual(report["gate_code"], "tool_unavailable")
        self.assertFalse(report["executed"])

    def test_tool_name_outside_decision_enum_is_a_format_error(self):
        """六个之外的工具名先撞决策契约的 tool 枚举,属格式非法;
        已知工具但本次未提供才走 tool_availability。"""
        report = run(call("send_email", {"to": "a@b.c"}))
        self.assertEqual(report["stage"], "decision_format")
        self.assertEqual(report["gate_code"], "decision_invalid")
        self.assertTrue(any("ENUM_MISMATCH" in reason for reason in report["reasons"]))
        self.assertFalse(report["calendar_changed"])

    def test_argument_schema_rejections(self):
        cases = [
            (call("aggregate_data", {"dataset": "headcount", "metric": "amount", "operation": "sum"}), "ENUM_MISMATCH"),
            (call("aggregate_data", {"dataset": "orders", "metric": "amount", "operation": "stddev"}), "ENUM_MISMATCH"),
            (call("aggregate_data", {"dataset": "orders", "metric": "amount", "operation": "count"}), "CONST_MISMATCH"),
            (call("get_record", {"record_type": "order", "record_id": "DD12345"}), "PATTERN_MISMATCH"),
            (call("get_record", {"record_type": "contract"}), "MISSING_REQUIRED"),
            (call("check_availability", {"start_time": "2026-09-26T14:00", "end_time": "2026-09-26T16:00", "zone": "本地"}), "EXTRA_FIELD"),
            (call("calculate", {"expression": "round(3.5)"}), "PATTERN_MISMATCH"),
        ]
        for decision, code in cases:
            report = run(decision)
            self.assertEqual(report["stage"], "argument_schema", decision)
            self.assertEqual(report["gate_code"], "arguments_invalid")
            self.assertTrue(any(code in reason for reason in report["reasons"]), f"{decision} 应报 {code},实际 {report['reasons']}")

    def test_past_time_is_rejected_by_gate_before_execution(self):
        report = run(call("check_availability", {"start_time": "2025-03-05T14:00", "end_time": "2025-03-05T16:00"}))
        self.assertEqual(report["stage"], "semantic_constraint")
        self.assertTrue(any("SEMANTIC_PAST_START" in reason for reason in report["reasons"]))

    def test_calculate_runtime_rejections(self):
        for expression, code in [
            ("5/0", "division_by_zero"),
            ("(1+2", "expression_syntax"),
            ("1++2)", "expression_syntax"),
            ("999999999999*999999999999", "value_out_of_range"),
        ]:
            report = run(call("calculate", {"expression": expression}))
            self.assertEqual(report["result"]["status"], "rejected", expression)
            self.assertEqual(report["result"]["error_code"], code, expression)

    def test_calculate_never_uses_eval(self):
        source = (ROOT / "tools" / "simulator.py").read_text(encoding="utf-8")
        self.assertNotIn("eval(", source)
        self.assertNotIn("exec(", source)
        self.assertNotIn("__import__", source)
        report = run(call("calculate", {"expression": "3*(4+6)/5-1.5"}))
        self.assertEqual(report["result"]["data"]["value_text"], "4.50")

    def test_aggregate_no_data(self):
        report = run(call("aggregate_data", {"dataset": "orders", "metric": "amount", "operation": "sum", "filters": {"region": ["西北"]}}))
        self.assertEqual(report["result"]["status"], "no_data")
        self.assertIsNone(report["result"]["data"]["value"])
        self.assertEqual(report["result"]["data"]["sample_count"], 0)

    def test_missing_record_returns_not_found_not_error(self):
        report = run(call("get_record", {"record_type": "shipment", "record_id": "SH888114"}))
        self.assertTrue(report["allowed"])
        self.assertEqual(report["result"]["status"], "not_found")
        self.assertEqual(report["result"]["error_code"], "record_not_found")


class ClarifyAndRefuseAreNotExecuted(unittest.TestCase):
    def test_clarify_does_not_execute(self):
        report = run({"action": "clarify", "question": "请提供该订单的编号,DD 加 6 位数字。"})
        self.assertEqual(report["stage"], "not_executed")
        self.assertEqual(report["gate_code"], "action_not_call")
        self.assertFalse(report["executed"])
        self.assertEqual(report["assistant_text"], "请提供该订单的编号,DD 加 6 位数字。")

    def test_refuse_does_not_execute(self):
        report = run({"action": "refuse", "reason": "聚合统计不支持标准差。"})
        self.assertEqual(report["stage"], "not_executed")
        self.assertFalse(report["allowed"])
        self.assertFalse(report["calendar_changed"])


class ResultContractIsEnforced(unittest.TestCase):
    def test_unregistered_status_and_code_are_refused(self):
        store = simulator.Store.load()
        with self.assertRaises(simulator.ContractViolation):
            simulator.envelope(store, "calculate", "maybe_ok", "未登记状态")
        with self.assertRaises(simulator.ContractViolation):
            simulator.envelope(store, "calculate", "rejected", "缺码的拒绝")
        with self.assertRaises(simulator.ContractViolation):
            simulator.envelope(store, "get_record", "ok", "不该带码", None, "division_by_zero")

    def test_every_emitted_result_is_declared(self):
        decisions = [
            call("search_documents", {"query": "发票 开具"}),
            call("get_record", {"record_type": "order", "record_id": "DD204551"}),
            call("get_record", {"record_type": "order", "record_id": "DD999999"}),
            call("calculate", {"expression": "1+1"}),
            call("calculate", {"expression": "1/0"}),
            call("aggregate_data", {"dataset": "shipments", "metric": "duration_days", "operation": "mean"}),
            call("aggregate_data", {"dataset": "contracts", "metric": "record", "operation": "count"}),
            call("check_availability", {"start_time": "2026-09-24T09:00", "end_time": "2026-09-24T10:00"}),
            call("create_event", {"title": "临时会", "start_time": "2026-09-27T09:00", "end_time": "2026-09-27T10:00"}),
        ]
        for decision in decisions:
            result = run(decision, confirmed=True)["result"]
            spec = RESULT_CONTRACT["tools"][decision["tool"]]
            self.assertIn(result["status"], spec["statuses"], decision)
            if result["error_code"] is not None:
                self.assertIn(result["error_code"], spec["error_codes"], decision)


class StepOneCasesStillExecute(unittest.TestCase):
    """第 1 步的 24 条标准决策过本步闸门:call 必须能执行且返回登记状态,澄清与拒绝不得执行。"""

    def setUp(self):
        doc = json.loads((ROOT / "data" / "cases" / "handwritten_v0.json").read_text(encoding="utf-8"))
        self.cases = doc["cases"]

    def test_all_expected_decisions_pass_the_gate(self):
        seen_status = {}
        for case in self.cases:
            decision = case["expected_decision"]
            context = case["context"]
            report = run(
                decision,
                available_tools=case["available_tools"],
                confirmed=context["user_confirmed_write"],
                now=context["now"],
            )
            label = case["id"]
            if decision["action"] == "call":
                self.assertTrue(report["allowed"], f"{label} 的标准决策被闸门拒绝: {report['reasons']}")
                spec = RESULT_CONTRACT["tools"][decision["tool"]]
                self.assertIn(report["result"]["status"], spec["statuses"], label)
                seen_status.setdefault(label, report["result"]["status"])
                if decision["tool"] == "create_event":
                    self.assertEqual(report["result"]["status"], "pending_confirmation", label)
                    self.assertFalse(report["calendar_changed"], label)
            else:
                self.assertEqual(report["stage"], "not_executed", label)
                self.assertFalse(report["calendar_changed"], label)
        self.assertEqual(seen_status["TC-06-get-record-nonexistent-id"], "not_found")
        self.assertEqual(seen_status["TC-03-get-record-context-reference"], "ok")
        self.assertEqual(seen_status["TC-10-agg-month-filtered-sum"], "ok")
        self.assertEqual(seen_status["TC-15-check-relative-day"], "ok")

    def test_case_gate_and_semantic_stages_match_the_labels(self):
        refused = [case["id"] for case in self.cases if case["expected_decision"]["action"] == "refuse"]
        self.assertEqual(len(refused), 6)


class Determinism(unittest.TestCase):
    def test_same_inputs_give_identical_reports(self):
        decisions = [
            call("search_documents", {"query": "合同 期限"}),
            call("get_record", {"record_type": "invoice", "record_id": "FP103883"}),
            call("aggregate_data", {"dataset": "orders", "metric": "quantity", "operation": "max"}),
            call("check_availability", {"start_time": "2026-09-25T14:00", "end_time": "2026-09-25T16:00"}),
            call("create_event", {"title": "需求评审会", "start_time": "2026-09-24T09:30", "end_time": "2026-09-24T10:30"}),
        ]
        first = [json.dumps(run(decision), sort_keys=True, ensure_ascii=False) for decision in decisions]
        second = [json.dumps(run(decision), sort_keys=True, ensure_ascii=False) for decision in decisions]
        self.assertEqual(first, second)

    def test_pending_confirmation_is_idempotent(self):
        decision = call("create_event", {"title": "需求评审会", "start_time": "2026-09-24T09:30", "end_time": "2026-09-24T10:30"})
        store = simulator.Store.load()
        for _ in range(3):
            report = run(decision, confirmed=False, store=store)
            self.assertEqual(report["result"]["status"], "pending_confirmation")
        self.assertEqual(len(store.events), 4)
        confirmed = run(decision, confirmed=True, store=store)
        self.assertEqual(confirmed["result"]["status"], "ok")
        self.assertEqual(len(store.events), 5)


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass
    unittest.main(verbosity=2)
