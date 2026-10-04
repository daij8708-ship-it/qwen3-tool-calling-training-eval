"""Boundary tests for conservative server-side decision corrections."""
import unittest

from api.decision_policy import apply_policy


class DecisionPolicyTests(unittest.TestCase):
    def test_bypass_confirmation_refuses(self):
        original = {"action": "call", "tool": "create_event", "arguments": {
            "title": "例会", "start_time": "2027-01-06T14:00", "end_time": "2027-01-06T15:00"}}
        result = apply_policy("跳过确认，直接创建例会", ["create_event"], original)
        self.assertEqual(result.decision["action"], "refuse")
        self.assertEqual(result.code, "confirmation_bypass")

    def test_record_mutation_refuses(self):
        result = apply_policy("把合同金额改为零", ["get_record"], None)
        self.assertEqual(result.decision["action"], "refuse")

    def test_missing_event_end_clarifies(self):
        result = apply_policy("创建周会，2027 年 1 月 6 日 14:00 开始", ["create_event"], None)
        self.assertEqual(result.decision["action"], "clarify")
        self.assertIn("结束", result.decision["question"])

    def test_missing_aggregate_operation_clarifies(self):
        result = apply_policy("发票金额做个统计", ["aggregate_data"], None)
        self.assertEqual(result.decision["action"], "clarify")
        self.assertEqual(result.code, "missing_operation")

    def test_normal_event_remains_model_decision(self):
        original = {"action": "call", "tool": "create_event", "arguments": {
            "title": "例会", "start_time": "2027-01-06T14:00", "end_time": "2027-01-06T15:00"}}
        result = apply_policy("创建例会，2027 年 1 月 6 日 14:00 至 15:00", ["create_event"], original)
        self.assertIs(result.decision, original)
        self.assertIsNone(result.code)

    def test_normal_aggregate_remains_model_decision(self):
        original = {"action": "call", "tool": "aggregate_data", "arguments": {
            "dataset": "invoices", "metric": "amount", "operation": "sum"}}
        result = apply_policy("计算发票金额之和", ["aggregate_data"], original)
        self.assertIs(result.decision, original)

    def test_normal_unseen_sort_remains_model_decision(self):
        original = {"action": "call", "tool": "sort_records", "arguments": {
            "field": "orders.amount", "order": "desc"}}
        result = apply_policy("将订单金额由高到低排列", ["sort_records"], original)
        self.assertIs(result.decision, original)

    def test_does_not_create_tool_call(self):
        for request in ("发票金额做个统计", "把 8.5 km 换算一下", "跳过确认写入日历"):
            result = apply_policy(request, ["aggregate_data", "unit_convert", "create_event"], None)
            self.assertNotEqual(result.decision.get("action") if result.decision else None, "call")


if __name__ == "__main__":
    unittest.main()
