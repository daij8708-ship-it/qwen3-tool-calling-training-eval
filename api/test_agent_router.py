import unittest

from api.agent_router import (
    TOOL_TO_ROUTE, split_two_tasks, validate_route_decision, validate_route_decision_v1,
)


class AgentRouteGateTest(unittest.TestCase):
    def test_all_five_routes_accept_exact_tool_names(self):
        for tool, expected in TOOL_TO_ROUTE.items():
            decision = {"action": "call", "tool": tool, "arguments": {"query": "测试请求"}}
            self.assertEqual(validate_route_decision("测试请求", decision), (expected, "model"))

    def test_unavailable_tool_cannot_be_executed(self):
        decision = {"action": "call", "tool": "delete_ticket", "arguments": {"query": "删除工单"}}
        self.assertEqual(validate_route_decision("删除工单", decision), (None, "unknown_tool"))

    def test_multi_task_returns_to_cloud_orchestrator(self):
        decision = {"action": "call", "tool": "consult_technical_expert", "arguments": {"query": "电脑黑屏"}}
        for query in ("电脑黑屏，并帮我找维修点", "查询故障资料，没解决就创建工单",
                      "查新闻并创建工单", "如果修不好就找维修站"):
            self.assertEqual(validate_route_decision(query, decision), (None, "multi_task"))

    def test_v1_model_can_explicitly_defer_cross_agent_tasks(self):
        self.assertEqual(
            validate_route_decision_v1({"action": "delegate", "reason": "multi_task"}),
            (None, "model_multi_task"),
        )

    def test_v1_single_route_stays_whitelisted(self):
        decision = {"action": "call", "tool": "query_service_station_and_navigate",
                    "arguments": {"query": "找维修店并导航过去"}}
        self.assertEqual(validate_route_decision_v1(decision), ("service", "model"))
        self.assertEqual(validate_route_decision_v1({"action": "call", "tool": "delete_ticket",
                                                     "arguments": {"query": "测试"}}),
                         (None, "unknown_tool"))

    def test_two_clause_boundary_requires_exactly_two_subtasks(self):
        self.assertEqual(
            split_two_tasks("先查本周手机新闻，再查本周芯片新闻。"),
            ("先查本周手机新闻", "查本周芯片新闻"),
        )
        self.assertIsNone(split_two_tasks("查手机新闻"))
        self.assertIsNone(split_two_tasks("先查天气，然后查工单，再写祝福。"))


if __name__ == "__main__":
    unittest.main()
