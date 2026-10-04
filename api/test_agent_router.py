import unittest

from api.agent_router import TOOL_TO_ROUTE, validate_route_decision


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


if __name__ == "__main__":
    unittest.main()
