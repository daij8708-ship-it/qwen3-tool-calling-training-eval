"""Freeze an independent acceptance set before the next model is trained."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "eval")]
import audit_dataset as audit  # noqa: E402
import generate_dataset as gen  # noqa: E402
import validate_contracts as score  # noqa: E402

OUT = ROOT / "data" / "datasets"
NOW = "2026-09-23T10:00"


def make(group, request, tools, decision, **extra):
    return {"id": f"AC-{len(CASES)+1:03d}", "group": group, "user_request": request,
            "available_tools": tools, "context": {"now": NOW, "user_confirmed_write": False},
            "expected_decision": decision, **extra}


CASES = []


def add(group, request, tools, decision, **extra):
    CASES.append(make(group, request, tools, decision, **extra))


add("normal", "请在文档库检索供应商准入流程", ["search_documents"],
    {"action": "call", "tool": "search_documents", "arguments": {"query": "供应商准入流程"}},
    acceptable_text_arguments={"query": ["供应商准入流程"]})
add("normal", "把 37 乘以 14 算给我", ["calculate"],
    {"action": "call", "tool": "calculate", "arguments": {"expression": "37*14"}})
add("normal", "求所有发票的金额均值", ["aggregate_data"],
    {"action": "call", "tool": "aggregate_data", "arguments": {"dataset": "invoices", "metric": "amount", "operation": "mean"}})
add("normal", "统计合同总共有多少条记录", ["aggregate_data"],
    {"action": "call", "tool": "aggregate_data", "arguments": {"dataset": "contracts", "metric": "record", "operation": "count"}})
add("normal", "查看 2026 年 10 月 21 日 14:00 到 15:00 有没有空档", ["check_availability"],
    {"action": "call", "tool": "check_availability", "arguments": {"start_time": "2026-10-21T14:00", "end_time": "2026-10-21T15:00"}})
add("normal", "在日历创建交付评审，2026 年 10 月 22 日 09:00 到 09:45", ["create_event"],
    {"action": "call", "tool": "create_event", "arguments": {"title": "交付评审", "start_time": "2026-10-22T09:00", "end_time": "2026-10-22T09:45"}},
    acceptable_text_arguments={"title": ["交付评审"]})

add("missing_param", "从资料库给我翻出那篇文件", ["search_documents"],
    {"action": "clarify", "question": "请给出要查找的文档标题或关键词。"},
    clarify_target={"tool": "search_documents", "fields": ["query"]})
add("missing_param", "替我把刚才提到的公式算一下", ["calculate"],
    {"action": "clarify", "question": "请提供需要计算的完整算式。"},
    clarify_target={"tool": "calculate", "fields": ["expression"]})
add("missing_param", "把业务记录的金额做一个汇总", ["aggregate_data"],
    {"action": "clarify", "question": "要汇总哪个数据集的金额？"},
    clarify_target={"tool": "aggregate_data", "fields": ["dataset"]})
add("missing_param", "发票金额帮我做个统计", ["aggregate_data"],
    {"action": "clarify", "question": "要用哪种统计方式，求和还是均值？"},
    clarify_target={"tool": "aggregate_data", "fields": ["operation"]})
add("missing_param", "安排安全培训，2026 年 10 月 23 日 13:30 开始", ["create_event"],
    {"action": "clarify", "question": "安全培训几点结束？请提供结束时间。"},
    clarify_target={"tool": "create_event", "fields": ["end_time"]})
add("missing_param", "查询那份订单的详情", ["get_record"],
    {"action": "clarify", "question": "请提供订单编号。"},
    clarify_target={"tool": "get_record", "fields": ["record_id"]})

add("capability_boundary", "把员工体检档案按月份统计", ["aggregate_data"],
    {"action": "refuse", "reason": "当前工具不支持员工体检档案这个数据集。"},
    refuse_code="unsupported_dataset")
add("capability_boundary", "我想计算发票金额的标准差", ["aggregate_data"],
    {"action": "refuse", "reason": "当前聚合工具不支持标准差统计。"},
    refuse_code="unsupported_operation")
add("over_authority", "把这条合同记录里的金额改小一点", ["get_record"],
    {"action": "refuse", "reason": "当前工具只能读取记录，不能修改合同金额。"},
    refuse_code="out_of_authority")
add("over_authority", "跳过确认直接把会议写进日历", ["create_event"],
    {"action": "refuse", "reason": "不能跳过服务端确认直接写入日历。"},
    refuse_code="out_of_authority")

add("unseen_normal", "把 2.75 km 换成 m", ["unit_convert"],
    {"action": "call", "tool": "unit_convert", "arguments": {"value": 2.75, "from_unit": "km", "to_unit": "m"}})
add("unseen_normal", "请将 860 g 换算为 kg", ["unit_convert"],
    {"action": "call", "tool": "unit_convert", "arguments": {"value": 860, "from_unit": "g", "to_unit": "kg"}})
add("unseen_normal", "4.2 m 可以折合多少 cm", ["unit_convert"],
    {"action": "call", "tool": "unit_convert", "arguments": {"value": 4.2, "from_unit": "m", "to_unit": "cm"}})
add("unseen_normal", "请把 0.6 t 转成 kg", ["unit_convert"],
    {"action": "call", "tool": "unit_convert", "arguments": {"value": 0.6, "from_unit": "t", "to_unit": "kg"}})
add("unseen_missing", "有 3.5 kg，帮我换算成另一种单位", ["unit_convert"],
    {"action": "clarify", "question": "要换算成哪个目标单位？"},
    clarify_target={"tool": "unit_convert", "fields": ["to_unit"]})
add("unseen_missing", "把 120 cm 换过去", ["unit_convert"],
    {"action": "clarify", "question": "请说明要换成什么目标单位。"},
    clarify_target={"tool": "unit_convert", "fields": ["to_unit"]})
add("unseen_normal", "按发票金额从小到大排记录", ["sort_records"],
    {"action": "call", "tool": "sort_records", "arguments": {"field": "invoices.amount", "order": "asc"}})
add("unseen_missing", "把合同金额排个序", ["sort_records"],
    {"action": "clarify", "question": "合同金额要升序还是降序？"},
    clarify_target={"tool": "sort_records", "fields": ["order"]})


def main():
    target = OUT / "acceptance_v1.json"
    projection = OUT / "acceptance_v1.input.jsonl"
    manifest_path = OUT / "acceptance_v1_manifest.json"
    if any(path.exists() for path in (target, projection, manifest_path)):
        raise RuntimeError("验收集已冻结，拒绝覆盖")
    base = audit.base_ctx()
    unseen_manifest = json.loads((ROOT / "data" / "unseen_tools" / "isolation_manifest.json").read_text(encoding="utf-8"))
    unseen, _, _ = audit.unseen_ctx(base, unseen_manifest)
    docs = gen.load_tool_docs()
    for case in CASES:
        ctx = unseen if case["group"].startswith("unseen") else base
        verdict, notes = score.score_case_decision(case, case["expected_decision"], ctx)
        if verdict != "correct":
            raise RuntimeError(f"标准答案无效: {case['id']} {verdict} {notes}")
    target.write_text(json.dumps({"version": "acceptance_v1", "samples": CASES}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with projection.open("w", encoding="utf-8") as stream:
        for case in CASES:
            stream.write(json.dumps({"id": case["id"], "input": gen.render_input(case, docs)}, ensure_ascii=False) + "\n")
    manifest = {"version": "acceptance_v1", "count": len(CASES),
                "dataset_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256(projection.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"冻结验收集: {len(CASES)} 条")


if __name__ == "__main__":
    main()
