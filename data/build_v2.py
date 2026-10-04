"""Build a targeted, paired missing-field training set from train/val only."""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "eval")]
import audit_dataset as audit
import generate_dataset as gen
import validate_contracts as score

OUT = ROOT / "data" / "datasets"
NOW = "2026-09-23T10:00"

# Each tuple is (missing request, complete request, complete arguments). The
# sixth pair in each family is validation-only; wording is disjoint from train.
PAIRS = {
"search_documents": [
 ("从资料库找找那篇文章", "从资料库找找费用报销问答", {"query":"费用报销问答"}),
 ("帮我定位那份流程文件", "帮我定位样品入库流程文件", {"query":"样品入库流程"}),
 ("查一下有关的制度", "查一下车辆使用管理制度", {"query":"车辆使用管理制度"}),
 ("搜搜刚刚说的材料", "搜搜仓储巡检材料", {"query":"仓储巡检"}),
 ("给我找份相关文档", "给我找份设备报废文档", {"query":"设备报废"}),
 ("去库里翻翻那个手册", "去库里翻翻新人入职手册", {"query":"新人入职手册"}),
],
"calculate": [
 ("把之前的题算出来", "把 41 加 36 算出来", {"expression":"41+36"}),
 ("算一下还没说的那个算式", "算一下 96 除以 8", {"expression":"96/8"}),
 ("直接给我运算结果", "直接给我 24 乘以 13 的运算结果", {"expression":"24*13"}),
 ("做一下那道算术", "做一下 73 减去 28 的算术", {"expression":"73-28"}),
 ("那个数学表达式求值", "数学表达式 5 加 17 求值", {"expression":"5+17"}),
 ("等我给你数字再计算", "现在计算 54 除以 6", {"expression":"54/6"}),
],
"get_record": [
 ("打开那笔订单看看", "打开订单 DD204552 看看", {"record_type":"order","record_id":"DD204552"}),
 ("把指定发票的内容发我", "把发票 FP103883 的内容发我", {"record_type":"invoice","record_id":"FP103883"}),
 ("看看那份合同的详情", "看看合同 HT045772 的详情", {"record_type":"contract","record_id":"HT045772"}),
 ("查询某张发货单", "查询发货单 SH220914", {"record_type":"shipment","record_id":"SH220914"}),
 ("把我说的订单调出来", "把订单 DD204555 调出来", {"record_type":"order","record_id":"DD204555"}),
 ("查看还没报编号的发票", "查看发票 FP103886", {"record_type":"invoice","record_id":"FP103886"}),
],
"aggregate_dataset": [
 ("把金额汇总成一个数", "把订单金额汇总成一个数", {"dataset":"orders","metric":"amount","operation":"sum"}),
 ("业务金额求个总和", "发票金额求个总和", {"dataset":"invoices","metric":"amount","operation":"sum"}),
 ("统计一下所有记录的条数", "统计一下合同记录的条数", {"dataset":"contracts","metric":"record","operation":"count"}),
 ("给我算金额平均值", "给我算订单金额平均值", {"dataset":"orders","metric":"amount","operation":"mean"}),
 ("把所有金额加起来", "把合同金额加起来", {"dataset":"contracts","metric":"amount","operation":"sum"}),
 ("看看总共有几条业务数据", "看看总共有几条发票数据", {"dataset":"invoices","metric":"record","operation":"count"}),
],
"aggregate_operation": [
 ("帮我分析订单的金额数据", "订单金额求总和", {"dataset":"orders","metric":"amount","operation":"sum"}),
 ("分析一下发票的金额", "分析一下发票金额的均值", {"dataset":"invoices","metric":"amount","operation":"mean"}),
 ("发货单数量统计一下", "统计发货单共有几条记录", {"dataset":"shipments","metric":"record","operation":"count"}),
 ("汇报合同金额的统计结果", "汇报合同金额的总和", {"dataset":"contracts","metric":"amount","operation":"sum"}),
 ("做一份订单金额统计", "做一份订单金额均值统计", {"dataset":"orders","metric":"amount","operation":"mean"}),
 ("发票金额给我统计一下", "发票金额给我算总和", {"dataset":"invoices","metric":"amount","operation":"sum"}),
],
"create_event": [
 ("新增团队碰头会，2026 年 10 月 5 日 09:00 开始", "新增团队碰头会，2026 年 10 月 5 日 09:00 到 10:00", {"title":"团队碰头会","start_time":"2026-10-05T09:00","end_time":"2026-10-05T10:00"}),
 ("安排需求对齐，2026 年 10 月 6 日 14:00 开始", "安排需求对齐，2026 年 10 月 6 日 14:00 至 15:00", {"title":"需求对齐","start_time":"2026-10-06T14:00","end_time":"2026-10-06T15:00"}),
 ("日历写入客户回访，2026 年 10 月 7 日 11:00 开始", "日历写入客户回访，2026 年 10 月 7 日 11:00 到 11:45", {"title":"客户回访","start_time":"2026-10-07T11:00","end_time":"2026-10-07T11:45"}),
 ("创建库存会议，2026 年 10 月 8 日 16:00 开始", "创建库存会议，2026 年 10 月 8 日 16:00 至 17:00", {"title":"库存会议","start_time":"2026-10-08T16:00","end_time":"2026-10-08T17:00"}),
 ("安排售后复盘，2026 年 10 月 9 日 13:30 开始", "安排售后复盘，2026 年 10 月 9 日 13:30 到 14:30", {"title":"售后复盘","start_time":"2026-10-09T13:30","end_time":"2026-10-09T14:30"}),
 ("创建交付讨论，2026 年 10 月 12 日 10:00 开始", "创建交付讨论，2026 年 10 月 12 日 10:00 到 11:00", {"title":"交付讨论","start_time":"2026-10-12T10:00","end_time":"2026-10-12T11:00"}),
],
"check_availability": [
 ("看看下周能不能约会议", "看看 2026 年 10 月 13 日 09:00 到 10:00 能不能约会议", {"start_time":"2026-10-13T09:00","end_time":"2026-10-13T10:00"}),
 ("查查某天下午的空闲情况", "查查 2026 年 10 月 14 日 15:00 到 16:00 的空闲情况", {"start_time":"2026-10-14T15:00","end_time":"2026-10-14T16:00"}),
 ("看看最近什么时候有空", "看看 2026 年 10 月 15 日 11:00 到 12:00 有没有空", {"start_time":"2026-10-15T11:00","end_time":"2026-10-15T12:00"}),
 ("查询后天上午的空档", "查询 2026 年 10 月 16 日 09:30 至 10:30 的空档", {"start_time":"2026-10-16T09:30","end_time":"2026-10-16T10:30"}),
 ("帮我查一段时间是否空闲", "帮我查 2026 年 10 月 19 日 14:00 至 15:00 是否空闲", {"start_time":"2026-10-19T14:00","end_time":"2026-10-19T15:00"}),
 ("找一下可以安排活动的时段", "查一下 2026 年 10 月 20 日 16:00 到 17:00 能否安排活动", {"start_time":"2026-10-20T16:00","end_time":"2026-10-20T17:00"}),
],
}

QUESTIONS = {
"search_documents": ("请提供文档标题或检索关键词。", ["query"]),
"calculate": ("请提供需要计算的完整算式。", ["expression"]),
"get_record": ("请提供这条记录的编号。", ["record_id"]),
"aggregate_dataset": ("请说明要统计哪个数据集。", ["dataset"]),
"aggregate_operation": ("请说明需要哪种统计方式。", ["operation"]),
"create_event": ("请提供这项日程的结束时间。", ["end_time"]),
"check_availability": ("请提供明确的开始和结束时间。", ["start_time", "end_time"]),
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    if any((OUT / f"{s}_v2.json").exists() for s in ("train", "val")):
        raise RuntimeError("v2 训练数据已存在，拒绝覆盖")
    docs = gen.load_tool_docs()
    ctx = audit.base_ctx()
    additions = {"train": [], "val": []}
    seq = 0
    for family, pairs in PAIRS.items():
        tool = "aggregate_data" if family.startswith("aggregate_") else family
        q, fields = QUESTIONS[family]
        for i, (missing, complete, args) in enumerate(pairs):
            split = "val" if i == 5 else "train"
            for is_complete, request in ((False, missing), (True, complete)):
                seq += 1
                case = {"id": f"V2-{seq:03d}", "user_request": request,
                    "available_tools": [tool], "context": {"now": NOW, "user_confirmed_write": False},
                    "category": "normal" if is_complete else "missing_param"}
                if is_complete:
                    case["expected_decision"] = {"action": "call", "tool": tool, "arguments": args}
                    if tool in ("search_documents", "create_event"):
                        key = "query" if tool == "search_documents" else "title"
                        case["acceptable_text_arguments"] = {key: [args[key]]}
                else:
                    case["expected_decision"] = {"action": "clarify", "question": q}
                    case["clarify_target"] = {"tool": tool, "fields": fields}
                verdict, notes = score.score_case_decision(case, case["expected_decision"], ctx)
                if verdict != "correct":
                    raise RuntimeError(f"无效标准答案 {case['id']} {verdict} {notes}")
                additions[split].append(case)
    # Fail on exact or near-duplicate requests across splits, frozen sets and prior acceptance.
    compared = []
    for name in ("train_v0", "val_v0", "test_v0", "test_unseen_v0", "acceptance_v1", "acceptance_v2b", "acceptance_v3"):
        compared += json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))["samples"]
    for case in additions["train"] + additions["val"]:
        left = audit.bigrams(case["user_request"])
        opposite = additions["val"] if case in additions["train"] else additions["train"]
        for other in compared + opposite:
            if audit.jaccard(left, audit.bigrams(other["user_request"])) >= audit.CROSS_SPLIT_SIMILARITY:
                raise RuntimeError(f"请求近重复 {case['id']} {other['id']}")
    for split in ("train", "val"):
        base = json.loads((OUT / f"{split}_v0.json").read_text(encoding="utf-8"))
        samples = base["samples"] + additions[split]
        target = OUT / f"{split}_v2.json"
        target.write_text(json.dumps({"version": "dataset_v2", "split": split,
            "description": "v0 + 缺参与齐全参数成对扩充，v3 验收集未参与训练", "samples": samples},
            ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (OUT / f"{split}_v2.input.jsonl").open("w", encoding="utf-8") as f:
            for case in samples:
                f.write(json.dumps({"id": case["id"], "input": gen.render_input(case, docs)}, ensure_ascii=False) + "\n")
        print(split, len(samples), "新增", len(additions[split]))
    (OUT / "build_manifest_v2.json").write_text(json.dumps({"version": "dataset_v2",
        "train_sha256": sha(OUT / "train_v2.json"), "val_sha256": sha(OUT / "val_v2.json"),
        "acceptance_v3_sha256": sha(OUT / "acceptance_v3.json")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if __name__ == "__main__": main()
