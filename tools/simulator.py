#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""六个工具的本地模拟执行。

固定合成数据 + 确定性算法:不读真实系统时钟(当前时间由调用方从服务端上下文传入)、
不用随机数、不联网、不调用大模型。calculate 用手写的递归下降解析器求值,不用 eval 或 exec。

    from simulator import Store, execute_tool

返回值的 status 与 error_code 必须在 tools/result_contract.json 里登记过,
否则抛 ContractViolation —— 防止实现随手造出前端无法处理的码。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "tools" / "fixtures"
RESULT_CONTRACT_FILE = ROOT / "tools" / "result_contract.json"

TIME_FORMAT = "%Y-%m-%dT%H:%M"
EXPRESSION_ALLOWED = re.compile(r"^[0-9+\-*/(). %]+$")
NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?|\.\d+")
MAX_EXPRESSION_LENGTH = 80
MAX_OPERAND = Decimal("1000000000000")
MAX_RESULT = Decimal("1000000000000")
MAX_CREATE_DURATION_MIN = 480
CENTS = Decimal("0.01")

def metric_fields(store):
    """聚合字段到记录字段的映射写在 result_contract 里,实现不再自带一份。"""
    return store.contract["tools"]["aggregate_data"]["metric_field"]


class ContractViolation(RuntimeError):
    """工具返回了未登记的状态或错误码。"""


class ExpressionError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ToolError(Exception):
    """可预期的拒绝执行,携带 result_contract 里登记过的错误码。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def parse_time(text):
    try:
        return datetime.strptime(text, TIME_FORMAT)
    except (TypeError, ValueError):
        return None


def round2(value: Decimal) -> Decimal:
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------
# 表达式求值:tokenizer + 递归下降,十进制度量,不用 eval
# --------------------------------------------------------------------------

def tokenize_expression(text):
    if not EXPRESSION_ALLOWED.match(text):
        bad = sorted({ch for ch in text if not EXPRESSION_ALLOWED.match(ch)})
        raise ExpressionError("expression_unsupported_char", f"算式含未允许的字符 {bad}")
    tokens = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "+-*/().%":
            tokens.append(char)
            index += 1
            continue
        match = NUMBER_PATTERN.match(text, index)
        if not match:
            raise ExpressionError("expression_syntax", f"算式在第 {index} 个字符处无法解析")
        tokens.append(("number", match.group()))
        index = match.end()
    if not tokens:
        raise ExpressionError("expression_syntax", "算式为空")
    return tokens


class ExpressionParser:
    """文法: expr = term (('+'|'-') term)*;term = unary (('*'|'/') unary)*;
    unary = ('-'|'+')* atom;atom = number ('%')* | '(' expr ')'。% 是百分号后缀,不是取模。"""

    def __init__(self, tokens):
        self.tokens = tokens
        self.position = 0

    def peek(self):
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ExpressionError("expression_syntax", "算式缺少操作数或括号不匹配")
        self.position += 1
        return token

    def parse(self):
        value = self.parse_expr()
        if self.peek() is not None:
            raise ExpressionError("expression_syntax", "算式尾部有多余内容")
        return value

    def parse_expr(self):
        value = self.parse_term()
        while self.peek() in ("+", "-"):
            operator = self.take()
            right = self.parse_term()
            value = value + right if operator == "+" else value - right
        return value

    def parse_term(self):
        value = self.parse_unary()
        while self.peek() in ("*", "/"):
            operator = self.take()
            right = self.parse_unary()
            if operator == "/":
                if right == 0:
                    raise ExpressionError("division_by_zero", "算式中出现除以零")
                value = value / right
            else:
                value = value * right
        return value

    def parse_unary(self):
        signs = 0
        while self.peek() in ("+", "-"):
            if self.take() == "-":
                signs += 1
        value = self.parse_atom()
        return -value if signs % 2 else value

    def parse_atom(self):
        token = self.peek()
        if token == "(":
            self.take("(")
            value = self.parse_expr()
            self.take(")")
            return self.apply_percent(value)
        if not isinstance(token, tuple) or token[0] != "number":
            raise ExpressionError("expression_syntax", "算式缺少操作数")
        self.take()
        try:
            value = Decimal(token[1])
        except InvalidOperation as exc:
            raise ExpressionError("expression_syntax", f"数值 {token[1]} 无法解析") from exc
        if value > MAX_OPERAND:
            raise ExpressionError("value_out_of_range", f"操作数 {token[1]} 超出允许范围")
        return self.apply_percent(value)

    def apply_percent(self, value):
        while self.peek() == "%":
            self.take("%")
            value = value / Decimal(100)
        return value


def evaluate_expression(text):
    if not isinstance(text, str):
        raise ExpressionError("expression_syntax", "算式必须是字符串")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ExpressionError("value_out_of_range", f"算式长度超过 {MAX_EXPRESSION_LENGTH}")
    value = ExpressionParser(tokenize_expression(text)).parse()
    if abs(value) > MAX_RESULT:
        raise ExpressionError("value_out_of_range", "算式结果超出允许范围")
    return round2(value)


# --------------------------------------------------------------------------
# 数据与结果封装
# --------------------------------------------------------------------------

class Store:
    """一次执行会话的数据。load 每次都从磁盘重新读,测试之间互不影响。"""

    def __init__(self, documents, records, events, contract):
        self.documents = documents
        self.records = records
        self.events = events
        self.contract = contract
        self.record_type_to_dataset = contract.get("record_type_to_dataset") or {}

    @classmethod
    def load(cls):
        documents = json.loads((FIXTURE_DIR / "documents.json").read_text(encoding="utf-8"))["documents"]
        records_doc = json.loads((FIXTURE_DIR / "records.json").read_text(encoding="utf-8"))
        contract = json.loads(RESULT_CONTRACT_FILE.read_text(encoding="utf-8"))
        store = cls(
            documents,
            {key: [dict(row) for row in rows] for key, rows in records_doc["records"].items()},
            [dict(event) for event in json.loads((FIXTURE_DIR / "calendar.json").read_text(encoding="utf-8"))["events"]],
            contract,
        )
        store.record_type_to_dataset = records_doc["record_type_to_dataset"]
        return store

    def find_record(self, record_type, record_id):
        dataset = self.record_type_to_dataset.get(record_type)
        for row in self.records.get(dataset, []):
            if row["record_id"] == record_id:
                return dataset, row
        return dataset, None

    def calendar_ids(self):
        return [event["event_id"] for event in self.events]

    def next_event_id(self):
        highest = 0
        for event in self.events:
            match = re.search(r"(\d+)$", event["event_id"])
            if match:
                highest = max(highest, int(match.group(1)))
        return f"EV-{highest + 1:04d}"


def envelope(store: Store, tool: str, status: str, message: str, data=None, error_code=None):
    contract = store.contract
    spec = contract["tools"].get(tool)
    if spec is None:
        raise ContractViolation(f"{tool} 不在执行结果契约里")
    if status not in spec["statuses"]:
        raise ContractViolation(f"{tool} 返回了未登记的状态 {status}")
    if error_code is not None and error_code not in spec["error_codes"]:
        raise ContractViolation(f"{tool} 返回了未登记的错误码 {error_code}")
    if status == contract["envelope"]["error_code_required_when"] and not error_code:
        raise ContractViolation(f"{tool} 的状态 {status} 必须带 error_code")
    return {"status": status, "message": message, "data": data, "error_code": error_code}


def _overlap(first_start, first_end, second_start, second_end):
    return first_start < second_end and second_start < first_end


def conflicting_events(events, start, end):
    return [
        {"event_id": event["event_id"], "title": event["title"], "start_time": event["start_time"], "end_time": event["end_time"]}
        for event in events
        if parse_time(event["start_time"]) and parse_time(event["end_time"])
        and _overlap(start, end, parse_time(event["start_time"]), parse_time(event["end_time"]))
    ]


# --------------------------------------------------------------------------
# 六个工具
# --------------------------------------------------------------------------

def search_documents(store, arguments, now=None, confirmed=False):
    del now, confirmed
    query = arguments["query"]
    tokens = [token for token in query.split(" ") if token]
    matches = []
    for document in store.documents:
        haystack = " ".join([document["title"], document["keywords"], document["summary"]])
        if tokens and all(token in haystack for token in tokens):
            matches.append({
                "id": document["id"],
                "title": document["title"],
                "snippet": document["summary"],
                "updated_at": document["updated_at"],
            })
    return envelope(
        store,
        "search_documents",
        "ok",
        f"命中 {len(matches)} 篇文档",
        {"query": query, "total": len(matches), "matches": matches},
    )


def get_record(store, arguments, now=None, confirmed=False):
    del now, confirmed
    record_type = arguments["record_type"]
    record_id = arguments["record_id"]
    dataset, row = store.find_record(record_type, record_id)
    if row is None:
        return envelope(
            store,
            "get_record",
            "not_found",
            f"{dataset} 里不存在编号 {record_id}",
            {"record_type": record_type, "record_id": record_id},
            "record_not_found",
        )
    return envelope(
        store,
        "get_record",
        "ok",
        f"已读取 {record_id}",
        {"record_type": record_type, "record_id": record_id, "fields": dict(row)},
    )


def calculate(store, arguments, now=None, confirmed=False):
    del now, confirmed
    expression = arguments["expression"]
    try:
        value = evaluate_expression(expression)
    except ExpressionError as exc:
        return envelope(store, "calculate", "rejected", exc.message, {"expression": expression}, exc.code)
    return envelope(
        store,
        "calculate",
        "ok",
        f"算式结果为 {value}",
        {"expression": expression, "value": float(value), "value_text": str(value)},
    )


def _filtered_rows(store, dataset, filters):
    rows = store.records.get(dataset, [])
    time_range = (filters or {}).get("time_range")
    if time_range:
        start = parse_time(time_range["start"])
        end = parse_time(time_range["end"])
        rows = [row for row in rows if start and end and start <= parse_time(row["business_time"]) <= end]
    status = (filters or {}).get("status")
    if status:
        rows = [row for row in rows if row.get("status") in set(status)]
    region = (filters or {}).get("region")
    if region:
        rows = [row for row in rows if row.get("region") in set(region)]
    return rows


def aggregate_data(store, arguments, now=None, confirmed=False):
    del now, confirmed
    dataset = arguments["dataset"]
    metric = arguments["metric"]
    operation = arguments["operation"]
    filters = arguments.get("filters") or {}
    rows = _filtered_rows(store, dataset, filters)
    data = {
        "dataset": dataset,
        "metric": metric,
        "operation": operation,
        "filters": filters,
        "sample_count": len(rows),
    }
    if not rows:
        return envelope(store, "aggregate_data", "no_data", "筛选条件没有命中任何数据行", {**data, "value": None}, "no_matching_rows")
    if operation == "count":
        value = Decimal(len(rows))
        data["value"] = int(value)
        data["value_text"] = str(int(value))
        return envelope(store, "aggregate_data", "ok", f"计数结果 {int(value)}", data)
    field = metric_fields(store)[metric]
    missing = [row["record_id"] for row in rows if field not in row]
    if missing:
        raise ContractViolation(f"{dataset} 的记录缺少字段 {field}: {missing}")
    values = [Decimal(str(row[field])) for row in rows]
    if operation == "sum":
        value = sum(values)
    elif operation == "mean":
        value = sum(values) / Decimal(len(values))
    elif operation == "max":
        value = max(values)
    else:
        value = min(values)
    value = round2(value)
    data["value"] = float(value)
    data["value_text"] = str(value)
    return envelope(store, "aggregate_data", "ok", f"{dataset}.{metric} 的 {operation} 为 {value}", data)


def check_availability(store, arguments, now=None, confirmed=False):
    del confirmed
    start = parse_time(arguments["start_time"])
    end = parse_time(arguments["end_time"])
    if start is None or end is None:
        raise ContractViolation("check_availability 收到无法解析的时间")
    if now is not None and start < now:
        return envelope(
            store,
            "check_availability",
            "rejected",
            f"起始时间 {arguments['start_time']} 早于服务端当前时间,不支持回溯查询",
            {"start_time": arguments["start_time"], "end_time": arguments["end_time"]},
            "past_time_not_supported",
        )
    conflicts = conflicting_events(store.events, start, end)
    return envelope(
        store,
        "check_availability",
        "ok",
        "该时段空闲" if not conflicts else f"该时段与 {len(conflicts)} 条已有日程冲突",
        {"start_time": arguments["start_time"], "end_time": arguments["end_time"], "available": not conflicts, "conflicts": conflicts},
    )


def create_event(store, arguments, now=None, confirmed=False):
    start = parse_time(arguments["start_time"])
    end = parse_time(arguments["end_time"])
    if start is None or end is None:
        raise ContractViolation("create_event 收到无法解析的时间")
    if now is not None and start < now:
        return envelope(
            store,
            "create_event",
            "rejected",
            f"开始时间 {arguments['start_time']} 早于服务端当前时间,不创建日程",
            {"start_time": arguments["start_time"], "end_time": arguments["end_time"]},
            "past_time_not_supported",
        )
    if (end - start).total_seconds() > MAX_CREATE_DURATION_MIN * 60:
        return envelope(
            store,
            "create_event",
            "rejected",
            f"日程时长超过 {MAX_CREATE_DURATION_MIN} 分钟上限",
            {"start_time": arguments["start_time"], "end_time": arguments["end_time"]},
            "duration_limit_exceeded",
        )
    conflicts = conflicting_events(store.events, start, end)
    if conflicts:
        return envelope(
            store,
            "create_event",
            "conflict",
            f"与已有日程 {', '.join(item['event_id'] for item in conflicts)} 时间重叠,未写入",
            {"conflicts": conflicts, "written": False},
        )
    draft = {"title": arguments["title"], "start_time": arguments["start_time"], "end_time": arguments["end_time"]}
    if not confirmed:
        return envelope(
            store,
            "create_event",
            "pending_confirmation",
            "服务端确认状态为 false,日程保持待确认,未写入日历",
            {"draft": draft, "written": False, "confirmation_source": "context.user_confirmed_write", "confirmation_value": False},
        )
    event = {"event_id": store.next_event_id(), **draft, "confirmed": True, "source": "create_event"}
    store.events.append(event)
    return envelope(store, "create_event", "ok", f"已创建日程 {event['event_id']}", {"event": event, "written": True})


TOOL_IMPL = {
    "search_documents": search_documents,
    "get_record": get_record,
    "calculate": calculate,
    "aggregate_data": aggregate_data,
    "check_availability": check_availability,
    "create_event": create_event,
}


def execute_tool(store, tool, arguments, now=None, confirmed=False):
    """调用方必须先过执行闸门;这里只做数据读写,不再判断决策格式。"""
    if tool not in TOOL_IMPL:
        raise ContractViolation(f"未实现的工具 {tool}")
    return TOOL_IMPL[tool](store, arguments, now=now, confirmed=confirmed)
