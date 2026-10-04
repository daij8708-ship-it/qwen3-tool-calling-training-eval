"""Conservative server-side correction for explicit missing fields and authority limits.

The policy can only change a model proposal to clarify or refuse. It never
manufactures a tool call or a write confirmation. This keeps model and system
scores separately attributable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RECORD_ID = re.compile(r"\b(?:DD|FP|HT|SH)\d{6}\b", re.IGNORECASE)
CLOCK = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):[0-5]\d(?!\d)")
UNIT = re.compile(r"(?<![a-z])(?:km|cm|mm|kg|g|m|t)(?![a-z])", re.IGNORECASE)

CONFIRM_BYPASS = re.compile(
    r"跳过.{0,5}确认|绕过.{0,5}确认|略过.{0,5}确认|省掉.{0,5}确认|"
    r"不必.{0,5}确认|无需.{0,5}确认|无须.{0,5}确认|不要.{0,5}确认|"
    r"别.{0,5}确认|不用.{0,5}确认|不要.{0,5}二次确认|关闭.{0,5}确认|"
    r"已经.{0,8}确认|确认过了|替.{0,5}确认|口头代替.{0,5}确认|"
    r"没有.{0,5}确认.{0,8}(保存|创建|写入)|别再确认"
)
RECORD_MUTATION = re.compile(
    r"删除|移除|清空|覆盖|改成|改为|改掉|改写|篡改|调高|调低|调小|"
    r"修改.{0,12}(金额|编号|内容|数量|记录)|更新.{0,12}(金额|编号|内容|数量|记录)"
)
DATASET = re.compile(r"订单|发票|合同|发货单|发货记录")
KNOWN_OPERATION = re.compile(r"求和|总和|之和|合计|总计|加起来|加总|总额|均值|平均|计数|多少条|几条|条数|总共有")
SORT_ORDER = re.compile(r"升序|降序|从小到大|从大到小|从低到高|从高到低|由小到大|由大到小|由低到高|由高到低")


@dataclass(frozen=True)
class PolicyResult:
    decision: dict | None
    code: str | None


def _clarify(question: str, code: str) -> PolicyResult:
    return PolicyResult({"action": "clarify", "question": question}, code)


def _refuse(reason: str, code: str) -> PolicyResult:
    return PolicyResult({"action": "refuse", "reason": reason}, code)


def apply_policy(user_request: str, available_tools: list[str], model_decision: dict | None,
                 context: dict | None = None) -> PolicyResult:
    """Guard clear cases only; abstain otherwise and return the model proposal."""
    request = user_request.strip()
    available = set(available_tools)
    proposed_tool = model_decision.get("tool") if isinstance(model_decision, dict) else None
    context = context or {}
    history = context.get("history")
    history_text = str(history) if history is not None else ""
    combined = request + " " + history_text

    # The server cannot treat a user's claim of confirmation as confirmation.
    calendar_intent = bool(re.search(r"日历|日程|会议|活动|培训|评审|复盘|安排|创建|新建", request))
    if "create_event" in available and calendar_intent and CONFIRM_BYPASS.search(request):
        return _refuse("不能跳过服务端确认直接写入日历。", "confirmation_bypass")

    record_intent = bool(re.search(r"订单|发票|合同|发货单|记录", request))
    if "get_record" in available and record_intent and RECORD_MUTATION.search(request):
        return _refuse("当前工具只能读取记录，没有权限修改或删除记录。", "record_mutation")

    if "export_excel" not in available and re.search(r"导出.{0,12}Excel|Excel.{0,12}导出", request, re.IGNORECASE):
        return _refuse("当前工具不支持导出 Excel。", "unavailable_export")

    if "aggregate_data" in available and re.search(r"员工.{0,8}(薪资|工资|绩效|住址|体检)|医院病历", request):
        return _refuse("当前工具不支持该数据集。", "unsupported_dataset")
    if "aggregate_data" in available and re.search(r"标准差|方差|百分位|中位数", request):
        return _refuse("当前聚合工具不支持该统计方式。", "unsupported_operation")
    if "get_record" in available and re.search(r"员工.{0,8}(档案|体检|考勤)|人事档案|医院病历", request):
        return _refuse("当前工具不支持这种记录类型。", "unsupported_record_type")

    if "unit_convert" in available and re.search(r"换算|转换|换个单位|换成|转成|折合", request):
        if len(UNIT.findall(request)) < 2:
            return _clarify("请说明要换算成哪个目标单位。", "missing_to_unit")
    if "sort_records" in available and re.search(r"排序|排列|排个|排成|整理|排列表|排先后", request):
        if not SORT_ORDER.search(request):
            return _clarify("请说明按升序还是降序排序。", "missing_sort_order")

    # Known tools: detect missing information before accepting fabricated args.
    if "aggregate_data" in available and (
        proposed_tool == "aggregate_data" or
        (re.search(r"金额|记录数|统计|汇总|平均|均值|总和|几条|多少条", request) and
         re.search(r"数据|业务|订单|发票|合同|发货|金额|记录", request))
    ):
        if not DATASET.search(request):
            return _clarify("请说明要统计哪个数据集。", "missing_dataset")
        if not KNOWN_OPERATION.search(request):
            return _clarify("请说明需要哪种统计方式，求和还是均值？", "missing_operation")

    if "create_event" in available and (proposed_tool == "create_event" or
        (calendar_intent and re.search(r"创建|新增|安排|添加|新建|写入", request))):
        if len(CLOCK.findall(request)) == 1 and not re.search(r"持续|时长|小时|分钟", request):
            return _clarify("请提供日程的结束时间。", "missing_event_end")

    if "check_availability" in available and (proposed_tool == "check_availability" or
        re.search(r"有空|空闲|空档|能不能约|能否安排", request)):
        if len(CLOCK.findall(request)) < 2:
            return _clarify("请提供明确的开始和结束时间。", "missing_availability_range")

    if "get_record" in available and (proposed_tool == "get_record" or
        (record_intent and re.search(r"查询|查看|读取|打开|调出|详情|给我看|检索", request))):
        if not RECORD_ID.search(combined):
            return _clarify("请提供记录编号。", "missing_record_id")

    if "calculate" in available and (proposed_tool == "calculate" or
        (re.search(r"算式|公式|算术|计算|算个数|运算", request) and not DATASET.search(request))):
        if not re.search(r"\d", combined):
            return _clarify("请提供需要计算的完整算式。", "missing_expression")

    if "search_documents" in available and (proposed_tool == "search_documents" or
        re.search(r"文档|文件|资料库|知识库|政策", request)):
        if re.search(r"那份|那篇|那个|刚才|前面|没说名字|未给|相关|某个", request) and not re.search(r"[\u4e00-\u9fff]{2,}(?:制度|指南|手册|流程|办法|规范)", request):
            return _clarify("请提供文档标题或检索关键词。", "missing_query")

    return PolicyResult(model_decision, None)
