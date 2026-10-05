"""Build a deterministic, route-only 200-case challenge set.

Keep this file and its output separate from training data. Each expected route
is explicit here so the test is auditable without any model-generated labels.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "agent_route_pairwise_v4.json"
rows: list[dict] = []


def add(query: str, routes: list[str], category: str) -> None:
    rows.append({"id": f"R{len(rows) + 1:03}", "query": query,
                 "expected_routes": routes, "category": category,
                 "writes_data": False})


single = {
    "technical": (
        ["无线鼠标断断续续", "外接显示器没有信号", "笔记本风扇一直高速转", "打印机总显示脱机", "电脑开机后黑屏"],
        ["应该怎么排查？", "请列出优先检查的地方。", "有哪些常见原因？", "给我一个处理顺序。", "先从哪里检查比较合适？"],
    ),
    "realtime": (
        ["北京今天的空气质量", "上海明天的降雨概率", "本周新能源汽车新闻", "今天的黄金价格", "昨晚欧冠比赛结果"],
        ["请帮我查一下。", "有最新数据吗？", "请按目前的信息核实。", "现在能查到什么？", "请给出最新情况。"],
    ),
    "service": (
        ["杭州东站附近的维修网点", "南京南站到夫子庙的路线", "武汉光谷附近的服务中心", "广州白云机场到广州塔的路线", "苏州工业园区的售后门店"],
        ["帮我找一下。", "请告诉我怎么过去。", "请查位置和交通方式。", "能给出导航信息吗？", "请帮我规划到达方式。"],
    ),
    "ticket": (
        ["我的电脑报修工单", "上周提交的售后申请", "尚未结案的维修单", "刚提交的客服工单", "名下的设备故障单"],
        ["现在处理到哪一步？", "请查一下负责人员。", "可以补充故障描述吗？", "请确认是否已经受理。", "请转人工继续处理。"],
    ),
    "general": (
        ["解释什么是边际成本", "给新人写一句欢迎语", "把这段通知改得简洁", "说明番茄工作法", "写一段面试自我介绍"],
        ["用通俗的话说。", "控制在两句话内。", "再举一个例子。", "语气正式一些。", "请用中文完成。"],
    ),
}

for route, (subjects, suffixes) in single.items():
    for subject in subjects:
        for suffix in suffixes:
            add(f"{subject}，{suffix}", [route], route)

# Each item has two independent, clearly different specialist requests.
cross_pairs = [
    ("technical", "realtime", "帮我排查蓝牙耳机连接失败", "查一下今天深圳的气温"),
    ("technical", "service", "说明电脑频繁重启如何检查", "找北京西站附近的维修网点"),
    ("technical", "ticket", "解释打印机卡纸怎么处理", "查看我的报修工单状态"),
    ("technical", "general", "排查笔记本无法充电", "把我的申请说明写得更礼貌"),
    ("realtime", "service", "查今天南京的天气", "规划到南京南站的路线"),
    ("realtime", "ticket", "核实本周新能源汽车新闻", "查看我的售后单是否受理"),
    ("realtime", "general", "查最新黄金报价", "解释什么是机会成本"),
    ("service", "ticket", "找杭州滨江的维修门店", "确认我的维修工单进度"),
    ("service", "general", "导航到上海虹桥站", "帮我写一句生日祝福"),
    ("ticket", "general", "查询我的客服工单", "把这段自我介绍改得简洁"),
]
connectors = ["，然后", "，还要", "，另外", "，同时", "，接着"]
for a, b, left, right in cross_pairs:
    for connector in connectors:
        add(f"{left}{connector}{right}。", sorted([a, b]), "multi_intent")

same_agent = {
    "technical": [
        "先检查无线网卡状态，再告诉我网络断开的排查顺序。",
        "先看打印机墨盒，再看看为什么打印颜色偏淡。",
        "先说明蓝屏代码含义，再列出同一故障的修复办法。",
        "先查电脑为什么过热，再说说如何清理散热口。",
        "先判断触控板失灵原因，再告诉我如何恢复设置。",
    ],
    "realtime": [
        "先核实本月平板行业动态，再核实本月存储器行业动态。",
        "先查北京今天的气温，再查天津今天的气温。",
        "先核实今天的金价，再核实今天的银价。",
        "先看本周汽车新闻，再看本周能源新闻。",
        "先查昨晚甲队比分，再查昨晚乙队比分。",
    ],
    "service": [
        "先找南京南站附近的服务店，再给我去那里的路线。",
        "先定位苏州的售后中心，再规划从火车站过去的路线。",
        "先找机场附近的维修点，再导航到其中最近的一家。",
        "先查杭州东站的位置，再告诉我怎么到西湖。",
        "先找武汉光谷的服务网点，再找武昌站附近的网点。",
    ],
    "ticket": [
        "先查我的报修单状态，再看最后一次处理记录。",
        "先确认售后申请是否受理，再核对负责客服。",
        "先查询工单编号，再补充设备的故障描述。",
        "先查看维修单进度，再确认预计完成时间。",
        "先看两张工单的状态，再比较各自的处理时间。",
    ],
    "general": [
        "先解释复利概念，再举一个储蓄例子。",
        "先写一句感谢语，再把它改成正式语气。",
        "先把这句话译成英文，再缩短译文。",
        "先概括这段文字，再拟一个标题。",
        "先给我一份学习计划，再把它改成表格提纲。",
    ],
}
for route, queries in same_agent.items():
    for query in queries:
        add(query, [route], "same_agent_steps")

assert len(rows) == 200
assert len({x["query"] for x in rows}) == len(rows)
OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(OUT)
