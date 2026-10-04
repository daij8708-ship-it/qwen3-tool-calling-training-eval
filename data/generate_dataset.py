#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按模板组生成第 3 步的中文工具调用数据集(只用标准库)。

顺序是硬性的:先有 data/templates/*.json 里的骨架与集合划分,再由本脚本产出样本。
一条样本 = 该组的一个说法;槽位的 surface 填进请求原文、value 填进期望入参,两处同源,
避免文本与答案各写一遍导致不一致。相对时间只以 context.now 为基准解析,
解析结果不晚于 now 时直接报错,不允许悄悄产出不可执行的期望调用。

    python data/generate_dataset.py            # 生成四个集合 + 模型可见输入投影
    python data/generate_dataset.py --summary  # 只统计不写文件
    python data/generate_dataset.py --print-id DS-0001-search-doc-by-title
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "eval", ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import validate_contracts as contracts  # noqa: E402  复用第 1 步的契约加载与紧凑视图渲染

DESIGN_DIR = ROOT / "data" / "templates"
DATASET_DIR = ROOT / "data" / "datasets"
UNSEEN_TOOL_DIR = ROOT / "data" / "unseen_tools"

DESIGN_FILES = ["template_groups.json", "groups_clarify.json", "groups_refuse.json", "groups_unseen.json"]
SPLIT_ORDER = ["train", "val", "test", "unseen_test"]
SPLIT_FILES = {
    "train": "train_v0.json",
    "val": "val_v0.json",
    "test": "test_v0.json",
    "unseen_test": "test_unseen_v0.json",
}
SPLIT_SOURCE = {"train": "generated_v0", "val": "generated_v0", "test": "generated_v0", "unseen_test": "unseen_v0"}
SPLIT_ID = {"train": ("DS", 1, 4), "val": ("DS", 2001, 4), "test": ("DS", 4001, 4), "unseen_test": ("UN", 1, 3)}
UNSEEN_TOOLS = ["unit_convert", "sort_records"]
PROMPT_HEAD = "你是中文工具调用助手。只输出一个 JSON 决策,字段仅限 action/tool/arguments/question/reason。"
PROMPT_TAIL = "当前时间与写操作确认状态由服务端掌握;模型输出不构成确认凭据。"
START_KEYS = ("start_time", "end_time")

DESCRIPTIONS = {
    "train": "训练集:按模板组划分的中文工具调用样本,同一骨架及其改写不跨集合;不含未见工具。",
    "val": "验证集:训练过程中间评估用,模板组与训练、测试互斥;不含未见工具。",
    "test": "测试集:冻结评估用;第 1 步 24 条手工案例对应的骨架固定计入本集合,拒绝类占比不低于 20%。",
    "unseen_test": "未见工具测试集:只提供 unit_convert 与 sort_records 的说明与 Schema,单独统计,不并入六工具总分。",
}


class DesignError(RuntimeError):
    """模板登记表自身不自洽,与样本内容无关。"""


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def parse(iso: str) -> datetime:
    return datetime.strptime(iso, contracts.TIME_FORMAT)


def fmt(moment: datetime) -> str:
    return moment.strftime(contracts.TIME_FORMAT)


def load_design():
    """读四份登记表:组号不得重复,划分必须已经写在组上。"""
    groups = []
    seed = None
    handwritten = []
    now_pool = []
    suffixes = []
    for name in DESIGN_FILES:
        path = DESIGN_DIR / name
        if not path.exists():
            raise DesignError(f"缺少模板登记表 {path.name}")
        payload = load_json(path)
        if name == "template_groups.json":
            seed = payload.get("seed")
            handwritten = payload.get("handwritten_groups") or []
            now_pool = payload.get("now_pool_default") or []
            suffixes = payload.get("generic_suffixes") or []
        groups.extend(payload.get("groups") or [])
    if seed is None:
        raise DesignError("主模板登记表缺少 seed")
    seen = {}
    for group in groups:
        gid = group.get("id")
        if gid in seen:
            raise DesignError(f"模板组编号重复: {gid}({seen[gid]} 与 {group.get('slug')})")
        seen[gid] = group.get("slug")
        for key in ("id", "slug", "label", "action", "category", "tags", "available_tools_pool", "phrasings", "rationale", "split"):
            if not group.get(key):
                raise DesignError(f"模板组 {gid} 缺少字段 {key}")
        if group["split"] not in SPLIT_FILES:
            raise DesignError(f"模板组 {gid} 的 split 非法: {group['split']}")
        if group["action"] == "call" and not group.get("arguments"):
            raise DesignError(f"模板组 {gid}: call 组必须有 arguments 模板")
        if group["action"] == "clarify" and not group.get("questions"):
            raise DesignError(f"模板组 {gid}: clarify 组必须有 questions")
        if group["action"] == "refuse" and not group.get("reasons"):
            raise DesignError(f"模板组 {gid}: refuse 组必须有 reasons")
    for entry in handwritten:
        if entry["id"] in seen:
            raise DesignError(f"手工模板组与生成组编号冲突: {entry['id']}")
    return groups, handwritten, seed, now_pool, suffixes


def option_rows(raw):
    """把槽位统一成 [(surface, value, extras)]。"""
    rows = []
    if not isinstance(raw, list) or not raw:
        raise DesignError(f"槽位必须是非空列表: {raw}")
    for item in raw:
        if isinstance(item, dict):
            if "surface" not in item:
                raise DesignError(f"map 槽位选项必须含 surface: {item}")
            extras = {key: value for key, value in item.items() if key not in ("surface", "value")}
            rows.append((item["surface"], item.get("value", item["surface"]), extras))
        else:
            rows.append((str(item), item, {}))
    return rows


def resolve_at(now: datetime, spec, where: str) -> str:
    """把相对时间说明解析成 YYYY-MM-DDTHH:MM,唯一基准是 context.now。"""
    if not isinstance(spec, dict):
        raise DesignError(f"{where}: 时间槽位的 value 必须是对象")
    clock = spec.get("clock", "00:00")
    if spec.get("abs"):
        return spec["abs"]
    if "month" in spec and "edge" in spec:
        year = spec.get("year", now.year)
        month = spec["month"]
        if spec["edge"] == "start":
            day, clock = date(year, month, 1), "00:00"
        elif spec["edge"] == "end":
            rollover = date(year + (month == 12), 1 if month == 12 else month + 1, 1)
            day, clock = rollover - timedelta(days=1), "23:59"
        else:
            raise DesignError(f"{where}: edge 只支持 start/end")
    elif "month" in spec:
        day = date(spec.get("year", now.year), spec["month"], spec["day"])
    elif "weekday" in spec:
        monday_this = now.date() - timedelta(days=now.weekday())
        day = monday_this + timedelta(days=7 * spec.get("weeks", 0) + spec["weekday"])
    elif "days" in spec:
        day = now.date() + timedelta(days=spec["days"])
    else:
        raise DesignError(f"{where}: 时间槽位缺 abs/month/weekday/days")
    hour, minute = (int(part) for part in clock.split(":"))
    return datetime(day.year, day.month, day.day, hour, minute).strftime(contracts.TIME_FORMAT)


CJK = re.compile(r"[一-鿿]")
ASCII_START = re.compile(r"[0-9A-Za-z$]")


def space_join(left: str, right: str) -> str:
    """中文与半角数字或字母相邻时补一个空格,避免"统计2026 年"这类粘连;归一化比较会去空格,不影响判重。"""
    if left and right and CJK.search(left[-1]) and ASCII_START.match(right[0]):
        return left + " " + right
    if left and right and ASCII_START.match(left[-1]) and CJK.search(right[0]):
        return left + " " + right
    return left + right


def fill(template: str, table: dict, where: str) -> str:
    out = template
    while "{" in out:
        start = out.index("{")
        end = out.find("}", start)
        if end < 0:
            raise DesignError(f"{where}: 占位符未闭合 {template}")
        key = out[start + 1:end]
        if key not in table:
            raise DesignError(f"{where}: 未知槽位 {key}")
        out = space_join(out[:start], str(table[key])) + out[end + 1:]
    return out


CJK_EDGE = re.compile(r"([一-鿿])([0-9A-Za-z])")
ASCII_EDGE = re.compile(r"([0-9A-Za-z])([一-鿿])")


def space_text(text: str) -> str:
    """中文与半角数字或字母相邻时补空格,消除"统计2026 年""15:30到"这类粘连;归一化比较会去空格,不影响判重。"""
    return ASCII_EDGE.sub(r"\g<1> \g<2>", CJK_EDGE.sub(r"\g<1> \g<2>", text))


def auto_alt(surface: str, suffixes) -> list:
    """登记写法:原写法 + 去掉结尾通用词的写法(去空格后仍是请求原文的子串)。"""
    out = [surface]
    for suffix in suffixes:
        if surface.endswith(suffix) and len(surface) - len(suffix) >= 2:
            trimmed = surface[: -len(suffix)]
            if trimmed != surface:
                out.append(trimmed)
            break
    return out


class Choice:
    """一条样本的槽位取值:文本用 surface,入参用 value,两处同源。"""

    def __init__(self, group, index, now):
        self.where = f"{group['id']}#{index + 1}"
        self.now = now
        self.surfaces = {}
        self.values = {}
        self.notes = []
        self.conversation = None
        slots = group.get("slots") or {}
        aligned = group.get("align") or []
        lengths = {name: len(option_rows(rows)) for name, rows in slots.items()}
        for name in aligned:
            if lengths.get(name) != lengths.get(aligned[0]):
                raise DesignError(f"{self.where}: align 组 {aligned} 取值数不同,无法同下标配对")
        for name, rows_raw in slots.items():
            rows = option_rows(rows_raw)
            surface, value, extras = rows[index % len(rows)]
            self.surfaces[name] = surface
            if isinstance(value, dict) and "$at" in value:
                resolved = resolve_at(now, value["$at"], f"{self.where}/{name}")
                self.values[name] = resolved
                self.notes.append(f"{name}「{surface}」→{resolved}")
            else:
                self.values[name] = value
                self.notes.append(f"{name}「{surface}」" + (f"→{value}" if value != surface else ""))
                if isinstance(value, dict):
                    for key, sub in value.items():
                        self.values[f"{name}.{key}"] = sub
            if extras.get("conversation"):
                self.conversation = extras["conversation"]

    def text(self, template: str) -> str:
        return space_text(fill(template, self.surfaces, f"{self.where} 文案"))

    def value_of(self, template: str):
        stripped = template.strip()
        if stripped.startswith("{") and stripped.endswith("}") and "{" not in stripped[1:-1]:
            key = stripped[1:-1]
            if key not in self.values:
                raise DesignError(f"{self.where}: 未知槽位 {key}")
            return self.values[key]
        return fill(template, self.values, f"{self.where} 入参")

    def arguments(self, template) -> dict:
        out = {}
        for key, value in template.items():
            if isinstance(value, str):
                out[key] = self.value_of(value)
            elif isinstance(value, dict):
                out[key] = self.arguments(value)
            elif isinstance(value, list):
                out[key] = [self.value_of(item) if isinstance(item, str) else item for item in value]
            else:
                out[key] = value
        return out


def decision_of(group, choice: Choice, index: int):
    action = group["action"]
    if action == "call":
        return {"action": "call", "tool": group["tool"], "arguments": choice.arguments(group["arguments"])}
    pool = group["questions"] if action == "clarify" else group["reasons"]
    text = choice.text(pool[index % len(pool)])
    return {"action": action, "question" if action == "clarify" else "reason": text}


def generate(groups, seed, now_pool_default, suffixes):
    buckets = {split: [] for split in SPLIT_ORDER}
    counters = {split: SPLIT_ID[split][1] for split in SPLIT_ORDER}
    ordered = sorted(groups, key=lambda g: (SPLIT_ORDER.index(g["split"]), g["id"]))
    for group in ordered:
        pool = group.get("now") or now_pool_default
        if not pool:
            raise DesignError(f"{group['id']}: 没有可用的 now 基准")
        phrasings = group["phrasings"]
        for index in range(len(phrasings)):
            now = parse(pool[index % len(pool)])
            choice = Choice(group, index, now)
            request = choice.text(phrasings[index])
            if len(request) > 200:
                raise DesignError(f"{group['id']}#{index + 1}: 请求原文 {len(request)} 字符超过 200 上限")
            prefix, _, width = SPLIT_ID[group["split"]]
            sample_id = f"{prefix}-{counters[group['split']]:0{width}d}-{group['slug']}"
            counters[group["split"]] += 1
            decision = decision_of(group, choice, index)
            if group["action"] == "call":
                for key in START_KEYS:
                    value = decision["arguments"].get(key)
                    if isinstance(value, str) and parse(value) <= now:
                        raise DesignError(f"{group['id']}#{index + 1}: 期望调用的 {key}={value} 不晚于 now,该样本不可执行")
            record = {
                "id": sample_id,
                "user_request": request,
                "available_tools": group["available_tools_pool"][index % len(group["available_tools_pool"])],
                "context": {"now": fmt(now), "user_confirmed_write": group.get("user_confirmed_write", False)},
                "expected_decision": decision,
                "rationale": choice.text(group["rationale"]),
                "tags": list(group["tags"]),
                "template_group": group["id"],
                "source": SPLIT_SOURCE[group["split"]],
                "split": group["split"],
                "category": group["category"],
                "generation_rule": (
                    f"模板组 {group['id']}({group['label']})第 {index + 1} 号说法;now={fmt(now)};" + ";".join(choice.notes)
                )[:300],
                "seed": seed,
            }
            if choice.conversation:
                record["context"]["conversation"] = choice.conversation
            for key, extra in (
                ("clarify_target", group.get("clarify_target") if group["action"] == "clarify" else None),
                ("refuse_code", group.get("refuse_code") if group["action"] == "refuse" else None),
                ("blocking_tools", group.get("blocking_tools") if group.get("refuse_code") == "unavailable_tool" else None),
                ("post_condition", group.get("post_condition") if group["action"] == "call" else None),
            ):
                if extra is not None:
                    record[key] = extra
            if group["action"] == "call":
                registry = text_registry(group, choice, suffixes)
                if registry:
                    record["acceptable_text_arguments"] = registry
            buckets[group["split"]].append(record)
    return buckets


def text_registry(group, choice: Choice, suffixes):
    out = {}
    for key, entries in (group.get("text_registry") or {}).items():
        values = []
        for entry in entries:
            surface = choice.text(entry)
            for alt in auto_alt(surface, suffixes):
                if alt not in values:
                    values.append(alt)
        if values:
            out[key] = values
    return out


def load_tool_docs():
    docs = {}
    for name in contracts.EXPECTED_TOOLS:
        errors, doc = contracts.check_one_tool_at(ROOT / "tools" / "schemas" / f"{name}.json")
        if errors:
            raise DesignError(f"工具契约 {name} 未通过检查: {errors}")
        docs[name] = doc
    for name in UNSEEN_TOOLS:
        errors, doc = contracts.check_one_tool_at(UNSEEN_TOOL_DIR / f"{name}.json")
        if errors:
            raise DesignError(f"未见工具契约 {name} 未通过检查: {errors}")
        docs[name] = doc
    return docs


def render_input(sample, docs):
    """模型可见输入:工具块 + 当前时间 + 可选历史 + 请求。标注侧元信息一律不进这里。"""
    lines = [PROMPT_HEAD, "本次可用工具:"]
    lines.append(contracts.render_compact_block([docs[name] for name in sample["available_tools"]]))
    context = sample["context"]
    lines.append(f"服务端当前时间: {context['now']}")
    for turn in context.get("conversation") or []:
        lines.append(f"历史对话({turn['role']}): {turn['content']}")
    lines.append(f"用户请求: {sample['user_request']}")
    lines.append(PROMPT_TAIL)
    return "\n".join(lines)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="按模板组生成训练、验证、测试与未见工具四个集合")
    parser.add_argument("--summary", action="store_true", help="只统计不写文件")
    parser.add_argument("--print-id", help="打印指定样本(含模型可见输入)后退出")
    args = parser.parse_args(argv)

    groups, handwritten, seed, now_pool, suffixes = load_design()
    docs = load_tool_docs()
    buckets = generate(groups, seed, now_pool, suffixes)

    if args.print_id:
        for samples in buckets.values():
            for sample in samples:
                if sample["id"] == args.print_id:
                    print(json.dumps(sample, ensure_ascii=False, indent=2))
                    print("-" * 66)
                    print(render_input(sample, docs))
                    return 0
        print(f"未找到样本 {args.print_id}")
        return 1

    if not args.summary:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for split in SPLIT_ORDER:
        samples = buckets[split]
        total += len(samples)
        filename = SPLIT_FILES[split]
        payload = {
            "version": filename.rsplit(".", 1)[0],
            "seed": seed,
            "generated_from": [f"data/templates/{name}" for name in DESIGN_FILES],
            "description": DESCRIPTIONS[split],
            "samples": samples,
        }
        if not args.summary:
            (DATASET_DIR / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (DATASET_DIR / filename.replace(".json", ".input.jsonl")).write_text(
                "\n".join(
                    json.dumps({"id": sample["id"], "input": render_input(sample, docs)}, ensure_ascii=False)
                    for sample in samples
                ) + "\n",
                encoding="utf-8",
            )
        groups_in = sorted({sample["template_group"] for sample in samples})
        actions = {}
        for sample in samples:
            action = sample["expected_decision"]["action"]
            actions[action] = actions.get(action, 0) + 1
        print(
            f"{split:11s} 样本 {len(samples):4d}  模板组 {len(groups_in):3d}  "
            + " ".join(f"{action}={actions.get(action, 0)}" for action in ("call", "clarify", "refuse"))
        )
    if not args.summary:
        (DATASET_DIR / "build_manifest.json").write_text(
            json.dumps(
                {
                    "version": "dataset_v0",
                    "seed": seed,
                    "design_files": [f"data/templates/{name}" for name in DESIGN_FILES],
                    "samples_per_group_rule": "每组样本数等于该组的说法数;槽位只让同一说法里的用词与取值变化,不按换编号或换日期扩量",
                    "counts": {split: len(buckets[split]) for split in SPLIT_ORDER},
                    "groups": {split: sorted({s["template_group"] for s in buckets[split]}) for split in SPLIT_ORDER},
                    "handwritten_groups_in_test": [entry["id"] for entry in handwritten],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    print(f"合计生成 {total} 条;第 1 步 24 条手工案例按骨架计入 test 集合(见 data/cases/handwritten_v0.json)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DesignError as exc:
        print(f"模板登记表不自洽: {exc}")
        sys.exit(1)
