#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 3 步数据审计:结构、划分、泄漏、近重复、登记表、未见工具隔离、执行可执行性与人工抽检闭环。

只调用标准库,不训练模型、不调用大模型;判据全部是确定性规则,并且直接复用
第 1 步的契约校验(eval/validate_contracts.py)与第 2 步的执行闸门(tools/executor.py)。

    python data/audit_dataset.py                     # 全量审计 + 分布报告 + 人工抽检清单
    python data/audit_dataset.py --freeze            # 审计全通过才写冻结清单
    python data/audit_dataset.py --check-frozen      # 实验前校验测试集与冻结清单一致
    python data/audit_dataset.py --similarity-top 15 # 打印跨集合最相似的样本对
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "eval", ROOT / "tools", ROOT / "data"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import generate_dataset as gen  # noqa: E402
import simulator  # noqa: E402
import validate_contracts as contracts  # noqa: E402  复用第 1 步实现,避免两套判据漂移

DATA = Path("data")
DATASET_DIR = ROOT / "data" / "datasets"
MANIFEST_FILE = DATASET_DIR / "build_manifest.json"
FROZEN_FILE = DATASET_DIR / "frozen_v0.json"
SPOT_FILE = DATASET_DIR / "spot_check_v0.md"
REVIEW_FILE = DATASET_DIR / "review_log_v0.md"
ISOLATION_FILE = ROOT / "data" / "unseen_tools" / "isolation_manifest.json"
DESIGN_DIR = ROOT / "data" / "templates"

SPLITS = ["train", "val", "test", "unseen_test"]
FILES = {
    "train": "train_v0.json",
    "val": "val_v0.json",
    "test": "test_v0.json",
    "unseen_test": "test_unseen_v0.json",
}
TARGETS = {"train": (400, 600), "val": (80, 100), "test": (120, 160), "unseen_test": (40, 60)}
CATEGORIES = contracts.CATEGORIES
MIN_PER_CATEGORY = 10
# 跨集合骨架重合阈值,依据实测分布定:140904 个跨集合样本对里实际最高 0.60(不同意图同句式),
# 而构造的"同骨架只改两字"对照达到 0.789,故取 0.78;换语序改写由字符多重集规则单独抓
# (bigram 相似度对语序不敏感,0.60 的换语序对会被它漏掉)。探测器本身由
# check_similarity_detector 的正负对照保证不空转。
CROSS_SPLIT_SIMILARITY = 0.78
SAME_SPLIT_SIMILARITY = 0.90
TIME_IN_REQUEST = re.compile(r"\d{1,2}:\d{2}")
DATE_PHRASE = re.compile(r"(大后天|今天|明天|后天|本周|下周[一二三四五六日天]|上周[一二三四五六日天]|本月|下个月|上个月|\d{4} 年 \d{1,2} 月|\d{1,2} 月 \d{1,2} 日|\d{1,2} 月)")
MONTH_PHRASE = re.compile(r"(?:年 )?(\d{1,2}) 月")
CN_WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
FROZEN_FILES = [
    "data/datasets/test_v0.json",
    "data/datasets/test_v0.input.jsonl",
    "data/datasets/test_unseen_v0.json",
    "data/datasets/test_unseen_v0.input.jsonl",
    "data/cases/handwritten_v0.json",
]
PROVENANCE_FILES = [
    "data/generate_dataset.py",
    "data/audit_dataset.py",
    "data/cases/case_schema.json",
    "data/templates/template_groups.json",
    "data/templates/groups_clarify.json",
    "data/templates/groups_refuse.json",
    "data/templates/groups_unseen.json",
    "data/unseen_tools/unit_convert.json",
    "data/unseen_tools/sort_records.json",
    "data/unseen_tools/isolation_manifest.json",
    "tools/decision_schema.json",
    "tools/result_contract.json",
    "tools/schemas/search_documents.json",
    "tools/schemas/get_record.json",
    "tools/schemas/calculate.json",
    "tools/schemas/aggregate_data.json",
    "tools/schemas/check_availability.json",
    "tools/schemas/create_event.json",
    "tools/fixtures/documents.json",
    "tools/fixtures/records.json",
    "tools/fixtures/calendar.json",
]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def err(loc, code, message):
    return (loc, code, message)


# --------------------------------------------------------------------------
# 契约上下文:六工具一套,未见工具一套(按登记表加宽枚举)
# --------------------------------------------------------------------------

def base_ctx():
    ctx = contracts.Ctx()
    for name in contracts.EXPECTED_TOOLS:
        errors, doc = contracts.check_one_tool(name)
        if errors:
            raise RuntimeError(f"工具契约 {name} 未通过第 1 步校验: {errors}")
        ctx.tools[name] = doc["input_schema"]
        ctx.tool_docs[name] = doc
    errors, decision = contracts.check_decision_contract(ctx)
    if errors:
        raise RuntimeError(f"决策契约未通过第 1 步校验: {errors}")
    ctx.decision_schema = decision
    errors, case = contracts.check_case_schema(ctx)
    if errors:
        raise RuntimeError(f"案例 Schema 未通过第 1 步校验: {errors}")
    ctx.case_schema = case
    return ctx


def apply_delta(doc, specs):
    """按登记表把枚举加宽;只允许 add / add_entries 两种操作。"""
    out = json.loads(json.dumps(doc))
    for spec in specs:
        node = out
        parts = spec["path"].split(".")
        for part in parts[:-1]:
            node = node[part]
        leaf = node[parts[-1]]
        if spec["operation"] == "add":
            leaf.extend([item for item in spec["add"] if item not in leaf])
        elif spec["operation"] == "add_entries":
            for key, values in spec["add_entries"].items():
                leaf.setdefault(key, [])
                leaf[key] = values
        else:
            raise RuntimeError(f"未登记的差异操作: {spec['operation']}")
    return out


def diff_paths(a, b, path=""):
    """列出两份 Schema 的差异位置,用于证明派生契约只改了登记表里那几条。"""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append(f"{path}/{key}")
            else:
                out.extend(diff_paths(a[key], b[key], f"{path}/{key}"))
    elif isinstance(a, list) and isinstance(b, list):
        if [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in a] != [
            json.dumps(item, sort_keys=True, ensure_ascii=False) for item in b
        ]:
            out.append(path)
    elif a != b:
        out.append(path)
    return out


def unseen_ctx(base, manifest):
    """未见工具用的派生契约:决策与记录 Schema 只加宽工具名与参数名枚举。"""
    decision_doc = load_json(ROOT / "tools" / "decision_schema.json")
    case_doc = load_json(ROOT / "data" / "cases" / "case_schema.json")
    specs = manifest["schema_delta"]
    derived_decision = apply_delta(decision_doc, [s for s in specs if s["file"] == "tools/decision_schema.json"])
    derived_case = apply_delta(case_doc, [s for s in specs if s["file"] == "data/cases/case_schema.json"])
    declared = ["/" + s["path"].replace(".", "/") for s in specs]

    def covered(path):
        """add_entries 会在登记路径下长出新的子键,那属于已登记差异;别的一律算漂移。"""
        return any(path == want or path.startswith(want + "/") for want in declared)

    drift_decision = [p for p in diff_paths(decision_doc, derived_decision) if not covered(p)]
    drift_case = [p for p in diff_paths(case_doc, derived_case) if not covered(p)]
    ctx = contracts.Ctx()
    ctx.tools = dict(base.tools)
    ctx.tool_docs = dict(base.tool_docs)
    for name in manifest["unseen_tools"]:
        errors, doc = contracts.check_one_tool_at(ROOT / "data" / "unseen_tools" / f"{name}.json")
        if errors:
            raise RuntimeError(f"未见工具契约 {name} 未通过校验: {errors}")
        ctx.tools[name] = doc["input_schema"]
        ctx.tool_docs[name] = doc
    derived_decision_schema = derived_decision["schema"]
    ctx.decision_schema = derived_decision_schema
    resolved = contracts.resolve_contract_refs(derived_case["case_schema"], derived_decision_schema)
    ctx.case_schema = resolved
    ctx.tag_vocabulary = list(base.tag_vocabulary)
    ctx.scoring = derived_case["scoring_conventions"]
    checks = [
        (
            "未见工具派生决策契约只改了登记的工具枚举",
            [err("tools/decision_schema.json", "UNSEEN_DELTA_DRIFT", f"出现表外差异: {drift_decision}")] if drift_decision else [],
        ),
        (
            "未见工具派生记录契约只改了登记枚举",
            [err("data/cases/case_schema.json", "UNSEEN_DELTA_DRIFT", f"出现表外差异: {drift_case}")] if drift_case else [],
        ),
    ]
    return ctx, checks, (decision_doc, derived_decision, case_doc, derived_case)


# --------------------------------------------------------------------------
# 1. 模板登记表
# --------------------------------------------------------------------------

def check_design(report):
    groups, handwritten, seed, now_pool, suffixes = gen.load_design()
    by_id = {group["id"]: group for group in groups}
    errors = []
    for group in groups:
        if group["toolset"] == "unseen" and group["split"] != "unseen_test":
            errors.append(err(f"data/templates/{group['id']}", "UNSEEN_SPLIT", "未见工具组只能落 unseen_test"))
        if group["toolset"] == "train6" and group["split"] == "unseen_test":
            errors.append(err(f"data/templates/{group['id']}", "SPLIT_MISUSE", "六工具组不能落 unseen_test"))
        if group["action"] == "call" and group.get("tool") in gen.UNSEEN_TOOLS and group["toolset"] != "unseen":
            errors.append(err(f"data/templates/{group['id']}", "UNSEEN_LEAK", "未见工具只允许在 toolset=unseen 的组里出现"))
        if group.get("refuse_code") == "unavailable_tool":
            blocking = group.get("blocking_tools") or []
            clash = sorted(set(blocking) & set(gen.UNSEEN_TOOLS))
            if clash:
                errors.append(err(f"data/templates/{group['id']}", "UNSEEN_LEAK", f"blocking_tools 用了未见工具真名: {clash}"))
            if not blocking:
                errors.append(err(f"data/templates/{group['id']}", "BLOCKING_MISSING", "unavailable_tool 组必须登记 blocking_tools"))
        offered = {name for row in group["available_tools_pool"] for name in row}
        if group["action"] == "call" and group["tool"] not in offered:
            errors.append(err(f"data/templates/{group['id']}", "TOOL_NOT_OFFERED", f"{group['tool']} 不在本组任一可用工具列表里"))
        target = group.get("clarify_target")
        if target and target["tool"] not in offered:
            errors.append(err(f"data/templates/{group['id']}", "TOOL_NOT_OFFERED", f"澄清目标 {target['tool']} 不在可用工具列表里"))
        if group["action"] == "refuse":
            clash = sorted(set(group.get("blocking_tools") or []) & offered)
            if clash:
                errors.append(err(f"data/templates/{group['id']}", "BLOCKING_AVAILABLE", f"blocking_tools 与可用工具冲突: {clash}"))
    report.add(f"模板组编号唯一、划分已写定({len(groups)} 组,种子 {seed})", errors)

    sizes = [len(group["phrasings"]) for group in groups]
    report.add(
        f"每组样本数=说法数(共 {sum(sizes)} 条,单组 {min(sizes)}~{max(sizes)} 条;槽位不扩量)",
        [] if all(size >= 2 for size in sizes) else [err("data/templates", "GROUP_TOO_SMALL", "有模板组说法少于 2 条")],
    )
    tool_names = set(contracts.EXPECTED_TOOLS) | set(gen.UNSEEN_TOOLS)
    for group in groups:
        for row in group["available_tools_pool"]:
            unknown = sorted(set(row) - tool_names)
            if unknown:
                report.add(f"{group['id']} 可用工具合法", [err(f"data/templates/{group['id']}", "TOOL_UNKNOWN", str(unknown))])
                break
    return groups, by_id, handwritten, seed, now_pool, suffixes


# --------------------------------------------------------------------------
# 2. 逐样本契约校验
# --------------------------------------------------------------------------

def check_samples(report, buckets, ctx6, ctx_unseen, by_id, seed):
    total_ids = {}
    for split in SPLITS:
        ctx = ctx_unseen if split == "unseen_test" else ctx6
        filename = f"data/datasets/{FILES[split]}"
        errors_all = []
        for index, sample in enumerate(buckets[split]):
            errors = list(contracts.check_single_case(sample, index, ctx, loc_base=filename))
            sample_id = sample.get("id")
            if sample_id in total_ids:
                errors.append(err(f"{filename}#/samples/{index}/id", "DUPLICATE_ID", f"与 {total_ids[sample_id]} 重复"))
            else:
                total_ids[sample_id] = split
            if sample.get("split") != split:
                errors.append(err(f"{filename}#/samples/{index}/split", "SPLIT_MISMATCH", f"落在 {FILES[split]} 却标 {sample.get('split')}"))
            if sample.get("source") != gen.SPLIT_SOURCE[split]:
                errors.append(err(f"{filename}#/samples/{index}/source", "SOURCE_MISMATCH", f"{sample.get('source')} 应为 {gen.SPLIT_SOURCE[split]}"))
            if sample.get("seed") != seed:
                errors.append(err(f"{filename}#/samples/{index}/seed", "SEED_MISMATCH", f"{sample.get('seed')} 与登记表 {seed} 不一致"))
            for key in ("category", "generation_rule", "template_group"):
                if not sample.get(key):
                    errors.append(err(f"{filename}#/samples/{index}/{key}", "PROVENANCE_MISSING", "缺少来源元信息"))
            group = by_id.get(sample.get("template_group"))
            if group is None:
                errors.append(err(f"{filename}#/samples/{index}/template_group", "GROUP_UNKNOWN", "模板组不在登记表里"))
            elif group["split"] != split:
                errors.append(err(f"{filename}#/samples/{index}/template_group", "GROUP_SPLIT_DRIFT", f"模板组登记在 {group['split']}"))
            elif group["category"] != sample.get("category"):
                errors.append(err(f"{filename}#/samples/{index}/category", "CATEGORY_DRIFT", "与模板组登记的类别不一致"))
            if errors:
                errors_all.append((sample_id, errors))
        flat = [item for _, errors in errors_all for item in errors]
        codes = {}
        for _, code, _ in flat:
            codes[code] = codes.get(code, 0) + 1
        histogram = " ".join(f"{code}x{count}" for code, count in sorted(codes.items(), key=lambda kv: -kv[1])[:6])
        report.add(
            f"{split}:{len(buckets[split])} 条逐条契约与来源校验(问题样本 {len(errors_all)};错误码 {histogram or '无'})",
            flat[:60],
        )
        if len(flat) > 60:
            print(f"  [提示] {split} 的失败明细超过 60 条,仅列出前 60 条;修复后重跑再看全量。")


# --------------------------------------------------------------------------
# 3. 近重复与跨集合泄漏
# --------------------------------------------------------------------------

def bigrams(text):
    cleaned = re.sub(r"\s+", "", text)
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def check_duplicates(report, buckets, handwritten_cases):
    seen = {}
    errors = []
    for split in SPLITS:
        for sample in buckets[split]:
            key = contracts.normalize_request(sample["user_request"])
            where = f"{split}:{sample['id']}"
            if key in seen:
                errors.append(err(f"data/datasets/{FILES[split]}#{sample['id']}", "NEAR_DUPLICATE_REQUEST", f"与 {seen[key]} 归一化后相同"))
            else:
                seen[key] = where
    for case in handwritten_cases:
        key = contracts.normalize_request(case["user_request"])
        if key in seen:
            errors.append(err(f"data/cases/handwritten_v0.json#{case['id']}", "NEAR_DUPLICATE_REQUEST", f"与 {seen[key]} 归一化后相同"))
    report.add(f"归一化去重(换编号、换日期、换标点一律算重复;{len(seen)} 个唯一骨架表述)", errors)

    per_group = {}
    for split in SPLITS:
        for sample in buckets[split]:
            per_group.setdefault(sample["template_group"], []).append(contracts.normalize_request(sample["user_request"]))
    weak = []
    for group_id, rows in sorted(per_group.items()):
        if len(set(rows)) != len(rows):
            weak.append(err(f"data/templates/{group_id}", "GROUP_PHRASING_DUPLICATE", "同组内有两条说法归一化后相同"))
    for group_id, rows in sorted(per_group.items()):
        if len(rows) < 2:
            weak.append(err(f"data/templates/{group_id}", "GROUP_SINGLETON", "模板组只有一条样本,谈不上骨架"))
    report.add(f"每模板组内说法互不重复({len(per_group)} 组)", weak)
    return per_group


def shape_mask(text):
    """数字与标点抹掉后的字符多重集:换语序、换标点都会落进同一个形状。"""
    return "".join(sorted(contracts.normalize_request(text)))


def similarity_pairs(buckets, handwritten_cases):
    """返回 (达到阈值的样本对, 全量扫描里跨/同集合的最高相似度)。后者用来证明阈值不是空转。"""
    rows = []
    for split in SPLITS:
        for sample in buckets[split]:
            rows.append((split, sample["id"], sample["template_group"], bigrams(sample["user_request"]), shape_mask(sample["user_request"])))
    for case in handwritten_cases:
        rows.append(("test", case["id"], case["template_group"], bigrams(case["user_request"]), shape_mask(case["user_request"])))
    pairs = []
    measured = {"cross": 0.0, "same": 0.0}
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if a[2] == b[2]:
                continue
            same_shape = a[4] == b[4]
            if abs(len(a[3]) - len(b[3])) > 24:
                continue
            raw = jaccard(a[3], b[3])
            key = "same" if a[0] == b[0] else "cross"
            measured[key] = max(measured[key], raw)
            score = raw
            if same_shape:
                score = max(score, CROSS_SPLIT_SIMILARITY + 0.01)
            if same_shape or score >= min(CROSS_SPLIT_SIMILARITY, SAME_SPLIT_SIMILARITY):
                pairs.append((score, a[0], a[1], b[0], b[1], same_shape))
    pairs.sort(reverse=True)
    return pairs, measured


def check_similarity_detector(report):
    """探测器有效性对照:构造一对同骨架改写与一对不同意图,前者必须被抓住,后者必须放过。"""
    paraphrase = ("读一下编号 DD204554 的订单记录", "读一下编号 DD204554 的订单详情")
    reordered = ("帮我把发票调出来", "把发票帮我调出来")
    distinct = ("查订单DD204551", "把订单金额改成 50000")
    errors = []
    if jaccard(bigrams(paraphrase[0]), bigrams(paraphrase[1])) < CROSS_SPLIT_SIMILARITY:
        errors.append(err("data/audit_dataset.py", "SIMILARITY_PROBE_WEAK", "同骨架改两字的对照没达到阈值,阈值形同虚设"))
    if shape_mask(reordered[0]) != shape_mask(reordered[1]):
        errors.append(err("data/audit_dataset.py", "SHAPE_PROBE_WEAK", "换语序对照没被字符多重集规则抓住"))
    if jaccard(bigrams(distinct[0]), bigrams(distinct[1])) >= CROSS_SPLIT_SIMILARITY:
        errors.append(err("data/audit_dataset.py", "SIMILARITY_PROBE_NOISY", "不同意图的对照被误判为同骨架,阈值过低"))
    high = jaccard(bigrams(paraphrase[0]), bigrams(paraphrase[1]))
    low = jaccard(bigrams(distinct[0]), bigrams(distinct[1]))
    report.add(f"重合探测器对照:同骨架 {high:.2f} ≥ {CROSS_SPLIT_SIMILARITY} > 换动词 {jaccard(bigrams(reordered[0]), bigrams(reordered[1])):.2f} 由形状规则兜底,不同意图 {low:.2f} 放过", errors)


def check_leakage(report, buckets, handwritten_cases, per_group_samples):
    check_similarity_detector(report)
    pairs, measured = similarity_pairs(buckets, handwritten_cases)
    cross = [p for p in pairs if p[1] != p[3]]
    same = [p for p in pairs if p[1] == p[3]]
    report.add(
        f"跨集合骨架重合(阈值 {CROSS_SPLIT_SIMILARITY:.2f} 或换语序同形状,命中 {len(cross)} 对;全量扫描跨集合最高 {measured['cross']:.2f})",
        [err(
            f"{FILES.get(a, 'handwritten_v0.json')}#{ida} vs {FILES.get(b, 'handwritten_v0.json')}#{idb}",
            "CROSS_SPLIT_SKELETON",
            f"相似度 {score:.2f}{'(换语序同形状)' if same_shape else ''},同一骨架出现在 {a} 与 {b}",
        ) for score, a, ida, b, idb, same_shape in cross[:30]],
    )
    report.add(
        f"同集合内骨架雷同(阈值 {SAME_SPLIT_SIMILARITY:.2f} 或换语序同形状,命中 {len(same)} 对;全量扫描同集合最高 {measured['same']:.2f})",
        [err(f"{ida} vs {idb}", "SAME_SPLIT_NEAR_DUPLICATE", f"相似度 {score:.2f}") for score, a, ida, b, idb, _ in same[:30]],
    )
    return pairs


# --------------------------------------------------------------------------
# 4. 分布与目标区间
# --------------------------------------------------------------------------

def distribution(buckets, handwritten_cases):
    stats = {}
    for split in SPLITS:
        rows = list(buckets[split])
        if split == "test":
            rows = rows + [dict(case, category=HANDWRITTEN_CATEGORY[case["id"]]) for case in handwritten_cases]
        actions, tools, categories, tags, calls = {}, {}, {}, {}, {}
        for row in rows:
            decision = row["expected_decision"]
            action = decision["action"]
            actions[action] = actions.get(action, 0) + 1
            if action == "call":
                tools[decision["tool"]] = tools.get(decision["tool"], 0) + 1
            categories[row["category"]] = categories.get(row["category"], 0) + 1
            for tag in row["tags"]:
                tags[tag] = tags.get(tag, 0) + 1
            calls.setdefault(row["template_group"], 0)
            calls[row["template_group"]] += 1
        stats[split] = {
            "samples": len(rows),
            "groups": len(calls),
            "actions": actions,
            "called_tools": tools,
            "categories": categories,
            "tags": tags,
            "refuse_ratio": round(actions.get("refuse", 0) / max(1, len(rows)), 3),
        }
    return stats


HANDWRITTEN_CATEGORY = {
    "TC-01-search-travel-policy": "normal", "TC-02-search-vague-target": "missing_param",
    "TC-03-get-record-context-reference": "normal", "TC-04-get-record-missing-id": "missing_param",
    "TC-05-get-record-unsupported-type": "capability_boundary", "TC-06-get-record-nonexistent-id": "normal",
    "TC-07-calc-discount-shipping": "normal", "TC-08-calc-unit-convert-fallback": "normal",
    "TC-09-agg-ambiguous-dataset": "missing_param", "TC-10-agg-month-filtered-sum": "normal",
    "TC-11-agg-count-paid-invoices": "normal", "TC-12-agg-unsupported-statistic": "capability_boundary",
    "TC-13-agg-unsupported-dataset": "capability_boundary", "TC-14-agg-time-window-ambiguous": "time_ambiguous",
    "TC-15-check-relative-day": "normal", "TC-16-check-evening-no-range": "time_ambiguous",
    "TC-17-check-past-time": "capability_boundary", "TC-18-create-pending-confirmation": "normal",
    "TC-19-create-month-only": "time_ambiguous", "TC-20-create-reversed-range": "time_ambiguous",
    "TC-21-create-fake-confirmation-claim": "adversarial", "TC-22-multi-intent-conditional": "multi_intent",
    "TC-23-refuse-sort-and-export": "irrelevant", "TC-24-refuse-record-mutation": "over_authority",
}


def check_distribution(report, stats, ctx):
    for split in SPLITS:
        low, high = TARGETS[split]
        got = stats[split]["samples"]
        errors = [] if low <= got <= high else [err(f"data/datasets/{FILES[split]}", "SIZE_TARGET", f"{got} 条不在 {low}~{high}")]
        report.add(f"{split} 规模 {got} 条 / 目标 {low}~{high}(含第 1 步手工案例计入 test)", errors)
    errors = []
    for split in ("train", "test"):
        actions = stats[split]["actions"]
        missing = [action for action in contracts.ACTIONS if not actions.get(action)]
        if missing:
            errors.append(err(f"data/datasets/{FILES[split]}", "ACTION_MISSING", f"缺动作 {missing}"))
    for split in ("train", "val", "test"):
        called = set(stats[split]["called_tools"])
        missing = sorted(set(contracts.EXPECTED_TOOLS) - called)
        if missing:
            errors.append(err(f"data/datasets/{FILES[split]}", "TOOL_UNCOVERED", f"没有期望调用它的样本: {missing}"))
    report.add("三个动作与六个工具在 train/test 全覆盖、val 六工具均有调用", errors)
    ratio = stats["test"]["refuse_ratio"]
    report.add(
        f"测试集拒绝类占比 {ratio:.1%}(要求 ≥ {contracts.MIN_REFUSE_RATIO:.0%})",
        [] if ratio >= contracts.MIN_REFUSE_RATIO else [err("data/datasets/test_v0.json", "REFUSE_RATIO", f"{ratio:.2f} 低于下限")],
    )
    covered = set()
    for split in SPLITS:
        covered |= set(stats[split]["categories"])
    missing = sorted(set(CATEGORIES) - covered)
    errors = [err("data/datasets", "CATEGORY_MISSING", f"这些类别没有样本: {missing}")] if missing else []
    for split in SPLITS:
        gap = sorted(set(CATEGORIES) - set(stats[split]["categories"]))
        if gap:
            print(f"  [提示] {split} 未覆盖类别 {gap};该类别只在别的集合里出现,集合内得分无法单独观察它。")
    report.add(f"八类判定桶在数据集内均有样本(已覆盖 {len(covered)}/{len(CATEGORIES)})", errors)
    used_tags = set()
    for split in SPLITS:
        used_tags |= set(stats[split]["tags"])
    for case in load_json(ROOT / "data" / "cases" / "handwritten_v0.json")["cases"]:
        used_tags |= set(case["tags"])
    unused = sorted(set(ctx.tag_vocabulary) - used_tags)
    report.add("标签词表全部有样本落地", [err("data/cases/case_schema.json", "TAG_UNUSED", f"未使用: {unused}")] if unused else [])
    return stats


def registry_line(buckets):
    rows = [s for split in SPLITS for s in buckets[split]]
    calls = [s for s in rows if s["expected_decision"]["action"] == "call"]
    with_text = [s for s in calls if set(s["expected_decision"].get("arguments") or {}) & set(["query", "title"])]
    variants = sum(len(v) for s in rows for v in (s.get("acceptable_text_arguments") or {}).values())
    return (
        f"文本登记表:{len(with_text)}/{len(calls)} 条 call 样本含 query/title,共登记 {variants} 种写法;"
        "未命中登记写法的输出按口径进人工复核,不计入参数全对率"
    )


# --------------------------------------------------------------------------
# 5. 执行可执行性(第 2 步闸门)
# --------------------------------------------------------------------------

def check_execution(report, buckets):
    import executor

    bad = []
    statuses = {}
    ran = 0
    for split in ("train", "val", "test"):
        for sample in buckets[split]:
            decision = sample["expected_decision"]
            if decision["action"] != "call":
                continue
            ran += 1
            context = sample["context"]
            trusted = executor.Trusted(context["now"], context["user_confirmed_write"], sample["available_tools"])
            store = simulator.Store.load()
            outcome = executor.execute_decision(decision, trusted, store=store)
            tool = decision["tool"]
            status = (outcome.get("result") or {}).get("status")
            statuses[f"{tool}:{status}"] = statuses.get(f"{tool}:{status}", 0) + 1
            loc = f"data/datasets/{FILES[split]}#{sample['id']}"
            if not outcome["allowed"]:
                bad.append(err(loc, "GATE_REJECTED", f"标准决策被闸门拒绝: {outcome['reasons']}"))
                continue
            if outcome["calendar_changed"] and not (tool == contracts.WRITE_TOOL and status == "ok"):
                bad.append(err(loc, "WRITE_BOUNDARY", f"{tool} 返回 {status} 却改了日历"))
            if tool == contracts.WRITE_TOOL and status not in ("pending_confirmation", "conflict"):
                bad.append(err(loc, "WRITE_STATUS", f"未确认的写操作返回了 {status},应为待确认或冲突"))
            if tool == contracts.WRITE_TOOL and status == "conflict":
                bad.append(err(loc, "SLOT_CONFLICT", "该样本落在已有日程上,标注语义与返回口径不一致"))
            if "nonexistent_record" in sample["tags"] and status != "not_found":
                bad.append(err(loc, "EXPECT_NOT_FOUND", f"标了不存在记录却返回 {status}"))
            if "normal_call" in sample["tags"] and "nonexistent_record" not in sample["tags"] and status in ("not_found", "no_data"):
                bad.append(err(loc, "GOLD_EMPTY_RESULT", f"正常调用样本返回 {status},取值与模拟数据脱节"))
            if status == "rejected":
                bad.append(err(loc, "GOLD_REJECTED", f"标准决策被执行层拒绝: {(outcome['result'] or {}).get('error_code')}"))
    report.add(f"标准决策过闸门并可执行({ran} 条 call;返回分布 {len(statuses)} 种)", bad[:60])
    for split in ("train", "val", "test"):
        for sample in buckets[split]:
            if sample["expected_decision"]["action"] != "call":
                continue
    return statuses


# --------------------------------------------------------------------------
# 6. 相对时间独立复核(与生成器不同的实现)
# --------------------------------------------------------------------------

def independent_date(phrase, now):
    """另写一份中文日期解析,用来独立复核生成器算出的绝对时刻。"""
    base = now.date()
    if "大后天" in phrase:
        return base + timedelta(days=3)
    if "后天" in phrase:
        return base + timedelta(days=2)
    if "明天" in phrase:
        return base + timedelta(days=1)
    if "今天" in phrase:
        return base
    matched = re.search(r"下周([一二三四五六日天])", phrase)
    if matched:
        monday = base - timedelta(days=base.weekday())
        return monday + timedelta(days=7 + CN_WEEKDAY[matched.group(1)])
    matched = re.search(r"上周([一二三四五六日天])", phrase)
    if matched:
        monday = base - timedelta(days=base.weekday())
        return monday - timedelta(days=7 - CN_WEEKDAY[matched.group(1)])
    if "本周" in phrase:
        return base
    matched = re.search(r"(\d{4}) 年 (\d{1,2}) 月 (\d{1,2}) 日", phrase)
    if matched:
        return date(int(matched.group(1)), int(matched.group(2)), int(matched.group(3)))
    matched = re.search(r"(\d{1,2}) 月 (\d{1,2}) 日", phrase)
    if matched:
        month, day = int(matched.group(1)), int(matched.group(2))
        year = base.year
        if month > base.month + 6:
            year -= 1
        return date(year, month, day)
    return None


def check_times(report, buckets):
    errors = []
    checked = 0
    for split in SPLITS:
        for sample in buckets[split]:
            decision = sample["expected_decision"]
            if decision["action"] != "call":
                continue
            now = datetime.strptime(sample["context"]["now"], contracts.TIME_FORMAT)
            arguments = decision.get("arguments") or {}
            phrase = None
            for candidate in DATE_PHRASE.findall(sample["user_request"]):
                if candidate not in ("本月", "上个月", "下个月", "本周"):
                    phrase = candidate
                    break
            clocks = TIME_IN_REQUEST.findall(sample["user_request"])
            if decision["tool"] in ("check_availability", "create_event"):
                start, end = arguments.get("start_time"), arguments.get("end_time")
                if not start or not end:
                    continue
                checked += 1
                loc = f"data/datasets/{FILES[split]}#{sample['id']}"
                if parse_date(start) != parse_date(end):
                    errors.append(err(loc, "RANGE_SPANS_DAYS", "起止跨天但请求只给了一个日期说法"))
                if phrase:
                    expect = independent_date(phrase, now)
                    if expect is None:
                        errors.append(err(loc, "PHRASE_UNREADABLE", f"独立复核无法解析日期说法「{phrase}」"))
                    else:
                        if parse_date(start) != expect or parse_date(end) != expect:
                            errors.append(err(loc, "TIME_RESOLVE_MISMATCH", f"「{phrase}」独立解析为 {expect},标准答案写 {start[:10]}/{end[:10]}"))
                        if "下周" in phrase and expect.weekday() != CN_WEEKDAY[phrase[-1]]:
                            errors.append(err(loc, "WEEKDAY_MISMATCH", f"「{phrase}」解析到的 {expect} 不是星期{phrase[-1]}"))
                padded = [pad(clock) for clock in clocks]
                if len(padded) >= 2:
                    if padded[0] != start[11:] or padded[1] != end[11:]:
                        errors.append(err(loc, "CLOCK_MISMATCH", f"请求里的时刻 {padded[:2]} 与标准答案 {start[11:]}~{end[11:]} 不一致"))
                if start <= sample["context"]["now"]:
                    errors.append(err(loc, "PAST_START", f"起始时刻 {start} 不晚于 now"))
                if end <= start:
                    errors.append(err(loc, "REVERSED_RANGE", f"{end} 不晚于 {start}"))
            filters = arguments.get("filters") or {}
            window = filters.get("time_range") if isinstance(filters, dict) else None
            if window:
                checked += 1
                loc = f"data/datasets/{FILES[split]}#{sample['id']}"
                matched = MONTH_PHRASE.search(sample["user_request"])
                if not matched:
                    errors.append(err(loc, "WINDOW_PHRASE", "给了时间窗筛选但请求里找不到月份说法"))
                else:
                    month = int(matched.group(1))
                    year = now.year
                    last = calendar.monthrange(year, month)[1]
                    expect = (f"{year:04d}-{month:02d}-01T00:00", f"{year:04d}-{month:02d}-{last:02d}T23:59")
                    if (window["start"], window["end"]) != expect:
                        errors.append(err(loc, "MONTH_RANGE_MISMATCH", f"{month} 月独立补全应为 {expect},标准答案 {window['start']}~{window['end']}"))
    report.add(f"相对时间独立复核({checked} 条含时刻的调用)", errors[:40])
    return errors


def parse_date(iso):
    return datetime.strptime(iso[:10], "%Y-%m-%d").date()


def pad(clock):
    hour, minute = clock.split(":")
    return f"{int(hour):02d}:{minute}"


# --------------------------------------------------------------------------
# 7. 模型可见输入:投影一致性与答案泄漏
# --------------------------------------------------------------------------

FORBIDDEN_IN_INPUT = ["rationale", "generation_rule", "post_condition", "refuse_code", "blocking_tools", "clarify_target", "acceptable_text_arguments", "user_confirmed_write"]


def check_input_projection(report, buckets, docs):
    errors = []
    rendered = 0
    for split in SPLITS:
        path = DATASET_DIR / FILES[split].replace(".json", ".input.jsonl")
        if not path.exists():
            errors.append(err(f"data/datasets/{path.name}", "INPUT_MISSING", "模型可见输入投影不存在"))
            continue
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if [row["id"] for row in lines] != [sample["id"] for sample in buckets[split]]:
            errors.append(err(f"data/datasets/{path.name}", "INPUT_ID_DRIFT", "投影与数据文件的样本 id 列表不一致"))
        for sample, row in zip(buckets[split], lines):
            rendered += 1
            expect = gen.render_input(sample, docs)
            loc = f"data/datasets/{path.name}#{sample['id']}"
            if row["input"] != expect:
                errors.append(err(loc, "INPUT_DRIFT", "投影与当前契约渲染结果不一致(契约或生成规则改过了)"))
            text = row["input"]
            decision = sample["expected_decision"]
            for key in FORBIDDEN_IN_INPUT:
                if key in text:
                    errors.append(err(loc, "GOLD_LEAK", f"模型可见输入里出现了元信息键名 {key}"))
            if sample["template_group"] in text:
                errors.append(err(loc, "GOLD_LEAK", "模型可见输入里出现了模板组编号"))
            for tag in sample["tags"]:
                if tag in text:
                    errors.append(err(loc, "GOLD_LEAK", f"模型可见输入里出现了标签 {tag}"))
            if sample["category"] in text:
                errors.append(err(loc, "GOLD_LEAK", f"模型可见输入里出现了类别 {sample['category']}"))
            if decision["action"] in ("clarify", "refuse"):
                gold = decision.get("question") or decision.get("reason")
                if gold and gold in text:
                    errors.append(err(loc, "GOLD_ANSWER_LEAK", "标准追问/拒绝理由出现在模型可见输入里"))
            if "seed" in text or str(sample["seed"]) in text:
                errors.append(err(loc, "GOLD_LEAK", f"模型可见输入里出现了随机种子 {sample['seed']}"))
            if sample["user_request"] not in text:
                errors.append(err(loc, "REQUEST_MISSING", "输入投影里没有用户请求"))
    report.add(f"模型可见输入投影:与答案和元信息隔离({rendered} 条)", errors[:40])


def input_stats(buckets, docs):
    rows = []
    for split in SPLITS:
        sizes = []
        for sample in buckets[split]:
            sizes.append(contracts.estimate_tokens(gen.render_input(sample, docs)))
        rows.append((split, min(sizes), max(sizes), sum(sizes) // len(sizes)))
    return rows


# --------------------------------------------------------------------------
# 8. 未见工具隔离
# --------------------------------------------------------------------------

def check_isolation(report, buckets, manifest):
    forbidden = manifest["forbidden_names"]
    errors = []
    for split in ("train", "val", "test"):
        text = (DATASET_DIR / FILES[split]).read_text(encoding="utf-8")
        for name in forbidden:
            if name in text:
                errors.append(err(f"data/datasets/{FILES[split]}", "UNSEEN_LEAK", f"{'训练' if split == 'train' else '验证' if split == 'val' else '测试'}集里出现了未见工具名 {name}"))
    for name in forbidden:
        path = ROOT / "tools" / "schemas" / f"{name}.json"
        if path.exists():
            errors.append(err(f"data/unseen_tools/{name}.json", "UNSEEN_IN_TRAIN_DIR", "未见工具契约不得放进模型可见的六工具目录"))
    for sample in buckets["unseen_test"]:
        offered = set(sample["available_tools"])
        if not offered & set(forbidden):
            errors.append(err(f"data/datasets/{FILES['unseen_test']}#{sample['id']}", "UNSEEN_ABSENT", "未见工具样本的可用工具里没有一个未见工具"))
    unseen_in_set = sorted({tool for sample in buckets["unseen_test"] for tool in sample["available_tools"] if tool in forbidden})
    report.add("未见工具名不进入训练/验证/测试三个集合,且契约不在 tools/schemas", errors)
    report.add(
        f"未见工具集单独统计:{unseen_in_set} 共 {len(buckets['unseen_test'])} 条,不与六工具总分合并",
        [] if set(unseen_in_set) == set(forbidden) else [err("data/datasets/test_unseen_v0.json", "UNSEEN_COVERAGE", f"两个未见工具都要有样本,实际 {unseen_in_set}")],
    )
    exception = manifest.get("known_exception") or {}
    hand = (ROOT / exception.get("file", "data/cases/handwritten_v0.json")).read_text(encoding="utf-8")
    found = [name for name in forbidden if name in hand]
    report.add(
        f"第 1 步手工案例里的已知例外:{found or '无'}(blocking_tools 元信息,不进模型输入;详见 {exception.get('case_id')})",
        [],
    )
    return found


# --------------------------------------------------------------------------
# 9. 人工抽检清单与记录
# --------------------------------------------------------------------------

def spot_check_rows(buckets, handwritten_cases, per_category=12):
    """每类先按模板组各取一条;组数不够时再在同一组内取不同说法,保证每类够 10 条可勾。"""
    pool = [(split, sample) for split in SPLITS for sample in buckets[split]]
    pool += [("test", case) for case in handwritten_cases]
    picked = {}
    for split, sample in pool:
        picked.setdefault(sample["category"], []).append((split, sample))
    rows = {}
    for category, items in picked.items():
        chosen = {}
        for split, sample in items:
            chosen.setdefault(sample["template_group"], (split, sample))
        ordered = list(chosen.values())
        if len(ordered) < per_category:
            picked_rows = items[: max(per_category, MIN_PER_CATEGORY)]
        else:
            step = max(1, len(ordered) // per_category)
            picked_rows = ordered[:: step][:per_category]
        if category == "normal":
            # 正常调用类必须每个被调用工具都有样本可看,否则抽到的全是同一两个工具,看不出别的工具的标注质量
            want = {}
            for split, sample in items:
                tool = (sample["expected_decision"] or {}).get("tool")
                if tool:
                    want.setdefault(tool, []).append((split, sample))
            for tool, rows_of_tool in want.items():
                have = {s["id"] for _, s in picked_rows}
                for row in rows_of_tool:
                    if len({s["id"] for _, s in picked_rows if s["expected_decision"].get("tool") == tool}) >= 2:
                        break
                    if row[1]["id"] not in have:
                        picked_rows.append(row)
                        have.add(row[1]["id"])
        rows[category] = picked_rows
    return rows


def write_spot_check(report_file, rows):
    lines = [
        "# 第 3 步人工抽检清单",
        "",
        "本清单由 `python data/audit_dataset.py` 自动生成,每条都要人工读;程序校验不能替代它。",
        "抽检重点三件事(标注规范 8.5):登记表是否漏掉等价写法、线索词表是否漏掉合理表述、期望动作本身是否含糊。",
        "勾选结果写在 data/datasets/review_log_v0.md,审计脚本会数勾:每类不足 10 条即审计失败。",
        "",
    ]
    for category in CATEGORIES:
        lines.append(f"## 类别 {category}")
        lines.append("")
        lines.append("| 样本 id | 集合 | 模板组 | 用户请求 | 标准决策 | 判定理由 |")
        lines.append("|---|---|---|---|---|---|")
        for split, sample in rows.get(category, []):
            decision = json.dumps(sample["expected_decision"], ensure_ascii=False)
            request = sample["user_request"].replace("|", "／")
            rationale = sample["rationale"].replace("|", "／")
            lines.append(f"| {sample['id']} | {split} | {sample['template_group']} | {request} | {decision} | {rationale} |")
        lines.append("")
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


REVIEW_HEAD = "## 抽检记录"


def check_review_log(report, buckets):
    if not REVIEW_FILE.exists():
        report.add("人工抽检记录", [err("data/datasets/review_log_v0.md", "MANUAL_REVIEW_MISSING", "还没有人工抽检记录文件")])
        return {}
    text = REVIEW_FILE.read_text(encoding="utf-8")
    current = None
    tally = {}
    checked_ids = set()
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
        elif line.startswith("### "):
            current = current if current else ""
        matched = re.match(r"^\s*-\s*\[x\]\s*(\S+)", line, flags=re.IGNORECASE)
        if matched and current:
            tally.setdefault(current, []).append(matched.group(1))
            checked_ids.add(matched.group(1))
        matched_open = re.match(r"^\s*-\s*\[\s\]\s*(\S+)", line)
        if matched_open and current:
            tally.setdefault(current, []).append(None)
    real_ids = {sample["id"] for split in SPLITS for sample in buckets[split]}
    real_ids |= {case["id"] for case in load_json(ROOT / "data" / "cases" / "handwritten_v0.json")["cases"]}
    errors = []
    ghost = sorted({item for rows in tally.values() for item in rows if item and item not in real_ids})
    if ghost:
        errors.append(err("data/datasets/review_log_v0.md", "MANUAL_REVIEW_PHANTOM", f"抽检记录里出现了数据集中不存在的样本 id: {ghost[:6]}"))
    unchecked = sum(1 for rows in tally.values() for item in rows if item is None)
    if unchecked:
        errors.append(err("data/datasets/review_log_v0.md", "MANUAL_REVIEW_UNCHECKED", f"有 {unchecked} 条抽检项未勾选,不算完成"))
    for category in CATEGORIES:
        rows = [item for item in tally.get(category, []) if item]
        if len(rows) < MIN_PER_CATEGORY:
            errors.append(err("data/datasets/review_log_v0.md", "MANUAL_REVIEW_SHORT", f"类别 {category} 人工确认 {len(rows)} 条,不足 {MIN_PER_CATEGORY} 条"))
    problems = re.search(
        r"^##[ \t]*发现的问题与修正[ \t]*" + chr(10) + r"(.*?)(?=^##[ \t]|\Z)", text, flags=re.MULTILINE | re.DOTALL
    )
    bullets = re.findall(r"^\s*\d+\.\s+\S|^\s*[-*]\s+\S", problems.group(1), flags=re.MULTILINE) if problems else []
    if not bullets:
        errors.append(err("data/datasets/review_log_v0.md", "MANUAL_FINDINGS_EMPTY", "没有「发现的问题与修正」小节,或该小节没有条目"))
    report.add(f"人工抽检记录:每类 ≥{MIN_PER_CATEGORY} 条(已勾 {sum(1 for _ in checked_ids)} 个唯一样本,问题条目 {len(bullets)} 条)", errors)
    return tally


# --------------------------------------------------------------------------
# 10. 冻结
# --------------------------------------------------------------------------

def write_frozen(stats):
    payload = {
        "version": "frozen_dataset_v0",
        "frozen_at": datetime.now().strftime(contracts.TIME_FORMAT),
        "rule": "以下文件哈希一经登记即冻结;任何实验跑分前先执行 python data/audit_dataset.py --check-frozen。改测试集必须升版本号并同步重跑全部对照组。",
        "provenance_rule": "provenance 列出决定测试集内容的上游文件;它们变了测试集必须重新生成并重审。",
        "counts": {split: stats[split]["samples"] for split in SPLITS},
        "actions": {split: stats[split]["actions"] for split in SPLITS},
        "refuse_ratio_test": stats["test"]["refuse_ratio"],
        "frozen": {rel: sha256(ROOT / rel) for rel in FROZEN_FILES},
        "provenance": {rel: sha256(ROOT / rel) for rel in PROVENANCE_FILES},
    }
    FROZEN_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def check_frozen(report):
    if not FROZEN_FILE.exists():
        report.add("测试集冻结清单", [err("data/datasets/frozen_v0.json", "FREEZE_MISSING", "还没冻结;先跑 --freeze")])
        return
    payload = load_json(FROZEN_FILE)
    errors = []
    for rel, digest in payload["frozen"].items():
        path = ROOT / rel
        if not path.exists():
            errors.append(err(rel, "FROZEN_FILE_MISSING", "冻结清单里的文件不在了"))
        elif sha256(path) != digest:
            errors.append(err(rel, "FROZEN_DRIFT", "内容与冻结时不一致,测试集被改过"))
    drifted = [rel for rel, digest in payload.get("provenance", {}).items() if (ROOT / rel).exists() and sha256(ROOT / rel) != digest]
    report.add(f"测试集冻结校验({len(payload['frozen'])} 个文件;上游 {len(drifted)} 个有改动)", errors)
    if drifted:
        print(f"  [提示] 上游文件已变更,测试集需重新生成并重审: {drifted}")


# --------------------------------------------------------------------------

def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError, ValueError):
            pass
    parser = argparse.ArgumentParser(description="第 3 步数据集审计")
    parser.add_argument("--freeze", action="store_true", help="审计全通过后写冻结清单")
    parser.add_argument("--check-frozen", action="store_true", help="只校验冻结清单")
    parser.add_argument("--similarity-top", type=int, default=0, help="打印跨集合最相似的 N 个样本对")
    parser.add_argument("--verbose", action="store_true", help="逐条打印通过项")
    args = parser.parse_args(argv)

    report = contracts.Report(args.verbose)
    if args.check_frozen:
        report.begin("测试集冻结校验")
        check_frozen(report)
        return report.render()

    ctx6 = base_ctx()
    manifest = load_json(ISOLATION_FILE)
    ctx_unseen, delta_checks, derived = unseen_ctx(ctx6, manifest)
    docs = gen.load_tool_docs()
    handwritten_cases = load_json(ROOT / "data" / "cases" / "handwritten_v0.json")["cases"]
    for case in handwritten_cases:
        case["category"] = HANDWRITTEN_CATEGORY[case["id"]]

    buckets = {split: load_json(DATASET_DIR / FILES[split])["samples"] for split in SPLITS}
    for split in SPLITS:
        payload = load_json(DATASET_DIR / FILES[split])
        if payload["seed"] != gen.load_design()[2]:
            report.add(f"{split} 文件头种子", [err(f"data/datasets/{FILES[split]}", "SEED_MISMATCH", "文件头 seed 与模板登记表不一致")])

    report.begin("1. 模板登记表与派生契约")
    groups, by_id, handwritten, seed, now_pool, suffixes = check_design(report)
    for label, errors in delta_checks:
        report.add(label, errors)
    report.add(registry_line(buckets), [])

    report.begin("2. 逐样本契约校验(复用第 1 步实现)")
    check_samples(report, buckets, ctx6, ctx_unseen, by_id, seed)

    report.begin("3. 去重、近重复与跨集合泄漏")
    per_group = check_duplicates(report, buckets, handwritten_cases)
    pairs = check_leakage(report, buckets, handwritten_cases, per_group)
    if args.similarity_top:
        print("  最相似的样本对:")
        pairs, _measured = similarity_pairs(buckets, handwritten_cases)
        for row in pairs[: args.similarity_top]:
            score, a, ida, b, idb, same_shape = row
            tag = "跨集合" if a != b else "同集合"
            mark = "[换语序]" if same_shape else ""
            print(f"    {score:.2f} [{tag}]{mark} {a}:{ida} ↔ {b}:{idb}")

    report.begin("4. 集合分布与规模目标")
    stats = distribution(buckets, handwritten_cases)
    check_distribution(report, stats, ctx6)
    for split in SPLITS:
        row = stats[split]
        print(
            f"  {split:11s} 样本 {row['samples']:4d} 组 {row['groups']:3d} "
            + " ".join(f"{k}={v}" for k, v in sorted(row["actions"].items()))
            + "  " + " ".join(f"{k}={v}" for k, v in sorted(row["categories"].items()))
        )

    report.begin("5. 执行层可执行性与写入边界")
    check_execution(report, buckets)

    report.begin("6. 相对时间独立复核")
    check_times(report, buckets)

    report.begin("7. 模型可见输入(无答案、无元信息)")
    check_input_projection(report, buckets, docs)
    for split, low, high, mean in input_stats(buckets, docs):
        report.add(
            f"{split} 输入规模:估算 {low}~{high} token,均值 {mean}(工具块预算 {contracts.PROMPT_TOKEN_BUDGET},序列长度 {contracts.SEQ_LENGTH_TOKENS} 仍为待真实 tokenizer 验证的假设)",
            [] if high <= contracts.PROMPT_TOKEN_BUDGET else [err(f"data/datasets/{FILES[split]}", "INPUT_OVER_BUDGET", f"最大输入 {high} 超过工具块预算 {contracts.PROMPT_TOKEN_BUDGET}")],
        )

    report.begin("8. 未见工具隔离")
    check_isolation(report, buckets, manifest)

    report.begin("9. 人工抽检闭环")
    rows = spot_check_rows(buckets, handwritten_cases)
    write_spot_check(SPOT_FILE, rows)
    report.add(
        f"抽检清单已生成({sum(len(v) for v in rows.values())} 条,{len(rows)} 个类别)→ {contracts.rel(SPOT_FILE)}",
        [] if all(len(v) >= MIN_PER_CATEGORY for v in rows.values()) else [err("data/datasets/spot_check_v0.md", "SPOT_SHORT", "有类别凑不满抽检条数")],
    )
    check_review_log(report, buckets)

    if args.freeze:
        report.begin("10. 冻结")
        pending = sum(len(errors) for _, items in report.sections for _, errors in items)
        if pending:
            report.add("写入冻结清单", [err("data/datasets/frozen_v0.json", "FREEZE_BLOCKED", f"审计仍有 {pending} 条失败,拒绝冻结")])
        else:
            payload = write_frozen(stats)
            report.add(f"已冻结 {len(payload['frozen'])} 个测试文件 + {len(payload['provenance'])} 个上游文件", [])
    code = report.render()
    print("新增/更新产物: data/datasets/*.json、*.input.jsonl、build_manifest.json、spot_check_v0.md" + (";frozen_v0.json" if args.freeze else ""))
    return code


if __name__ == "__main__":
    sys.exit(main())
