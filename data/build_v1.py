"""Add contrastive train/val examples while preserving every v0 and frozen test file."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "eval"))
import audit_dataset as audit  # noqa: E402
import generate_dataset as generator  # noqa: E402
import validate_contracts as scoring  # noqa: E402

DESIGN = ROOT / "data" / "templates" / "contrastive_v1.json"
OUT = ROOT / "data" / "datasets"
NOW = "2026-09-23T10:00"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(text):
    return re.sub(r"[^\w一-鿿]", "", text).lower()


def make_case(pair, index, complete):
    tool = pair["tool"]
    request = pair["complete" if complete else "missing"]
    case = {
        "id": f"V1-{index:03d}-{'call' if complete else 'clarify'}",
        "user_request": request,
        "available_tools": [tool],
        "context": {"now": NOW, "user_confirmed_write": False},
        "rationale": "同一工具的参数齐全/缺参对照;只根据请求中的明确取值决定调用或追问。",
        "tags": ["normal_call"] if complete else ["missing_required_field"],
        "template_group": f"TG-{600 + index * 2 + int(complete):03d}",
        "source": "generated_v1",
        "split": pair["split"],
        "category": "normal" if complete else "missing_param",
        "generation_rule": f"contrastive_v1 第 {index} 对;工具 {tool};参数{'齐全' if complete else '缺失'}。",
        "seed": 20260924,
    }
    if tool == "search_documents":
        if complete:
            case["expected_decision"] = {"action": "call", "tool": tool,
                                         "arguments": {"query": pair["value"]}}
            case["acceptable_text_arguments"] = {"query": [pair["value"]]}
        else:
            case["expected_decision"] = {"action": "clarify", "question": "请给出要检索的文档标题或关键词。"}
            case["clarify_target"] = {"tool": tool, "fields": ["query"]}
    elif tool == "calculate":
        if complete:
            case["expected_decision"] = {"action": "call", "tool": tool,
                                         "arguments": {"expression": pair["value"]}}
        else:
            case["expected_decision"] = {"action": "clarify", "question": "请给出需要计算的完整算式。"}
            case["clarify_target"] = {"tool": tool, "fields": ["expression"]}
    elif tool == "create_event":
        if complete:
            case["expected_decision"] = {"action": "call", "tool": tool,
                                         "arguments": {"title": pair["title"], "start_time": pair["start"], "end_time": pair["end"]}}
            case["acceptable_text_arguments"] = {"title": [pair["title"]]}
            case["post_condition"] = "requires_server_confirmation"
        else:
            case["expected_decision"] = {"action": "clarify", "question": "这项日程的结束时间是几点?"}
            case["clarify_target"] = {"tool": tool, "fields": ["end_time"]}
    elif tool == "aggregate_data":
        if complete:
            case["expected_decision"] = {"action": "call", "tool": tool,
                                         "arguments": {"dataset": pair["dataset"], "metric": "amount", "operation": "sum"}}
        else:
            case["expected_decision"] = {"action": "clarify", "question": "请说明要统计哪个数据集的金额总和。"}
            case["clarify_target"] = {"tool": tool, "fields": ["dataset"]}
    else:
        raise ValueError(f"未支持的工具: {tool}")
    if complete and tool != "create_event":
        case["post_condition"] = "none"
    return case


def main():
    check = subprocess.run([sys.executable, str(ROOT / "data" / "audit_dataset.py"), "--check-frozen"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check.returncode:
        raise RuntimeError("冻结测试集校验失败，拒绝构造第二轮数据")
    design = read_json(DESIGN)
    pairs = design["pairs"]
    if Counter(pair["split"] for pair in pairs) != {"train": 24, "val": 8}:
        raise RuntimeError("第二轮对照组必须为 train 24 对、val 8 对")
    if any(pair["tool"] not in ("search_documents", "calculate", "create_event", "aggregate_data") for pair in pairs):
        raise RuntimeError("对照设计含未登记工具")
    docs = generator.load_tool_docs()
    ctx = audit.base_ctx()
    # v0 契约文件属于冻结来源，v1 只在内存中扩展新样本的编号与来源。
    ctx.case_schema["properties"]["id"]["pattern"] = r"^(TC-[0-9]{2}|DS-[0-9]{4}|UN-[0-9]{3}|V1-[0-9]{3})-[a-z][a-z0-9-]{2,40}$"
    ctx.case_schema["properties"]["source"]["enum"].append("generated_v1")
    new_by_split = {"train": [], "val": []}
    for index, pair in enumerate(pairs, 1):
        for complete in (False, True):
            case = make_case(pair, index, complete)
            errors = scoring.check_single_case(case, index, ctx, loc_base="data/templates/contrastive_v1.json")
            if errors:
                raise RuntimeError(f"{case['id']} 契约或评分口径不合格: {errors}")
            new_by_split[pair["split"]].append(case)
    all_seen = set()
    for split in ("train", "val", "test", "test_unseen"):
        for sample in read_json(OUT / f"{split}_v0.json")["samples"]:
            all_seen.add(normalize(sample["user_request"]))
    for sample in read_json(ROOT / "data" / "cases" / "handwritten_v0.json")["cases"]:
        all_seen.add(normalize(sample["user_request"]))
    for split in ("train", "val"):
        for case in new_by_split[split]:
            key = normalize(case["user_request"])
            if key in all_seen:
                raise RuntimeError(f"新样本与现有样本请求重复: {case['id']}")
            all_seen.add(key)
    # 新样本与其他集合做近重复检查；只看请求文本，不依据测试预测改标签。
    old_by_split = {split: read_json(OUT / f"{stem}_v0.json")["samples"]
                    for split, stem in (("train", "train"), ("val", "val"),
                                        ("test", "test"), ("unseen_test", "test_unseen"))}
    old_by_split["test"] += read_json(ROOT / "data" / "cases" / "handwritten_v0.json")["cases"]
    for split, new_cases in new_by_split.items():
        others = [case for other, cases in old_by_split.items() if other != split for case in cases]
        others += new_by_split["val" if split == "train" else "train"]
        for case in new_cases:
            left = audit.bigrams(case["user_request"])
            for other in others:
                score = audit.jaccard(left, audit.bigrams(other["user_request"]))
                if score >= audit.CROSS_SPLIT_SIMILARITY:
                    raise RuntimeError(f"跨集合近重复: {case['id']} / {other['id']} ({score:.2f})")
    for split in ("train", "val"):
        base = read_json(OUT / f"{split}_v0.json")
        samples = base["samples"] + new_by_split[split]
        ids = [sample["id"] for sample in samples]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"{split} 样本 ID 重复")
        output = {"version": "dataset_v1", "split": split,
                  "description": "v0 原始样本 + 独立的参数齐全/缺参对照样本;冻结测试集仍为 v0。",
                  "samples": samples}
        (OUT / f"{split}_v1.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        input_path = OUT / f"{split}_v1.input.jsonl"
        with input_path.open("w", encoding="utf-8") as stream:
            for sample in samples:
                stream.write(json.dumps({"id": sample["id"], "input": generator.render_input(sample, docs)}, ensure_ascii=False) + "\n")
        print(f"{split}: {len(base['samples'])} + {len(new_by_split[split])} = {len(samples)}")
    manifest = {"version": "dataset_v1", "design_sha256": digest(DESIGN),
                "train_v0_sha256": digest(OUT / "train_v0.json"), "val_v0_sha256": digest(OUT / "val_v0.json"),
                "frozen_v0_sha256": digest(OUT / "frozen_v0.json"),
                "counts": {split: len(read_json(OUT / f"{split}_v1.json")["samples"]) for split in ("train", "val")}}
    (OUT / "build_manifest_v1.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    check = subprocess.run([sys.executable, str(ROOT / "data" / "audit_dataset.py"), "--check-frozen"],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check.returncode:
        raise RuntimeError("构造后冻结测试集校验失败")
    print("冻结测试集校验通过")


if __name__ == "__main__":
    main()
