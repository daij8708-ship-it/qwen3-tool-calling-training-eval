"""Run reproducible basic/enhanced Qwen3 baselines on frozen test sets.

Run with the CUDA-enabled fllam Python environment. Predictions are appended
after every item, so an interrupted run can resume with the same --run-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import audit_dataset as audit  # noqa: E402
import generate_dataset as generator  # noqa: E402
import validate_contracts as scoring  # noqa: E402

MODEL_DIR = Path(r"models/Qwen3-0.6B")
MAX_NEW_TOKENS = 128
MODES = ("basic", "enhanced")
ENHANCED_RULES = """决策规则：
1. 只可调用“本次可用工具”中的工具，不要自行执行工具，也不要编造查询结果。
2. 请求可执行且必填参数明确时输出 call；缺少必填信息或存在多种意图时输出 clarify；能力不支持或越权时输出 refuse。
3. 参数只能来自用户请求和上下文；时间以服务端当前时间为基准，不自行补充用户没有给出的时刻。
4. 写操作只提出 call 建议，不能声称已经得到服务端确认。
5. 仅输出一个严格 JSON 对象：call 使用 action/tool/arguments，clarify 使用 action/question，refuse 使用 action/reason；不要输出解释、Markdown 或额外字段。
以下是训练集中的格式示例，不代表本次可用工具：
{examples}
现在处理下面的真实任务。"""


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_samples():
    docs = generator.load_tool_docs()
    result = {}
    for split, stem in (("test", "test"), ("unseen_test", "test_unseen")):
        dataset = json.loads((ROOT / "data" / "datasets" / f"{stem}_v0.json").read_text(encoding="utf-8"))["samples"]
        inputs = [json.loads(line) for line in (ROOT / "data" / "datasets" / f"{stem}_v0.input.jsonl").read_text(encoding="utf-8").splitlines()]
        by_id = {sample["id"]: sample for sample in dataset}
        if {item["id"] for item in inputs} != set(by_id):
            raise RuntimeError(f"{split} 输入投影与标注样本不匹配")
        rows = [(by_id[item["id"]], item["input"]) for item in inputs]
        if split == "test":
            handwritten = json.loads((ROOT / "data" / "cases" / "handwritten_v0.json").read_text(encoding="utf-8"))["cases"]
            rows += [(dict(case, category=audit.HANDWRITTEN_CATEGORY[case["id"]]), generator.render_input(case, docs)) for case in handwritten]
        result[split] = rows
    return result


def enhanced_prefix():
    train = json.loads((ROOT / "data" / "datasets" / "train_v0.json").read_text(encoding="utf-8"))["samples"]
    examples = []
    for action in ("call", "clarify", "refuse"):
        sample = next(sample for sample in train if sample["expected_decision"]["action"] == action)
        examples.append(
            f"用户请求：{sample['user_request']}\n"
            f"可用工具：{', '.join(sample['available_tools'])}\n"
            f"决策：{canonical_json(sample['expected_decision'])}"
        )
    return ENHANCED_RULES.format(examples="\n".join(examples))


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], "all")].append(row)
        groups[(row["split"], row["category"])].append(row)
    output = {}
    for (split, category), items in sorted(groups.items()):
        counts = Counter(item["verdict"] for item in items)
        valid = [item for item in items if item["verdict"] != "invalid"]
        action_ok = [item for item in valid if item["predicted_action"] == item["expected_decision"]["action"]]
        gold_calls = [item for item in items if item["expected_decision"]["action"] == "call"]
        tool_ok = [item for item in gold_calls if item["predicted_action"] == "call" and item["predicted_tool"] == item["expected_decision"]["tool"] and item["verdict"] != "invalid"]
        output.setdefault(split, {})[category] = {
            "count": len(items),
            "verdict_counts": dict(sorted(counts.items())),
            "format_valid_rate": round(len(valid) / len(items), 4),
            "action_correct_rate_among_valid": round(len(action_ok) / len(valid), 4) if valid else None,
            "tool_correct_rate_among_gold_calls": round(len(tool_ok) / len(gold_calls), 4) if gold_calls else None,
            "fully_correct_rate": round(counts["correct"] / len(items), 4),
            "manual_review_count": counts["manual_review"],
            "mean_latency_seconds": round(sum(item["latency_seconds"] for item in items) / len(items), 3),
        }
    return output


def write_comparison(run_dir):
    summaries = {}
    for mode in MODES:
        path = run_dir / f"{mode}_summary.json"
        if not path.exists():
            return
        summaries[mode] = json.loads(path.read_text(encoding="utf-8"))
    lines = [
        "# Qwen3-0.6B 微调前基线对比",
        "",
        "两组使用同一模型版本、冻结测试集、贪心解码和确定性评分器。test 含 120 条生成样本与 24 条手工案例；unseen_test 单独统计。",
        "",
        "| 提示词 | 集合 | 样本 | 格式合法率 | 动作正确率（合法样本内） | 工具选择正确率（应调用样本内） | 全对率 | 待人工复核 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for split in ("test", "unseen_test"):
            item = summaries[mode][split]["all"]
            action = item["action_correct_rate_among_valid"]
            lines.append(
                f"| {mode} | {split} | {item['count']} | {item['format_valid_rate']:.1%} | "
                f"{action:.1%}" if action is not None else
                f"| {mode} | {split} | {item['count']} | {item['format_valid_rate']:.1%} | —"
            )
            lines[-1] += (
                f" | {item['tool_correct_rate_among_gold_calls']:.1%} | "
                f"{item['fully_correct_rate']:.1%} | {item['manual_review_count']} |"
            )
    lines += ["", "## 增强提示词的类别结果", "", "| 集合 | 类别 | 样本 | 全对 | 无效 | 判错 |", "|---|---|---:|---:|---:|---:|"]
    for split in ("test", "unseen_test"):
        for category, item in sorted(summaries["enhanced"][split].items()):
            if category == "all":
                continue
            counts = item["verdict_counts"]
            lines.append(f"| {split} | {category} | {item['count']} | {counts.get('correct', 0)} | {counts.get('invalid', 0)} | {counts.get('wrong', 0)} |")
    lines += [
        "", "## 判读", "",
        "基础提示词没有产生合格决策；原始输出保存在 basic_predictions.jsonl。增强提示词改善了格式，但动作与参数错误仍多。",
        "`manual_review` 为 0 仅表示本轮没有落入文本参数复核通道，不表示所有文本参数都正确。",
        "每题的输入、原始输出、解析结果、标准答案、判定和原因见两个 predictions.jsonl 文件。",
    ]
    (run_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="运行 Qwen3 微调前基线")
    parser.add_argument("--mode", choices=(*MODES, "both"), default="both")
    parser.add_argument("--limit", type=int, help="每个测试集合只跑前 N 条；仅用于冒烟，不作为正式成绩")
    parser.add_argument("--run-dir", type=Path, help="指定已有运行目录以续跑")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="本地模型目录")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")

    check = subprocess.run(
        [sys.executable, str(ROOT / "data" / "audit_dataset.py"), "--check-frozen"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if check.returncode:
        raise RuntimeError("测试集冻结校验失败，停止基线实验：\n" + check.stdout + check.stderr)
    print("冻结测试集校验通过", flush=True)
    model_dir = args.model_dir
    manifest_path = model_dir / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not ((model_dir / "model.safetensors").is_file() or (model_dir / "model.safetensors.index.json").is_file()):
        raise RuntimeError("模型权重不存在")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 没有 CUDA；请使用 fllam 环境")
    rows_by_split = load_samples()
    prefix = enhanced_prefix()
    config = {
        "model_repo": manifest["repo_id"],
        "model_revision": manifest["revision"],
        "model_dir": str(model_dir),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "enable_thinking": False,
        "enhanced_prompt": prefix,
        "frozen_manifest_sha256": sha256(ROOT / "data" / "datasets" / "frozen_v0.json"),
        "limit_per_split": args.limit,
    }
    run_dir = args.run_dir or ROOT / "reports" / "baselines" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise RuntimeError("续跑目录的模型、提示词或解码配置与当前运行不一致")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, torch_dtype=torch.float16).to("cuda")
    model.eval()
    model.generation_config = GenerationConfig.from_model_config(model.config)
    base_ctx = audit.base_ctx()
    unseen_manifest = json.loads((ROOT / "data" / "unseen_tools" / "isolation_manifest.json").read_text(encoding="utf-8"))
    unseen_ctx, _, _ = audit.unseen_ctx(base_ctx, unseen_manifest)

    modes = MODES if args.mode == "both" else (args.mode,)
    for mode in modes:
        prediction_path = run_dir / f"{mode}_predictions.jsonl"
        existing_rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()] if prediction_path.exists() else []
        completed = {(row["split"], row["id"]) for row in existing_rows}
        with prediction_path.open("a", encoding="utf-8") as output:
            for split in ("test", "unseen_test"):
                dataset_rows = rows_by_split[split][:args.limit] if args.limit else rows_by_split[split]
                ctx = unseen_ctx if split == "unseen_test" else base_ctx
                for index, (case, base_input) in enumerate(dataset_rows, 1):
                    if (split, case["id"]) in completed:
                        continue
                    input_text = base_input if mode == "basic" else prefix + "\n\n" + base_input
                    encoded = tokenizer.apply_chat_template(
                        [{"role": "user", "content": input_text}],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=False,
                        return_tensors="pt",
                    ).to("cuda")
                    torch.cuda.synchronize()
                    start = perf_counter()
                    with torch.inference_mode():
                        generated = model.generate(
                            encoded,
                            attention_mask=torch.ones_like(encoded),
                            max_new_tokens=MAX_NEW_TOKENS,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    torch.cuda.synchronize()
                    latency = perf_counter() - start
                    output_ids = generated[0][encoded.shape[-1]:]
                    raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
                    try:
                        decision = json.loads(raw)
                    except json.JSONDecodeError:
                        decision = None
                    verdict, notes = scoring.score_case_decision(case, decision, ctx)
                    row = {
                        "id": case["id"], "split": split, "category": case.get("category"),
                        "user_request": case["user_request"], "available_tools": case["available_tools"],
                        "expected_decision": case["expected_decision"],
                        "input": input_text, "raw_output": raw, "parsed_decision": decision,
                        "predicted_action": decision.get("action") if isinstance(decision, dict) else None,
                        "predicted_tool": decision.get("tool") if isinstance(decision, dict) else None,
                        "verdict": verdict, "notes": notes,
                        "input_tokens": encoded.shape[-1], "output_tokens": len(output_ids),
                        "latency_seconds": round(latency, 3),
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    existing_rows.append(row)
                    completed.add((split, case["id"]))
                    print(f"{mode} {split} {index}/{len(dataset_rows)} {case['id']} {verdict}", flush=True)
        summary = summarize(existing_rows)
        (run_dir / f"{mode}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{mode} 完成；结果：{prediction_path}", flush=True)
    if args.limit is None and all((run_dir / f"{mode}_summary.json").exists() for mode in MODES):
        write_comparison(run_dir)
    print(f"运行目录：{run_dir}", flush=True)


if __name__ == "__main__":
    main()
