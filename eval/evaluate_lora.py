"""Evaluate the validation-selected LoRA adapter on the frozen test sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import audit_dataset as audit  # noqa: E402
import validate_contracts as scoring  # noqa: E402
from run_baselines import MAX_NEW_TOKENS, MODEL_DIR, load_samples, sha256, summarize  # noqa: E402

BASELINE_DIR = ROOT / "reports" / "baselines" / "qwen3_0p6b_pre_lora_v0"
TRAIN_DIR = ROOT / "train" / "outputs" / "lora_v0"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(run_dir, adapter_summary, baseline_summary):
    lines = [
        "# LoRA 与增强提示词基线对比",
        "",
        "两组使用同一基座模型、同一增强提示词、同一冻结测试集、贪心解码和确定性评分器。",
        "适配器按验证集损失选择；测试集未参与训练或选参。",
        "",
        "| 模型 | 集合 | 样本 | 格式合法率 | 动作正确率（合法样本内） | 工具选择正确率（应调用样本内） | 全对率 | 人工复核 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in (("原模型增强提示词", baseline_summary), ("LoRA 最佳适配器", adapter_summary)):
        for split in ("test", "unseen_test"):
            item = summary[split]["all"]
            action = item["action_correct_rate_among_valid"]
            lines.append(
                f"| {label} | {split} | {item['count']} | {item['format_valid_rate']:.1%} | "
                f"{action:.1%}" if action is not None else
                f"| {label} | {split} | {item['count']} | {item['format_valid_rate']:.1%} | —"
            )
            lines[-1] += (
                f" | {item['tool_correct_rate_among_gold_calls']:.1%} | "
                f"{item['fully_correct_rate']:.1%} | {item['manual_review_count']} |"
            )
    lines += ["", "## 类别变化", "", "| 集合 | 类别 | 样本 | 基线全对 | LoRA 全对 | 变化 | LoRA 格式无效 |", "|---|---|---:|---:|---:|---:|---:|"]
    for split in ("test", "unseen_test"):
        for category, item in sorted(adapter_summary[split].items()):
            if category == "all":
                continue
            counts = item["verdict_counts"]
            baseline_correct = baseline_summary[split][category]["verdict_counts"].get("correct", 0)
            lora_correct = counts.get("correct", 0)
            lines.append(
                f"| {split} | {category} | {item['count']} | {baseline_correct} | "
                f"{lora_correct} | {lora_correct - baseline_correct:+d} | {counts.get('invalid', 0)} |"
            )
    lines += [
        "", "## 代表性未解决错误", "",
        "- `DS-4059`：缺结束时间却调用 `create_event`，参数不完整；基线原本正确追问。",
        "- `TC-19`：用户只给月份，模型自行补出具体日期和时刻；基线原本正确追问。",
        "- `UN-044`：排序方向未给出，模型猜测 `desc`；基线原本正确追问。",
        "- `DS-4023`：聚合筛选字段被放在错误层级，未通过工具 Schema。",
        "- `DS-4085`：要求删除记录时调用只读 `get_record`，没有按能力边界拒绝。",
        "", "完整逐题输入、输出与判分原因见 adapter_predictions.jsonl；模型与数据指纹见 run_config.json。",
        "测试集类别变化只用于报告局限，不用于挑选新超参数；后续改动应先在 train/val 上决定。",
    ]
    (run_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="评估验证集选出的 LoRA 适配器")
    parser.add_argument("--limit", type=int, help="每个集合只跑前 N 条，仅用于冒烟")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR, help="训练输出目录")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR, help="本地基座模型目录")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports" / "lora_v0_eval")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")
    check = subprocess.run(
        [sys.executable, str(ROOT / "data" / "audit_dataset.py"), "--check-frozen"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if check.returncode:
        raise RuntimeError("冻结测试集校验失败:\n" + check.stdout + check.stderr)
    baseline_config = read_json(BASELINE_DIR / "run_config.json")
    train_dir = args.train_dir if args.train_dir.is_absolute() else ROOT / args.train_dir
    adapter_dir = train_dir / "best"
    train_manifest = read_json(train_dir / "run_manifest.json")
    model_manifest = read_json(args.model_dir / "download_manifest.json")
    epochs = [json.loads(line) for line in (train_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    if len(epochs) != train_manifest["config"]["epochs"]:
        raise RuntimeError("训练尚未完成全部轮次")
    best_epoch = min(epochs, key=lambda row: row["val_loss"])["epoch"]
    adapter_file = adapter_dir / "adapter_model.safetensors"
    selected_file = train_dir / f"epoch_{best_epoch}" / "adapter_model.safetensors"
    if not adapter_file.is_file() or sha256(adapter_file) != sha256(selected_file):
        raise RuntimeError("best 适配器与最小验证损失轮次不一致")
    prefix = baseline_config["enhanced_prompt"]
    prompt_hash = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
    if prompt_hash != train_manifest["baseline_prompt_sha256"]:
        raise RuntimeError("训练提示词与增强基线不一致")
    if model_manifest["revision"] != train_manifest["base_revision"]:
        raise RuntimeError("评测基座模型版本与训练清单不一致")
    if not train_manifest["config"].get("prompt_source_only", False) and baseline_config["model_revision"] != train_manifest["base_revision"]:
        raise RuntimeError("基座模型版本与增强基线不一致")
    data_version = train_manifest["config"].get("data_version", "v0")
    for split in ("train", "val"):
        source = ROOT / "data" / "datasets" / f"{split}_{data_version}.json"
        if sha256(source) != train_manifest[f"{split}_dataset_sha256"]:
            raise RuntimeError(f"{split} 训练数据与运行清单不一致")
    if baseline_config["frozen_manifest_sha256"] != sha256(ROOT / "data" / "datasets" / "frozen_v0.json"):
        raise RuntimeError("冻结清单与基线不一致")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("当前环境不可使用 CUDA")
    config = {
        "model_revision": baseline_config["model_revision"],
        "adapter_path": str(adapter_dir),
        "adapter_sha256": sha256(adapter_file),
        "best_epoch": best_epoch,
        "prompt_sha256": prompt_hash,
        "frozen_manifest_sha256": baseline_config["frozen_manifest_sha256"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "enable_thinking": False,
        "limit_per_split": args.limit,
    }
    if train_manifest["config"].get("prompt_source_only", False):
        config["evaluated_model_revision"] = model_manifest["revision"]
        config["model_dir"] = str(args.model_dir.resolve())
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        if read_json(config_path) != config:
            raise RuntimeError("续跑目录的模型、提示词或数据配置不一致")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(args.model_dir), local_files_only=True, torch_dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    model.eval()
    model.generation_config = GenerationConfig.from_model_config(model.config)
    base_ctx = audit.base_ctx()
    unseen_manifest = read_json(ROOT / "data" / "unseen_tools" / "isolation_manifest.json")
    unseen_ctx, _, _ = audit.unseen_ctx(base_ctx, unseen_manifest)
    rows_by_split = load_samples()
    prediction_path = run_dir / "adapter_predictions.jsonl"
    existing = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines()] if prediction_path.exists() else []
    completed = {(row["split"], row["id"]) for row in existing}
    with prediction_path.open("a", encoding="utf-8") as stream:
        for split in ("test", "unseen_test"):
            samples = rows_by_split[split][:args.limit] if args.limit else rows_by_split[split]
            ctx = unseen_ctx if split == "unseen_test" else base_ctx
            for index, (case, base_input) in enumerate(samples, 1):
                if (split, case["id"]) in completed:
                    continue
                input_text = prefix + "\n\n" + base_input
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": input_text}], tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
                ).to("cuda")
                torch.cuda.synchronize()
                start = perf_counter()
                with torch.inference_mode():
                    generated = model.generate(
                        encoded, attention_mask=torch.ones_like(encoded),
                        max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
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
                    "expected_decision": case["expected_decision"], "input": input_text,
                    "raw_output": raw, "parsed_decision": decision,
                    "predicted_action": decision.get("action") if isinstance(decision, dict) else None,
                    "predicted_tool": decision.get("tool") if isinstance(decision, dict) else None,
                    "verdict": verdict, "notes": notes,
                    "input_tokens": encoded.shape[-1], "output_tokens": len(output_ids),
                    "latency_seconds": round(latency, 3),
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                existing.append(row)
                completed.add((split, case["id"]))
                print(f"{split} {index}/{len(samples)} {case['id']} {verdict}", flush=True)
    summary = summarize(existing)
    (run_dir / "adapter_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.limit is None and baseline_config["model_revision"] == train_manifest["base_revision"]:
        compare(run_dir, summary, read_json(BASELINE_DIR / "enhanced_summary.json"))
    print(f"LoRA 评测完成：{run_dir}")


if __name__ == "__main__":
    main()
