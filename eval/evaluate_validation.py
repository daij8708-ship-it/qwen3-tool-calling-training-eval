"""Score a LoRA adapter on a train/validation split, never on frozen tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import audit_dataset as audit  # noqa: E402
import validate_contracts as scoring  # noqa: E402
from run_baselines import MAX_NEW_TOKENS, MODEL_DIR, summarize  # noqa: E402


def load_rows(split: str, version: str):
    if split not in ("train", "val"):
        raise ValueError("只允许分析 train/val，冻结测试集不能用于第二轮设计")
    data_dir = ROOT / "data" / "datasets"
    samples = json.loads((data_dir / f"{split}_{version}.json").read_text(encoding="utf-8"))["samples"]
    inputs = [json.loads(line) for line in (data_dir / f"{split}_{version}.input.jsonl").read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in samples}
    if len(by_id) != len(samples) or {row["id"] for row in inputs} != set(by_id):
        raise RuntimeError("输入投影与标准答案不匹配")
    return [(by_id[row["id"]], row["input"]) for row in inputs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--version", choices=("v0", "v1"), default="v0")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--adapter", type=Path, default=ROOT / "train" / "outputs" / "lora_v0" / "best")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "reports" / "lora_v0_validation")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    baseline = json.loads((ROOT / "reports" / "baselines" / "qwen3_0p6b_pre_lora_v0" / "run_config.json").read_text(encoding="utf-8"))
    prefix = baseline["enhanced_prompt"]
    model_manifest = json.loads((args.model_dir / "download_manifest.json").read_text(encoding="utf-8"))
    ctx = audit.base_ctx()
    rows = load_rows(args.split, args.version)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {"split": args.split, "version": args.version, "adapter": str(args.adapter.resolve()),
              "model_dir": str(args.model_dir.resolve()), "model_revision": model_manifest["revision"], "max_new_tokens": MAX_NEW_TOKENS,
              "do_sample": False, "enable_thinking": False}
    config_file = args.run_dir / "run_config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")) != config:
        raise RuntimeError("续跑配置不一致")
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prediction_file = args.run_dir / "predictions.jsonl"
    existing = [json.loads(line) for line in prediction_file.read_text(encoding="utf-8").splitlines()] if prediction_file.exists() else []
    completed = {row["id"] for row in existing}

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(args.model_dir), local_files_only=True, torch_dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False)
    model.eval()
    model.generation_config = GenerationConfig.from_model_config(model.config)
    with prediction_file.open("a", encoding="utf-8") as stream:
        for index, (case, base_input) in enumerate(rows, 1):
            if case["id"] in completed:
                continue
            encoded = tokenizer.apply_chat_template(
                [{"role": "user", "content": prefix + "\n\n" + base_input}],
                tokenize=True, add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt",
            ).to("cuda")
            torch.cuda.synchronize()
            started = perf_counter()
            with torch.inference_mode():
                generated = model.generate(encoded, attention_mask=torch.ones_like(encoded),
                                           max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                           pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            raw = tokenizer.decode(generated[0][encoded.shape[-1]:], skip_special_tokens=True).strip()
            try:
                decision = json.loads(raw)
            except json.JSONDecodeError:
                decision = None
            verdict, notes = scoring.score_case_decision(case, decision, ctx)
            item = {"id": case["id"], "split": args.split, "category": case["category"],
                    "user_request": case["user_request"], "expected_decision": case["expected_decision"],
                    "raw_output": raw, "parsed_decision": decision,
                    "predicted_action": decision.get("action") if isinstance(decision, dict) else None,
                    "predicted_tool": decision.get("tool") if isinstance(decision, dict) else None,
                    "verdict": verdict, "notes": notes,
                    "latency_seconds": round(perf_counter() - started, 3)}
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
            existing.append(item)
            completed.add(case["id"])
            print(f"{args.split} {index}/{len(rows)} {case['id']} {verdict}", flush=True)
    (args.run_dir / "summary.json").write_text(json.dumps(summarize(existing), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"验证集评估完成: {args.run_dir}")


if __name__ == "__main__":
    main()
