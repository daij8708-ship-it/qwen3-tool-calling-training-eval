"""Evaluate a pinned base model or adapter on the frozen new acceptance set."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "data"), str(ROOT / "eval")]
import audit_dataset as audit  # noqa: E402
import validate_contracts as scoring  # noqa: E402
from run_baselines import MAX_NEW_TOKENS  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["acceptance_v1", "acceptance_v2b", "acceptance_v3", "acceptance_v4", "acceptance_v5b"], default="acceptance_v1")
    args = parser.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    dataset_file = ROOT / "data" / "datasets" / f"{args.dataset}.json"
    input_file = ROOT / "data" / "datasets" / f"{args.dataset}.input.jsonl"
    manifest = json.loads((ROOT / "data" / "datasets" / f"{args.dataset}_manifest.json").read_text(encoding="utf-8"))
    if sha(dataset_file) != manifest["dataset_sha256"] or sha(input_file) != manifest["input_sha256"]:
        raise RuntimeError("验收集哈希与冻结清单不一致")
    samples = json.loads(dataset_file.read_text(encoding="utf-8"))["samples"]
    inputs = {row["id"]: row["input"] for row in map(json.loads, input_file.read_text(encoding="utf-8").splitlines())}
    if len(inputs) != len(samples) or set(inputs) != {row["id"] for row in samples}:
        raise RuntimeError("验收集输入投影不一致")
    model_manifest = json.loads((args.model_dir / "download_manifest.json").read_text(encoding="utf-8"))
    prompt = json.loads((ROOT / "reports" / "baselines" / "qwen3_0p6b_pre_lora_v0" / "run_config.json").read_text(encoding="utf-8"))["enhanced_prompt"]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {"dataset": args.dataset, "model_dir": str(args.model_dir.resolve()), "model_revision": model_manifest["revision"],
              "adapter": str(args.adapter.resolve()) if args.adapter else None,
              "adapter_sha256": sha(args.adapter / "adapter_model.safetensors") if args.adapter else None,
              "dataset_sha256": manifest["dataset_sha256"], "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
              "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False}
    config_file = args.run_dir / "run_config.json"
    if config_file.exists() and json.loads(config_file.read_text(encoding="utf-8")) != config:
        raise RuntimeError("续跑配置不一致")
    config_file.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions = args.run_dir / "predictions.jsonl"
    existing = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()] if predictions.exists() else []
    done = {row["id"] for row in existing}
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    tok = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(str(args.model_dir), local_files_only=True, torch_dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, str(args.adapter), is_trainable=False) if args.adapter else base
    model.eval()
    model.generation_config = GenerationConfig.from_model_config(model.config)
    base_ctx = audit.base_ctx()
    unseen_manifest = json.loads((ROOT / "data" / "unseen_tools" / "isolation_manifest.json").read_text(encoding="utf-8"))
    unseen_ctx, _, _ = audit.unseen_ctx(base_ctx, unseen_manifest)
    with predictions.open("a", encoding="utf-8") as stream:
        for i, case in enumerate(samples, 1):
            if case["id"] in done:
                continue
            text = prompt + "\n\n" + inputs[case["id"]]
            encoded = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                              add_generation_prompt=True, enable_thinking=False,
                                              return_tensors="pt").to("cuda")
            torch.cuda.synchronize()
            started = perf_counter()
            with torch.inference_mode():
                generated = model.generate(encoded, attention_mask=torch.ones_like(encoded),
                                           max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                           pad_token_id=tok.eos_token_id)
            torch.cuda.synchronize()
            output_ids = generated[0][encoded.shape[-1]:]
            raw = tok.decode(output_ids, skip_special_tokens=True).strip()
            try:
                decision = json.loads(raw)
            except json.JSONDecodeError:
                decision = None
            ctx = unseen_ctx if case["group"].startswith("unseen") else base_ctx
            verdict, notes = scoring.score_case_decision(case, decision, ctx)
            row = {"id": case["id"], "group": case["group"], "user_request": case["user_request"],
                   "expected_decision": case["expected_decision"], "raw_output": raw,
                   "parsed_decision": decision, "verdict": verdict, "notes": notes,
                   "output_tokens": len(output_ids), "latency_seconds": round(perf_counter() - started, 3)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            existing.append(row)
            print(f"acceptance {i}/{len(samples)} {case['id']} {verdict}", flush=True)
    groups = defaultdict(list)
    for row in existing:
        groups["all"].append(row)
        groups[row["group"]].append(row)
    summary = {group: {"count": len(rows), "verdict_counts": dict(Counter(row["verdict"] for row in rows)),
                       "correct_rate": round(sum(row["verdict"] == "correct" for row in rows) / len(rows), 4),
                       "max_output_tokens": max(row["output_tokens"] for row in rows)}
               for group, rows in sorted(groups.items())}
    (args.run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"验收评测完成: {args.run_dir}")


if __name__ == "__main__":
    main()
