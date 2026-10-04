"""Measure frozen dataset lengths with the downloaded Qwen3 tokenizer."""

import json
import sys
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
import generate_dataset as dataset_generator

MODEL_DIR = Path(r"models/Qwen3-0.6B")
SPLITS = ("train", "val", "test", "test_unseen")


def describe(values):
    return {
        "min": min(values),
        "mean": round(mean(values), 1),
        "max": max(values),
    }


def main():
    manifest = json.loads((MODEL_DIR / "download_manifest.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
    report = {"model": manifest["repo_id"], "revision": manifest["revision"], "splits": {}}

    for split in SPLITS:
        inputs_path = ROOT / "data" / "datasets" / f"{split}_v0.input.jsonl"
        dataset_path = ROOT / "data" / "datasets" / f"{split}_v0.json"
        inputs = [json.loads(line) for line in inputs_path.read_text(encoding="utf-8").splitlines()]
        samples = json.loads(dataset_path.read_text(encoding="utf-8"))["samples"]
        if split == "test":
            handwritten = json.loads((ROOT / "data" / "cases" / "handwritten_v0.json").read_text(encoding="utf-8"))["cases"]
            docs = dataset_generator.load_tool_docs()
            inputs.extend({"id": case["id"], "input": dataset_generator.render_input(case, docs)} for case in handwritten)
            samples.extend(handwritten)
        by_id = {sample["id"]: sample for sample in samples}
        raw_lengths, chat_lengths, answer_lengths, total_lengths = [], [], [], []

        for item in inputs:
            prompt = item["input"]
            raw = len(tokenizer.encode(prompt, add_special_tokens=False))
            chat = len(tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            answer = json.dumps(by_id[item["id"]]["expected_decision"], ensure_ascii=False, separators=(",", ":"))
            answer_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
            raw_lengths.append(raw)
            chat_lengths.append(chat)
            answer_lengths.append(answer_tokens)
            total_lengths.append(chat + answer_tokens + 1)

        report["splits"][split] = {
            "samples": len(inputs),
            "raw_input_tokens": describe(raw_lengths),
            "chat_prompt_tokens": describe(chat_lengths),
            "gold_answer_tokens": describe(answer_lengths),
            "chat_plus_gold_and_eos_tokens": describe(total_lengths),
            "over_1280": sum(length > 1280 for length in total_lengths),
        }

    output = ROOT / "reports" / "token_budget_v0.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告: {output}")


if __name__ == "__main__":
    main()
