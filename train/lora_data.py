"""Build response-only LoRA samples from train/val without touching test data."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "lora_v0.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(config_path: Path = CONFIG_PATH):
    config = load_json(config_path)
    baseline = load_json(ROOT / config["baseline_config"])
    model_dir = Path(config["model_dir"])
    manifest = load_json(model_dir / "download_manifest.json")
    if config["base_revision"] != manifest["revision"]:
        raise RuntimeError("训练模型 revision 与本地模型清单不一致")
    if not config.get("prompt_source_only", False) and config["base_revision"] != baseline["model_revision"]:
        raise RuntimeError("训练模型 revision 与基线不一致")
    if not baseline["enhanced_prompt"]:
        raise RuntimeError("增强提示词基线不存在")
    return config, baseline


def load_split(split, tokenizer, config, enhanced_prompt):
    if split not in ("train", "val"):
        raise ValueError("训练准备只允许读取 train 和 val")
    version = config.get("data_version", "v0")
    if version not in ("v0", "v1", "v2", "v3"):
        raise ValueError(f"不支持的数据版本: {version}")
    dataset = load_json(ROOT / "data" / "datasets" / f"{split}_{version}.json")["samples"]
    input_path = ROOT / "data" / "datasets" / f"{split}_{version}.input.jsonl"
    inputs = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
    by_id = {case["id"]: case for case in dataset}
    if len(by_id) != len(dataset) or {item["id"] for item in inputs} != set(by_id):
        raise RuntimeError(f"{split} 输入投影与标准答案不匹配")
    result = []
    for item in inputs:
        case = by_id[item["id"]]
        prompt = enhanced_prompt + "\n\n" + item["input"]
        answer = json.dumps(case["expected_decision"], ensure_ascii=False, separators=(",", ":"))
        user_turn = [{"role": "user", "content": prompt}]
        prefix = tokenizer.apply_chat_template(
            user_turn, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        full = tokenizer.apply_chat_template(
            [*user_turn, {"role": "assistant", "content": answer}],
            tokenize=True, add_generation_prompt=False, enable_thinking=False,
        )
        if full[:len(prefix)] != prefix:
            raise RuntimeError(f"{item['id']} 对话模板前缀不一致，不能安全掩码")
        if len(full) > config["max_seq_length"]:
            raise RuntimeError(f"{item['id']} 长度 {len(full)} 超过上限 {config['max_seq_length']}，拒绝截断")
        labels = [-100] * len(prefix) + full[len(prefix):]
        if all(token == -100 for token in labels):
            raise RuntimeError(f"{item['id']} 没有可训练的答案 token")
        result.append({"id": item["id"], "input_ids": full, "labels": labels,
                       "prompt_tokens": len(prefix), "answer_tokens": len(full) - len(prefix)})
    return result


def print_stats(split, samples):
    lengths = [len(sample["input_ids"]) for sample in samples]
    answers = [sample["answer_tokens"] for sample in samples]
    print(f"{split}: {len(samples)} 条，总长度均值 {mean(lengths):.1f} / 最大 {max(lengths)}，"
          f"答案 token 均值 {mean(answers):.1f} / 最小 {min(answers)}")
