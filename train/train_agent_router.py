"""在现有 1.7B LoRA 上继续训练五路 Agent 分诊适配器。"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api.agent_router import TOOL_TO_ROUTE, build_route_prompt  # noqa: E402


DATA = ROOT / "data" / "agent_route_training_v0.json"
SOURCE = ROOT / "train" / "outputs" / "lora_1p7b_v0"
OUTPUT = ROOT / "train" / "outputs" / "agent_router_v0"
NOW = "2026-10-01T10:00"
SEED = 20261004


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(tokenizer, seeds: dict[str, list[str]], augment: bool) -> list[dict]:
    route_to_tool = {route: tool for tool, route in TOOL_TO_ROUTE.items()}
    items = []
    for route, queries in seeds.items():
        if route not in route_to_tool:
            raise ValueError(f"未知路由: {route}")
        for query in queries:
            variants = [query, "麻烦" + query] if augment else [query]
            for user_request in variants:
                prompt = build_route_prompt(user_request, NOW)
                answer = json.dumps({"action": "call", "tool": route_to_tool[route],
                                     "arguments": {"query": user_request}}, ensure_ascii=False,
                                    separators=(",", ":"))
                turn = [{"role": "user", "content": prompt}]
                prefix = tokenizer.apply_chat_template(turn, tokenize=True,
                                                       add_generation_prompt=True, enable_thinking=False)
                full = tokenizer.apply_chat_template(
                    [*turn, {"role": "assistant", "content": answer}], tokenize=True,
                    add_generation_prompt=False, enable_thinking=False,
                )
                if full[:len(prefix)] != prefix or len(full) > 1280:
                    raise RuntimeError(f"样本不能安全掩码或超长: {query}")
                items.append({"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]})
    return items


def batch(sample: dict) -> dict:
    ids = torch.tensor([sample["input_ids"]], device="cuda")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([sample["labels"]], device="cuda")}


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"训练目录已存在，拒绝覆盖: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 显卡训练")
    manifest = json.loads((SOURCE / "run_manifest.json").read_text(encoding="utf-8"))
    model_dir = Path(manifest["config"]["model_dir"])
    if not model_dir.is_dir():
        raise RuntimeError(f"基座模型不存在: {model_dir}")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    seeds = json.loads(DATA.read_text(encoding="utf-8"))
    train = prepare(tokenizer, seeds["train"], augment=True)
    val = prepare(tokenizer, seeds["val"], augment=False)
    print(f"训练 {len(train)} 条；验证 {len(val)} 条；最长 {max(len(x['input_ids']) for x in train)} token", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float16,
    ).to("cuda")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, SOURCE / "best", is_trainable=True)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not trainable:
        raise RuntimeError("没有可训练的 LoRA 参数")
    print(f"LoRA 可训练参数 {trainable:,}", flush=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=5e-5)
    scaler = torch.amp.GradScaler("cuda")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "run_manifest.json").write_text(json.dumps({
        "model": "Qwen3-1.7B", "source_adapter_sha256": sha256(SOURCE / "best" / "adapter_model.safetensors"),
        "training_data_sha256": sha256(DATA), "train_count": len(train), "val_count": len(val),
        "seed": SEED, "epochs": 2, "learning_rate": 5e-5, "trainable_parameters": trainable,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    start = perf_counter()
    best = float("inf")
    torch.cuda.reset_peak_memory_stats()
    for epoch in (1, 2):
        order = list(range(len(train)))
        random.Random(SEED + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for position, index in enumerate(order, 1):
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**batch(train[index])).loss
            losses.append(loss.item())
            scaler.scale(loss / 8).backward()
            if position % 8 == 0 or position == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if position % 40 == 0 or position == len(order):
                print(f"epoch {epoch}/2 sample {position}/{len(train)} loss {loss.item():.4f}", flush=True)
        model.eval()
        with torch.inference_mode():
            val_loss = sum(model(**batch(item)).loss.item() for item in val) / len(val)
        model.train()
        if val_loss < best:
            best = val_loss
            model.save_pretrained(OUTPUT / "best", safe_serialization=True)
            tokenizer.save_pretrained(OUTPUT / "best")
        row = {"epoch": epoch, "train_loss": round(sum(losses) / len(losses), 6),
               "val_loss": round(val_loss, 6),
               "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2),
               "elapsed_seconds": round(perf_counter() - start, 1)}
        with (OUTPUT / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"epoch {epoch} 完成: {row}", flush=True)
    print(f"训练完成；最佳验证损失 {best:.4f}；适配器 {OUTPUT / 'best'}", flush=True)


if __name__ == "__main__":
    main()
