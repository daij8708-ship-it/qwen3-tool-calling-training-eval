"""从五路 LoRA v0 继续训练：增加跨 Agent 多任务交还云端的输出。"""

from __future__ import annotations

import argparse
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
from api.agent_router import TOOL_TO_ROUTE, build_route_prompt_v1  # noqa: E402

SINGLE_DATA = ROOT / "data" / "agent_route_training_v0.json"
MULTI_DATA = ROOT / "data" / "agent_route_multi_v1.json"
BASE_ADAPTER = ROOT / "train" / "outputs" / "agent_router_v0" / "best"
MODEL_MANIFEST = ROOT / "train" / "outputs" / "lora_1p7b_v0" / "run_manifest.json"
OUTPUT = ROOT / "train" / "outputs" / "agent_router_v1"
NOW = "2026-10-01T10:00"
SEED = 20261005


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def examples(single: dict[str, list[str]], multi: list[str]) -> list[tuple[str, str | None]]:
    return [(query, route) for route, queries in single.items() for query in queries] + [
        (query, None) for query in multi
    ]


def prepare(tokenizer, items: list[tuple[str, str | None]], augment: bool) -> list[dict]:
    route_to_tool = {route: tool for tool, route in TOOL_TO_ROUTE.items()}
    encoded_items = []
    for query, route in items:
        if route is not None and route not in route_to_tool:
            raise ValueError(f"未知路由: {route}")
        for user_request in ([query, "麻烦" + query] if augment else [query]):
            prompt = build_route_prompt_v1(user_request, NOW)
            answer = (
                {"action": "delegate", "reason": "multi_task"}
                if route is None else
                {"action": "call", "tool": route_to_tool[route],
                 "arguments": {"query": user_request}}
            )
            rendered = json.dumps(answer, ensure_ascii=False, separators=(",", ":"))
            turn = [{"role": "user", "content": prompt}]
            prefix = tokenizer.apply_chat_template(
                turn, tokenize=True, add_generation_prompt=True, enable_thinking=False,
            )
            full = tokenizer.apply_chat_template(
                [*turn, {"role": "assistant", "content": rendered}], tokenize=True,
                add_generation_prompt=False, enable_thinking=False,
            )
            if full[:len(prefix)] != prefix or len(full) > 1280:
                raise RuntimeError(f"样本不能安全掩码或超长: {query}")
            encoded_items.append({
                "input_ids": full,
                "labels": [-100] * len(prefix) + full[len(prefix):],
            })
    return encoded_items


def batch(item: dict) -> dict:
    ids = torch.tensor([item["input_ids"]], device="cuda")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "labels": torch.tensor([item["labels"]], device="cuda")}


def main() -> None:
    parser = argparse.ArgumentParser(description="继续训练五路 Agent 与多意图分诊 LoRA")
    parser.add_argument("--source-adapter", type=Path, default=BASE_ADAPTER)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--extra-data", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"训练目录已存在，拒绝覆盖: {args.output}")
    if args.epochs < 1 or args.learning_rate <= 0:
        parser.error("epochs 和 learning-rate 必须为正数")
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 显卡训练")
    base_manifest = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    model_dir = Path(base_manifest["config"]["model_dir"])
    if not model_dir.is_dir() or not (args.source_adapter / "adapter_model.safetensors").is_file():
        raise RuntimeError("基座模型或源路由适配器缺失")
    original = json.loads(SINGLE_DATA.read_text(encoding="utf-8"))
    added = json.loads(MULTI_DATA.read_text(encoding="utf-8"))
    extra = json.loads(args.extra_data.read_text(encoding="utf-8")) if args.extra_data else None
    train_single = {
        route: (original["train"][route] + added["train_single_boundary"].get(route, [])
                + (extra["train_single"].get(route, []) if extra else []))
        for route in original["train"]
    }
    val_single = {
        route: (original["val"][route] + added["val_single_boundary"].get(route, [])
                + (extra["val_single"].get(route, []) if extra else []))
        for route in original["val"]
    }
    train_multi = added["train_multi"] + (extra["train_multi"] if extra else [])
    val_multi = added["val_multi"] + (extra["val_multi"] if extra else [])
    train_examples = examples(train_single, train_multi)
    val_examples = examples(val_single, val_multi)
    train_queries = {query for query, _ in train_examples}
    val_queries = {query for query, _ in val_examples}
    if len(train_queries) != len(train_examples) or len(val_queries) != len(val_examples) or train_queries & val_queries:
        raise RuntimeError("训练/验证样本重复")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    train = prepare(tokenizer, train_examples, augment=True)
    val = prepare(tokenizer, val_examples, augment=False)
    print(f"训练 {len(train)} 条（多任务 {len(train_multi) * 2} 条）；验证 {len(val)} 条；最长 {max(len(x['input_ids']) for x in train)} token", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float16,
    ).to("cuda")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, args.source_adapter, is_trainable=True)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    scaler = torch.amp.GradScaler("cuda")
    args.output.mkdir(parents=True)
    (args.output / "run_manifest.json").write_text(json.dumps({
        "model": "Qwen3-1.7B", "source_adapter_sha256": sha256(args.source_adapter / "adapter_model.safetensors"),
        "single_data_sha256": sha256(SINGLE_DATA), "multi_data_sha256": sha256(MULTI_DATA),
        "extra_data_sha256": sha256(args.extra_data) if args.extra_data else None,
        "train_count": len(train), "val_count": len(val), "train_multi_count": len(train_multi) * 2,
        "seed": args.seed, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "trainable_parameters": trainable,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    start = perf_counter()
    best = float("inf")
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        random.Random(args.seed + epoch).shuffle(order)
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
            if position % 40 == 0 or position == len(train):
                print(f"epoch {epoch}/{args.epochs} sample {position}/{len(train)} loss {loss.item():.4f}", flush=True)
        model.eval()
        with torch.inference_mode():
            val_loss = sum(model(**batch(item)).loss.item() for item in val) / len(val)
        model.train()
        if val_loss < best:
            best = val_loss
            model.save_pretrained(args.output / "best", safe_serialization=True)
            tokenizer.save_pretrained(args.output / "best")
        row = {"epoch": epoch, "train_loss": round(sum(losses) / len(losses), 6),
               "val_loss": round(val_loss, 6),
               "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2),
               "elapsed_seconds": round(perf_counter() - start, 1)}
        with (args.output / "epochs.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"epoch {epoch} 完成: {row}", flush=True)
    print(f"训练完成；最佳验证损失 {best:.4f}；适配器 {args.output / 'best'}", flush=True)


if __name__ == "__main__":
    main()
