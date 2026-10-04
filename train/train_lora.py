"""Preflight or train a response-only Qwen3-0.6B LoRA adapter.

Default action is a no-update preflight. Actual optimization requires --train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch
import peft
from peft import LoraConfig, TaskType, get_peft_model
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_data import ROOT, CONFIG_PATH, load_config, load_split, print_stats


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batch_of(sample):
    return {
        "input_ids": torch.tensor([sample["input_ids"]], device="cuda"),
        "attention_mask": torch.ones((1, len(sample["input_ids"])), dtype=torch.long, device="cuda"),
        "labels": torch.tensor([sample["labels"]], device="cuda"),
    }


def load_model(config):
    base = AutoModelForCausalLM.from_pretrained(
        config["model_dir"], local_files_only=True, torch_dtype=torch.float16
    ).to("cuda")
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
    )
    model = get_peft_model(base, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA 可训练参数 {trainable:,} / 总参数 {total:,} ({trainable / total:.3%})", flush=True)
    return model, trainable


def validation_loss(model, samples):
    model.eval()
    losses = []
    with torch.inference_mode():
        for sample in samples:
            losses.append(model(**batch_of(sample)).loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B LoRA 训练；默认只做预检")
    parser.add_argument("--train", action="store_true", help="真正更新权重并保存 LoRA 适配器")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="训练配置路径，默认首轮配置")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config, baseline = load_config(config_path)
    version = config.get("data_version", "v0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；请使用项目 .venv-train 环境")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_dir"], local_files_only=True)
    train_samples = load_split("train", tokenizer, config, baseline["enhanced_prompt"])
    val_samples = load_split("val", tokenizer, config, baseline["enhanced_prompt"])
    print_stats("train", train_samples)
    print_stats("val", val_samples)
    output_dir = ROOT / config["output_dir"]
    if args.train and output_dir.exists():
        raise RuntimeError(f"训练输出目录已存在，拒绝覆盖: {output_dir}")

    model, trainable = load_model(config)
    torch.cuda.reset_peak_memory_stats()
    if not args.train:
        longest = max(train_samples, key=lambda sample: len(sample["input_ids"]))
        model.train()
        probe_optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=config["learning_rate"], weight_decay=config["weight_decay"],
        )
        probe_scaler = torch.amp.GradScaler("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**batch_of(longest)).loss
        probe_scaler.scale(loss).backward()
        probe_scaler.unscale_(probe_optimizer)
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        # 不调用 optimizer.step，也不保存权重。
        torch.cuda.synchronize()
        val_probe_loss = validation_loss(model, val_samples[:2])
        peak = round(torch.cuda.max_memory_allocated() / 1024**2)
        report = {
            "status": "preflight_only_no_optimizer_step",
            "base_revision": config["base_revision"],
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "max_train_tokens": max(len(sample["input_ids"]) for sample in train_samples),
            "max_val_tokens": max(len(sample["input_ids"]) for sample in val_samples),
            "longest_train_id": longest["id"],
            "longest_sample_loss": round(loss.item(), 6),
            "two_sample_val_loss": round(val_probe_loss, 6),
            "peak_backward_vram_mib": peak,
            "trainable_parameters": trainable,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "config_sha256": sha256(config_path),
        }
        report_path = ROOT / config.get("preflight_report", "reports/lora_preflight_v0.json")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"预检通过: {longest['id']}，loss {loss.item():.4f}，反向传播显存峰值 {peak} MiB")
        print(f"预检报告: {report_path}")
        print("没有执行优化器更新；尚未训练。")
        return

    output_dir.mkdir(parents=True)
    run_info = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_revision": config["base_revision"],
        "model_manifest_sha256": sha256(Path(config["model_dir"]) / "download_manifest.json"),
        "train_config_sha256": sha256(config_path),
        "train_dataset_sha256": sha256(ROOT / "data" / "datasets" / f"train_{version}.json"),
        "val_dataset_sha256": sha256(ROOT / "data" / "datasets" / f"val_{version}.json"),
        "baseline_prompt_sha256": hashlib.sha256(baseline["enhanced_prompt"].encode("utf-8")).hexdigest(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "trainable_parameters": trainable,
        "config": config,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    best_val = float("inf")
    log_path = output_dir / "epochs.jsonl"
    start = perf_counter()
    model.train()
    for epoch in range(1, config["epochs"] + 1):
        order = list(range(len(train_samples)))
        random.Random(config["seed"] + epoch).shuffle(order)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        accum = config["gradient_accumulation_steps"]
        for position, index in enumerate(order, 1):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch_of(train_samples[index])).loss
            losses.append(loss.item())
            scaler.scale(loss / accum).backward()
            if position % accum == 0 or position == len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if position % 40 == 0 or position == len(order):
                print(f"epoch {epoch}/{config['epochs']} sample {position}/{len(order)} loss {loss.item():.4f}", flush=True)
        val_loss = validation_loss(model, val_samples)
        epoch_dir = output_dir / f"epoch_{epoch}"
        model.save_pretrained(epoch_dir, safe_serialization=True)
        if val_loss < best_val:
            best_val = val_loss
            best_dir = output_dir / "best"
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
        record = {
            "epoch": epoch,
            "train_loss": round(sum(losses) / len(losses), 6),
            "val_loss": round(val_loss, 6),
            "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 1024**2),
            "elapsed_seconds": round(perf_counter() - start, 1),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"epoch {epoch} 完成: {record}", flush=True)
    print(f"训练完成；最佳验证损失 {best_val:.4f}；适配器 {output_dir / 'best'}")


if __name__ == "__main__":
    main()
