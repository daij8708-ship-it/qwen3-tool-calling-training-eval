"""Model inference with server-owned tool availability and write confirmation."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "data", ROOT / "tools"):
    sys.path.insert(0, str(extra))

import executor  # noqa: E402
import generate_dataset as generator  # noqa: E402
import simulator  # noqa: E402
from api.decision_policy import apply_policy  # noqa: E402

PROFILES = {
    "all": ["search_documents", "get_record", "calculate", "aggregate_data", "check_availability", "create_event"],
    "query": ["search_documents", "get_record"],
    "calculate": ["calculate", "aggregate_data"],
    "calendar": ["check_availability", "create_event"],
}
PENDING_SECONDS = 600


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ModelService:
    def __init__(self, now_provider=None):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，无法加载本地 LoRA 推理服务")
        self.torch = torch
        self.now_provider = now_provider or (lambda: datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M"))
        train_dir = ROOT / "train" / "outputs" / "lora_1p7b_v0"
        baseline = read_json(ROOT / "reports" / "baselines" / "qwen3_0p6b_pre_lora_v0" / "run_config.json")
        manifest = read_json(train_dir / "run_manifest.json")
        adapter_dir = train_dir / "best"
        adapter_hash = file_hash(adapter_dir / "adapter_model.safetensors")
        epochs = [json.loads(line) for line in (train_dir / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
        best_epoch = min(epochs, key=lambda row: row["val_loss"])["epoch"]
        if adapter_hash != file_hash(train_dir / f"epoch_{best_epoch}" / "adapter_model.safetensors"):
            raise RuntimeError("服务加载的适配器不是验证集选出的轮次")
        config = manifest["config"]
        model_dir = Path(config["model_dir"])
        if not config.get("prompt_source_only") and baseline["model_revision"] != manifest["base_revision"]:
            raise RuntimeError("LoRA 基座版本与基线不一致")
        if config.get("prompt_source_only") and file_hash(model_dir / "download_manifest.json") != manifest["model_manifest_sha256"]:
            raise RuntimeError("LoRA 基座本地清单与训练时不一致")
        prompt_hash = hashlib.sha256(baseline["enhanced_prompt"].encode("utf-8")).hexdigest()
        if prompt_hash != manifest["baseline_prompt_sha256"]:
            raise RuntimeError("LoRA 训练提示词与基线不一致")
        self.model_revision = manifest["base_revision"]
        self.model_name = model_dir.name
        self.adapter_hash = adapter_hash
        self.prefix = baseline["enhanced_prompt"]
        self.max_new_tokens = baseline["max_new_tokens"]
        self.docs = generator.load_tool_docs()
        self.store = simulator.Store.load()
        self.gate = executor.Executor()
        self.pending = {}
        self.lock = threading.RLock()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, torch_dtype=torch.float16
        ).to("cuda")
        self.model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        self.model.eval()
        self.model.generation_config = GenerationConfig.from_model_config(self.model.config)

    def decide(self, user_request, profile):
        if profile not in PROFILES:
            raise ValueError("未知工具场景")
        with self.lock:
            now = self.now_provider()
            available = PROFILES[profile]
            sample = {
                "user_request": user_request,
                "available_tools": available,
                "context": {"now": now, "user_confirmed_write": False},
            }
            prompt = self.prefix + "\n\n" + generator.render_input(sample, self.docs)
            encoded = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_tensors="pt",
            ).to("cuda")
            self.torch.cuda.synchronize()
            start = time.perf_counter()
            with self.torch.inference_mode():
                generated = self.model.generate(
                    encoded, attention_mask=self.torch.ones_like(encoded),
                    max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            self.torch.cuda.synchronize()
            latency = time.perf_counter() - start
            raw = self.tokenizer.decode(generated[0][encoded.shape[-1]:], skip_special_tokens=True).strip()
            try:
                model_decision = json.loads(raw)
            except json.JSONDecodeError:
                model_decision = None
            policy = apply_policy(user_request, available, model_decision, sample["context"])
            decision = policy.decision
            report = self.gate.run(
                decision, executor.Trusted(now, False, available), store=self.store
            )
            token = None
            if (report["result"] or {}).get("status") == "pending_confirmation":
                token = uuid.uuid4().hex
                self.pending[token] = {
                    "decision": decision, "available_tools": available,
                    "expires_at": time.monotonic() + PENDING_SECONDS,
                }
            return {
                "model_revision": self.model_revision,
                "profile": profile,
                "server_now": now,
                "raw_model_output": raw,
                "model_decision": model_decision,
                "policy_code": policy.code,
                "decision": decision,
                "gate": report,
                "confirmation_token": token,
                "latency_seconds": round(latency, 3),
            }

    def confirm(self, token):
        with self.lock:
            pending = self.pending.pop(token, None)
            if pending is None or time.monotonic() > pending["expires_at"]:
                return None
            now = self.now_provider()
            report = self.gate.run(
                pending["decision"],
                executor.Trusted(now, True, pending["available_tools"]),
                store=self.store,
            )
            return {"server_now": now, "gate": report}
