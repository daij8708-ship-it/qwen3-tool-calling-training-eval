"""Download a pinned Qwen3-1.7B model to the local models directory."""

import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
MODEL_DIR = Path(r"models/Qwen3-1.7B")
CACHE_DIR = Path(r"models/hf_cache")


def main():
    snapshot_download(repo_id=REPO_ID, revision=REVISION,
                      local_dir=str(MODEL_DIR), cache_dir=str(CACHE_DIR))
    needed = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    if any(not (MODEL_DIR / name).is_file() for name in needed):
        raise RuntimeError("模型配置、分词器或权重索引下载不完整")
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = set(index["weight_map"].values())
    if any(not (MODEL_DIR / name).is_file() for name in shards):
        raise RuntimeError("模型权重分片下载不完整")
    manifest = {"repo_id": REPO_ID, "revision": REVISION, "local_path": str(MODEL_DIR),
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "shards": sorted(shards)}
    (MODEL_DIR / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"下载完成: {REPO_ID}@{REVISION} -> {MODEL_DIR}")


if __name__ == "__main__":
    main()
