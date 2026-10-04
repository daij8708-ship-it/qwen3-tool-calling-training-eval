"""Fetch Qwen3-1.7B from Qwen's ModelScope repository when HF CDN is unavailable."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from modelscope import snapshot_download

MODEL_ID = "Qwen/Qwen3-1.7B"
MODEL_DIR = Path(r"models/Qwen3-1.7B")


def sha256(path):
    block_size = 4 * 1024 * 1024
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    snapshot_download(model_id=MODEL_ID, revision="master", local_dir=str(MODEL_DIR), max_workers=2)
    index = json.loads((MODEL_DIR / "model.safetensors.index.json").read_text(encoding="utf-8"))
    files = ["config.json", "tokenizer.json", "model.safetensors.index.json", *sorted(set(index["weight_map"].values()))]
    if any(not (MODEL_DIR / name).is_file() for name in files):
        raise RuntimeError("模型权重或配置下载不完整")
    manifest = {"repo_id": MODEL_ID, "source": "ModelScope", "revision": "master",
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "local_path": str(MODEL_DIR), "sha256": {name: sha256(MODEL_DIR / name) for name in files}}
    (MODEL_DIR / "download_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"下载完成: {MODEL_DIR}")


if __name__ == "__main__":
    main()
