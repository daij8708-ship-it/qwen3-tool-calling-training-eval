"""Download and pin the Qwen3-0.6B baseline model to the local models directory."""

import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


REPO_ID = "Qwen/Qwen3-0.6B"
MODEL_DIR = Path(r"models/Qwen3-0.6B")
CACHE_DIR = Path(r"models/hf_cache")


def main() -> None:
    api = HfApi()
    revision = api.model_info(REPO_ID).sha
    if not revision:
        raise RuntimeError("无法取得模型 revision，下载未开始")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"模型: {REPO_ID}")
    print(f"固定版本: {revision}")
    print(f"保存位置: {MODEL_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        revision=revision,
        local_dir=str(MODEL_DIR),
        cache_dir=str(CACHE_DIR),
    )

    required = ["config.json", "tokenizer.json", "model.safetensors"]
    missing = [name for name in required if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"下载不完整，缺少: {', '.join(missing)}")

    metadata = {
        "repo_id": REPO_ID,
        "revision": revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(MODEL_DIR),
    }
    (MODEL_DIR / "download_manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("下载完成；版本已记录在 download_manifest.json")


if __name__ == "__main__":
    main()
