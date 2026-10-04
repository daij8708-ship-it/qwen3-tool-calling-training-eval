"""Score unchanged model outputs and server policy decisions side by side."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "data"), str(ROOT / "eval")]
from api.decision_policy import apply_policy
import audit_dataset as audit
import validate_contracts as scoring

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    ds = ROOT / "data" / "datasets"
    dataset_file = ds / f"{args.dataset}.json"
    input_file = ds / f"{args.dataset}.input.jsonl"
    manifest = json.loads((ds / f"{args.dataset}_manifest.json").read_text(encoding="utf-8"))
    if sha(dataset_file) != manifest["dataset_sha256"] or sha(input_file) != manifest["input_sha256"]:
        raise RuntimeError("冻结验收集哈希不匹配")
    cases = json.loads(dataset_file.read_text(encoding="utf-8"))["samples"]
    model_run = ROOT / args.model_run_dir
    raw_config = json.loads((model_run / "run_config.json").read_text(encoding="utf-8"))
    if raw_config["dataset_sha256"] != manifest["dataset_sha256"]:
        raise RuntimeError("模型逐题输出与指定验收集不一致")
    raw_rows = {row["id"]: row for row in map(json.loads, (model_run / "predictions.jsonl").read_text(encoding="utf-8").splitlines())}
    if set(raw_rows) != {case["id"] for case in cases}:
        raise RuntimeError("模型逐题输出数量或 ID 不匹配")
    base = audit.base_ctx()
    unseen_config = json.loads((ROOT / "data" / "unseen_tools" / "isolation_manifest.json").read_text(encoding="utf-8"))
    unseen, _, _ = audit.unseen_ctx(base, unseen_config)
    rows = []
    groups = defaultdict(list)
    for case in cases:
        raw = raw_rows[case["id"]]
        policy = apply_policy(case["user_request"], case["available_tools"], raw["parsed_decision"], case["context"])
        ctx = unseen if case["group"].startswith("unseen") else base
        verdict, notes = scoring.score_case_decision(case, policy.decision, ctx)
        row = {"id": case["id"], "group": case["group"], "user_request": case["user_request"],
               "expected_decision": case["expected_decision"], "raw_model_output": raw["raw_output"],
               "model_decision": raw["parsed_decision"], "model_verdict": raw["verdict"],
               "policy_decision": policy.decision, "policy_code": policy.code,
               "pipeline_verdict": verdict, "notes": notes}
        rows.append(row)
        groups["all"].append(row)
        groups[case["group"]].append(row)
    summary = {}
    for group, items in sorted(groups.items()):
        summary[group] = {"count": len(items),
            "model_verdict_counts": dict(Counter(x["model_verdict"] for x in items)),
            "pipeline_verdict_counts": dict(Counter(x["pipeline_verdict"] for x in items)),
            "model_correct": sum(x["model_verdict"] == "correct" for x in items),
            "pipeline_correct": sum(x["pipeline_verdict"] == "correct" for x in items),
            "policy_changed": sum(x["policy_code"] is not None for x in items)}
    run_dir = ROOT / args.run_dir
    if run_dir.exists(): raise RuntimeError(f"结果目录已存在，拒绝覆盖: {run_dir}")
    run_dir.mkdir(parents=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps({"dataset": args.dataset,
        "dataset_sha256": manifest["dataset_sha256"], "model_run_dir": str(model_run),
        "policy_sha256": sha(ROOT / "api" / "decision_policy.py")}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
