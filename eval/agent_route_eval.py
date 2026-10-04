"""对本地 LoRA 的五路 Agent 决策做逐题评测。"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data" / "agent_route_eval_v0.json"
OUT = ROOT / "reports" / "agent_route_eval_v0"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011/route")
    parser.add_argument("--cases", type=Path, default=CASES)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))["cases"]
    rows = []
    for index, case in enumerate(cases, 1):
        request = urllib.request.Request(
            args.url,
            data=json.dumps({"user_request": case["query"]}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        actual = result.get("route") or "mixed"
        row = {"id": case["id"], "query": case["query"], "expected": case["expected"],
               "actual": actual, "correct": actual == case["expected"],
               "reason": result.get("reason"), "raw_model_output": result.get("raw_model_output"),
               "latency_seconds": result.get("latency_seconds")}
        rows.append(row)
        print(f"{index:02d}/{len(cases)} {case['id']} {case['expected']} -> {actual}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "predictions.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8"
    )
    groups = defaultdict(list)
    for row in rows:
        groups[row["expected"]].append(row)
    summary = {
        "dataset": args.cases.name,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": len(rows),
        "correct": sum(row["correct"] for row in rows),
        "by_route": {name: {"count": len(items), "correct": sum(row["correct"] for row in items)}
                     for name, items in sorted(groups.items())},
        "confusion": {name: dict(Counter(row["actual"] for row in items))
                      for name, items in sorted(groups.items())},
        "mean_latency_seconds": round(sum(row["latency_seconds"] for row in rows) / len(rows), 3),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
