"""Compare first-layer route decisions without running specialist Agents or tools.

Cloud mode keeps the ITS orchestrator's prompt, model, settings and tool schemas,
but replaces each specialist tool body with a short, side-effect-free result. The
orchestrator can therefore make further routing decisions for multi-intent cases.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ITS = ROOT.parent / "ITS-Multi-Agent-System-main"
TOOL_ROUTES = {
    "consult_technical_expert": "technical",
    "search_realtime_information": "realtime",
    "query_service_station_and_navigate": "service",
    "manage_support_ticket": "ticket",
    "handle_general_conversation": "general",
}


def load_cloud_agent(its_project: Path):
    app_dir = its_project / "backend" / "app"
    if not app_dir.is_dir():
        raise RuntimeError(f"找不到 ITS 后端: {app_dir}")
    sys.path.insert(0, str(app_dir))
    from multi_agent.orchestrator_agent import orchestrator_agent

    agent = copy.copy(orchestrator_agent)
    calls: list[str] = []
    stub_tools = []
    for original in orchestrator_agent.tools:
        if original.name not in TOOL_ROUTES:
            raise RuntimeError(f"未知调度工具: {original.name}")
        tool = copy.copy(original)
        route = TOOL_ROUTES[tool.name]

        async def record_only(_ctx, _input: str, selected=route) -> str:
            calls.append(selected)
            return f"{selected} 专家已处理当前子任务。"

        tool.on_invoke_tool = record_only
        stub_tools.append(tool)
    agent.tools = stub_tools
    return agent, calls


async def cloud_decision(agent, calls: list[str], query: str) -> dict:
    from agents import Runner
    from agents.run import RunConfig

    calls.clear()
    started = time.perf_counter()
    error = None
    usage = None
    try:
        result = await Runner.run(
            agent, input=query, context=query, max_turns=5,
            run_config=RunConfig(tracing_disabled=True),
        )
        raw_usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        if raw_usage is not None:
            usage = {
                "requests": int(getattr(raw_usage, "requests", 0) or 0),
                "input_tokens": int(getattr(raw_usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(raw_usage, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(raw_usage, "total_tokens", 0) or 0),
            }
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    return {
        "routes": list(dict.fromkeys(calls)), "calls": list(calls),
        "reason": "orchestrator", "error": error, "usage": usage,
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


def local_decision(url: str, query: str) -> dict:
    started = time.perf_counter()
    try:
        request = Request(
            url,
            data=json.dumps({"user_request": query}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        route = payload.get("route")
        return {
            "routes": [route] if route else [], "calls": [route] if route else [],
            "reason": payload.get("reason"), "error": None,
            "adapter_version": payload.get("adapter_version"),
            "model_latency_ms": round(float(payload.get("latency_seconds") or 0) * 1000, 3),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
    except Exception as exc:
        return {"routes": [], "calls": [], "reason": None,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.perf_counter() - started) * 1000, 3)}


def score(case: dict, actual: dict, mode: str) -> bool:
    expected = set(case["expected_routes"])
    if mode == "local" and len(expected) > 1:
        return (not actual["error"] and not actual["routes"]
                and actual.get("reason") == "model_multi_task")
    return not actual["error"] and set(actual["routes"]) == expected


def summary(rows: list[dict], mode: str, cases_hash: str) -> dict:
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["case"]["category"]].append(row)
    latencies = sorted(row["actual"]["latency_ms"] for row in rows)
    def pct(p: float) -> float | None:
        if not latencies:
            return None
        pos = (len(latencies) - 1) * p
        low = int(pos)
        high = min(low + 1, len(latencies) - 1)
        return round(latencies[low] + (latencies[high] - latencies[low]) * (pos - low), 3)
    usages = [row["actual"].get("usage") for row in rows]
    usages = [usage for usage in usages if usage]
    return {
        "mode": mode, "cases_sha256": cases_hash,
        "count": len(rows), "correct": sum(row["correct"] for row in rows),
        "by_category": {
            name: {"count": len(items), "correct": sum(item["correct"] for item in items)}
            for name, items in sorted(by_category.items())
        },
        "latency_ms": {"mean": round(statistics.fmean(latencies), 3),
                       "p50": pct(0.5), "p95": pct(0.95)},
        "cloud_usage": {
            "available_rows": len(usages),
            "requests": sum(item["requests"] for item in usages),
            "input_tokens": sum(item["input_tokens"] for item in usages),
            "output_tokens": sum(item["output_tokens"] for item in usages),
            "total_tokens": sum(item["total_tokens"] for item in usages),
        } if mode == "cloud" else None,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="只测第一层路由，不运行专业 Agent")
    parser.add_argument("--mode", choices=("cloud", "local"), required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--its-project", type=Path, default=DEFAULT_ITS)
    parser.add_argument("--local-url", default="http://127.0.0.1:8011/route")
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    cases_hash = hashlib.sha256(raw).hexdigest()
    cases = json.loads(raw)
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("题库必须是非空 JSON 数组")
    ids = [case["id"] for case in cases]
    if len(ids) != len(set(ids)):
        raise RuntimeError("题号重复")
    agent, calls = load_cloud_agent(args.its_project) if args.mode == "cloud" else (None, None)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with (args.out / "rows.jsonl").open("w", encoding="utf-8") as stream:
        for index, case in enumerate(cases, 1):
            actual = (await cloud_decision(agent, calls, case["query"])
                      if args.mode == "cloud" else
                      await asyncio.to_thread(local_decision, args.local_url, case["query"]))
            row = {"case": case, "actual": actual,
                   "correct": score(case, actual, args.mode)}
            rows.append(row)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{index}/{len(cases)} {case['id']} expected={case['expected_routes']} "
                  f"actual={actual['routes']} reason={actual.get('reason')} "
                  f"{'OK' if row['correct'] else 'FAIL'}", flush=True)
    report = summary(rows, args.mode, cases_hash)
    (args.out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
