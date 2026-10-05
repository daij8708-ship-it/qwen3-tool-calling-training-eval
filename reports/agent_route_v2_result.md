# 五路 Agent 路由 LoRA v2：多意图修正与验收

## 为什么上一版输给云端

旧路由适配器 v0 用 120 条单任务种子扩成 240 条训练样本，训练目标始终是从五个 Agent 里选一个。多意图没有作为模型输出类别训练，而是由服务端少量连接词规则直接交还云端。2026-10-04 固定的 80 条新题中，原云端主调度路由 Exact Match 为 79/80，本地 v0 为 69/80；本地的 8 条明确分诊遗漏均是跨 Agent 多任务只处理了其中一个。另有 3 条本地请求在下游云端专业 Agent 阶段异常，路由结束记录不完整，不应算作模型本身的 3 次错误。云端主调度能连续调用多个 Agent，本地 1.7B 模型及小规模定向数据则需要显式学习“交还云端”的边界。

## 改动

v1 从 v0 路由 LoRA 继续训练，新增 `{"action":"delegate","reason":"multi_task"}` 作为跨 Agent 多任务输出；仍保留五路 `call` JSON 和服务端工具名白名单校验。新增 50 条跨 Agent 多任务训练种子及 20 条同领域多步骤反例，每条训练种子做两种表述，结合原单任务种子得到 380 条训练样本。v1 的新冻结集首次评测为 54/60，其中跨 Agent 多任务 20/20；其余 6 错包括 5 条同领域请求过度回退及 1 条工单记录错分。

v2 从 v1 继续训练，再加 50 条同领域多步骤训练种子、10 条跨 Agent 多任务种子，以及独立验证样本。v2 共 500 条训练、50 条验证，学习率 `1e-5`，训练 2 轮；第 1/2 轮验证损失为 `0.006137` / `0.004810`，选第 2 轮。最佳适配器 SHA-256：`04dbaf8456b155f233a20730591b1e50c34a29b11bd41dc96c575a8130416dc8`。模型服务 `/health` 返回 `route_adapter_version=v2`；`/route` 保持原有 `route` 契约，多意图返回 `null` 交云端主调度。原六工具 `/decide` 适配器未被覆盖。

## 验收与限制

| 题集 | 结果 | 解释 |
|---|---:|---|
| v2 训练前冻结的 50 条新题，首次运行 | 48/50（96%） | 20 条普通单任务 20/20，15 条跨 Agent 多任务 14/15，15 条同领域多步骤 14/15；这是 v2 的独立成绩 |
| v1 训练前冻结的 60 条题，用 v2 重跑 | 57/60 | v1 首次为 54/60；v2 已参考 v1 首次失误补充训练，此项只能作回归 |
| 上轮 80 条端到端题，仅测试 v2 本地 `/route` | 80/80 | 旧题已用于定位问题，此项只能作回归；其中原来漏判的 8 条多意图均修复 |
| 上轮 20 条多意图真实链路，v2 接入 ITS | 路由 20/20；任务 17/20 | 旧题回归；原云端路由也是 20/20、任务 17/20，v0 为路由 12/20、任务 12/20 |

v2 新冻结题的两处错误：一条跨 Agent 的祝福加导航请求只选了服务站；一条同领域的“查昨天比分，再查今天比赛预告”被交还云端。前者会漏掉任务，后者通常能由云端完成但失去本地路由的用量收益。v1 冻结题回归还有 2 条跨 Agent 漏判和 1 条工单记录误分流。不能据 48/50 或旧题 80/80 宣称已达到云端主调度的总体准确率。

真实链路的 3 条任务失败 H065、H067、H073 均记录了正确的两个路由，但答案不可用；测试时联网搜索 MCP 不可用。这说明第一层分诊修正不能替代下游搜索可靠性。真实链路仍是顺序运行的旧题回归，不是第二次独立 A/B。

## 复现

先训练原六工具 `lora_1p7b_v0` 和五路路由 v0，再运行：

```powershell
& '.\.venv-train\Scripts\python.exe' '.\train\train_agent_router_v1.py'
& '.\.venv-train\Scripts\python.exe' '.\train\train_agent_router_v1.py' --source-adapter '.\train\outputs\agent_router_v1\best' --extra-data '.\data\agent_route_boundaries_v2.json' --output '.\train\outputs\agent_router_v2' --learning-rate 0.00001 --epochs 2 --seed 20261006
& '.\.venv-train\Scripts\python.exe' -m uvicorn api.app:app --host 127.0.0.1 --port 8011 --workers 1
```

每个训练脚本都拒绝覆盖已有输出目录。公开源码不含基座模型与 LoRA 权重。v2 首次评测逐题原始输出在 `reports/agent_route_frozen_v2_first/predictions.jsonl`；旧题回归在 `reports/agent_route_old80_v2_regression/`；多智能体真实链路报告在 ITS 项目 `docs/evaluation/results/router_v2_multi20_dev_20261005_114645.json`。
