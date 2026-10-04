# 本地 LoRA 推理服务

服务加载验证集选出的 Qwen3-1.7B LoRA 适配器，使用与原增强基线相同的提示词和解码参数。训练与评测见 [`../reports/qwen3_1p7b_result.md`](../reports/qwen3_1p7b_result.md)。
模型只产生决策建议；`api/decision_policy.py` 对明确缺参或越权请求保守地改为追问/拒绝，`tools/executor.py` 对最终决策、工具可用性和参数做服务端校验。规则不会生成工具调用。最终验收见 [`../reports/final_project_delivery.md`](../reports/final_project_delivery.md)。
本服务仅用于本地演示，日历状态保存在当前进程内，重启后恢复固定模拟数据。

## 启动

在项目根目录运行：

```powershell
& '.\.venv-train\Scripts\python.exe' -m uvicorn api.app:app --host 127.0.0.1 --port 8011 --workers 1
```

打开 `http://127.0.0.1:8011/` 使用中文演示页面；`/docs` 保留自动生成的接口文档。

- `GET /health`：模型版本、适配器指纹和服务端工具场景；
- `POST /decide`：输入 `user_request` 与 `profile`（`all/query/calculate/calendar`），分别返回模型原始输出、规则修正后的系统决策、修正原因、闸门结果和耗时；
- `POST /confirm`：仅接受 `/decide` 返回的一次性 `confirmation_token`，确认后由服务端重新通过闸门执行待确认的日程创建。
- `POST /route`：输入 `user_request`，用独立的五路 LoRA 适配器选择多智能体项目的第一层 Agent；只返回路由，不执行工具。多任务或适配器缺失时返回空路由，调用方交给原云端主调度。

服务端自己提供当前时间、工具列表和确认状态。`/decide` 不接受 `confirmed` 等自造字段；
模型文本说“已确认”不会改变写入状态。确认令牌 10 分钟有效，仅能用一次。
`/route` 的五路适配器需先运行 `train/train_agent_router.py` 生成；端口 8011 与多智能体知识库的 8001 分开。

## 集成验证

```powershell
& '.\.venv-train\Scripts\python.exe' -m unittest api.test_decision_policy api.test_api -v
```

其中 8 项直接检查规则边界，6 项使用真实本地模型，覆盖中文页面、文档查询、缺参追问、越权拒绝、写操作待确认、显式确认后写入、令牌重放拒绝和客户端伪造确认字段。
