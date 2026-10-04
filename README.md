# 中文工具调用模型训练与评测系统

这是一个围绕中文工具调用决策搭建的训练与评测项目。模型根据用户请求和本次可用工具，输出 `call`、`clarify` 或 `refuse` 三种 JSON 决策；服务端再检查参数和权限，决定是否执行工具。项目包含合成数据生成、LoRA 微调、逐题评分、本地推理接口和中文演示页面。

项目使用 **Qwen3-1.7B** 作为最终基座。训练得到的是 LoRA 适配器，模型权重和适配器文件均不包含在仓库中。工具调用均基于本地模拟数据，日历写入只发生在进程内演示环境。

## 结果概览

最终验收集包含 48 条合成请求。使用同一基座和相同推理设置比较原模型与 LoRA 原始输出；服务端策略的结果单独列出。

| 被测对象 | 决策全对数 | 说明 |
| --- | ---: | --- |
| Qwen3-1.7B 原模型 | 25/48 | 微调前的同基座对照 |
| Qwen3-1.7B + LoRA 原始输出 | 32/48 | 模型训练后的原始决策 |
| LoRA + 服务端保守策略 | 46/48 | 完整系统的最终决策 |

完整系统在这套题中的缺参场景为 **14/14**，越权场景为 **10/10**。策略层改写了 31 条模型候选决策，只会把明确风险降级为追问或拒绝，不会凭规则生成工具调用。**25→32 是模型微调的收益；32→46 是加入服务端策略后的系统收益。** 46/48 衡量的是决策评分，不是实际工具调用成功率，也不是生产环境准确率。

最终验收集在 LoRA 训练完成后、服务端策略实现和最终评分前冻结。初稿因近重复样本作废，正式版本为 `acceptance_v5b`。公开仓库保留三组汇总评分与错误分析，完整逐题推理输出留在本地实验目录；完整结论见 [`reports/final_project_delivery.md`](reports/final_project_delivery.md)。

## 系统结构

```text
用户请求 + 服务端提供的可用工具
              ↓
      Qwen3-1.7B + LoRA
              ↓
        原始 JSON 决策
              ↓
   保守策略：必要时追问或拒绝
              ↓
   执行闸门：格式、白名单、参数、确认
              ↓
       六个本地模拟工具
```

模型可以建议调用工具，但不能自行获得执行权限。写操作 `create_event` 需要服务端生成的一次性确认令牌；用户文本或模型输出中的“已确认”无效。页面和 API 均区分模型原始建议、策略后的最终决策及执行结果。

仓库中的六个工具包括文档检索、记录查询、算式计算、数据聚合、空闲查询和日程创建。`unit_convert`、`sort_records` 只用于离线未见工具评测，不属于本地服务的执行白名单。

## 训练与数据

- 最终 LoRA 训练使用 **408 条训练样本、96 条验证样本**；数据按模板组切分，避免简单替换数字或日期后跨集合泄漏。
- 用 Qwen3 的非思考模式对话模板构造输入，只对标准答案 token 计算交叉熵损失，输入部分的 label 设为 `-100`。
- LoRA 挂载在 `q_proj`、`v_proj`，rank 8、alpha 16、dropout 0.05；可训练参数约 **160 万，占总参数 0.093%**。
- batch size 1、梯度累积 8、学习率 `1e-4`、训练 2 轮；按验证损失选择第 2 轮适配器，测试集不参与选轮次。
- 额外训练过的三个 1.7B 候选未同时满足总分与安全门槛，未替换最终适配器；过程见 [`reports/model_optimization_followup.md`](reports/model_optimization_followup.md)。

训练采用 Transformers 加载模型和 tokenizer、PEFT 注入 LoRA，前向、反向、梯度累积和 AdamW 更新由 PyTorch 训练循环显式完成。模型、数据、提示词和配置的版本信息写入运行清单。

## 本地运行

已验证的环境主要是 Windows、Python 3.10、CUDA 版 PyTorch。先根据自己的显卡和驱动安装对应版本的 PyTorch，再安装其余依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-train.txt
python -m pip install -r api/requirements.txt
```

从 [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) 获取基座权重，放到 `models/Qwen3-1.7B`，或者把 [`configs/lora_1p7b_v0.json`](configs/lora_1p7b_v0.json) 的 `model_dir` 改成自己的模型目录；API 在启动时检查基座和适配器清单。`train/outputs/` 被 Git 忽略，不会覆盖原有实验输出。

```powershell
# 无 --train 时执行数据和显存预检，不更新参数
python train/train_lora.py --config configs/lora_1p7b_v0.json

# 正式训练；输出目录已存在时脚本会拒绝覆盖
python train/train_lora.py --config configs/lora_1p7b_v0.json --train

# 启动本地服务，浏览器访问 http://127.0.0.1:8001/
python -m uvicorn api.app:app --host 127.0.0.1 --port 8001 --workers 1
```

公开仓库不包含 LoRA 权重，因此首次启动服务前需要先完成训练，或自行提供与配置、清单一致的适配器。低显存设备可先运行不加载大模型的契约、数据和策略测试。

## 复核与测试

```powershell
python data/audit_dataset.py --check-frozen
python -m unittest api.test_decision_policy -v
python tools/test_execution.py
python eval/validate_contracts.py

# 以下测试会加载本地 Qwen3-1.7B 和训练好的适配器
python -m unittest api.test_api -v
```

最终交付前，8 项策略单测和 6 项真实模型 API 测试通过。数据冻结清单在 `data/datasets/`；最终集的原模型、LoRA 原始输出与完整系统评分汇总分别位于 `reports/acceptance_v5b_qwen1p7b_base/`、`reports/acceptance_v5b_qwen1p7b_lora_raw/` 和 `reports/acceptance_v5b_final_pipeline/`。

## 目录说明

| 目录 | 内容 |
| --- | --- |
| `tools/` | 六工具契约、本地模拟执行和服务端执行闸门 |
| `data/` | 标注规范、模板、数据生成与冻结审计 |
| `configs/` | LoRA 配置和 token 长度测量 |
| `train/` | 模型下载辅助脚本、数据编码和训练循环 |
| `eval/` | 基线、模型原始输出及完整系统评分 |
| `api/` | FastAPI 服务、保守策略、中文页面和测试 |
| `reports/` | 汇总评分、训练复验与最终结论 |

## 使用边界

这套数据主要是合成和定向设计的请求，适合比较同条件下的模型与策略，不代表真实业务流量。保守策略依赖明确的词和字段线索，仍可能漏判或误拦；本地工具和日历不接入真实企业系统。若要用于生产，还需要真实请求评测、身份认证、持久化、审计和更完整的安全测试。
