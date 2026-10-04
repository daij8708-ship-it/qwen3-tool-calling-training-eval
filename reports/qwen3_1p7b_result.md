# 第二轮优化交付：Qwen3-1.7B LoRA

## 交付结论

第一轮 Qwen3-0.6B LoRA 在旧测试集为 32/144、未见工具集为 21/45，在新验收集为 13/24，尚不足以作为最终模型。第二轮原始 0.6B 方案退步，因此另用 Qwen3-1.7B 进行独立训练，并保留原始失败日志。最终交付 `train/outputs/lora_1p7b_v0/best`，本地 API 已切换到这个适配器。后续三次 1.7B 优化候选均未达到事前门槛；扩充验收及复盘见[模型能力优化复验](model_optimization_followup.md)。

新验收集在 1.7B 训练前冻结，24 条均有逐题预测和判分。训练轮次只依据验证损失选择，未依据验收集选轮次。

| 模型 | 新验收集全对 | 已知工具缺参 | 未见工具正常调用 | 未见工具缺参 |
| --- | ---: | ---: | ---: | ---: |
| 0.6B 原模型 | 5/24 | 见逐题报告 | 见逐题报告 | 见逐题报告 |
| 0.6B 首轮 LoRA | 13/24 | 见逐题报告 | 见逐题报告 | 见逐题报告 |
| 1.7B 原模型 | 12/24 | 2/6 | 5/5 | 0/3 |
| **1.7B LoRA** | **18/24** | **3/6** | **5/5** | **1/3** |

1.7B 原模型和 LoRA 使用同一权重底座、提示词、验收集与解码设置，6 条净增益可归于这次适配器；与 0.6B 的差值同时包含模型规模变化，不全部归因于微调。

## 训练与诊断

- 训练/验证：408/96 条；回答 token 参与损失，输入部分屏蔽。LoRA `q_proj/v_proj`，rank 8，alpha 16；可训练参数 1,605,632 / 1,722,180,608（0.093%）。
- 两轮验证损失：0.683448、**0.639254**。选第 2 轮，`best` 与 `epoch_2` 适配器哈希一致。训练耗时 261.8 秒，峰值显存约 5086 MiB。
- 旧冻结集仅作诊断性对照：最终模型 test **54/144**、unseen_test **27/45**；0.6B 首轮分别为 32/144、21/45。该旧集参与过先前问题定位，不能作为独立最终验收证据。
- 仍有明显短板：旧集多意图 0/10、未见工具缺参 1/18；新验收集未见工具缺参 1/3。不能声称复杂请求已经可靠解决。

## 可复核材料

- 预设门槛：[`qwen3_1p7b_experiment_plan.md`](qwen3_1p7b_experiment_plan.md)。四项门槛均达到：18/24 ≥16/24，相对自身基座 +6 ≥3，缺参 3/6，未见正常 5/5，API 四项测试通过。
- 训练配置：[`../configs/lora_1p7b_v0.json`](../configs/lora_1p7b_v0.json)；公开的逐轮损失与关键哈希：[`qwen3_1p7b_training_metrics.json`](qwen3_1p7b_training_metrics.json)。本地适配器和完整运行清单不随仓库发布。
- 验收集及哈希：[`../data/datasets/acceptance_v1.json`](../data/datasets/acceptance_v1.json)、[`../data/datasets/acceptance_v1_manifest.json`](../data/datasets/acceptance_v1_manifest.json)。
- 逐题输出与评分：[`acceptance_v1_qwen1p7b_base/`](acceptance_v1_qwen1p7b_base/)、[`acceptance_v1_qwen1p7b_lora/`](acceptance_v1_qwen1p7b_lora/)、[`qwen1p7b_lora_diagnostic/`](qwen1p7b_lora_diagnostic/)。
- 旧第二轮失败证据：[`lora_v1_eval/round_comparison.md`](lora_v1_eval/round_comparison.md)、[`lora_v1b_result.md`](lora_v1b_result.md)。

本地服务只把模型输出当成建议；工具白名单、参数校验、日历写入的单次确认均在服务端执行。四项集成测试使用真实 1.7B 模型通过。
