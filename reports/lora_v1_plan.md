# 第二轮 LoRA 实验：数据对照与训练交接

状态：**第二轮训练和评测已完成**。本文保留训练前预设的实验方案与交接步骤；实测结论见 [`lora_v1_eval/round_comparison.md`](lora_v1_eval/round_comparison.md)。

## 问题与假设

首轮普通冻结测试集的 `missing_param` 从增强提示词基线 5/21 全对降为 LoRA 1/21；未见工具测试集从 5/18 降为 1/18。这些测试结果只用来识别项目局限，不用逐题改训练标签。

为决定第二轮改动，另用首轮适配器在**原验证集**推理：总计 28/96 全对，其中 `missing_param` 0/9。9 条缺参题均是 `aggregate_data` 没有说明数据集，模型全部输出 `call`（其中 6 条格式无效）。逐题记录见 [`lora_v0_validation/predictions.jsonl`](lora_v0_validation/predictions.jsonl)。

本轮假设：把“同一工具缺少必填值”和“明确给出该值”做成成对监督样本，可以改善模型擅自补参数的问题。保持基座、提示词、LoRA 结构、优化器、学习率、轮数和解码规则不变，以便把主要变化归因于训练数据。

## 数据改动

在 v0 上新增 24 对训练样本、8 对验证样本，每对一条 `clarify` 和一条 `call`；四类工具分别是 `search_documents`、`calculate`、`create_event`、`aggregate_data`。其中 `aggregate_data` 的缺参项是数据集，直接对应原验证集的失误。其余三类帮助检查这种追问判断是否能跨工具保持，而不是只记住一个聚合统计说法。

| 集合 | v0 | 本轮新增 | v1 |
|---|---:|---:|---:|
| train | 408 | 48 | 456 |
| val | 96 | 16 | 112 |
| test | 144 | 0 | 仍用冻结 v0 |
| unseen_test | 45 | 0 | 仍用冻结 v0 |

设计文件为 [`data/templates/contrastive_v1.json`](../data/templates/contrastive_v1.json)，构造脚本为 [`data/build_v1.py`](../data/build_v1.py)。新样本逐条通过案例契约和标准答案评分检查；与其他集合做请求近重复检查；冻结测试集哈希在构造前后均通过校验。原有 v0 数据、模板和冻结清单均未修改。

在新的 112 条验证集上，**首轮适配器**作为第二轮的预先对照得到 35/112 全对（31.25%）。其中新增 16 条为 7/16 全对，新增缺参 8 条为 1/8 全对。这是验证集对照数字，不应与冻结测试集的 22.2% 混称同一个指标。记录见 [`lora_v0_validation_v1/`](lora_v0_validation_v1/)。

## 训练配置与预检

[`configs/lora_v1.json`](../configs/lora_v1.json) 除数据版本、输出目录和预检报告路径外，与首轮参数相同：rank 8、alpha 16、dropout 0.05、`q_proj/v_proj`、学习率 2e-4、batch 1、梯度累积 8、2 轮、最大序列长度 1280。

预检已通过：train 456 条、val 112 条；最长分别 944/885 token，无截断；最长样本前向与反向传播成功，峰值 PyTorch 已分配显存 2850 MiB。**预检没有执行优化器更新，也没有创建第二轮适配器。** 记录见 [`lora_preflight_v1.json`](lora_preflight_v1.json)。

## 用户亲自执行的训练步骤

在项目根目录运行：

```powershell
& '.\.venv-train\Scripts\python.exe' '.\train\train_lora.py' --config '.\configs\lora_v1.json' --train
```

正常结束时应看到两轮的 `train_loss`、`val_loss`，以及 `train/outputs/lora_v1/best`。程序按**最低验证损失**自动保存最佳适配器；不会覆盖首轮 `lora_v0`。输出目录若已存在会拒绝重训，避免误覆盖。请保留完整终端输出；不要只依据某一步的瞬时 loss 判断效果。

训练后由助手进行：

1. 核对两轮日志、数据和配置哈希、最佳轮次及适配器文件。
2. 用第二轮适配器在 112 条 v1 验证集逐题评测，与首轮适配器在**同一集合**的 35/112 对照，并分别看新增/原有缺参案例和正常调用案例。验证损失仍是 checkpoint 的预设选择规则，决策指标是结果分析。
3. 冻结校验通过后，使用同一冻结测试集和评分器做最终对照；同时报告总分、追问类别、格式错误及退步，不因不理想而删除结果。

第二轮是否真正改善追问，目前未知。新增数据规模有限，且原验证集缺参题集中在一个聚合统计模板组；即使该组改善，也要观察其他工具、普通测试集与未见工具测试集是否一致。
