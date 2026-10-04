# LoRA 训练说明

最终实验使用 Qwen3-1.7B，配置在 [`../configs/lora_1p7b_v0.json`](../configs/lora_1p7b_v0.json)。模型权重和适配器不随仓库发布。运行前需下载基座模型，并把配置中的 `model_dir` 改成自己的本地目录；模型目录还需包含下载脚本生成的 `download_manifest.json`，用于训练与服务启动时核对文件清单。

训练环境需要 CUDA 版 PyTorch、Transformers 和 PEFT。主仓库 [`requirements-train.txt`](../requirements-train.txt) 列出其余依赖。训练脚本默认只执行预检，真正更新参数需显式传入 `--train`：

```powershell
python train/train_lora.py --config configs/lora_1p7b_v0.json
python train/train_lora.py --config configs/lora_1p7b_v0.json --train
```

预检会检查训练/验证样本的对话模板、答案掩码、长度与最长样本的前后向显存，不执行优化器更新。正式训练对输入 token 的 label 使用 `-100`，只对标准答案计算损失；batch size 1，梯度累积 8，训练 2 轮，并按验证损失选择适配器。输出目录已存在时脚本拒绝覆盖，复现实验前需设置一个新的 `output_dir`。

最终实验的训练集为 408 条、验证集为 96 条。两轮验证损失分别为 0.683448、0.639254，因此选择第 2 轮。逐轮损失、显存、配置与文件哈希见 [`../reports/qwen3_1p7b_training_metrics.json`](../reports/qwen3_1p7b_training_metrics.json)。模型对照和系统验收结果见 [`../reports/final_project_delivery.md`](../reports/final_project_delivery.md)。
