# configs/

`measure_tokens.py` 用真实 Qwen3 tokenizer 测量四个集合的输入与标准答案长度，
结果见 `reports/token_budget_v0.json`。

`lora_v0.json` 固定 LoRA 首轮训练配置，并绑定基座模型 revision 与增强提示词基线。
预检结果见 `reports/lora_preflight_v0.json`。`max_seq_length=1280` 对当前训练/验证样本
均无截断。首轮训练已完成，验证损失与冻结测试结果见 `train/README.md` 和
`reports/lora_v0_eval/comparison.md`。
