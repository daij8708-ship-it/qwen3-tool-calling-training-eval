# Qwen3-0.6B 微调前基线对比

两组使用同一模型版本、冻结测试集、贪心解码和确定性评分器。test 含 120 条生成样本与 24 条手工案例；unseen_test 单独统计。

| 提示词 | 集合 | 样本 | 格式合法率 | 动作正确率（合法样本内） | 工具选择正确率（应调用样本内） | 全对率 | 待人工复核 |
|---|---|---:|---:|---:|---:|---:|---:|
| basic | test | 144 | 0.0% | — | 0.0% | 0.0% | 0 |
| basic | unseen_test | 45 | 0.0% | — | 0.0% | 0.0% | 0 |
| enhanced | test | 144 | 76.4% | 30.0% | 12.1% | 6.2% | 0 |
| enhanced | unseen_test | 45 | 51.1% | 52.2% | 14.8% | 17.8% | 0 |

## 增强提示词的类别结果

| 集合 | 类别 | 样本 | 全对 | 无效 | 判错 |
|---|---|---:|---:|---:|---:|
| test | adversarial | 11 | 0 | 1 | 10 |
| test | capability_boundary | 13 | 0 | 2 | 11 |
| test | irrelevant | 19 | 0 | 3 | 16 |
| test | missing_param | 21 | 5 | 6 | 10 |
| test | multi_intent | 10 | 0 | 8 | 2 |
| test | normal | 47 | 3 | 9 | 35 |
| test | over_authority | 10 | 0 | 0 | 10 |
| test | time_ambiguous | 13 | 1 | 5 | 7 |
| unseen_test | missing_param | 18 | 5 | 8 | 5 |
| unseen_test | normal | 27 | 3 | 14 | 10 |

## 判读

基础提示词没有产生合格决策；原始输出保存在 basic_predictions.jsonl。增强提示词改善了格式，但动作与参数错误仍多。
`manual_review` 为 0 仅表示本轮没有落入文本参数复核通道，不表示所有文本参数都正确。
每题的输入、原始输出、解析结果、标准答案、判定和原因见两个 predictions.jsonl 文件。
